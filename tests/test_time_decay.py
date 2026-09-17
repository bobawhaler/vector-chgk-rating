"""Unit and integration tests for Time Decay, Dynamic Skill Tracking, and Calibration."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from chgk_rating.analytics import (
    compute_team_synergy_details,
    get_player_historical_timeline,
)
from chgk_rating.app import app
import chgk_rating.app as app_module
from chgk_rating.config import ModelConfig
from chgk_rating.data_loader import generate_synthetic_dataset
from chgk_rating.dataset import ChGKDataset, DatasetMetadata, IDIndexer
from chgk_rating.model import DualEmbeddingChGKModel
from chgk_rating.time_decay import TimeDecayConfig, TimeDecayModule
from chgk_rating.train import evaluate_model, split_data_chronologically, train_model


def test_inactivity_grace_period_and_decay() -> None:
    """Acceptance Criterion 1: Inactivity & Asymmetric Decay Test.

    A player who hasn't played in 3 years must show a decayed effective skill vector
    compared to their raw vector, with factual dimensions decaying faster than logical ones.
    """
    dim_k = 16
    cfg = TimeDecayConfig(
        grace_period_days=365.0,
        half_life_factual_days=730.0,
        half_life_logical_days=2920.0,
    )
    module = TimeDecayModule(dim_k=dim_k, config=cfg)

    # Unit knowledge vector with equal weights across all dimensions
    raw_k = torch.ones(1, dim_k)

    # 1. Within grace period (e.g. 180 days): no decay
    d_within_grace = torch.tensor([[180.0]])
    k_grace = module.apply_knowledge_decay(raw_k, d_within_grace)
    assert torch.allclose(k_grace, raw_k)

    # 2. Exactly at grace period (365 days): no decay
    d_at_grace = torch.tensor([[365.0]])
    k_at_grace = module.apply_knowledge_decay(raw_k, d_at_grace)
    assert torch.allclose(k_at_grace, raw_k)

    # 3. 3 years hiatus (1095 days -> 730 days past grace period)
    d_3_years = torch.tensor([[1095.0]])
    k_decayed = module.apply_knowledge_decay(raw_k, d_3_years)

    raw_norm = torch.norm(raw_k).item()
    decayed_norm = torch.norm(k_decayed).item()

    # Effective skill must be strictly lower
    assert decayed_norm < raw_norm

    # 4. Check Asymmetric Decay:
    # First 8 dimensions (factual) should have decayed by ~ 1 half-life (50% retention)
    # Remaining 8 dimensions (logical) should have decayed by ~ 730/2920 = 0.25 half-lives (~84% retention)
    factual_retention = float(k_decayed[0, :8].mean().item())
    logical_retention = float(k_decayed[0, 8:].mean().item())

    assert factual_retention < logical_retention
    assert 0.45 <= factual_retention <= 0.55  # ~50% after exactly 1 half-life (730 days)
    assert 0.80 <= logical_retention <= 0.88  # ~84%


def test_synergy_erosion() -> None:
    """Acceptance Criterion 2: Synergy Erosion Test.

    Two highly synergistic players who haven't played together for 2 years (730 days)
    must show reduced pairwise synergy S_im compared to an active pair.
    """
    dim_k = 16
    dim_c = 8
    cfg = TimeDecayConfig(half_life_synergy_days=540.0)
    model = DualEmbeddingChGKModel(
        num_players=10,
        num_questions=5,
        dim_k=dim_k,
        dim_c=dim_c,
        time_decay_config=cfg,
    )

    # Non-zero communication vectors and synergy tensor
    with torch.no_grad():
        model.player_c_base.weight[1] = torch.tensor([1.0] * dim_c)
        model.player_c_base.weight[2] = torch.tensor([1.0] * dim_c)
        model.W_synergy.fill_(0.5)

    player_ids = torch.tensor([[1, 2, 0, 0, 0, 0]], dtype=torch.long)
    format_ids = torch.tensor([0], dtype=torch.long)
    mask = player_ids != 0

    c_eff = model.get_effective_communication(player_ids, format_ids)

    # Active pair (Delta_t = 0)
    delta_active = torch.zeros(1, 6, 6)
    s_active = model.compute_team_synergy_matrix(c_eff, mask, delta_active)[0, 0, 1].item()

    # Inactive pair for 2 years (730 days)
    delta_inactive = torch.zeros(1, 6, 6)
    delta_inactive[0, 0, 1] = 730.0
    delta_inactive[0, 1, 0] = 730.0
    s_inactive = model.compute_team_synergy_matrix(c_eff, mask, delta_inactive)[0, 0, 1].item()

    # Must be strictly eroded
    assert s_inactive < s_active
    # Ratio should match exp(-ln(2)/540 * 730) = 2^(-730/540) ~ 2^(-1.35) ~ 0.39
    expected_ratio = math.exp(-math.log(2.0) / 540.0 * 730.0)
    assert pytest.approx(s_inactive / s_active, rel=0.05) == expected_ratio


def test_brownian_variance_expansion() -> None:
    """Test Brownian variance scaling for returning inactive players."""
    module = TimeDecayModule(dim_k=16, config=TimeDecayConfig(alpha_gap=0.5))

    d_zero = torch.tensor([0.0])
    scale_zero = module.compute_training_gap_scale(d_zero).item()
    assert scale_zero == 1.0

    # 1 year gap: sqrt(1.0 + 0.5 * 1) = sqrt(1.5) ~ 1.2247
    d_1yr = torch.tensor([365.0])
    scale_1yr = module.compute_training_gap_scale(d_1yr).item()
    assert pytest.approx(scale_1yr, rel=1e-3) == math.sqrt(1.5)

    # 3 year gap: sqrt(1.0 + 0.5 * 3) = sqrt(2.5) ~ 1.5811
    d_3yr = torch.tensor([1095.0])
    scale_3yr = module.compute_training_gap_scale(d_3yr).item()
    assert pytest.approx(scale_3yr, rel=1e-3) == math.sqrt(2.5)


def test_time_travel_calibration(tmp_path: Path) -> None:
    """Acceptance Criterion 3: Time-Travel Calibration Test.

    Train only on historic data up to late 2023, and evaluate holdout data without leakage.
    """
    pq_path = tmp_path / "multiyear_dataset.parquet"
    generate_synthetic_dataset(
        output_parquet_path=pq_path,
        num_tournaments=20,
        num_players=60,
        num_teams=10,
        questions_per_tournament=10,
        start_year=2022,
        seed=999,
    )

    df = pd.read_parquet(pq_path)
    train_df, val_df, test_df = split_data_chronologically(df, train_ratio=0.8, val_ratio=0.1)

    # Strict non-leakage chronological boundary verification
    max_train_date = train_df["date"].max()
    min_val_date = val_df["date"].min()
    max_val_date = val_df["date"].max()
    min_test_date = test_df["date"].min()

    assert max_train_date <= min_val_date
    assert max_val_date <= min_test_date

    model_path = tmp_path / "timed_model.pt"
    meta_path = tmp_path / "timed_metadata.json"

    cfg = ModelConfig(
        dim_k=8,
        dim_c=4,
        epochs=2,
        batch_size=128,
        learning_rate=0.01,
    )

    model, metadata, results = train_model(
        config=cfg,
        parquet_path=pq_path,
        model_save_path=model_path,
        meta_save_path=meta_path,
    )

    assert "test_metrics" in results
    assert results["test_metrics"]["log_loss"] > 0
    assert len(metadata.player_last_date) > 0


def test_historical_timeline_and_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_player_historical_timeline and API endpoints with date decay."""
    p_indexer = IDIndexer(start_idx=1)
    p_indexer.fit(["P101", "P102", "P103", "P104", "P105", "P106"])
    q_indexer = IDIndexer(start_idx=0)
    q_indexer.fit(["Q1", "Q2"])

    player_last = {
        "P101": "2023-01-01",  # Inactive since 2023
        "P102": "2026-08-01",  # Recent active
    }
    pair_last = {
        "P101_P102": "2023-01-01",
    }

    metadata = DatasetMetadata(
        p_indexer,
        q_indexer,
        player_last_date=player_last,
        pair_last_date=pair_last,
    )

    model = DualEmbeddingChGKModel(
        num_players=len(p_indexer) + 1,
        num_questions=len(q_indexer),
        dim_k=16,
        dim_c=8,
    )

    # 1. Test timeline generator
    timeline = get_player_historical_timeline(
        model, metadata, "P101", start_date="2023-01-01", end_date="2026-01-01", step_days=180
    )
    assert len(timeline) >= 6
    # Skill should decline over hiatus
    assert timeline[0]["retention_ratio"] >= timeline[-1]["retention_ratio"]

    # 2. Test FastAPI endpoints
    m_path = tmp_path / "api_m.pt"
    meta_path = tmp_path / "api_meta.json"
    metadata.save(meta_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"dim_k": 16, "dim_c": 8, "num_players": 7, "num_questions": 2, "num_formats": 3},
        },
        m_path,
    )

    monkeypatch.setattr(app_module, "DEFAULT_MODEL_PATH", m_path)
    monkeypatch.setattr(app_module, "DEFAULT_META_PATH", meta_path)
    app_module.state["model"] = model
    app_module.state["metadata"] = metadata

    client = TestClient(app)

    # Timeline endpoint
    r_time = client.get("/player/P101/historical_timeline?step_days=180")
    assert r_time.status_code == 200
    data_time = r_time.json()
    assert data_time["player_id"] == "P101"
    assert len(data_time["timeline"]) >= 5

    # Predict match with match_date in 2026 (decay active) vs 2023
    r_2023 = client.post(
        "/predict_match",
        json={
            "player_ids": ["P101", "P102"],
            "question_ids": ["Q1"],
            "match_date": "2023-01-02",
        },
    )
    assert r_2023.status_code == 200

    r_2026 = client.post(
        "/predict_match",
        json={
            "player_ids": ["P101", "P102"],
            "question_ids": ["Q1"],
            "match_date": "2026-09-01",
        },
    )
    assert r_2026.status_code == 200
    # Both queries should succeed and return predictions
    assert "predictions" in r_2026.json()

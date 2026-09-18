"""Tests for FastAPI endpoints and Click CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from click.testing import CliRunner
from fastapi.testclient import TestClient

import chgk_rating.app as app_module
from chgk_rating.app import app
from chgk_rating.cli import cli
from chgk_rating.data_loader import generate_synthetic_dataset
from chgk_rating.dataset import DatasetMetadata, IDIndexer
from chgk_rating.model import DualEmbeddingChGKModel


@pytest.fixture(autouse=True)
def setup_mock_model_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Setup mock model state in app_module."""
    p_indexer = IDIndexer(start_idx=1)
    p_indexer.fit(["27403", "28751", "21026", "74001", "40393", "7008", "99999"])

    ed_indexer = IDIndexer(start_idx=1)
    ed_indexer.fit(["2421", "3438", "100"])

    q_indexer = IDIndexer(start_idx=0)
    q_indexer.fit(["Q_100_1", "Q_100_2", "Q_100_3"])

    metadata = DatasetMetadata(
        player_indexer=p_indexer,
        editor_indexer=ed_indexer,
        question_indexer=q_indexer,
    )
    model = DualEmbeddingChGKModel(
        num_players=len(p_indexer) + 1,
        num_editors=len(ed_indexer) + 1,
        dim_k=8,
        dim_c=4,
    )

    # Save to disk as well
    m_path = tmp_path / "best_model.pt"
    meta_path = tmp_path / "metadata.json"
    metadata.save(meta_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "dim_k": 8,
                "dim_c": 4,
                "num_players": len(p_indexer) + 1,
                "num_editors": len(ed_indexer) + 1,
                "num_formats": 3,
            },
        },
        m_path,
    )

    monkeypatch.setattr(app_module, "DEFAULT_MODEL_PATH", m_path)
    monkeypatch.setattr(app_module, "DEFAULT_META_PATH", meta_path)
    app_module.state["model"] = model
    app_module.state["metadata"] = metadata


def test_api_health_and_root() -> None:
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_api_predict_match() -> None:
    client = TestClient(app)
    payload = {
        "player_ids": ["27403", "28751", "21026", "74001", "40393", "7008"],
        "question_ids": ["Q_100_1", "Q_100_2"],
        "format": "offline",
    }
    r = client.post("/predict_match", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["format"] == "offline"
    assert len(data["predictions"]) == 2
    assert "solved_probability" in data["predictions"][0]
    assert 0.0 <= data["predictions"][0]["solved_probability"] <= 1.0


def test_api_team_synergy() -> None:
    client = TestClient(app)
    payload = {
        "player_ids": ["27403", "28751", "21026", "74001", "40393", "7008"],
        "format": "online",
    }
    r = client.post("/team_synergy", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["format"] == "online"
    assert len(data["players"]) == 6
    assert len(data["synergy_matrix"]) == 6
    assert len(data["optimal_seating_order"]) == 6


def test_api_player_profile() -> None:
    client = TestClient(app)
    r = client.get("/player/27403/profile")
    assert r.status_code == 200
    data = r.json()
    assert data["player_id"] == "27403"
    assert "knowledge_norm" in data
    assert "communication_base_norm" in data
    assert "top_synergistic_teammates" in data

    # Unknown player should 404
    r_unknown = client.get("/player/non_existent_player/profile")
    assert r_unknown.status_code == 404


def test_cli_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()

    # 1. Test fetch-data --synthetic
    pq_path = tmp_path / "cli_test.parquet"
    res = runner.invoke(cli, ["fetch-data", "--synthetic", "--output-path", str(pq_path)])
    assert res.exit_code == 0
    assert pq_path.exists()

    # 2. Test train
    model_path = tmp_path / "cli_model.pt"
    meta_path = tmp_path / "cli_meta.json"
    res = runner.invoke(
        cli,
        [
            "train",
            "--data-path", str(pq_path),
            "--model-path", str(model_path),
            "--meta-path", str(meta_path),
            "--epochs", "2",
            "--batch-size", "128",
        ],
    )
    assert res.exit_code == 0
    assert model_path.exists()
    assert meta_path.exists()


def test_editor_calibration_api():
    client = TestClient(app)
    response = client.get("/editors/leaderboard?min_tournaments=1&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        first_id = data[0]["editor_id"]
        res_single = client.get(f"/editor/{first_id}/calibration")
        assert res_single.status_code == 200
        assert res_single.json()["editor_id"] == first_id


def test_simulate_tournament_api():
    client = TestClient(app)
    payload = {
        "teams": {
            "Team A": [1585, 23737],
            "Team B": [62, 707],
        },
        "format": "offline",
        "target_dl": 4.5,
        "editor_ids": [2421],
        "num_questions": 36,
    }
    response = client.post("/simulate_tournament", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "standings" in data
    assert len(data["standings"]) == 2
    assert "prob_1st_place" in data["standings"][0]

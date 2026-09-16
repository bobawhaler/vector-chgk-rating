"""Unit tests for model inspection, synergy matrix, online anomaly detection, and embedding export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from chgk_rating.analytics import (
    compute_team_synergy_details,
    detect_online_anomalies,
    export_embeddings_to_parquet,
    get_top_players,
)
from chgk_rating.config import FORMAT_OFFLINE, FORMAT_ONLINE
from chgk_rating.data_loader import generate_synthetic_dataset
from chgk_rating.dataset import DatasetMetadata, IDIndexer
from chgk_rating.model import DualEmbeddingChGKModel


@pytest.fixture
def mock_model_and_metadata() -> tuple[DualEmbeddingChGKModel, DatasetMetadata]:
    p_indexer = IDIndexer(start_idx=1)
    p_indexer.fit([f"P_{i}" for i in range(1, 21)])

    q_indexer = IDIndexer(start_idx=0)
    q_indexer.fit([f"Q_{j}" for j in range(1, 31)])

    metadata = DatasetMetadata(p_indexer, q_indexer)
    model = DualEmbeddingChGKModel(
        num_players=len(p_indexer) + 1,
        num_questions=len(q_indexer),
        dim_k=8,
        dim_c=4,
    )
    return model, metadata


def test_get_top_players(mock_model_and_metadata: tuple[DualEmbeddingChGKModel, DatasetMetadata]) -> None:
    model, metadata = mock_model_and_metadata
    top_5 = get_top_players(model, metadata, top_n=5)

    assert len(top_5) == 5
    assert top_5[0]["rank"] == 1
    # Check sorted descending
    for i in range(len(top_5) - 1):
        assert top_5[i]["knowledge_norm"] >= top_5[i + 1]["knowledge_norm"]


def test_compute_team_synergy_details(mock_model_and_metadata: tuple[DualEmbeddingChGKModel, DatasetMetadata]) -> None:
    model, metadata = mock_model_and_metadata
    players = ["P_1", "P_2", "P_3", "P_4"]

    details_off = compute_team_synergy_details(model, metadata, players, format_name=FORMAT_OFFLINE)
    details_on = compute_team_synergy_details(model, metadata, players, format_name=FORMAT_ONLINE)

    assert details_off["format"] == FORMAT_OFFLINE
    assert details_on["format"] == FORMAT_ONLINE
    assert len(details_off["players"]) == 4
    assert len(details_off["synergy_matrix"]) == 4
    assert len(details_off["synergy_matrix"][0]) == 4
    assert isinstance(details_off["total_synergy"], float)
    # Number of pairs for 4 players is 4 * 3 / 2 = 6
    assert len(details_off["pairwise_rankings"]) == 6


def test_detect_online_anomalies(mock_model_and_metadata: tuple[DualEmbeddingChGKModel, DatasetMetadata], tmp_path: Path) -> None:
    model, metadata = mock_model_and_metadata
    pq_path = tmp_path / "anom_dataset.parquet"
    df = generate_synthetic_dataset(
        output_parquet_path=pq_path,
        num_tournaments=6,
        num_players=20,
        num_teams=4,
        questions_per_tournament=10,
    )

    anomalies = detect_online_anomalies(df, model, metadata, min_games=5)
    assert isinstance(anomalies, list)
    if anomalies:
        first = anomalies[0]
        assert "team_id" in first
        assert "delta_cheat_score" in first
        assert "is_suspicious" in first


def test_export_embeddings(mock_model_and_metadata: tuple[DualEmbeddingChGKModel, DatasetMetadata], tmp_path: Path) -> None:
    model, metadata = mock_model_and_metadata
    out_dir = tmp_path / "exported"
    paths = export_embeddings_to_parquet(model, metadata, out_dir)

    assert paths["player_parquet"].exists()
    assert paths["player_csv"].exists()
    assert paths["question_parquet"].exists()
    assert paths["question_csv"].exists()

    p_df = pd.read_parquet(paths["player_parquet"])
    assert "player_id" in p_df.columns
    assert "knowledge_norm" in p_df.columns
    assert "k_0" in p_df.columns

"""Unit tests for dataset split, training loop, and evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from chgk_rating.config import ModelConfig
from chgk_rating.data_loader import generate_synthetic_dataset
from chgk_rating.dataset import DatasetMetadata, IDIndexer
from chgk_rating.train import split_data_chronologically, train_model


def test_split_data_chronologically() -> None:
    data = {
        "tournament_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "date": [
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
            "2023-01-06",
            "2023-01-07",
            "2023-01-08",
            "2023-01-09",
            "2023-01-10",
        ],
        "solved": [1] * 10,
    }
    df = pd.DataFrame(data)
    train_df, val_df, test_df = split_data_chronologically(df, train_ratio=0.8, val_ratio=0.1)

    assert len(train_df) == 8
    assert len(val_df) == 1
    assert len(test_df) == 1
    # Check strict chronological ordering
    assert train_df["date"].max() <= val_df["date"].min()
    assert val_df["date"].max() <= test_df["date"].min()


def test_id_indexer_serialization(tmp_path: Path) -> None:
    indexer = IDIndexer(start_idx=1)
    indexer.fit(["player_A", "player_B", "player_C"])

    assert indexer.get_idx("player_A") == 1
    assert indexer.get_idx("player_B") == 2
    assert indexer.get_idx("player_C") == 3
    assert indexer.get_item(2) == "player_B"

    meta_file = tmp_path / "metadata.json"
    ed_indexer = IDIndexer(start_idx=1)
    ed_indexer.fit(["ed_1", "ed_2"])
    q_indexer = IDIndexer(start_idx=0)
    q_indexer.fit(["q_1", "q_2"])

    meta = DatasetMetadata(indexer, editor_indexer=ed_indexer, question_indexer=q_indexer)
    meta.save(meta_file)

    loaded_meta = DatasetMetadata.load(meta_file)
    assert loaded_meta.player_indexer.get_idx("player_A") == 1
    assert loaded_meta.question_indexer.get_idx("q_2") == 1


def test_train_model_pipeline(tmp_path: Path) -> None:
    # 1. Generate small synthetic dataset
    pq_path = tmp_path / "train_synth.parquet"
    generate_synthetic_dataset(
        output_parquet_path=pq_path,
        num_tournaments=4,
        num_players=30,
        num_teams=6,
        questions_per_tournament=10,
        seed=123,
    )

    model_path = tmp_path / "best_model.pt"
    meta_path = tmp_path / "metadata.json"

    cfg = ModelConfig(
        dim_k=8,
        dim_c=4,
        epochs=3,
        batch_size=64,
        learning_rate=0.01,
    )

    model, metadata, results = train_model(
        config=cfg,
        parquet_path=pq_path,
        model_save_path=model_path,
        meta_save_path=meta_path,
    )

    assert model_path.exists()
    assert meta_path.exists()
    assert "test_metrics" in results
    test_metrics = results["test_metrics"]
    assert "log_loss" in test_metrics
    assert "roc_auc" in test_metrics
    assert "brier_score" in test_metrics
    assert test_metrics["log_loss"] > 0
    assert 0.0 <= test_metrics["brier_score"] <= 1.0

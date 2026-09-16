"""Training engine for the ChGK Dual Latent Embedding Rating Model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from chgk_rating.config import (
    DEFAULT_META_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PARQUET_PATH,
    ModelConfig,
)
from chgk_rating.dataset import ChGKDataset, DatasetMetadata, IDIndexer
from chgk_rating.model import DualEmbeddingChGKModel

logger = logging.getLogger(__name__)


def split_data_chronologically(
    df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically based on tournament date."""
    # Ensure sorted by date and tournament_id
    sorted_df = df.sort_values(by=["date", "tournament_id"]).reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = sorted_df.iloc[:train_end].copy()
    val_df = sorted_df.iloc[train_end:val_end].copy()
    test_df = sorted_df.iloc[val_end:].copy()

    logger.info(
        "Split chronologically: %d train (%.1f%%), %d val (%.1f%%), %d test (%.1f%%)",
        len(train_df),
        train_ratio * 100,
        len(val_df),
        val_ratio * 100,
        len(test_df),
        (1.0 - train_ratio - val_ratio) * 100,
    )
    return train_df, val_df, test_df


def evaluate_model(
    model: DualEmbeddingChGKModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute LogLoss, ROC-AUC, and Brier Score on evaluation DataLoader."""
    model.eval()
    all_targets: list[float] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            player_ids = batch["player_ids"].to(device)
            question_id = batch["question_id"].to(device)
            format_id = batch["format_id"].to(device)
            labels = batch["label"].to(device)

            out = model(player_ids, question_id, format_id)
            probs = out["probs"].squeeze(-1).cpu().numpy().tolist()
            targets = labels.squeeze(-1).cpu().numpy().tolist()

            all_probs.extend(probs)
            all_targets.extend(targets)

    y_true = np.array(all_targets, dtype=np.float64)
    y_prob = np.clip(np.array(all_probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)

    # Calculate metrics
    ll = float(log_loss(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    # ROC-AUC requires at least two classes present
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc = 0.5

    return {
        "log_loss": ll,
        "roc_auc": auc,
        "brier_score": brier,
    }


def train_model(
    config: Optional[ModelConfig] = None,
    parquet_path: Path = DEFAULT_PARQUET_PATH,
    model_save_path: Path = DEFAULT_MODEL_PATH,
    meta_save_path: Path = DEFAULT_META_PATH,
) -> Tuple[DualEmbeddingChGKModel, DatasetMetadata, Dict[str, Any]]:
    """Complete training loop with chronological validation and checkpointing."""
    cfg = config or ModelConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device != "cpu" else "cpu")
    logger.info("Training on device: %s", device)

    # 1. Load data
    df = pd.read_parquet(parquet_path)
    train_df, val_df, test_df = split_data_chronologically(df)

    # 2. Build indexers
    # Player indexing starts at 1; 0 is reserved for padding
    all_players: set[str] = set()
    for plist in df["player_ids"]:
        if isinstance(plist, (list, np.ndarray)):
            for p in plist:
                all_players.add(str(p))

    player_indexer = IDIndexer(start_idx=1)
    player_indexer.fit(sorted(list(all_players)))

    question_indexer = IDIndexer(start_idx=0)
    question_indexer.fit(sorted(list(df["question_id"].astype(str).unique())))

    metadata = DatasetMetadata(player_indexer, question_indexer)
    metadata.save(meta_save_path)

    # 3. Create datasets & dataloaders
    train_dataset = ChGKDataset(train_df, metadata)
    val_dataset = ChGKDataset(val_df, metadata)
    test_dataset = ChGKDataset(test_df, metadata)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False
    )

    # 4. Instantiate model
    model = DualEmbeddingChGKModel(
        num_players=len(player_indexer) + 1,  # account for 0 padding
        num_questions=len(question_indexer),
        dim_k=cfg.dim_k,
        dim_c=cfg.dim_c,
        num_formats=cfg.num_formats,
    ).to(device)

    # 5. Optimizer & Criterion
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_metrics: Dict[str, float] = {}

    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    # 6. Training loop
    logger.info("Starting training for %d epochs...", cfg.epochs)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        for batch in train_loader:
            player_ids = batch["player_ids"].to(device)
            question_id = batch["question_id"].to(device)
            format_id = batch["format_id"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            out = model(player_ids, question_id, format_id)
            loss = criterion(out["logits"], labels)

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(epoch_losses))

        # Validation evaluation
        val_metrics = evaluate_model(model, val_loader, device)
        val_loss = val_metrics["log_loss"]

        logger.info(
            "Epoch %02d/%02d - Train Loss: %.4f | Val LogLoss: %.4f | Val AUC: %.4f | Val Brier: %.4f",
            epoch,
            cfg.epochs,
            train_loss,
            val_metrics["log_loss"],
            val_metrics["roc_auc"],
            val_metrics["brier_score"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": {
                        "dim_k": cfg.dim_k,
                        "dim_c": cfg.dim_c,
                        "num_players": len(player_indexer) + 1,
                        "num_questions": len(question_indexer),
                        "num_formats": cfg.num_formats,
                    },
                },
                model_save_path,
            )

    # 7. Load best model and evaluate on Test set
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, device)

    logger.info(
        "Final Test Evaluation: LogLoss: %.4f | ROC-AUC: %.4f | Brier: %.4f",
        test_metrics["log_loss"],
        test_metrics["roc_auc"],
        test_metrics["brier_score"],
    )

    results = {
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "epochs_trained": cfg.epochs,
        "total_train_samples": len(train_dataset),
        "total_test_samples": len(test_dataset),
    }
    return model, metadata, results

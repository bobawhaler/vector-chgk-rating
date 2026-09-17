"""End-to-end pipeline script: fetches all tournaments from 2020-01-01 and trains the Dual Embedding Model."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from chgk_rating.analytics import get_format_analysis, load_model_and_metadata
from chgk_rating.config import (
    DEFAULT_META_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PARQUET_PATH,
    DataIngestConfig,
    ModelConfig,
)
from chgk_rating.data_loader import ChGKDataIngester
from chgk_rating.train import train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


async def main() -> None:
    start_time = time.time()
    logger.info("=== Starting ChGK Rating Pipeline ===")

    # 1. Ingest historical data if not already present
    if DEFAULT_PARQUET_PATH.exists() and DEFAULT_PARQUET_PATH.stat().st_size > 1000000:
        logger.info("Found existing dataset at %s (size: %.1f MB). Loading...", DEFAULT_PARQUET_PATH, DEFAULT_PARQUET_PATH.stat().st_size / 1024 / 1024)
        df = pd.read_parquet(DEFAULT_PARQUET_PATH)
        logger.info("Successfully loaded %d records from disk.", len(df))
    else:
        logger.info("Target: Historical data from 2020-01-01 to present")
        ingest_config = DataIngestConfig(
            parquet_path=DEFAULT_PARQUET_PATH,
            max_requests_per_second=10.0,
            timeout_seconds=20.0,
            max_retries=4,
        )
        ingester = ChGKDataIngester(ingest_config)

        t0 = time.time()
        logger.info("Fetching tournament list and results...")

        def progress_callback(msg: str, current: int, total: int) -> None:
            elapsed = time.time() - t0
            rate = current / max(elapsed, 0.1)
            remaining_s = (total - current) / max(rate, 0.01)
            if current % 50 == 0 or current == total:
                logger.info(
                    "Progress: [%d/%d] (%.1f%%) | Rate: %.1f t/s | ETA: %.0fs | %s",
                    current,
                    total,
                    (current / total) * 100,
                    rate,
                    remaining_s,
                    msg,
                )

        df = await ingester.ingest_dataset(
            start_date="2020-01-01",
            end_date=datetime.now().strftime("%Y-%m-%d"),
            progress_cb=progress_callback,
        )

        fetch_duration = time.time() - t0
        logger.info(
            "Data ingestion finished in %.1f seconds. Total records: %d",
            fetch_duration,
            len(df),
        )

    if df.empty:
        logger.error("Dataset is empty! Exiting.")
        sys.exit(1)

    logger.info("Unique tournaments: %d", df["tournament_id"].nunique())
    logger.info("Unique questions:   %d", df["question_id"].nunique())
    logger.info("Unique teams:       %d", df["team_id"].nunique())

    # 2. Train Dual Latent Embedding Rating Model
    logger.info("=== Starting Model Training ===")
    target_device = "cpu"

    model_config = ModelConfig(
        dim_k=16,
        dim_c=8,
        epochs=15,
        batch_size=8192,
        learning_rate=1e-3,
        weight_decay=1e-4,
        device=target_device,
    )

    t_train_start = time.time()
    model, metadata, results = train_model(
        config=model_config,
        parquet_path=DEFAULT_PARQUET_PATH,
        model_save_path=DEFAULT_MODEL_PATH,
        meta_save_path=DEFAULT_META_PATH,
    )
    train_duration = time.time() - t_train_start
    logger.info("Training completed in %.1f seconds.", train_duration)

    # 3. Print Final Summary
    logger.info("=== Final Evaluation Metrics ===")
    logger.info("Best Val LogLoss: %.4f", results["best_val_metrics"].get("log_loss", 0.0))
    logger.info("Test LogLoss:     %.4f", results["test_metrics"]["log_loss"])
    logger.info("Test ROC-AUC:     %.4f", results["test_metrics"]["roc_auc"])
    logger.info("Test Brier Score: %.4f", results["test_metrics"]["brier_score"])

    if "calibration" in results:
        cal = results["calibration"]
        logger.info("=== Learned Data-Driven Temporal Parameters ===")
        logger.info("Learned Grace Period:   %.1f days", cal["learned_grace_period_days"])
        logger.info("Factual Half-Life:      %.1f days", cal["factual_half_life_days"])
        logger.info("Logical Half-Life:      %.1f days", cal["logical_half_life_days"])
        logger.info("Synergy Half-Life:      %.1f days", cal["synergy_half_life_days"])
        logger.info("Brownian Alpha Gap:     %.4f", cal["alpha_gap"])

    # 4. Format Similarity Analysis
    logger.info("=== Format Similarity & Baseline Biases ===")
    format_info = get_format_analysis(model, metadata)
    for fmt_name, data in format_info.get("formats", {}).items():
        logger.info(
            "Format '%s': Bias = %.4f, Norm = %.4f",
            fmt_name,
            data.get("bias", 0.0),
            data.get("norm", 0.0),
        )
    logger.info("Format Cosine Similarities: %s", format_info.get("cosine_similarities", {}))

    total_pipeline_time = time.time() - start_time
    logger.info(
        "=== Full Pipeline Completed Successfully in %.1f minutes ===",
        total_pipeline_time / 60.0,
    )


if __name__ == "__main__":
    asyncio.run(main())

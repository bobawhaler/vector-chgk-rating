"""Configuration and constants for the ChGK Rating System."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final

# Base Paths
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
MODELS_DIR: Final[Path] = BASE_DIR / "models"

# Format categorical mapping
FORMAT_OFFLINE: Final[str] = "offline"
FORMAT_SYNC: Final[str] = "sync"
FORMAT_ONLINE: Final[str] = "online"

FORMAT_TO_ID: Final[Dict[str, int]] = {
    FORMAT_OFFLINE: 0,
    FORMAT_SYNC: 1,
    FORMAT_ONLINE: 2,
}

ID_TO_FORMAT: Final[Dict[int, str]] = {v: k for k, v in FORMAT_TO_ID.items()}
NUM_FORMATS: Final[int] = len(FORMAT_TO_ID)

# Official API defaults
DEFAULT_API_BASE_URL: Final[str] = "https://api.rating.chgk.info"
DEFAULT_CACHE_DB_PATH: Final[Path] = DATA_DIR / "chgk_cache.db"
DEFAULT_PARQUET_PATH: Final[Path] = DATA_DIR / "dataset.parquet"
DEFAULT_MODEL_PATH: Final[Path] = MODELS_DIR / "best_model.pt"
DEFAULT_META_PATH: Final[Path] = MODELS_DIR / "metadata.json"


@dataclass
class ModelConfig:
    """Hyperparameters for DualEmbeddingChGKModel."""

    dim_k: int = 16  # Knowledge latent dimension D
    dim_c: int = 8   # Communication & format latent dimension M
    num_formats: int = NUM_FORMATS
    weight_decay: float = 1e-4
    learning_rate: float = 1e-3
    batch_size: int = 4096
    epochs: int = 20
    device: str = "cpu"


@dataclass
class DataIngestConfig:
    """Configuration for data fetching and caching."""

    api_base_url: str = DEFAULT_API_BASE_URL
    cache_db_path: Path = DEFAULT_CACHE_DB_PATH
    parquet_path: Path = DEFAULT_PARQUET_PATH
    max_requests_per_second: float = 10.0
    timeout_seconds: float = 15.0
    max_retries: int = 4

"""Paths for the local question corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RATING_PARQUET = BASE_DIR.parent / "data" / "dataset.parquet"

DEFAULT_GOTQUESTIONS_URL = "https://gotquestions.online"
DEFAULT_GQ_CACHE_DB_PATH = DATA_DIR / "gotquestions_cache.db"
DEFAULT_GQ_QUESTIONS_PARQUET = DATA_DIR / "gotquestions_questions.parquet"
DEFAULT_GQ_PACKS_PARQUET = DATA_DIR / "gotquestions_packs.parquet"
DEFAULT_KNOWLEDGE_DB = DATA_DIR / "knowledge.sqlite"


@dataclass
class GotQuestionsConfig:
    """Configuration for a local full crawl of gotquestions.online.

    The scheduled nightly delta lives in the sync project. This crawl builds
    or refreshes the parquet this package analyzes.
    """

    base_url: str = DEFAULT_GOTQUESTIONS_URL
    cache_db_path: Path = DEFAULT_GQ_CACHE_DB_PATH
    questions_parquet: Path = DEFAULT_GQ_QUESTIONS_PARQUET
    packs_parquet: Path = DEFAULT_GQ_PACKS_PARQUET
    max_requests_per_second: float = 1.0
    timeout_seconds: float = 15.0
    max_retries: int = 4
    consecutive_404_stop: int = 30

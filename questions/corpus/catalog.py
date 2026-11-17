"""Local query table: current question texts plus sources and dates from the cache.

The questions parquet is the актуа corpus (snapshot and nightly parts). It does
not yet store sources or package dates. Those fields are still in the sqlite
cache of raw packs, so the catalog joins them without a new crawl.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from corpus.config import GotQuestionsConfig
from corpus.gotquestions import GotQuestionsCache
from corpus.records import QUESTION_COLUMNS, day, frames_from_packs

SCHEMA = "1"
ENRICH = [
    "source",
    "zachet",
    "nezachet",
    "pack_title",
    "start_date",
    "end_date",
    "pub_date",
    "number_in_tour",
    "pack_editor_names",
    "pack_editor_player_ids",
]
_LISTS = {"pack_editor_names", "pack_editor_player_ids"}
_STRINGS = [name for name in ENRICH if name not in _LISTS and name != "number_in_tour"]


def _say(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _mtime(path: Path) -> int:
    if not path.exists():
        return 0
    return path.stat().st_mtime_ns


def _stamp_text(questions_path: Path, cache_path: Path) -> str:
    return f"{SCHEMA}|{_mtime(questions_path)}|{_mtime(cache_path)}"


def _paths(config: GotQuestionsConfig) -> tuple[Path, Path]:
    return (
        config.questions_parquet.with_name("gotquestions_catalog.parquet"),
        config.questions_parquet.with_name("gotquestions_catalog.stamp"),
    )


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip() in ("", "nan", "None", "<NA>")
    return False


def _blank(series: pd.Series) -> pd.Series:
    return series.map(_is_blank)


def _overlay(current: pd.DataFrame, cached: pd.DataFrame) -> pd.DataFrame:
    """Keep texts from ``current`` and fill sources, dates and titles from ``cached``."""
    base = current.copy()
    base["question_id"] = base["question_id"].astype(str)
    base = base.drop_duplicates("question_id", keep="last")
    extra = [name for name in ENRICH if name in cached.columns]
    enrich = cached.drop_duplicates("question_id", keep="last")[["question_id", *extra]].copy()
    enrich["question_id"] = enrich["question_id"].astype(str)
    renamed = enrich.rename(columns={name: f"{name}__cache" for name in extra})
    merged = base.merge(renamed, on="question_id", how="left")
    for name in extra:
        cached_name = f"{name}__cache"
        if name not in merged.columns:
            merged[name] = merged[cached_name]
        elif name in _LISTS:
            merged[name] = [
                left if isinstance(left, (list, tuple)) and len(left) > 0 else right
                for left, right in zip(merged[name], merged[cached_name], strict=True)
            ]
        else:
            merged[name] = merged[name].where(~_blank(merged[name]), merged[cached_name])
        merged = merged.drop(columns=[cached_name])
    return merged


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for name in _STRINGS:
        if name not in cleaned.columns:
            cleaned[name] = ""
        else:
            cleaned[name] = cleaned[name].map(lambda value: "" if _is_blank(value) else value)
    for name in ("start_date", "end_date", "pub_date"):
        cleaned[name] = cleaned[name].map(day)
    for name in _LISTS:
        if name not in cleaned.columns:
            cleaned[name] = [[] for _ in range(len(cleaned))]
        else:
            cleaned[name] = cleaned[name].map(
                lambda value: list(value) if isinstance(value, (list, tuple)) else []
            )
    if "number_in_tour" not in cleaned.columns:
        cleaned["number_in_tour"] = cleaned["question_number"] if "question_number" in cleaned.columns else pd.NA
    else:
        fallback = cleaned["question_number"] if "question_number" in cleaned.columns else pd.NA
        cleaned["number_in_tour"] = cleaned["number_in_tour"].where(~_blank(cleaned["number_in_tour"]), fallback)
    if "question_id" in cleaned.columns:
        cleaned["question_id"] = cleaned["question_id"].astype(str)
    ordered = [name for name in QUESTION_COLUMNS if name in cleaned.columns]
    rest = [name for name in cleaned.columns if name not in ordered]
    return cleaned[ordered + rest]


def _read_questions(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if "question_id" in frame.columns:
        frame["question_id"] = frame["question_id"].astype(str)
    return frame


def load_catalog(
    config: GotQuestionsConfig | None = None,
    *,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Load the query table, building it when the cache or the parquet changed."""
    settings = config or GotQuestionsConfig()
    questions_path = settings.questions_parquet
    cache_path = settings.cache_db_path
    catalog_path, stamp_path = _paths(settings)
    stamp = _stamp_text(questions_path, cache_path)
    if not force and catalog_path.exists() and stamp_path.exists() and stamp_path.read_text() == stamp:
        return pd.read_parquet(catalog_path)

    if not questions_path.exists() and not cache_path.exists():
        raise FileNotFoundError(
            f"Нет файла {questions_path} и нет кэша {cache_path}. "
            "Соберите базу: sync materialize-questions или questions fetch."
        )

    current = _read_questions(questions_path)
    cached_questions: pd.DataFrame | None = None
    if cache_path.exists():
        _say(progress, "Читаю локальный кэш пакетов и присоединяю источники, даты и названия.")
        cache = GotQuestionsCache(cache_path)
        cached_questions, packages = frames_from_packs(cache.get_all_packs())
        if not packages.empty:
            settings.packs_parquet.parent.mkdir(parents=True, exist_ok=True)
            packages.to_parquet(settings.packs_parquet, index=False)
            _say(progress, f"Таблица пакетов: {len(packages)}.")

    cache_has_rows = cached_questions is not None and not cached_questions.empty
    if current is None:
        frame = cached_questions if cache_has_rows else pd.DataFrame(columns=QUESTION_COLUMNS)
    elif not cache_has_rows:
        frame = current
    elif _mtime(questions_path) >= _mtime(cache_path):
        frame = _overlay(current, cached_questions)
    else:
        fresh = cached_questions.drop_duplicates("question_id", keep="last")
        known = set(fresh["question_id"].astype(str))
        extra = current[~current["question_id"].astype(str).isin(known)]
        frame = pd.concat([fresh, extra], ignore_index=True) if not extra.empty else fresh

    frame = _clean(frame)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    # An empty column list still has a schema the next launch can read.
    if frame.empty:
        frame = pd.DataFrame(columns=QUESTION_COLUMNS)
    frame.to_parquet(catalog_path, index=False)
    stamp_path.write_text(stamp)
    packages_count = int(frame["pack_id"].nunique()) if "pack_id" in frame.columns else 0
    _say(progress, f"Каталог: {len(frame)} строк, пакетов {packages_count}.")
    return frame


def corpus_meta(questions: pd.DataFrame) -> dict[str, object]:
    """Counts shown in the page header."""
    if questions.empty:
        return {"rows": 0, "questions": 0, "packs": 0, "span": "", "with_source": 0}
    unique = questions
    if {"pack_id", "question_number"}.issubset(questions.columns):
        unique = questions.drop_duplicates(["pack_id", "question_number"])
    span = ""
    if "start_date" in questions.columns:
        dates = questions["start_date"].fillna("").astype(str)
        dates = dates[dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)]
        if not dates.empty:
            span = f"{dates.min()} — {dates.max()}"
    with_source = 0
    if "source" in unique.columns:
        with_source = int(unique["source"].fillna("").astype(str).str.strip().ne("").sum())
    packs = int(questions["pack_id"].nunique()) if "pack_id" in questions.columns else 0
    return {
        "rows": int(len(questions)),
        "questions": int(len(unique)),
        "packs": packs,
        "span": span,
        "with_source": with_source,
    }

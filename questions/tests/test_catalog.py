"""Catalog keeps the newer text and still fills sources from the cache."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from corpus.catalog import load_catalog
from corpus.config import GotQuestionsConfig
from corpus.gotquestions import GotQuestionsCache


def _config(tmp_path: Path) -> GotQuestionsConfig:
    return GotQuestionsConfig(
        cache_db_path=tmp_path / "cache.db",
        questions_parquet=tmp_path / "questions.parquet",
        packs_parquet=tmp_path / "packs.parquet",
    )


def _pack(text: str) -> dict:
    return {
        "id": 9,
        "title": "Свежий пакет",
        "startDate": "$D2024-03-01T00:00:00.000Z",
        "editors": [{"person": {"name": "Редактор пакета", "playerId": 7}}],
        "tIds": [],
        "tours": [{
            "id": 1,
            "editors": [{"person": {"name": "Редактор", "playerId": 8}}],
            "questions": [{
                "number": 1,
                "text": text,
                "answer": "ответ",
                "comment": "из кэша",
                "source": "https://example.test/src",
                "authors": [{"person": {"name": "Автор", "playerId": 3}}],
            }],
        }],
    }


def _write_questions(path: Path, text: str) -> None:
    pd.DataFrame([{
        "question_id": "gq_9_1",
        "rating_tournament_id": 0,
        "pack_id": 9,
        "tour_id": 1,
        "tour_number": 1,
        "question_number": 1,
        "text": text,
        "answer": "ответ из таблицы",
        "comment": "из таблицы",
        "author_names": ["Автор"],
        "author_player_ids": [3],
        "editor_names": ["Редактор"],
        "editor_player_ids": [8],
    }]).to_parquet(path, index=False)


def test_newer_parquet_keeps_its_text_and_takes_the_source(tmp_path: Path) -> None:
    config = _config(tmp_path)
    GotQuestionsCache(config.cache_db_path).save_pack(9, _pack("старый"))
    _write_questions(config.questions_parquet, "актуальный")

    frame = load_catalog(config)
    row = frame[frame["question_id"] == "gq_9_1"].iloc[0]
    assert row["text"] == "актуальный"
    assert row["comment"] == "из таблицы"
    assert row["source"] == "https://example.test/src"
    assert row["start_date"] == "2024-03-01"
    assert row["pack_title"] == "Свежий пакет"
    packs = pd.read_parquet(config.packs_parquet)
    assert list(packs["title"]) == ["Свежий пакет"]

    again = load_catalog(config)
    assert again.iloc[0]["text"] == "актуальный"


def test_newer_cache_replaces_the_question_text(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_questions(config.questions_parquet, "устаревший")
    os.utime(config.questions_parquet, (1, 1))
    GotQuestionsCache(config.cache_db_path).save_pack(9, _pack("из кэша"))

    frame = load_catalog(config)
    row = frame[frame["question_id"] == "gq_9_1"].iloc[0]
    assert row["text"] == "из кэша"
    assert row["source"] == "https://example.test/src"

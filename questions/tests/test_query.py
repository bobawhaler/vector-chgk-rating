"""Filters over a tiny question table. No parquet files and no network."""

from __future__ import annotations

import pandas as pd
import pytest

from corpus.query import QueryError, QuestionFilter, run
from corpus.records import frames_from_packs


def _questions() -> pd.DataFrame:
    cup = {
        "question_id": "10_1",
        "rating_tournament_id": 10,
        "pack_id": 5,
        "pack_title": "Кубок",
        "tour_number": 1,
        "question_number": 1,
        "number_in_tour": 1,
        "text": "Кот на крыше",
        "answer": "кот",
        "zachet": "",
        "nezachet": "",
        "comment": "про кота",
        "source": "https://example.test/cat",
        "start_date": "2024-05-01",
        "end_date": "2024-05-20",
        "pub_date": "2024-06-01",
        "author_names": ["Анна"],
        "author_player_ids": [3],
        "editor_names": ["Редактор тура"],
        "editor_player_ids": [8],
        "pack_editor_names": ["Редактор пакета"],
        "pack_editor_player_ids": [7],
    }
    mirror = dict(cup)
    mirror["question_id"] = "11_1"
    mirror["rating_tournament_id"] = 11
    winter = {
        "question_id": "gq_6_1",
        "rating_tournament_id": 0,
        "pack_id": 6,
        "pack_title": "Зима",
        "tour_number": 2,
        "question_number": 4,
        "number_in_tour": 1,
        "text": "Пёс во дворе",
        "answer": "пёс",
        "zachet": "",
        "nezachet": "",
        "comment": "",
        "source": "",
        "start_date": "2020-01-01",
        "end_date": "2020-01-01",
        "pub_date": "2020-02-01",
        "author_names": ["Борис"],
        "author_player_ids": [4],
        "editor_names": [],
        "editor_player_ids": [],
        "pack_editor_names": [],
        "pack_editor_player_ids": [],
    }
    undated = dict(winter)
    undated["question_id"] = "gq_7_1"
    undated["pack_id"] = 7
    undated["pack_title"] = "Без даты"
    undated["text"] = "Без даты"
    undated["start_date"] = ""
    undated["end_date"] = ""
    undated["pub_date"] = ""
    return pd.DataFrame([cup, mirror, winter, undated])


def _find(**kwargs: object) -> list[str]:
    page = run(_questions(), QuestionFilter(**kwargs))  # type: ignore[arg-type]
    return [str(row["question_id"]) for row in page.rows]


def test_author_editor_and_date_narrow_one_package() -> None:
    assert _find(author="анна") == ["10_1"]
    assert _find(author_id=3, editor="пакета") == ["10_1"]
    assert _find(editor="тура") == ["10_1"]
    assert _find(editor_id=7) == ["10_1"]
    assert _find(since="2024-05-10", until="2024-05-10") == ["10_1"]
    assert _find(since="2024-07-01") == []
    assert _find(pack="куб") == ["10_1"]
    assert _find(pack_id=6) == ["gq_6_1"]


def test_unique_collapses_tournament_copies_unless_asked() -> None:
    assert _find(author="Анна") == ["10_1"]
    found = _find(author="Анна", unique=False)
    assert found == ["10_1", "11_1"]


def test_text_field_does_not_search_the_source() -> None:
    assert _find(text="кот") == ["10_1"]
    assert _find(text="example.test") == []
    assert _find(text="example.test", where="source") == ["10_1"]
    assert _find(text="a.b", where="all") == []


def test_tour_number_uses_the_number_inside_the_tour() -> None:
    assert _find(pack_id=6, tour=2, number=1) == ["gq_6_1"]
    assert _find(number=4) == []


def test_newest_packages_come_first_and_undated_go_last() -> None:
    order = _find(limit=10)
    assert order == ["10_1", "gq_6_1", "gq_7_1"]


def test_bad_day_is_rejected() -> None:
    with pytest.raises(QueryError):
        QuestionFilter(since="2024/05/01")


def test_second_tour_keeps_its_own_question_number() -> None:
    questions, packs = frames_from_packs([{
        "id": 3,
        "title": "Пакет",
        "startDate": "$D2020-01-02T00:00:00.000Z",
        "endDate": "2020-01-09",
        "pubDate": "2020-02-01T00:00:00.000Z",
        "tIds": [],
        "editors": [{"person": {"name": "Редактор пакета", "playerId": 7}}],
        "tours": [
            {
                "id": 1,
                "editors": [{"person": {"name": "Редактор тура", "playerId": 8}}],
                "questions": [{
                    "number": 1,
                    "text": "A",
                    "answer": "a",
                    "source": "http://a",
                    "zachet": "з",
                    "nezachet": "н",
                    "comment": "к",
                    "authors": [{"person": {"name": "Автор", "playerId": 9}}],
                }],
            },
            {"id": 2, "editors": [], "questions": [{"number": 1, "text": "B", "answer": "b"}]},
        ],
    }])
    assert list(questions["question_number"]) == [1, 2]
    assert list(questions["number_in_tour"]) == [1, 1]
    assert questions.iloc[0]["source"] == "http://a"
    assert questions.iloc[0]["start_date"] == "2020-01-02"
    assert questions.iloc[1]["editor_names"] == []
    assert questions.iloc[1]["pack_editor_names"] == ["Редактор пакета"]
    assert packs.iloc[0]["date"] == "2020-01-02"
    assert packs.iloc[0]["editor_player_ids"] == [7]

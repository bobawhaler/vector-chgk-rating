"""Corpus summaries on a tiny frame. No parquet files and no network."""

from __future__ import annotations

import pandas as pd

from corpus.analyze import link_coverage, people, search, summarize


def _questions() -> pd.DataFrame:
    return pd.DataFrame({
        "question_id": ["10_1", "10_2", "gq_3_1"],
        "rating_tournament_id": [10, 10, 0],
        "pack_id": [1, 1, 2],
        "question_number": [1, 2, 1],
        "text": ["Кот на крыше", "Пёс во дворе", "Снова кот"],
        "answer": ["кот", "пёс", "кот"],
        "author_names": [["Анна"], ["Анна", "Борис"], ["Борис"]],
        "editor_names": [["Редактор"], ["Редактор"], []],
    })


def test_summary_counts_rows_people_and_links() -> None:
    text = summarize(_questions(), pd.DataFrame({"pack_id": [1, 2]}))
    assert "Вопросов: 3" in text
    assert "Пакетов в таблице пакетов: 2" in text
    assert "С привязкой к турниру рейтинга: 2" in text
    assert "Авторов: 2" in text
    assert "Редакторов туров: 1" in text


def test_people_ranks_by_question_rows() -> None:
    frame = people(_questions(), "author_names", limit=10)
    assert list(frame["name"]) == ["Анна", "Борис"]
    assert list(frame["questions"]) == [2, 2]


def test_link_coverage_uses_question_id() -> None:
    rating = pd.DataFrame({"question_id": ["10_1", "10_2", "11_1"]})
    text = link_coverage(rating, _questions())
    assert "Совпало по question_id: 2" in text
    assert "66.67%" in text


def test_search_is_case_insensitive_substring() -> None:
    found = search(_questions(), "КОТ")
    assert list(found["question_id"]) == ["10_1", "gq_3_1"]

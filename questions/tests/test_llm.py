"""Samples, prompts, and parsed model replies. No catalog and no network."""

from __future__ import annotations

import pandas as pd
import pytest

from corpus.prompt import batches, each, render
from corpus.query import QuestionFilter
from corpus.results import ResultsError, analyze, coverage, same_text, tally
from corpus.sample import take


def _questions() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "question_id": "a",
            "pack_id": 2,
            "pack_title": "Новый",
            "question_number": 1,
            "number_in_tour": 1,
            "text": "Про кота",
            "answer": "кот",
            "zachet": "",
            "nezachet": "",
            "comment": "котёнок",
            "source": "http://cat",
            "start_date": "2024-05-01",
            "author_names": ["Анна"],
            "editor_names": ["Ира"],
            "pack_editor_names": [],
        },
        {
            "question_id": "b",
            "pack_id": 2,
            "pack_title": "Новый",
            "question_number": 2,
            "number_in_tour": 2,
            "text": "Про пса",
            "answer": "пёс",
            "zachet": "",
            "nezachet": "",
            "comment": "",
            "source": "",
            "start_date": "2024-05-01",
            "author_names": ["Борис"],
            "editor_names": ["Ира"],
            "pack_editor_names": [],
        },
        {
            "question_id": "c",
            "pack_id": 1,
            "pack_title": "Старый",
            "question_number": 1,
            "number_in_tour": 1,
            "text": "Про дом",
            "answer": "дом",
            "zachet": "",
            "nezachet": "",
            "comment": "",
            "source": "http://house",
            "start_date": "2020-01-01",
            "author_names": ["Анна"],
            "editor_names": [],
            "pack_editor_names": ["Глеб"],
        },
    ])


def test_take_without_a_seed_is_the_newest_slice() -> None:
    sample = take(_questions(), size=2)
    assert sample.population == 3
    assert sample.ids() == ("a", "b")


def test_seed_is_stable_and_changes_the_draw() -> None:
    assert take(_questions(), size=2, seed=1).ids() == take(_questions(), size=2, seed=1).ids()
    draws = {take(_questions(), size=2, seed=seed).ids() for seed in range(8)}
    assert len(draws) > 1
    assert set().union(*draws) <= {"a", "b", "c"}


def test_by_author_alternates_groups() -> None:
    sample = take(_questions(), size=3, by="author")
    authors = [row["author_names"][0] for row in sample.rows]
    assert authors == ["Анна", "Борис", "Анна"]


def test_filter_limit_does_not_shrink_the_population() -> None:
    sample = take(_questions(), QuestionFilter(author="Анна", limit=1), size=10)
    assert sample.population == 2
    assert sample.ids() == ("a", "c")


def test_render_hides_the_answer_when_asked_and_marks_ids() -> None:
    sample = take(_questions(), size=1)
    prompt = render(sample, "Назови тему в поле theme.", reveal=("text",))
    assert "Про кота" in prompt.text
    assert "Ответ:" not in prompt.text
    assert "[id=a]" in prompt.text
    assert "JSON-массив" in prompt.text
    assert prompt.ids == ("a",)


def test_batches_keep_every_id_and_respect_a_tight_budget() -> None:
    rows = _questions().to_dict("records")
    one = render([rows[0]], "Кратко.", reveal=("text",))
    parts = batches(rows, "Кратко.", max_chars=len(one.text), reveal=("text",))
    assert [part.ids for part in parts] == [("a",), ("b",), ("c",)]
    assert all(len(part.text) <= len(one.text) or len(part.ids) == 1 for part in parts)


def test_each_makes_one_prompt_per_question() -> None:
    prompts = each(_questions().iloc[:2], "Ответь.", reveal=("text",))
    assert [prompt.ids for prompt in prompts] == [("a",), ("b",)]


def test_analyze_joins_a_fenced_reply_and_counts_a_field() -> None:
    prompt = render(_questions(), "Назови тему в поле theme.", reveal=("text",))
    reply = """Вот результат:
```json
[
  {"id": "a", "theme": "животные"},
  {"id": "b", "theme": "животные"},
  {"id": "c", "theme": "дом"},
  {"id": "a", "theme": "повтор"},
  {"id": "z", "theme": "чужой"}
]
```
"""
    report = analyze(prompt, reply)
    assert [item.question_id for item in report.findings] == ["a", "b", "c"]
    assert report.findings[0].fields == {"theme": "животные"}
    assert report.duplicate_ids == ("a",)
    assert report.unknown_ids == ("z",)
    assert coverage(report) == {"asked": 3, "matched": 3, "missing": 0, "unknown": 1, "duplicate": 1}
    assert tally(report, "theme") == [("животные", 2), ("дом", 1)]


def test_missing_id_and_exact_answer_match() -> None:
    prompt = render([_questions().iloc[0].to_dict()], "Дай ответ в поле guess.", reveal=("text",))
    report = analyze(prompt, '{"answers": [{"id": "a", "guess": "  Кот "}]}')
    assert report.missing_ids == ()
    score = same_text(report, {"a": {"answer": "кот"}}, "guess", "answer")
    assert score == {"compared": 1, "equal": 1, "rate": 1.0}
    empty = analyze(prompt, "[]")
    assert empty.missing_ids == ("a",)


def test_prose_without_json_is_rejected() -> None:
    prompt = render([{"question_id": "a", "text": "x"}], "Тема.", reveal=("text",))
    with pytest.raises(ResultsError):
        analyze(prompt, "Не могу ответить списком.")


def test_bad_sample_and_prompt_arguments() -> None:
    with pytest.raises(ValueError):
        take(_questions(), size=0)
    with pytest.raises(ValueError):
        render(_questions(), "   ")
    with pytest.raises(ValueError):
        render(_questions(), "Тема.", reveal=("secret",))

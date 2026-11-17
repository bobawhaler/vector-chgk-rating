"""Corpus card for one realia: links and editors of the neighborhood."""

from __future__ import annotations

import pandas as pd
import pytest

from corpus.knowledge import KnowledgeError, about


def _frame() -> pd.DataFrame:
    rows = [
        ("q1", "Батончик калорийный, это Сникерс для солдат.", "Марс", "Анна"),
        ("q2", "Рядом со Сникерсом в витрине лежит «Милки Вэй».", "Twix", "Борис"),
        ("q3", "Компания носит имя бога.", "Сникерс", "Анна"),
        ("q4", "В рекламе Twix спорит, а Марс молчит.", "пауза", "Анна"),
        ("q5", "Четвёртая планета.", "Марс", "Вера"),
        ("q6", "Twix спорит с Марсом в рекламе.", "Баунти", "Глеб"),
    ]
    return pd.DataFrame([
        {
            "question_id": question_id,
            "pack_id": index + 1,
            "question_number": 1,
            "pack_title": "Пакет",
            "start_date": "2020-01-01",
            "text": text,
            "answer": answer,
            "comment": "",
            "editor_names": [editor],
            "pack_editor_names": [],
        }
        for index, (question_id, text, answer, editor) in enumerate(rows)
    ])


def test_snickers_links_the_answer_and_keeps_mars_out_of_the_planet():
    card = about(_frame(), "сникерс", broad_at=2)
    assert {link.name for link in card.links} == {"Марс", "Twix", "Милки Вэй"}
    assert card.links[0].questions == 1
    editors = {editor.name: editor.questions for editor in card.editors}
    assert editors == {"Анна": 2, "Борис": 1}
    similar = {editor.name: editor.realia for editor in card.similar_editors}
    assert "Вера" not in similar
    assert "Марс" in similar["Анна"] and "Twix" in similar["Анна"]
    assert "Twix" in similar["Борис"]
    assert "Глеб" in similar


def test_empty_name_is_rejected():
    with pytest.raises(KnowledgeError):
        about(_frame(), "  ")

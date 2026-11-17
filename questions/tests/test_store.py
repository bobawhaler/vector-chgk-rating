"""Decomposition store. No catalog and no model."""

from __future__ import annotations

from corpus.store import (
    answers_by_form,
    editors_for_realia,
    open_base,
    put,
    questions_by_area,
    questions_by_editor,
    questions_by_move,
    questions_by_source_language,
    questions_by_source_theme,
    questions_by_source_type,
    unlisted,
)


def _law() -> dict:
    return {
        "realia": [
            {"id": "nevada", "name": "Невада", "kind": "place", "status": "given"},
            {"id": "law", "name": "сухой закон", "kind": "thing", "status": "yielded"},
        ],
        "properties": [
            {"id": "ban", "bearer": "law", "type": "право", "text": "в Неваде дешевле на доллар"},
        ],
        "facts": [],
        "relations": [
            {"id": "from-place", "link": "происхождение", "text": "слово от топонима", "realia": ["law", "nevada"]},
        ],
        "moves": [{"code": "statute", "reads": ["ban"], "yields": ["law"]}],
        "techniques": ["форма"],
        "forms": [{"wording": "два слова на парные согласные", "answer": "сухой закон"}],
        "areas": ["новости"],
        "sources": [
            {
                "citation": "Телепрограмма «Новости», эфир «Первого канала», 25.11.11",
                "types": ["телепередача"],
                "languages": ["русский"],
                "themes": ["запреты"],
            }
        ],
    }


def test_lookup_by_editor_theme_form_and_new_codes(tmp_path):
    connection = open_base(tmp_path / "knowledge.sqlite")
    put(connection, "q1", _law(), editors=("Анна",))
    other = _law()
    other["forms"] = [{"wording": "два слова на парные согласные", "answer": "голубая кровь"}]
    other["realia"][1]["name"] = "голубая кровь"
    other["areas"] = ["новости", "спорт"]
    other["sources"] = [
        {
            "citation": "https://example.test/blood",
            "types": ["комикс"],
            "languages": ["испанский"],
            "themes": ["аристократия"],
        }
    ]
    put(connection, "q2", other, editors=("Борис", "Анна"))

    assert questions_by_editor(connection, "анна") == ("q1", "q2")
    assert questions_by_area(connection, "Новости") == ("q1", "q2")
    assert questions_by_source_type(connection, "телепередача") == ("q1",)
    assert questions_by_source_type(connection, "комикс") == ("q2",)
    assert questions_by_source_language(connection, "испанский") == ("q2",)
    assert questions_by_source_theme(connection, "аристократия") == ("q2",)
    assert questions_by_move(connection, "statute") == ("q1", "q2")
    assert answers_by_form(connection, "Два слова на парные согласные") == ("голубая кровь", "сухой закон")
    assert editors_for_realia(connection, "Сухой закон") == (("Анна", 1),)
    fresh = unlisted(connection)
    assert fresh["moves"] == ("statute",)
    assert fresh["links"] == ("происхождение",)
    assert fresh["property_types"] == ("право",)
    assert fresh["techniques"] == ("форма",)
    assert fresh["areas"] == ("спорт",)
    assert fresh["source_types"] == ("комикс",)
    assert fresh["source_languages"] == ("испанский",)
    assert fresh["source_themes"] == ("аристократия", "запреты")

    again = _law()
    again["areas"] = []
    put(connection, "q1", again, editors=("Вера",))
    assert questions_by_editor(connection, "Анна") == ("q2",)
    assert questions_by_area(connection, "новости") == ("q2",)
    connection.close()

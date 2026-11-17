"""Decomposition of a question into what its moves read."""

from __future__ import annotations

import pytest

from corpus.prompt import render
from corpus.realia import INSTRUCTION, RealiaError, novel, parse


def _blowhole() -> dict:
    return {
        "realia": [
            {"id": "dolphin", "name": "дельфин", "kind": "thing", "status": "given"},
            {"id": "blowhole", "name": "дыхало", "kind": "thing", "status": "yielded"},
        ],
        "properties": [
            {"id": "on-the-head", "bearer": "blowhole", "text": "после превращения остаётся на затылке, поэтому шляпу не снимают"},
        ],
        "facts": [],
        "relations": [],
        "moves": [{"code": "property", "reads": ["on-the-head"], "yields": ["blowhole"]}],
    }


def _mausoleum() -> dict:
    return {
        "realia": [
            {"id": "lenin", "name": "Ленин", "kind": "person", "status": "given"},
            {"id": "hammer", "name": "Хаммер", "kind": "person", "status": "given"},
            {"id": "mausoleum", "name": "Мавзолей", "kind": "place", "status": "yielded"},
        ],
        "properties": [],
        "facts": [
            {"id": "pass", "text": "Ленин выдал Хаммеру пропуск в любое время", "realia": ["lenin", "hammer"]},
            {"id": "night", "text": "ночью по этому пропуску Хаммера пустили в закрытое место", "realia": ["hammer", "mausoleum"]},
        ],
        "relations": [],
        "moves": [{"code": "precedent", "reads": ["pass", "night"], "yields": ["mausoleum"]}],
    }


def _snickers() -> dict:
    return {
        "realia": [
            {"id": "bar", "name": "Сникерс", "kind": "thing", "status": "given"},
            {"id": "soldiers", "name": "солдаты", "kind": "group", "status": "yielded"},
            {"id": "mars", "name": "Марс", "kind": "person", "status": "yielded"},
        ],
        "properties": [
            {"id": "ration", "bearer": "soldiers", "text": "калорийно, мало весит и занимает мало места"},
        ],
        "facts": [
            {"id": "patron", "text": "покровитель римских солдат — Марс", "realia": ["soldiers", "mars"]},
        ],
        "relations": [],
        "moves": [
            {"code": "property", "reads": ["ration"], "yields": ["soldiers"]},
            {"code": "precedent", "reads": ["patron"], "yields": ["mars"]},
        ],
    }


def test_property_move_names_the_bearer():
    found = parse(_blowhole())
    assert found.codes() == ("property",)
    assert found.moves[0].yields == ("blowhole",)


def test_precedent_reads_facts_of_one_case():
    found = parse(_mausoleum())
    assert found.codes() == ("precedent",)
    assert found.facts[0].realia == ("lenin", "hammer")


def test_chain_keeps_move_order():
    assert parse(_snickers()).codes() == ("property", "precedent")


def test_word_move_reads_a_relation():
    found = parse({
        "realia": [
            {"id": "source", "name": "Starbucks", "kind": "word", "status": "given"},
            {"id": "result", "name": "Bartsucks", "kind": "word", "status": "yielded"},
        ],
        "properties": [],
        "facts": [],
        "relations": [
            {"id": "swap", "kind": "word", "text": "перестановка букв", "realia": ["source", "result"]},
        ],
        "moves": [{"code": "word", "reads": ["swap"], "yields": ["result"]}],
    })
    assert found.codes() == ("word",)


def test_property_cannot_read_a_fact():
    payload = _mausoleum()
    payload["moves"] = [{"code": "property", "reads": ["pass"], "yields": ["mausoleum"]}]
    with pytest.raises(RealiaError, match="property"):
        parse(payload)


def test_yielded_realia_must_be_named_by_a_move():
    payload = _blowhole()
    payload["realia"].append({"id": "hat", "name": "шляпа", "kind": "thing", "status": "yielded"})
    with pytest.raises(RealiaError, match="hat"):
        parse(payload)


def test_unknown_realia_kind_is_kept():
    payload = _snickers()
    payload["realia"][2]["kind"] = "candy"
    found = parse(payload)
    assert found.realia[2].kind == "candy"
    assert novel(found)["realia_kinds"] == ("candy",)


def test_relation_needs_two_realia():
    with pytest.raises(RealiaError, match="две реалии"):
        parse({
            "realia": [{"id": "only", "name": "Марс", "kind": "person", "status": "given"}],
            "properties": [],
            "facts": [],
            "relations": [{"id": "alone", "kind": "word", "text": "то же слово", "realia": ["only"]}],
            "moves": [{"code": "word", "reads": ["alone"], "yields": ["only"]}],
        })


def test_author_and_meter_are_links_beside_the_move():
    found = parse({
        "realia": [
            {"id": "homer", "name": "Гомер", "kind": "person", "status": "yielded"},
            {"id": "odyssey", "name": "Одиссея", "kind": "work", "status": "given"},
            {"id": "iliad", "name": "Илиада", "kind": "work", "status": "given"},
            {"id": "hexameter", "name": "гекзаметр", "kind": "work", "status": "given"},
        ],
        "properties": [
            {"id": "poet", "bearer": "homer", "text": "древнегреческий поэт"},
            {"id": "blind", "bearer": "homer", "text": "слепой"},
            {"id": "when", "bearer": "homer", "text": "время жизни известно приблизительно"},
        ],
        "facts": [],
        "relations": [
            {"id": "wrote", "link": "автор", "text": "Гомер написал «Одиссею» и «Илиаду»", "realia": ["homer", "odyssey", "iliad"]},
            {"id": "meter", "link": "размер", "text": "обе поэмы написаны гекзаметром", "realia": ["odyssey", "iliad", "hexameter"]},
        ],
        "moves": [{"code": "property", "reads": ["blind", "poet"], "yields": ["homer"]}],
    })
    assert found.relations[0].link == "автор"
    assert found.relations[0].kind == ""
    assert found.codes() == ("property",)


def test_property_move_cannot_read_an_author_link():
    with pytest.raises(RealiaError, match="автор"):
        parse({
            "realia": [
                {"id": "homer", "name": "Гомер", "kind": "person", "status": "yielded"},
                {"id": "iliad", "name": "Илиада", "kind": "work", "status": "given"},
            ],
            "properties": [],
            "facts": [],
            "relations": [
                {"id": "wrote", "link": "автор", "text": "Гомер написал «Илиаду»", "realia": ["homer", "iliad"]},
            ],
            "moves": [{"code": "property", "reads": ["wrote"], "yields": ["homer"]}],
        })


def test_new_move_form_and_technique_are_kept():
    found = parse({
        "realia": [
            {"id": "bar", "name": "Сникерс", "kind": "thing", "status": "given"},
            {"id": "law", "name": "сухой закон", "kind": "thing", "status": "yielded"},
        ],
        "properties": [
            {"id": "dry", "bearer": "law", "type": "право", "text": "запрет спиртного в штате"},
        ],
        "facts": [],
        "relations": [],
        "moves": [{"code": "statute", "reads": ["dry"], "yields": ["law"]}],
        "techniques": ["форма", "проверка"],
        "forms": [{"wording": "два слова на парные согласные", "answer": "сухой закон"}],
        "areas": ["мемы", "спорт"],
        "sources": [
            {
                "citation": "https://example.test/law",
                "types": ["сайт", "комикс"],
                "languages": ["английский"],
                "themes": ["запреты"],
            }
        ],
    })
    assert found.codes() == ("statute",)
    assert found.forms[0].answer == "сухой закон"
    assert novel(found)["moves"] == ("statute",)
    assert novel(found)["techniques"] == ("форма",)
    assert novel(found)["property_types"] == ("право",)
    assert novel(found)["areas"] == ("спорт",)
    assert novel(found)["source_types"] == ("комикс",)
    assert novel(found)["source_themes"] == ("запреты",)


def test_instruction_renders_for_a_question():
    prompt = render(
        [{"question_id": "q1", "text": "Дельфин не снимает шляпу.", "answer": "дыхало"}],
        INSTRUCTION,
        reveal=("text", "answer"),
    )
    assert prompt.ids == ("q1",)
    assert "свойства" in prompt.text
    assert "Дельфин не снимает шляпу." in prompt.text

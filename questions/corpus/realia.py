"""Facts, realia, properties and relations a move reads.

This module does not call a model. ``INSTRUCTION`` is the text to pass to
``corpus.prompt.render``. ``parse`` checks the object the model returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

MOVES = (
    "word",
    "precedent",
    "property",
    "parallel",
    "formula",
    "literal",
    "function",
    "count",
    "school",
    "meta",
    "concept",
)

# What a move is allowed to read. property and precedent are not relations.
_READS = {
    "property": "property",
    "precedent": "fact",
    "word": "word",
    "parallel": "parallel",
    "formula": "formula",
    "literal": "literal",
    "function": "function",
    "count": "count",
    "school": "school",
    "meta": "meta",
    "concept": "concept",
}

RELATION_KINDS = tuple(code for code, kind in _READS.items() if kind == code)

# A realia is one object another question could name again. Homonyms differ in kind or in which object.
REALIA_KINDS = ("person", "group", "thing", "place", "work", "event", "word")

# Seen so far. The store keeps a value that is not in these tuples.
TECHNIQUES = ("упаковка", "проверка", "подмена", "окраска", "костыль")
LINKS = ("автор", "размер", "покровитель")
PROPERTY_TYPES: tuple[str, ...] = ()
AREAS = ("искусство", "литература", "история", "новости", "мемы")
SOURCE_TYPES = ("книга", "статья", "сайт", "фильм", "энциклопедия", "телепередача", "газета")
LANGUAGES = ("русский", "английский", "французский", "немецкий", "латынь", "древнегреческий", "японский")
SOURCE_THEMES: tuple[str, ...] = ()

INSTRUCTION = """\
Разложи вопрос на реалии, свойства, факты и связи. Выписывай каждую явно названную реалию и каждое предложение, которое даёт ей свойство или связывает её с другой названной реалией. Ход читает те записи, которыми берётся ответ. Упаковку, проверку, подмену синонимом, окраску, костыль и форму не выписывай свойством реалии: это приёмы редактора. Форма — требование из вопроса и ответ, который под него подошёл. Приём — не класс хода и не запись базы. Имя, которое приём всё-таки называет, остаётся реалией. Одинаковая строка не делает одну реалию: бог Марс, планета Марс и батончик «Марс» — три реалии.

Реалия — объект, который другой вопрос может назвать снова. kind: person, group, thing, place, work, event или word. status "given", если она названа в вопросе до хода, и "yielded", если её называет ход. Выписывай и явно названные в тексте, и ту, которую называет ход.

Свойство — черта одного носителя, которую можно повторить в другом вопросе про него, без даты и без «однажды». bearer — id этой реалии. Если черта называет другую реалию, это уже связь, не свойство.
Факт — один эпизод. С другими участниками или в другой момент это уже другое событие. realia — id участников.
Связь — устойчивое отношение минимум двух реалий. link — имя связи: автор, размер, покровитель. kind — код хода, и только если ход читает эту связь: word, parallel, formula, literal, function, count, school, meta, concept. Связь без хода kind не имеет.

Ходы идут по порядку. code — класс хода. reads — id свойств, фактов или связей. yields — id реалий, которые ход называет.
property читает только свойства, precedent — только факты, остальные известные коды — связь своего kind. Код, kind реалии или приём, которого нет в списках выше, всё равно выпиши как есть.

Приёмы — отдельный список name: упаковка, проверка, подмена, окраска, костыль или новое имя.
Формы — отдельный список. wording — требование как в вопросе, answer — ответ, который под него подошёл.
Области вопроса — список areas: искусство, литература, история, новости, мемы или новая область.
Источники — список. citation — один пункт поля «источник». types, languages и themes — списки: тип (книга, статья, сайт, фильм, энциклопедия, телепередача, газета или новый), язык и тема источника. Несколько пунктов поля «источник» — несколько элементов. Новое значение всё равно выпиши.
""".strip()


class RealiaError(ValueError):
    """The decomposition does not match the move it claims to make."""


@dataclass(frozen=True)
class Realia:
    id: str
    name: str
    kind: str
    status: str


@dataclass(frozen=True)
class Property:
    id: str
    bearer: str
    text: str
    type: str = ""


@dataclass(frozen=True)
class Fact:
    id: str
    text: str
    realia: tuple[str, ...]


@dataclass(frozen=True)
class Relation:
    id: str
    kind: str
    text: str
    realia: tuple[str, ...]
    link: str = ""


@dataclass(frozen=True)
class Move:
    code: str
    reads: tuple[str, ...]
    yields: tuple[str, ...]


@dataclass(frozen=True)
class Technique:
    name: str


@dataclass(frozen=True)
class Form:
    wording: str
    answer: str


@dataclass(frozen=True)
class Source:
    citation: str
    types: tuple[str, ...]
    languages: tuple[str, ...]
    themes: tuple[str, ...]


@dataclass(frozen=True)
class Decomposition:
    realia: tuple[Realia, ...]
    properties: tuple[Property, ...]
    facts: tuple[Fact, ...]
    relations: tuple[Relation, ...]
    moves: tuple[Move, ...]
    techniques: tuple[Technique, ...] = ()
    forms: tuple[Form, ...] = ()
    areas: tuple[str, ...] = ()
    sources: tuple[Source, ...] = ()

    def codes(self) -> tuple[str, ...]:
        """Move codes in the order the question is solved."""
        return tuple(move.code for move in self.moves)


def novel(found: Decomposition) -> dict[str, tuple[str, ...]]:
    """Values this decomposition uses that are not in the inventories above."""
    return {
        "moves": _fresh(found.codes(), MOVES),
        "realia_kinds": _fresh((item.kind for item in found.realia), REALIA_KINDS),
        "relation_kinds": _fresh((item.kind for item in found.relations if item.kind), RELATION_KINDS),
        "links": _fresh((item.link for item in found.relations if item.link), LINKS),
        "property_types": _fresh((item.type for item in found.properties if item.type), PROPERTY_TYPES),
        "techniques": _fresh((item.name for item in found.techniques), TECHNIQUES),
        "areas": _fresh(found.areas, AREAS),
        "source_types": _fresh((name for item in found.sources for name in item.types), SOURCE_TYPES),
        "source_languages": _fresh((name for item in found.sources for name in item.languages), LANGUAGES),
        "source_themes": _fresh((name for item in found.sources for name in item.themes), SOURCE_THEMES),
    }


def _fresh(values, known: tuple[str, ...]) -> tuple[str, ...]:
    seen = {item.casefold() for item in known}
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return tuple(result)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RealiaError(f"{label}: нужна непустая строка.")
    return value.strip()


def _ids(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RealiaError(f"{label}: нужен непустой список id.")
    result = tuple(_text(item, label) for item in value)
    if len(set(result)) != len(result):
        raise RealiaError(f"{label}: id повторяется.")
    return result


def parse(payload: Mapping[str, object]) -> Decomposition:
    """Build a decomposition and reject a move that reads the wrong layer."""
    if not isinstance(payload, Mapping):
        raise RealiaError("Разбор должен быть объектом.")
    realia = _realia(payload.get("realia"))
    properties = _properties(payload.get("properties"))
    facts = _facts(payload.get("facts"))
    relations = _relations(payload.get("relations"))
    known = {item.id: "realia" for item in realia}
    kinds: dict[str, str] = {}
    for item in properties:
        _claim(known, item.id, "property")
        kinds[item.id] = "property"
    for item in facts:
        _claim(known, item.id, "fact")
        kinds[item.id] = "fact"
    for item in relations:
        _claim(known, item.id, item.kind or item.link)
        kinds[item.id] = item.kind or item.link
    by_realia = {item.id for item in realia}
    for item in properties:
        if item.bearer not in by_realia:
            raise RealiaError(f"Свойство {item.id} ссылается на неизвестную реалию {item.bearer}.")
    for item in (*facts, *relations):
        missing = [name for name in item.realia if name not in by_realia]
        if missing:
            raise RealiaError(f"{item.id} ссылается на неизвестные реалии: {', '.join(missing)}.")
    moves = _moves(payload.get("moves"), kinds, by_realia)
    techniques = _techniques(payload.get("techniques"))
    forms = _forms(payload.get("forms"))
    areas = _labels(payload.get("areas"), "areas")
    sources = _sources(payload.get("sources"))
    yielded = {name for move in moves for name in move.yields}
    for item in realia:
        if item.status == "yielded" and item.id not in yielded:
            raise RealiaError(f"Реалия {item.id} помечена yielded, но ни один ход её не называет.")
        if item.status == "given" and item.id in yielded:
            raise RealiaError(f"Реалия {item.id} уже названа в вопросе и не может быть результатом хода.")
    if not moves:
        raise RealiaError("Нужен хотя бы один ход.")
    return Decomposition(realia, properties, facts, relations, moves, techniques, forms, areas, sources)


def _claim(known: dict[str, str], item_id: str, kind: str) -> None:
    if item_id in known:
        raise RealiaError(f"Id {item_id} повторяется.")
    known[item_id] = kind


def _realia(value: object) -> tuple[Realia, ...]:
    rows = _rows(value, "realia")
    result = []
    for row in rows:
        status = _text(row.get("status"), "status")
        if status not in ("given", "yielded"):
            raise RealiaError("status реалии: given или yielded.")
        kind = _text(row.get("kind"), "kind")
        result.append(Realia(_text(row.get("id"), "id"), _text(row.get("name"), "name"), kind, status))
    if not result:
        raise RealiaError("Нужна хотя бы одна реалия.")
    return tuple(result)


def _properties(value: object) -> tuple[Property, ...]:
    return tuple(
        Property(
            _text(row.get("id"), "id"),
            _text(row.get("bearer"), "bearer"),
            _text(row.get("text"), "text"),
            _optional(row.get("type")),
        )
        for row in _rows(value, "properties", empty=True)
    )


def _facts(value: object) -> tuple[Fact, ...]:
    return tuple(
        Fact(_text(row.get("id"), "id"), _text(row.get("text"), "text"), _ids(row.get("realia"), "realia"))
        for row in _rows(value, "facts", empty=True)
    )


def _relations(value: object) -> tuple[Relation, ...]:
    result = []
    for row in _rows(value, "relations", empty=True):
        raw_kind = row.get("kind")
        kind = raw_kind.strip() if isinstance(raw_kind, str) else ""
        raw_link = row.get("link")
        link = raw_link.strip() if isinstance(raw_link, str) else ""
        item_id = _text(row.get("id"), "id")
        if not kind and not link:
            raise RealiaError(f"Связь {item_id}: нужно имя связи или kind хода.")
        ends = _ids(row.get("realia"), "realia")
        if len(ends) < 2:
            raise RealiaError(f"Связь {item_id} соединяет минимум две реалии.")
        result.append(Relation(item_id, kind, _text(row.get("text"), "text"), ends, link))
    return tuple(result)


def _moves(value: object, kinds: Mapping[str, str], realia: set[str]) -> tuple[Move, ...]:
    result = []
    for row in _rows(value, "moves"):
        code = _text(row.get("code"), "code")
        reads = _ids(row.get("reads"), "reads")
        yields = _ids(row.get("yields"), "yields")
        expected = _READS.get(code)
        for item_id in reads:
            if item_id not in kinds:
                raise RealiaError(f"Ход {code} читает неизвестный id {item_id}.")
            if expected is not None and kinds[item_id] != expected:
                raise RealiaError(f"Ход {code} читает {item_id} вида {kinds[item_id]}, а нужен {expected}.")
        missing = [item_id for item_id in yields if item_id not in realia]
        if missing:
            raise RealiaError(f"Ход {code} называет неизвестные реалии: {', '.join(missing)}.")
        result.append(Move(code, reads, yields))
    return tuple(result)


def _optional(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _techniques(value: object) -> tuple[Technique, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise RealiaError("techniques: нужен список.")
    result: list[Technique] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, Mapping):
            name = _text(item.get("name"), "name")
        else:
            raise RealiaError("techniques: нужна строка или объект с name.")
        if not name:
            raise RealiaError("techniques: нужно имя.")
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(Technique(name))
    return tuple(result)


def _labels(value: object, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise RealiaError(f"{label}: нужен список.")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise RealiaError(f"{label}: нужна строка.")
        name = item.strip()
        if not name:
            raise RealiaError(f"{label}: нужна непустая строка.")
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(name)
    return tuple(result)


def _sources(value: object) -> tuple[Source, ...]:
    result: list[Source] = []
    for row in _rows(value, "sources", empty=True):
        result.append(
            Source(
                _text(row.get("citation"), "citation"),
                _labels(row.get("types"), "types"),
                _labels(row.get("languages"), "languages"),
                _labels(row.get("themes"), "themes"),
            )
        )
    return tuple(result)


def _forms(value: object) -> tuple[Form, ...]:
    result: list[Form] = []
    seen: set[tuple[str, str]] = set()
    for row in _rows(value, "forms", empty=True):
        wording = _text(row.get("wording"), "wording")
        answer = _text(row.get("answer"), "answer")
        key = (wording.casefold(), answer.casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append(Form(wording, answer))
    return tuple(result)


def _rows(value: object, label: str, empty: bool = False) -> list[Mapping[str, object]]:
    if value is None and empty:
        return []
    if not isinstance(value, list):
        raise RealiaError(f"{label}: нужен список.")
    if not value and not empty:
        raise RealiaError(f"{label}: список пуст.")
    rows = []
    for item in value:
        if not isinstance(item, Mapping):
            raise RealiaError(f"{label}: элемент должен быть объектом.")
        rows.append(item)
    return rows

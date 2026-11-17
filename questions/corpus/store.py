"""Sqlite store for one decomposition per question.

The catalog stays the source of question text. This file keeps the
decomposition, the question areas, the classified sources and the editors
copied in with it, so a lookup by editor, area, source or move does not open
the parquet. A code that is not in
the written inventories is stored and shows up in ``unlisted``. This module
does not call a model.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Mapping

from corpus.realia import (
    AREAS,
    LANGUAGES,
    LINKS,
    MOVES,
    PROPERTY_TYPES,
    REALIA_KINDS,
    RELATION_KINDS,
    SOURCE_THEMES,
    SOURCE_TYPES,
    TECHNIQUES,
    Decomposition,
    parse,
)

_TABLES = (
    "move_read",
    "move_yield",
    "move",
    "technique",
    "form",
    "relation_realia",
    "relation",
    "fact_realia",
    "fact",
    "property",
    "realia",
    "source_type",
    "source_language",
    "source_theme",
    "source",
    "area",
    "editor",
    "question",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS question (
    question_id TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS editor (
    question_id TEXT NOT NULL,
    name TEXT NOT NULL,
    name_fold TEXT NOT NULL,
    PRIMARY KEY (question_id, name_fold)
);
CREATE INDEX IF NOT EXISTS editor_name ON editor (name_fold);
CREATE TABLE IF NOT EXISTS area (
    question_id TEXT NOT NULL,
    name TEXT NOT NULL,
    name_fold TEXT NOT NULL,
    PRIMARY KEY (question_id, name_fold)
);
CREATE INDEX IF NOT EXISTS area_name ON area (name_fold);
CREATE TABLE IF NOT EXISTS source (
    question_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    citation TEXT NOT NULL,
    PRIMARY KEY (question_id, position)
);
CREATE TABLE IF NOT EXISTS source_type (
    question_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    name TEXT NOT NULL,
    name_fold TEXT NOT NULL,
    PRIMARY KEY (question_id, position, name_fold)
);
CREATE INDEX IF NOT EXISTS source_type_name ON source_type (name_fold);
CREATE TABLE IF NOT EXISTS source_language (
    question_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    name TEXT NOT NULL,
    name_fold TEXT NOT NULL,
    PRIMARY KEY (question_id, position, name_fold)
);
CREATE INDEX IF NOT EXISTS source_language_name ON source_language (name_fold);
CREATE TABLE IF NOT EXISTS source_theme (
    question_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    name TEXT NOT NULL,
    name_fold TEXT NOT NULL,
    PRIMARY KEY (question_id, position, name_fold)
);
CREATE INDEX IF NOT EXISTS source_theme_name ON source_theme (name_fold);
CREATE TABLE IF NOT EXISTS realia (
    question_id TEXT NOT NULL,
    local_id TEXT NOT NULL,
    name TEXT NOT NULL,
    name_fold TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    PRIMARY KEY (question_id, local_id)
);
CREATE INDEX IF NOT EXISTS realia_name ON realia (name_fold);
CREATE INDEX IF NOT EXISTS realia_kind ON realia (kind);
CREATE TABLE IF NOT EXISTS property (
    question_id TEXT NOT NULL,
    local_id TEXT NOT NULL,
    bearer_id TEXT NOT NULL,
    type TEXT NOT NULL,
    type_fold TEXT NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (question_id, local_id)
);
CREATE INDEX IF NOT EXISTS property_type ON property (type_fold);
CREATE TABLE IF NOT EXISTS fact (
    question_id TEXT NOT NULL,
    local_id TEXT NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (question_id, local_id)
);
CREATE TABLE IF NOT EXISTS fact_realia (
    question_id TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    realia_id TEXT NOT NULL,
    PRIMARY KEY (question_id, fact_id, realia_id)
);
CREATE TABLE IF NOT EXISTS relation (
    question_id TEXT NOT NULL,
    local_id TEXT NOT NULL,
    link TEXT NOT NULL,
    link_fold TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (question_id, local_id)
);
CREATE INDEX IF NOT EXISTS relation_link ON relation (link_fold);
CREATE TABLE IF NOT EXISTS relation_realia (
    question_id TEXT NOT NULL,
    relation_id TEXT NOT NULL,
    realia_id TEXT NOT NULL,
    PRIMARY KEY (question_id, relation_id, realia_id)
);
CREATE TABLE IF NOT EXISTS move (
    question_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    code TEXT NOT NULL,
    code_fold TEXT NOT NULL,
    PRIMARY KEY (question_id, position)
);
CREATE INDEX IF NOT EXISTS move_code ON move (code_fold);
CREATE TABLE IF NOT EXISTS move_read (
    question_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    record_id TEXT NOT NULL,
    PRIMARY KEY (question_id, position, record_id)
);
CREATE TABLE IF NOT EXISTS move_yield (
    question_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    realia_id TEXT NOT NULL,
    PRIMARY KEY (question_id, position, realia_id)
);
CREATE TABLE IF NOT EXISTS technique (
    question_id TEXT NOT NULL,
    name TEXT NOT NULL,
    name_fold TEXT NOT NULL,
    PRIMARY KEY (question_id, name_fold)
);
CREATE INDEX IF NOT EXISTS technique_name ON technique (name_fold);
CREATE TABLE IF NOT EXISTS form (
    question_id TEXT NOT NULL,
    wording TEXT NOT NULL,
    wording_fold TEXT NOT NULL,
    answer TEXT NOT NULL,
    answer_fold TEXT NOT NULL,
    PRIMARY KEY (question_id, wording_fold, answer_fold)
);
CREATE INDEX IF NOT EXISTS form_wording ON form (wording_fold);
"""


def open_base(path: Path) -> sqlite3.Connection:
    """Open or create the decomposition store."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(_SCHEMA)
    return connection


def put(
    connection: sqlite3.Connection,
    question_id: str,
    payload: Decomposition | Mapping[str, object],
    *,
    editors: tuple[str, ...] | list[str] = (),
) -> Decomposition:
    """Replace this question's rows with one decomposition."""
    found = payload if isinstance(payload, Decomposition) else parse(payload)
    qid = question_id.strip()
    if not qid:
        raise ValueError("question_id: нужна непустая строка.")
    names = _unique(editors)
    topics = found.areas
    with connection:
        for table in _TABLES:
            connection.execute(f"DELETE FROM {table} WHERE question_id = ?", (qid,))
        connection.execute("INSERT INTO question (question_id) VALUES (?)", (qid,))
        connection.executemany(
            "INSERT INTO editor (question_id, name, name_fold) VALUES (?, ?, ?)",
            ((qid, name, name.casefold()) for name in names),
        )
        connection.executemany(
            "INSERT INTO area (question_id, name, name_fold) VALUES (?, ?, ?)",
            ((qid, name, name.casefold()) for name in topics),
        )
        connection.executemany(
            "INSERT INTO realia (question_id, local_id, name, name_fold, kind, status) VALUES (?, ?, ?, ?, ?, ?)",
            ((qid, item.id, item.name, item.name.casefold(), item.kind, item.status) for item in found.realia),
        )
        connection.executemany(
            "INSERT INTO property (question_id, local_id, bearer_id, type, type_fold, text) VALUES (?, ?, ?, ?, ?, ?)",
            (
                (qid, item.id, item.bearer, item.type, item.type.casefold(), item.text)
                for item in found.properties
            ),
        )
        connection.executemany(
            "INSERT INTO fact (question_id, local_id, text) VALUES (?, ?, ?)",
            ((qid, item.id, item.text) for item in found.facts),
        )
        connection.executemany(
            "INSERT INTO fact_realia (question_id, fact_id, realia_id) VALUES (?, ?, ?)",
            ((qid, item.id, realia_id) for item in found.facts for realia_id in item.realia),
        )
        connection.executemany(
            "INSERT INTO relation (question_id, local_id, link, link_fold, kind, text) VALUES (?, ?, ?, ?, ?, ?)",
            ((qid, item.id, item.link, item.link.casefold(), item.kind, item.text) for item in found.relations),
        )
        connection.executemany(
            "INSERT INTO relation_realia (question_id, relation_id, realia_id) VALUES (?, ?, ?)",
            ((qid, item.id, realia_id) for item in found.relations for realia_id in item.realia),
        )
        connection.executemany(
            "INSERT INTO move (question_id, position, code, code_fold) VALUES (?, ?, ?, ?)",
            ((qid, position, move.code, move.code.casefold()) for position, move in enumerate(found.moves)),
        )
        connection.executemany(
            "INSERT INTO move_read (question_id, position, record_id) VALUES (?, ?, ?)",
            ((qid, position, record_id) for position, move in enumerate(found.moves) for record_id in move.reads),
        )
        connection.executemany(
            "INSERT INTO move_yield (question_id, position, realia_id) VALUES (?, ?, ?)",
            ((qid, position, realia_id) for position, move in enumerate(found.moves) for realia_id in move.yields),
        )
        connection.executemany(
            "INSERT INTO technique (question_id, name, name_fold) VALUES (?, ?, ?)",
            ((qid, item.name, item.name.casefold()) for item in found.techniques),
        )
        connection.executemany(
            "INSERT INTO source (question_id, position, citation) VALUES (?, ?, ?)",
            ((qid, position, item.citation) for position, item in enumerate(found.sources)),
        )
        connection.executemany(
            "INSERT INTO source_type (question_id, position, name, name_fold) VALUES (?, ?, ?, ?)",
            (
                (qid, position, name, name.casefold())
                for position, item in enumerate(found.sources)
                for name in item.types
            ),
        )
        connection.executemany(
            "INSERT INTO source_language (question_id, position, name, name_fold) VALUES (?, ?, ?, ?)",
            (
                (qid, position, name, name.casefold())
                for position, item in enumerate(found.sources)
                for name in item.languages
            ),
        )
        connection.executemany(
            "INSERT INTO source_theme (question_id, position, name, name_fold) VALUES (?, ?, ?, ?)",
            (
                (qid, position, name, name.casefold())
                for position, item in enumerate(found.sources)
                for name in item.themes
            ),
        )
        connection.executemany(
            "INSERT INTO form (question_id, wording, wording_fold, answer, answer_fold) VALUES (?, ?, ?, ?, ?)",
            (
                (qid, item.wording, item.wording.casefold(), item.answer, item.answer.casefold())
                for item in found.forms
            ),
        )
    return found


def questions_by_editor(connection: sqlite3.Connection, name: str) -> tuple[str, ...]:
    return _ids(connection, "SELECT question_id FROM editor WHERE name_fold = ? ORDER BY question_id", name)


def questions_by_area(connection: sqlite3.Connection, name: str) -> tuple[str, ...]:
    return _ids(connection, "SELECT question_id FROM area WHERE name_fold = ? ORDER BY question_id", name)


def questions_by_theme(connection: sqlite3.Connection, name: str) -> tuple[str, ...]:
    """Question thematic area. Same lookup as ``questions_by_area``."""
    return questions_by_area(connection, name)


def questions_by_source_type(connection: sqlite3.Connection, name: str) -> tuple[str, ...]:
    return _ids(
        connection,
        "SELECT DISTINCT question_id FROM source_type WHERE name_fold = ? ORDER BY question_id",
        name,
    )


def questions_by_source_language(connection: sqlite3.Connection, name: str) -> tuple[str, ...]:
    return _ids(
        connection,
        "SELECT DISTINCT question_id FROM source_language WHERE name_fold = ? ORDER BY question_id",
        name,
    )


def questions_by_source_theme(connection: sqlite3.Connection, name: str) -> tuple[str, ...]:
    return _ids(
        connection,
        "SELECT DISTINCT question_id FROM source_theme WHERE name_fold = ? ORDER BY question_id",
        name,
    )


def questions_by_move(connection: sqlite3.Connection, code: str) -> tuple[str, ...]:
    return _ids(
        connection,
        "SELECT DISTINCT question_id FROM move WHERE code_fold = ? ORDER BY question_id",
        code,
    )


def answers_by_form(connection: sqlite3.Connection, wording: str) -> tuple[str, ...]:
    rows = connection.execute(
        """
        SELECT answer FROM form
        WHERE wording_fold = ?
        GROUP BY answer_fold
        ORDER BY answer
        """,
        (wording.casefold(),),
    )
    return tuple(row[0] for row in rows)


def editors_for_realia(connection: sqlite3.Connection, name: str) -> tuple[tuple[str, int], ...]:
    """Editors of questions that name this realia, with how many such questions each has."""
    rows = connection.execute(
        """
        SELECT e.name, COUNT(DISTINCT e.question_id) AS questions
        FROM editor e
        JOIN realia r ON r.question_id = e.question_id
        WHERE r.name_fold = ?
        GROUP BY e.name_fold
        ORDER BY questions DESC, e.name
        """,
        (name.casefold(),),
    )
    return tuple((row[0], int(row[1])) for row in rows)


def unlisted(connection: sqlite3.Connection) -> dict[str, tuple[str, ...]]:
    """Values already stored that the written inventories do not name yet."""
    return {
        "moves": _outside(connection, "SELECT code FROM move", MOVES),
        "realia_kinds": _outside(connection, "SELECT kind FROM realia", REALIA_KINDS),
        "relation_kinds": _outside(connection, "SELECT kind FROM relation WHERE kind != ''", RELATION_KINDS),
        "links": _outside(connection, "SELECT link FROM relation WHERE link != ''", LINKS),
        "property_types": _outside(connection, "SELECT type FROM property WHERE type != ''", PROPERTY_TYPES),
        "techniques": _outside(connection, "SELECT name FROM technique", TECHNIQUES),
        "areas": _outside(connection, "SELECT name FROM area", AREAS),
        "source_types": _outside(connection, "SELECT name FROM source_type", SOURCE_TYPES),
        "source_languages": _outside(connection, "SELECT name FROM source_language", LANGUAGES),
        "source_themes": _outside(connection, "SELECT name FROM source_theme", SOURCE_THEMES),
    }


def _ids(connection: sqlite3.Connection, sql: str, name: str) -> tuple[str, ...]:
    rows = connection.execute(sql, (name.casefold(),))
    return tuple(dict.fromkeys(row[0] for row in rows))


def _outside(connection: sqlite3.Connection, sql: str, known: tuple[str, ...]) -> tuple[str, ...]:
    seen = {item.casefold() for item in known}
    found: dict[str, str] = {}
    for (label,) in connection.execute(sql):
        if not label:
            continue
        key = label.casefold()
        if key in seen or key in found:
            continue
        found[key] = label
    return tuple(found[key] for key in sorted(found))


def _unique(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = value.strip()
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(name)
    return tuple(result)

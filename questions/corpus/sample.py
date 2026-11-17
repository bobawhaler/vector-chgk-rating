"""Draw a reproducible question sample from a filtered population.

``QuestionFilter.limit`` pages the web and the CLI. A sample ignores it and
draws ``size`` questions from everything that matches.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import pandas as pd

from corpus.query import QuestionFilter, matched, materialize

_BY = {"editor", "author", "year", "pack"}
_YEAR = re.compile(r"\d{4}")


@dataclass(frozen=True)
class Sample:
    """Questions drawn for one analysis run.

    ``population`` is how many questions matched before drawing. ``rows`` keeps
    the draw order: catalog order, the seeded random order, or round-robin
    across groups.
    """

    rows: tuple[dict[str, object], ...]
    population: int
    size: int
    seed: int | None
    by: str | None
    spec: QuestionFilter

    def __len__(self) -> int:
        return len(self.rows)

    def ids(self) -> tuple[str, ...]:
        return tuple(str(row["question_id"]) for row in self.rows)


def _names(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _first(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return [""] * len(frame)
    return [(_names(value) or [""])[0] for value in frame[column].tolist()]


def _keys(frame: pd.DataFrame, by: str) -> list[str]:
    if by == "author":
        return _first(frame, "author_names")
    if by == "editor":
        editors = _first(frame, "editor_names")
        package = _first(frame, "pack_editor_names")
        return [editor or fallback for editor, fallback in zip(editors, package, strict=True)]
    if by == "year":
        if "start_date" not in frame.columns:
            return [""] * len(frame)
        years: list[str] = []
        for value in frame["start_date"].tolist():
            text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
            years.append(text[:4] if _YEAR.fullmatch(text[:4]) and (len(text) == 4 or text[4:5] == "-") else "")
        return years
    if "pack_id" not in frame.columns:
        return [""] * len(frame)
    keys: list[str] = []
    for value in frame["pack_id"].tolist():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            keys.append("")
            continue
        try:
            keys.append(str(int(value)))
        except (TypeError, ValueError):
            keys.append("")
    return keys


def _round_robin(groups: dict[str, list[int]], size: int, seed: int | None) -> list[int]:
    order = sorted(groups)
    rng = random.Random(seed) if seed is not None else None
    if rng is not None:
        rng.shuffle(order)
        for key in order:
            rng.shuffle(groups[key])
    picked: list[int] = []
    cursors = dict.fromkeys(order, 0)
    while len(picked) < size:
        progressed = False
        for key in order:
            cursor = cursors[key]
            bucket = groups[key]
            if cursor >= len(bucket):
                continue
            picked.append(bucket[cursor])
            cursors[key] = cursor + 1
            progressed = True
            if len(picked) == size:
                break
        if not progressed:
            break
    return picked


def take(
    questions: pd.DataFrame,
    spec: QuestionFilter | None = None,
    *,
    size: int = 20,
    seed: int | None = None,
    by: str | None = None,
) -> Sample:
    """Draw up to ``size`` questions.

    Without ``seed`` the draw is the newest matching questions. With ``seed``
    it is a random sample of that population. ``by`` (``editor``, ``author``,
    ``year``, or ``pack``) takes turns across groups so one group cannot fill
    the sample. ``spec.limit`` is not the sample size.
    """
    if size < 1:
        raise ValueError("size: нужно положительное число.")
    if by is not None and by not in _BY:
        raise ValueError("by: editor, author, year или pack.")
    chosen_spec = spec or QuestionFilter()
    found = matched(questions, chosen_spec).reset_index(drop=True)
    population = int(len(found))
    if population == 0:
        return Sample((), population, size, seed, by, chosen_spec)

    if by is None:
        if seed is None:
            picked = list(range(min(size, population)))
        else:
            picked = random.Random(seed).sample(range(population), min(size, population))
    else:
        groups: dict[str, list[int]] = {}
        for index, key in enumerate(_keys(found, by)):
            groups.setdefault(key, []).append(index)
        picked = _round_robin(groups, size, seed)

    rows = tuple(materialize(found.iloc[picked]))
    return Sample(rows, population, size, seed, by, chosen_spec)

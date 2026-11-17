"""Join a model's JSON reply back onto the question ids in a prompt.

No model is called here. ``analyze`` reads the text the model already returned.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass


class ResultsError(ValueError):
    """The reply is not the JSON array the prompt asked for."""


@dataclass(frozen=True)
class Finding:
    question_id: str
    fields: dict[str, object]


@dataclass(frozen=True)
class Alignment:
    """Model objects lined up with the ids that were asked."""

    findings: tuple[Finding, ...]
    missing_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    duplicate_ids: tuple[str, ...]


def _decode(text: str) -> object:
    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(raw, index)
        except json.JSONDecodeError:
            continue
        return value
    raise ResultsError("В ответе нет JSON.")


def _as_records(data: object) -> list[dict[str, object]]:
    if isinstance(data, list):
        if data and not all(isinstance(item, dict) for item in data):
            raise ResultsError("JSON-массив должен состоять из объектов.")
        return [dict(item) for item in data]
    if isinstance(data, dict):
        lists = [
            value
            for value in data.values()
            if isinstance(value, list) and all(isinstance(item, dict) for item in value)
        ]
        if len(lists) == 1:
            return [dict(item) for item in lists[0]]
        if "id" in data:
            return [dict(data)]
    raise ResultsError("JSON должен быть массивом объектов с полем id.")


def parse_reply(reply: str) -> list[dict[str, object]]:
    """Objects from a reply. A markdown fence or a single wrapping object is accepted."""
    return _as_records(_decode(reply))


def align(ids: tuple[str, ...] | list[str], records: list[dict[str, object]]) -> Alignment:
    """Keep the first object for each requested id, in the requested order."""
    wanted = [str(question_id) for question_id in ids]
    wanted_set = set(wanted)
    seen: set[str] = set()
    by_id: dict[str, Finding] = {}
    unknown: list[str] = []
    duplicate: list[str] = []
    for record in records:
        raw_id = record.get("id")
        question_id = "" if raw_id is None else str(raw_id).strip()
        if not question_id or question_id not in wanted_set:
            unknown.append(question_id)
            continue
        if question_id in seen:
            duplicate.append(question_id)
            continue
        seen.add(question_id)
        fields = {key: value for key, value in record.items() if key != "id"}
        by_id[question_id] = Finding(question_id, fields)
    return Alignment(
        findings=tuple(by_id[question_id] for question_id in wanted if question_id in by_id),
        missing_ids=tuple(question_id for question_id in wanted if question_id not in by_id),
        unknown_ids=tuple(unknown),
        duplicate_ids=tuple(duplicate),
    )


def analyze(prompt: object, reply: str) -> Alignment:
    """Parse ``reply`` and align it with ``prompt.ids``."""
    ids = getattr(prompt, "ids", None)
    if not isinstance(ids, tuple):
        raise TypeError("Нужен промпт с ids.")
    return align(ids, parse_reply(reply))


def coverage(alignment: Alignment) -> dict[str, int]:
    """How many requested questions came back, and what did not match."""
    return {
        "asked": len(alignment.findings) + len(alignment.missing_ids),
        "matched": len(alignment.findings),
        "missing": len(alignment.missing_ids),
        "unknown": len(alignment.unknown_ids),
        "duplicate": len(alignment.duplicate_ids),
    }


def tally(alignment: Alignment, field: str) -> list[tuple[str, int]]:
    """Counts of one field in the model reply, most frequent first."""
    counts: Counter[str] = Counter()
    for finding in alignment.findings:
        value = finding.fields.get(field, "")
        if isinstance(value, list):
            text = ", ".join(str(item).strip() for item in value if str(item).strip())
        elif value is None:
            text = ""
        else:
            text = str(value).strip()
        counts[text] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _norm(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def same_text(
    alignment: Alignment,
    questions: dict[str, dict[str, object]],
    model_field: str,
    question_field: str,
) -> dict[str, float | int]:
    """Share of replies whose field equals the question field after normalizing space and case.

    Empty sides are not compared. This is an exact match, not a judgement of meaning.
    """
    compared = 0
    equal = 0
    for finding in alignment.findings:
        question = questions.get(finding.question_id)
        if question is None:
            continue
        left = _norm(finding.fields.get(model_field, ""))
        right = _norm(question.get(question_field, ""))
        if not left or not right:
            continue
        compared += 1
        equal += int(left == right)
    rate = equal / compared if compared else 0.0
    return {"compared": compared, "equal": equal, "rate": rate}

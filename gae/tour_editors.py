"""Tour editors from a rating.chgk.info tournament card.

``GET /tournaments/{id}`` returns ``credits``: each item is a tour or a question
range and the people who edited it. ``tour: null`` and ``questions: null`` is
the whole package. Question numbers are 1-based across the package, the same
index as ``{tournament_id}_{n}`` in the results table. The tournament list
endpoint does not include ``credits``.
"""

from __future__ import annotations

from typing import Any


def player_ids(people: list[Any] | None) -> list[int]:
    """Confirmed player ids. Ambiguous namesakes (``players`` + ``tentative``) are skipped."""
    found: list[int] = []
    for person in people or []:
        if not isinstance(person, dict) or person.get("tentative"):
            continue
        player = person.get("player")
        if isinstance(player, dict) and player.get("id"):
            found.append(int(player["id"]))
    return found


def credit_record(tournament: dict[str, Any]) -> dict[str, Any] | None:
    """Compact credits for one tournament, or None when the card has no credits array."""
    if "credits" not in tournament:
        return None
    raw = tournament.get("credits")
    if not isinstance(raw, list):
        return None
    blocks: list[dict[str, Any]] = []
    for block in raw:
        if not isinstance(block, dict):
            continue
        questions = block.get("questions")
        span: list[int] | None = None
        if isinstance(questions, list) and len(questions) == 2 and questions[0] and questions[1]:
            span = [int(questions[0]), int(questions[1])]
        blocks.append({
            "tour": block.get("tour"),
            "questions": span,
            "editor_ids": player_ids(block.get("editors")),
        })
    package: list[int] = []
    for person in tournament.get("editors") or []:
        if isinstance(person, dict) and person.get("id"):
            package.append(int(person["id"]))
    return {
        "tournament_id": int(tournament["id"]),
        "package_editor_ids": package,
        "blocks": blocks,
    }


def editors_for_question(record: dict[str, Any] | None, question_number: int, fallback: list[int] | None = None) -> list[int]:
    """Editors of one question. A narrower credit range wins. Empty credits use ``fallback``."""
    fallback = list(fallback or [])
    if not record:
        return fallback
    recorded = [int(item) for item in record.get("package_editor_ids") or []]
    if not record.get("blocks"):
        return recorded or fallback
    package: list[int] | None = None
    best: list[int] | None = None
    best_span = 10**9
    for block in record["blocks"]:
        ids = [int(item) for item in block.get("editor_ids") or []]
        span = block.get("questions")
        if block.get("tour") is None and not span:
            package = ids
            continue
        if not span or not (int(span[0]) <= question_number <= int(span[1])):
            continue
        width = int(span[1]) - int(span[0])
        if ids and width < best_span:
            best = ids
            best_span = width
    if best:
        return best
    if package:
        return package
    recorded = [int(item) for item in record.get("package_editor_ids") or []]
    return recorded or fallback


def editors_for_span(
    record: dict[str, Any] | None,
    first: int,
    last: int,
    fallback: list[int] | None = None,
) -> list[int]:
    """Editors of a tour, in credit order, without duplicates."""
    seen: list[int] = []
    for number in range(first, last + 1):
        for editor_id in editors_for_question(record, number, fallback):
            if editor_id not in seen:
                seen.append(editor_id)
    return seen

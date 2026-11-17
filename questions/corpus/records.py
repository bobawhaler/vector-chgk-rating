"""Turn a gotquestions pack dict into question and package rows.

Question ids stay compatible with the existing parquet: the number in
``question_id`` is the position of the question inside the package, while
``number_in_tour`` is the number printed in that tour.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

QUESTION_COLUMNS = [
    "question_id",
    "rating_tournament_id",
    "pack_id",
    "pack_title",
    "tour_id",
    "tour_number",
    "question_number",
    "number_in_tour",
    "text",
    "answer",
    "zachet",
    "nezachet",
    "comment",
    "source",
    "start_date",
    "end_date",
    "pub_date",
    "author_names",
    "author_player_ids",
    "editor_names",
    "editor_player_ids",
    "pack_editor_names",
    "pack_editor_player_ids",
]

PACK_COLUMNS = [
    "pack_id",
    "title",
    "rating_tournament_ids",
    "date",
    "start_date",
    "end_date",
    "pub_date",
    "tours_count",
    "editor_names",
    "editor_player_ids",
]


def day(value: object) -> str:
    """Calendar day as YYYY-MM-DD. Accepts the site's ``$D`` prefix."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.startswith("$D"):
        text = text[2:]
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return ""


def people(items: object) -> tuple[list[str], list[int]]:
    """Names and rating player ids from a gotquestions person list."""
    names: list[str] = []
    player_ids: list[int] = []
    if not isinstance(items, list):
        return names, player_ids
    for item in items:
        if not isinstance(item, dict):
            continue
        person = item.get("person")
        if not isinstance(person, dict):
            continue
        name = str(person.get("name") or "").strip()
        if name:
            names.append(name)
        player_id = person.get("playerId")
        if not player_id:
            continue
        try:
            player_ids.append(int(player_id))
        except (TypeError, ValueError):
            continue
    return names, player_ids


def _tournament_ids(pack: dict[str, Any]) -> list[int]:
    raw = pack.get("tIds") or []
    if not isinstance(raw, list):
        return []
    return [int(item) for item in raw if str(item).isdigit()]


def _tour_number(question: dict[str, Any], fallback: int) -> int:
    raw = question.get("number")
    if raw in (None, ""):
        return fallback
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback


def frames_from_packs(packs: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Question rows (one per rating tournament) and one row per package."""
    pack_records: list[dict[str, Any]] = []
    question_records: list[dict[str, Any]] = []

    for pack in packs:
        pack_id = int(pack.get("id") or 0)
        title = str(pack.get("title") or "")
        tournament_ids = _tournament_ids(pack)
        start_date = day(pack.get("startDate"))
        end_date = day(pack.get("endDate"))
        pub_date = day(pack.get("pubDate"))
        pack_editor_names, pack_editor_player_ids = people(pack.get("editors"))
        tours = pack.get("tours") or []
        if not isinstance(tours, list):
            tours = []

        pack_records.append({
            "pack_id": pack_id,
            "title": title,
            "rating_tournament_ids": tournament_ids,
            "date": start_date or pub_date or end_date,
            "start_date": start_date,
            "end_date": end_date,
            "pub_date": pub_date,
            "tours_count": len(tours),
            "editor_names": pack_editor_names,
            "editor_player_ids": pack_editor_player_ids,
        })

        position = 0
        for tour_index, tour in enumerate(tours, 1):
            if not isinstance(tour, dict):
                continue
            tour_id = int(tour.get("id") or 0)
            editor_names, editor_player_ids = people(tour.get("editors"))
            questions = tour.get("questions") or []
            if not isinstance(questions, list):
                continue
            for question in questions:
                if not isinstance(question, dict):
                    continue
                position += 1
                author_names, author_player_ids = people(question.get("authors"))
                base = {
                    "pack_id": pack_id,
                    "pack_title": title,
                    "tour_id": tour_id,
                    "tour_number": tour_index,
                    "question_number": position,
                    "number_in_tour": _tour_number(question, position),
                    "text": str(question.get("text") or "").strip(),
                    "answer": str(question.get("answer") or "").strip(),
                    "zachet": str(question.get("zachet") or "").strip(),
                    "nezachet": str(question.get("nezachet") or "").strip(),
                    "comment": str(question.get("comment") or "").strip(),
                    "source": str(question.get("source") or "").strip(),
                    "start_date": start_date,
                    "end_date": end_date,
                    "pub_date": pub_date,
                    "author_names": author_names,
                    "author_player_ids": author_player_ids,
                    "editor_names": editor_names,
                    "editor_player_ids": editor_player_ids,
                    "pack_editor_names": pack_editor_names,
                    "pack_editor_player_ids": pack_editor_player_ids,
                }
                targets = tournament_ids or [0]
                for tournament_id in targets:
                    question_id = (
                        f"{tournament_id}_{position}" if tournament_id else f"gq_{pack_id}_{position}"
                    )
                    question_records.append({
                        "question_id": question_id,
                        "rating_tournament_id": tournament_id,
                        **base,
                    })

    if not question_records:
        questions = pd.DataFrame(columns=QUESTION_COLUMNS)
    else:
        questions = pd.DataFrame(question_records, columns=QUESTION_COLUMNS)
    if not pack_records:
        packages = pd.DataFrame(columns=PACK_COLUMNS)
    else:
        packages = pd.DataFrame(pack_records, columns=PACK_COLUMNS)
    return questions, packages

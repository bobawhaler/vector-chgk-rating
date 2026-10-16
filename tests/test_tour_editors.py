"""Tour-editor credits from a rating.chgk.info tournament card."""

from __future__ import annotations

import sys
from pathlib import Path

_GAE = Path(__file__).resolve().parents[1] / "gae"
if str(_GAE) not in sys.path:
    sys.path.append(str(_GAE))

from tour_editors import credit_record, editors_for_question, editors_for_span


def _card() -> dict:
    return {
        "id": 13575,
        "editors": [
            {"id": 1, "name": "Package"},
            {"id": 2, "name": "Also"},
        ],
        "credits": [
            {
                "tour": 1,
                "questions": [1, 12],
                "editors": [{"player": {"id": 3671}}],
            },
            {
                "tour": 2,
                "questions": [13, 24],
                "editors": [{"player": {"id": 80561}}],
            },
            {
                "tour": 3,
                "questions": [25, 36],
                "editors": [{"player": {"id": 170136}}, {"player": {"id": 72211}}],
            },
            {
                "tour": None,
                "questions": None,
                "editors": [{"player": {"id": 53426}}],
            },
        ],
    }


def test_question_uses_the_narrowest_range() -> None:
    record = credit_record(_card())
    assert editors_for_question(record, 1) == [3671]
    assert editors_for_question(record, 24) == [80561]
    assert editors_for_question(record, 30) == [170136, 72211]


def test_gap_falls_back_to_package_credits_then_tournament_editors() -> None:
    card = _card()
    card["credits"].append({
        "tour": 2,
        "questions": [17, 17],
        "editors": [],
    })
    record = credit_record(card)
    assert editors_for_question(record, 17) == [80561]

    bare = {"id": 9, "editors": [{"id": 4}], "credits": [{"tour": 1, "questions": [1, 1], "editors": []}]}
    assert editors_for_question(credit_record(bare), 5, fallback=[4]) == [4]


def test_package_editors_come_from_the_tournament_card() -> None:
    record = credit_record({"id": 3, "editors": [{"id": 8, "name": "Editor"}], "credits": []})
    assert record is not None
    assert record["package_editor_ids"] == [8]
    assert editors_for_question(record, 2) == [8]


def test_missing_credits_keep_the_tournament_editor_list() -> None:
    assert credit_record({"id": 3, "editors": [{"id": 8}]}) is None
    assert editors_for_question(None, 1, fallback=[8]) == [8]


def test_tentative_namesake_is_not_an_editor() -> None:
    card = {
        "id": 1,
        "editors": [],
        "credits": [{
            "tour": 1,
            "questions": [1, 12],
            "editors": [{"tentative": True, "players": [{"id": 1}, {"id": 2}]}],
        }],
    }
    assert editors_for_question(credit_record(card), 3, fallback=[9]) == [9]


def test_span_collects_tour_editors_in_order() -> None:
    record = credit_record(_card())
    assert editors_for_span(record, 13, 24) == [80561]
    assert editors_for_span(record, 1, 13) == [3671, 80561]

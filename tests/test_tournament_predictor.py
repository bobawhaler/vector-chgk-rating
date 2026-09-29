"""Tests for generic TournamentPredictor."""

import json
import sqlite3
import pytest
from pathlib import Path
from chgk_rating.tournament_predictor import TournamentPredictor


def test_tournament_predictor_synthetic(tmp_path):
    # Create mock SQLite cache DB with tournament 99999
    db_file = tmp_path / "test_cache.db"
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE ingested_tournaments (
            tournament_id INTEGER PRIMARY KEY,
            title TEXT,
            tournament_type TEXT,
            date_start TEXT,
            date_end TEXT,
            raw_data TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE api_cache (
            url TEXT PRIMARY KEY,
            response_json TEXT,
            cached_at TEXT
        )
    """)

    t_data = {
        "id": 99999,
        "name": "Test Tournament",
        "dateStart": "2026-05-01",
        "difficultyForecast": 4.5,
        "type": {"name": "Синхрон"},
        "editors": [{"id": 1, "name": "Editor", "surname": "One"}],
        "questionQty": {"1": 12, "2": 12, "3": 12}
    }
    cur.execute(
        "INSERT INTO ingested_tournaments VALUES (99999, 'Test Tournament', 'sync', '2026-05-01', '2026-05-02', ?)",
        (json.dumps(t_data),)
    )

    results_data = [
        {
            "team": {"id": 10, "name": "Alpha", "town": {"name": "Moscow"}},
            "teamMembers": [{"player": {"id": 101}}, {"player": {"id": 102}}],
            "mask": "111000111000111000111000111000111000"
        },
        {
            "team": {"id": 20, "name": "Beta", "town": {"name": "SPb"}},
            "teamMembers": [{"player": {"id": 201}}, {"player": {"id": 202}}],
            "mask": "000000000000000000000000000000000000"
        }
    ]
    cur.execute(
        "INSERT INTO api_cache VALUES (?, ?, datetime('now'))",
        (
            "https://api.rating.chgk.info/tournaments/99999/results?includeMasksAndControversials=1&includeTeamMembers=1",
            json.dumps(results_data)
        )
    )
    con.commit()
    con.close()

    predictor = TournamentPredictor(db_path=db_file)
    res = predictor.predict_tournament(tournament_id=99999, num_simulations=100, tactical=False)

    assert res["tournament_id"] == 99999
    assert res["tournament_name"] == "Test Tournament"
    assert res["total_questions"] == 36
    assert len(res["standings"]) == 2

    # Check export functions
    json_path = predictor.export_json(res, tmp_path / "forecast.json")
    assert json_path.exists()

    html_path = predictor.export_html(res, tmp_path / "forecast.html")
    assert html_path.exists()

    table_text = predictor.format_terminal_table(res)
    assert "Test Tournament" in table_text
    assert "Alpha" in table_text
    assert "Beta" in table_text

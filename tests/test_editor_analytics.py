import pytest
from pathlib import Path
from chgk_rating.editor_analytics import EditorCalibrationRegistry, EditorProfile

def test_editor_calibration_synthetic(tmp_path):
    import sqlite3, json

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
    # Insert 2 tournaments with editor 100
    t1 = {"id": 1, "name": "T1", "difficultyForecast": 4.0, "trueDL": 5.0, "editors": [{"id": 100, "name": "Ivan", "surname": "Ivanov"}]}
    t2 = {"id": 2, "name": "T2", "difficultyForecast": 5.0, "trueDL": 6.0, "editors": [{"id": 100, "name": "Ivan", "surname": "Ivanov"}]}
    cur.execute("INSERT INTO ingested_tournaments VALUES (1, 'T1', 'offline', '2023-01-01', '2023-01-02', ?)", (json.dumps(t1),))
    cur.execute("INSERT INTO ingested_tournaments VALUES (2, 'T2', 'offline', '2023-02-01', '2023-02-02', ?)", (json.dumps(t2),))
    con.commit()
    con.close()

    reg = EditorCalibrationRegistry(db_path=db_file, prior_n0=3.0, prior_mu0=-0.02)
    reg.load_data()

    prof = reg.get_editor_profile(100)
    assert prof is not None
    assert prof.tournaments_count == 2
    assert prof.raw_bias == 1.0  # (1.0 + 1.0)/2
    # Bayesian shrinkage: (2.0 + 3 * (-0.02)) / (2 + 3) = 1.94 / 5 = 0.388
    assert abs(prof.calibrated_bias - 0.388) < 1e-4
    assert prof.tendency == "harder"

    # Predict TrueDL for a 4.0 tournament with editor 100
    pred_dl = reg.predict_tournament_truedl(forecast_dl=4.0, editor_ids=[100])
    assert abs(pred_dl - (4.0 + 0.388)) < 1e-4

def test_editor_leaderboard(tmp_path):
    reg = EditorCalibrationRegistry(db_path=tmp_path / "empty.db")
    reg.load_data()
    assert len(reg.get_top_editors()) == 0

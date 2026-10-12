"""Rating-site sync into Cloud Storage. No network."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["ENV_YAML"] = "/nonexistent-gae-env.yaml"
os.environ["GCS_BUCKET"] = ""
os.environ["RATING_INTERVAL_SECONDS"] = "0"

_GAE = Path(__file__).resolve().parents[1] / "gae"
if str(_GAE) not in sys.path:
    sys.path.append(str(_GAE))

from questions_sync import MemoryStorage
from rating_sync import materialize, records_from_results, run_incremental, upload_snapshot


def _result(mask: str) -> dict:
    return {
        "team": {"id": 3},
        "teamMembers": [{"player": {"id": 1}}, {"player": {"id": 2}}],
        "mask": mask,
        "synchRequest": {"venue": {"name": "Москва"}},
    }


def _tournament(tournament_id: int) -> dict:
    return {"id": tournament_id, "name": "Cup", "format": "offline", "dateStart": "2026-10-01"}


def _upload(tmp_path: Path, storage: MemoryStorage) -> None:
    import pandas as pd

    frame = pd.DataFrame([
        {
            "tournament_id": 10,
            "tournament_type": "offline",
            "date": "2026-01-01",
            "team_id": 3,
            "player_ids": [1, 2],
            "question_id": "10_1",
            "solved": 0,
        },
        {
            "tournament_id": 11,
            "tournament_type": "sync",
            "date": "2026-02-01",
            "team_id": 4,
            "player_ids": [8],
            "question_id": "11_1",
            "solved": 1,
        },
    ])
    path = tmp_path / "rating.parquet"
    frame.to_parquet(path, index=False)
    upload_snapshot(path, storage=storage)


def test_mask_becomes_question_rows() -> None:
    rows = records_from_results(_tournament(10), [_result("10x1")])
    assert [row["question_id"] for row in rows] == ["10_1", "10_2", "10_3"]
    assert [row["solved"] for row in rows] == [1, 0, 1]
    assert rows[0]["player_ids"] == [1, 2]


def test_materialize_replaces_only_updated_tournament(tmp_path: Path) -> None:
    storage = MemoryStorage()
    _upload(tmp_path, storage)

    def listing(since: str, page: int) -> list[dict]:
        assert page == 1
        return [_tournament(10)]

    result = run_incremental(
        storage=storage,
        list_tournaments=listing,
        fetch_results=lambda tournament_id: [_result("11")],
        sleep=lambda _seconds: None,
        lookback=3,
        max_tournaments=10,
    )
    assert result.ok
    assert result.written == 1

    dest = tmp_path / "merged.parquet"
    materialize(dest, storage=storage)
    import pandas as pd

    merged = pd.read_parquet(dest)
    updated = merged[merged["tournament_id"] == 10].sort_values("question_id")
    assert updated["question_id"].tolist() == ["10_1", "10_2"]
    assert updated["solved"].tolist() == [1, 1]
    assert merged.loc[merged["tournament_id"] == 11, "solved"].tolist() == [1]


def test_cap_resumes_on_the_next_run(tmp_path: Path) -> None:
    storage = MemoryStorage()
    _upload(tmp_path, storage)
    calls = {"n": 0}

    def listing(since: str, page: int) -> list[dict]:
        return [_tournament(10), _tournament(12)]

    def results(tournament_id: int) -> list[dict]:
        calls["n"] += 1
        return [_result("1")]

    first = run_incremental(
        storage=storage,
        list_tournaments=listing,
        fetch_results=results,
        sleep=lambda _seconds: None,
        max_tournaments=1,
    )
    assert first.written == 1
    assert "продолжение" in first.message

    second = run_incremental(
        storage=storage,
        list_tournaments=listing,
        fetch_results=results,
        sleep=lambda _seconds: None,
        max_tournaments=1,
    )
    assert second.written == 1
    assert calls["n"] == 2
    state_raw = storage.get_bytes("state/rating.json")
    assert state_raw is not None
    assert b'"pass_since": ""' in state_raw or b'"pass_since":""' in state_raw


def test_missing_upload_is_reported() -> None:
    result = run_incremental(
        storage=MemoryStorage(),
        list_tournaments=lambda since, page: [],
        fetch_results=lambda tournament_id: [],
        sleep=lambda _seconds: None,
    )
    assert not result.ok
    assert "upload-rating" in result.message

"""Cloud Storage question sync. No network and no Google credentials."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ["ENV_YAML"] = "/nonexistent-gae-env.yaml"
os.environ["GCS_BUCKET"] = ""
os.environ["USE_DATASTORE"] = "0"

_GAE = Path(__file__).resolve().parents[1] / "gae"
if str(_GAE) not in sys.path:
    sys.path.append(str(_GAE))

import questions_sync
from questions_sync import (
    MemoryStorage,
    SyncResult,
    hashes_from_frame,
    materialize,
    pack_hash,
    records_from_pack,
    run_incremental,
    upload_snapshot,
)


def _pack(pack_id: int, text: str = "hello") -> dict:
    return {
        "id": pack_id,
        "title": "Pack",
        "tIds": [100],
        "tours": [
            {
                "id": 7,
                "editors": [],
                "questions": [
                    {
                        "number": 1,
                        "text": text,
                        "answer": "answer",
                        "comment": "",
                        "authors": [{"person": {"name": "Author", "playerId": 3}}],
                    }
                ],
            }
        ],
    }


def _upload(tmp_path: Path, storage: MemoryStorage, pack_id: int = 5, text: str = "hello") -> None:
    import pandas as pd

    frame = pd.DataFrame(records_from_pack(_pack(pack_id, text)))
    path = tmp_path / "questions.parquet"
    frame.to_parquet(path, index=False)
    upload_snapshot(path, storage=storage, lookback=20)


def test_hash_matches_between_pack_and_frame() -> None:
    import pandas as pd

    records = records_from_pack(_pack(5))
    frame = pd.DataFrame(records)
    assert hashes_from_frame(frame) == {"5": pack_hash(records)}


def test_sync_skips_unchanged_lookback(tmp_path: Path) -> None:
    storage = MemoryStorage()
    _upload(tmp_path, storage)

    def fetch(pack_id: int) -> dict | None:
        if pack_id == 5:
            return _pack(5)
        return None

    result = run_incremental(
        storage=storage,
        fetch_pack=fetch,
        interval=0,
        lookback=20,
        max_new=5,
        not_found_stop=2,
    )
    assert result.ok
    assert result.written == 0
    assert not result.notify
    assert storage.list_prefix("questions/parts/") == []


def test_sync_writes_a_part_when_text_changes(tmp_path: Path) -> None:
    storage = MemoryStorage()
    _upload(tmp_path, storage)

    def fetch(pack_id: int) -> dict | None:
        if pack_id == 5:
            return _pack(5, text="edited")
        return None

    result = run_incremental(
        storage=storage,
        fetch_pack=fetch,
        interval=0,
        lookback=20,
        max_new=1,
        not_found_stop=1,
    )
    assert result.written == 1
    assert result.notify
    assert len(storage.list_prefix("questions/parts/")) == 1

    dest = tmp_path / "merged.parquet"
    materialize(dest, storage=storage)
    import pandas as pd

    merged = pd.read_parquet(dest)
    row = merged[merged["question_id"] == "100_1"].iloc[0]
    assert row["text"] == "edited"


def test_sync_advances_past_new_packs(tmp_path: Path) -> None:
    storage = MemoryStorage()
    _upload(tmp_path, storage)

    def fetch(pack_id: int) -> dict | None:
        if pack_id == 5:
            return _pack(5)
        if pack_id == 6:
            return _pack(6, text="brand new")
        return None

    result = run_incremental(
        storage=storage,
        fetch_pack=fetch,
        interval=0,
        lookback=1,
        max_new=5,
        not_found_stop=2,
    )
    assert result.ok
    assert result.max_pack_id == 6
    assert result.written == 1


def test_missing_snapshot_asks_for_upload() -> None:
    result = run_incremental(storage=MemoryStorage(), fetch_pack=lambda _pack_id: None, interval=0)
    assert not result.ok
    assert result.notify
    assert "upload" in result.message


def test_missing_bucket_is_a_quiet_skip(monkeypatch) -> None:
    monkeypatch.setenv("GCS_BUCKET", "")
    result = run_incremental()
    assert result.ok
    assert not result.notify


def test_reupload_drops_old_parts(tmp_path: Path) -> None:
    storage = MemoryStorage()
    _upload(tmp_path, storage)
    storage.put_bytes("questions/parts/old.jsonl.gz", b"stale")
    _upload(tmp_path, storage)
    assert storage.list_prefix("questions/parts/") == []


def test_cron_route_requires_header_and_notifies(monkeypatch) -> None:
    from gae_platform import telegram_api

    sent: list[tuple[object, str]] = []
    monkeypatch.setattr(telegram_api, "send_message", lambda chat_id, text: sent.append((chat_id, text)) or True)
    monkeypatch.setenv("ADMIN_CHAT_ID", "42")
    monkeypatch.setattr(
        questions_sync,
        "run_incremental",
        lambda: SyncResult(True, True, 1, 2, 9, "Вопросы обновлены"),
    )

    spec = importlib.util.spec_from_file_location("gae_questions_main", _GAE / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = module.app.test_client()

    denied = client.get("/cron/questions")
    assert denied.status_code == 403
    allowed = client.get("/cron/questions", headers={"X-Appengine-Cron": "true"})
    assert allowed.status_code == 200
    assert "обновлены" in allowed.get_data(as_text=True)
    assert sent == [("42", "Вопросы обновлены")]

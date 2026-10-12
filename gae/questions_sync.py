"""Question corpus in Cloud Storage, with a small nightly delta.

The readable database is ``questions/snapshot.parquet`` (uploaded once from the
machine that already has it) plus ``questions/parts/*.jsonl.gz`` written by
cron. Cron never downloads the snapshot: F1 has 256 MB of RAM, and the parquet
is 171 MB before it expands into rows. Each night it refetches a short lookback
window and any new gotquestions packs, and writes a part only when a pack
actually changed.

Datastore is the wrong place for this table: the free tier is about 1 GB and
20 000 writes a day, and the corpus is half a million questions.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from gae_platform import config

logger = logging.getLogger(__name__)

SNAPSHOT_OBJECT = "questions/snapshot.parquet"
STATE_OBJECT = "state/questions.json"
PARTS_PREFIX = "questions/parts/"
SOURCE = "https://gotquestions.online"

DEFAULT_LOOKBACK = 20
DEFAULT_MAX_NEW = 30
DEFAULT_NOT_FOUND_STOP = 15


class QuestionsBlocked(Exception):
    """gotquestions.online returned 403 or 429."""


@dataclass
class SyncResult:
    ok: bool
    notify: bool
    written: int
    fetched: int
    max_pack_id: int
    message: str


class Storage(Protocol):
    def get_bytes(self, name: str) -> bytes | None: ...

    def put_bytes(self, name: str, data: bytes, content_type: str = "application/octet-stream") -> None: ...

    def delete(self, name: str) -> None: ...

    def list_prefix(self, prefix: str) -> list[str]: ...


class MemoryStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def get_bytes(self, name: str) -> bytes | None:
        return self.objects.get(name)

    def put_bytes(self, name: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.objects[name] = data

    def delete(self, name: str) -> None:
        self.objects.pop(name, None)

    def list_prefix(self, prefix: str) -> list[str]:
        return sorted(name for name in self.objects if name.startswith(prefix))


class GcsStorage:
    def __init__(self, bucket: str) -> None:
        from google.cloud import storage

        self._client = storage.Client()
        self._bucket_name = bucket
        self._bucket = self._client.bucket(bucket)

    def get_bytes(self, name: str) -> bytes | None:
        from google.api_core.exceptions import NotFound

        try:
            return self._bucket.blob(name).download_as_bytes()
        except NotFound:
            return None

    def put_bytes(self, name: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self._bucket.blob(name).upload_from_string(data, content_type=content_type)

    def delete(self, name: str) -> None:
        from google.api_core.exceptions import NotFound

        try:
            self._bucket.blob(name).delete()
        except NotFound:
            return

    def list_prefix(self, prefix: str) -> list[str]:
        return sorted(blob.name for blob in self._client.list_blobs(self._bucket_name, prefix=prefix))


class GcloudCliStorage:
    """Used on a laptop when google-cloud-storage is not installed."""

    def __init__(self, bucket: str) -> None:
        self._bucket = bucket

    def _uri(self, name: str) -> str:
        return f"gs://{self._bucket}/{name}"

    def get_bytes(self, name: str) -> bytes | None:
        proc = subprocess.run(
            ["gcloud", "storage", "cat", self._uri(name)],
            capture_output=True,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout

    def put_bytes(self, name: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        handle = tempfile.NamedTemporaryFile(delete=False)
        try:
            handle.write(data)
            handle.close()
            subprocess.run(
                ["gcloud", "storage", "cp", handle.name, self._uri(name)],
                check=True,
                capture_output=True,
            )
        finally:
            os.unlink(handle.name)

    def delete(self, name: str) -> None:
        subprocess.run(["gcloud", "storage", "rm", self._uri(name)], capture_output=True)

    def list_prefix(self, prefix: str) -> list[str]:
        proc = subprocess.run(
            ["gcloud", "storage", "ls", self._uri(prefix) + "**"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return []
        base = f"gs://{self._bucket}/"
        names = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith(base):
                names.append(line[len(base):])
        return sorted(names)


def open_storage(bucket: str | None = None) -> Storage:
    name = bucket or config.gcs_bucket()
    if not name:
        raise RuntimeError("GCS_BUCKET is not set")
    try:
        from google.cloud import storage  # noqa: F401
    except ImportError:
        return GcloudCliStorage(name)
    return GcsStorage(name)


def _names(value: object) -> list[str]:
    if value is None or isinstance(value, float):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if item is not None and str(item) != ""]


def _ints(value: object) -> list[int]:
    if value is None or isinstance(value, float):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    out: list[int] = []
    for item in value:
        if item is None or item == "":
            continue
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def pack_hash(records: list[dict[str, Any]]) -> str:
    slim = [
        {
            "question_number": int(row["question_number"]),
            "rating_tournament_id": int(row["rating_tournament_id"]),
            "text": str(row.get("text") or ""),
            "answer": str(row.get("answer") or ""),
            "comment": str(row.get("comment") or ""),
            "author_names": _names(row.get("author_names")),
            "author_player_ids": _ints(row.get("author_player_ids")),
            "editor_names": _names(row.get("editor_names")),
            "editor_player_ids": _ints(row.get("editor_player_ids")),
        }
        for row in records
    ]
    slim.sort(key=lambda row: (row["question_number"], row["rating_tournament_id"]))
    payload = json.dumps(slim, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def hashes_from_frame(frame: Any) -> dict[str, str]:
    revisions: dict[str, str] = {}
    for pack_id, group in frame.groupby("pack_id"):
        revisions[str(int(pack_id))] = pack_hash(group.to_dict("records"))
    return revisions


def extract_pack_from_rsc(text: str) -> dict[str, Any] | None:
    """Same RSC scrape as chgk_rating.gotquestions.extract_pack_from_rsc."""
    if "NEXT_HTTP_ERROR_FALLBACK;404" in text:
        return None
    decoder = json.JSONDecoder()
    pos = 0
    target = '"pack":{'
    while pos < len(text):
        idx = text.find(target, pos)
        if idx == -1:
            break
        try:
            obj, _ = decoder.raw_decode(text, idx + len('"pack":'))
            if isinstance(obj, dict) and "id" in obj:
                return obj
        except Exception:
            pass
        pos = idx + len(target)
    return None


def records_from_pack(pack: dict[str, Any]) -> list[dict[str, Any]]:
    pack_id = int(pack.get("id") or 0)
    raw_t_ids = pack.get("tIds") or []
    t_ids = [int(item) for item in raw_t_ids if str(item).isdigit()]
    records: list[dict[str, Any]] = []
    global_q_idx = 0
    for tour_idx, tour in enumerate(pack.get("tours") or [], 1):
        tour_id = int(tour.get("id") or 0)
        editor_names: list[str] = []
        editor_player_ids: list[int] = []
        for editor in tour.get("editors") or []:
            person = editor.get("person") or {}
            if person.get("name"):
                editor_names.append(str(person["name"]))
            if person.get("playerId"):
                editor_player_ids.append(int(person["playerId"]))
        for question in tour.get("questions") or []:
            global_q_idx += 1
            q_num = int(question.get("number") or global_q_idx)
            author_names: list[str] = []
            author_player_ids: list[int] = []
            for author in question.get("authors") or []:
                person = author.get("person") or {}
                if person.get("name"):
                    author_names.append(str(person["name"]))
                if person.get("playerId"):
                    author_player_ids.append(int(person["playerId"]))
            base = {
                "pack_id": pack_id,
                "tour_id": tour_id,
                "tour_number": tour_idx,
                "question_number": q_num,
                "text": str(question.get("text") or "").strip(),
                "answer": str(question.get("answer") or "").strip(),
                "comment": str(question.get("comment") or "").strip(),
                "author_names": author_names,
                "author_player_ids": author_player_ids,
                "editor_names": editor_names,
                "editor_player_ids": editor_player_ids,
            }
            targets = t_ids or [0]
            for tournament_id in targets:
                question_id = (
                    f"{tournament_id}_{global_q_idx}" if tournament_id else f"gq_{pack_id}_{global_q_idx}"
                )
                records.append({
                    "question_id": question_id,
                    "rating_tournament_id": tournament_id,
                    **base,
                })
    return records


def _stamp(moment: datetime | None = None) -> str:
    current = moment or datetime.now(timezone.utc)
    return current.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_state(storage: Storage) -> dict[str, Any] | None:
    raw = storage.get_bytes(STATE_OBJECT)
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        return None
    return data


def save_state(storage: Storage, state: dict[str, Any]) -> None:
    storage.put_bytes(
        STATE_OBJECT,
        json.dumps(state, ensure_ascii=False).encode(),
        content_type="application/json",
    )


def _write_part(storage: Storage, records: list[dict[str, Any]], moment: datetime) -> str:
    name = PARTS_PREFIX + moment.strftime("%Y%m%dT%H%M%SZ") + ".jsonl.gz"
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False).encode())
            handle.write(b"\n")
    storage.put_bytes(name, buffer.getvalue(), content_type="application/gzip")
    return name


def _fetch_live(pack_id: int) -> dict[str, Any] | None:
    import requests

    response = requests.get(
        f"{SOURCE}/pack/{pack_id}",
        headers={
            "RSC": "1",
            "Accept": "*/*",
            "User-Agent": "Mozilla/5.0 (compatible; ChGKRating/1.0)",
        },
        timeout=30,
    )
    if response.status_code in (403, 429):
        raise QuestionsBlocked(f"HTTP {response.status_code} on pack {pack_id}")
    if response.status_code == 404 or "NEXT_HTTP_ERROR_FALLBACK;404" in response.text:
        return None
    if response.status_code != 200:
        return None
    pack = extract_pack_from_rsc(response.text)
    if pack is None or int(pack.get("id") or 0) != pack_id:
        return None
    return pack


def _settings(
    lookback: int | None,
    max_new: int | None,
    interval: float | None,
) -> tuple[int, int, float]:
    if lookback is None:
        lookback = int(os.environ.get("QUESTIONS_LOOKBACK") or DEFAULT_LOOKBACK)
    if max_new is None:
        max_new = int(os.environ.get("QUESTIONS_MAX_NEW") or DEFAULT_MAX_NEW)
    if interval is None:
        interval = float(os.environ.get("QUESTIONS_INTERVAL_SECONDS") or "1")
    return lookback, max_new, interval


def run_incremental(
    *,
    storage: Storage | None = None,
    fetch_pack: Callable[[int], dict[str, Any] | None] | None = None,
    sleep: Callable[[float], None] | None = None,
    lookback: int | None = None,
    max_new: int | None = None,
    interval: float | None = None,
    not_found_stop: int = DEFAULT_NOT_FOUND_STOP,
    now: datetime | None = None,
) -> SyncResult:
    lookback, max_new, interval = _settings(lookback, max_new, interval)
    if storage is None:
        if not config.gcs_bucket():
            return SyncResult(True, False, 0, 0, 0, "GCS_BUCKET не задан, синхронизация пропущена.")
        storage = open_storage()
    fetch = fetch_pack or _fetch_live
    pause = sleep or __import__("time").sleep
    moment = now or datetime.now(timezone.utc)

    state = load_state(storage)
    if not state or "max_pack_id" not in state:
        return SyncResult(
            False,
            True,
            0,
            0,
            0,
            "База вопросов ещё не загружена в Cloud Storage. Нужен upload snapshot.",
        )

    max_pack_id = int(state["max_pack_id"])
    revisions = {str(key): str(value) for key, value in (state.get("pack_rev") or {}).items()}
    changed: list[dict[str, Any]] = []
    written = 0
    fetched = 0

    def pause_once() -> None:
        if interval > 0 and fetched:
            pause(interval)

    def take(pack_id: int) -> dict[str, Any] | None:
        nonlocal fetched
        pause_once()
        fetched += 1
        return fetch(pack_id)

    try:
        start = max(1, max_pack_id - lookback + 1)
        for pack_id in range(start, max_pack_id + 1):
            pack = take(pack_id)
            if not pack:
                continue
            records = records_from_pack(pack)
            digest = pack_hash(records)
            if revisions.get(str(pack_id)) == digest:
                continue
            revisions[str(pack_id)] = digest
            changed.extend(records)
            written += 1

        streak = 0
        scan = max_pack_id + 1
        new_count = 0
        while new_count < max_new and streak < not_found_stop:
            pack = take(scan)
            if not pack:
                streak += 1
                scan += 1
                continue
            streak = 0
            records = records_from_pack(pack)
            pack_key = str(int(pack.get("id") or scan))
            revisions[pack_key] = pack_hash(records)
            changed.extend(records)
            written += 1
            new_count += 1
            max_pack_id = max(max_pack_id, int(pack.get("id") or scan))
            scan += 1
        status = "ok"
    except QuestionsBlocked as exc:
        logger.warning("Question sync stopped: %s", exc)
        status = "blocked"
        _persist(storage, state, revisions, max_pack_id, fetched, written, changed, status, moment)
        return SyncResult(
            False,
            True,
            written,
            fetched,
            max_pack_id,
            f"Синхронизация вопросов остановлена: {exc}. Записано пакетов: {written}.",
        )

    part = _persist(storage, state, revisions, max_pack_id, fetched, written, changed, status, moment)
    message = (
        f"Вопросы: max pack {max_pack_id}, запросов {fetched}, "
        f"изменённых пакетов {written}"
        + (f", часть {part}" if part else "")
        + "."
    )
    logger.info(message)
    return SyncResult(True, written > 0, written, fetched, max_pack_id, message)


def _persist(
    storage: Storage,
    state: dict[str, Any],
    revisions: dict[str, str],
    max_pack_id: int,
    fetched: int,
    written: int,
    changed: list[dict[str, Any]],
    status: str,
    moment: datetime,
) -> str:
    part = ""
    if changed:
        part = _write_part(storage, changed, moment)
    state.update({
        "max_pack_id": max_pack_id,
        "pack_rev": revisions,
        "updated_at": _stamp(moment),
        "last_status": status,
        "last_fetched": fetched,
        "last_written": written,
        "snapshot": state.get("snapshot") or SNAPSHOT_OBJECT,
    })
    if part:
        state["last_part"] = part
    save_state(storage, state)
    return part


def upload_snapshot(path: Path, *, storage: Storage | None = None, lookback: int = DEFAULT_LOOKBACK) -> str:
    import pandas as pd

    frame = pd.read_parquet(path)
    if frame.empty or "pack_id" not in frame.columns:
        raise RuntimeError(f"{path} has no question rows")
    if storage is None:
        storage = open_storage()
    for name in storage.list_prefix(PARTS_PREFIX):
        storage.delete(name)
    storage.put_bytes(SNAPSHOT_OBJECT, Path(path).read_bytes(), content_type="application/octet-stream")
    max_pack_id = int(frame["pack_id"].max())
    recent = frame[frame["pack_id"] >= max(1, max_pack_id - lookback + 1)]
    state = {
        "max_pack_id": max_pack_id,
        "pack_rev": hashes_from_frame(recent),
        "updated_at": _stamp(),
        "last_status": "uploaded",
        "last_fetched": 0,
        "last_written": 0,
        "snapshot": SNAPSHOT_OBJECT,
        "rows": int(len(frame)),
    }
    save_state(storage, state)
    return (
        f"Загружен snapshot ({len(frame)} вопросов, max pack {max_pack_id}) "
        f"в gs://{config.gcs_bucket()}/{SNAPSHOT_OBJECT}"
    )


def materialize(dest: Path, *, storage: Storage | None = None) -> str:
    import pandas as pd

    if storage is None:
        storage = open_storage()
    frames = []
    snapshot = storage.get_bytes(SNAPSHOT_OBJECT)
    if snapshot:
        frames.append(pd.read_parquet(io.BytesIO(snapshot)))
    for name in storage.list_prefix(PARTS_PREFIX):
        raw = storage.get_bytes(name)
        if not raw:
            continue
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as handle:
            payload = handle.read().decode()
        # json.loads, not read_json: ids like "100_1" must stay strings.
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        raise RuntimeError("В бакете нет snapshot и нет частей")
    merged = pd.concat(frames, ignore_index=True)
    merged["question_id"] = merged["question_id"].astype(str)
    merged = merged.drop_duplicates(subset=["question_id"], keep="last")
    dest.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(dest, index=False)
    return f"Собрано {len(merged)} вопросов в {dest}"


def status_text(storage: Storage | None = None) -> str:
    bucket = config.gcs_bucket()
    if storage is None:
        if not bucket:
            return (
                "База вопросов — Cloud Storage (US regional, 5 ГБ free), не Datastore.\n"
                "GCS_BUCKET пока не задан."
            )
        try:
            storage = open_storage(bucket)
        except Exception as exc:
            logger.exception("question status failed")
            return f"Не удалось открыть бакет: {exc.__class__.__name__}"
    state = load_state(storage)
    if not state:
        return f"gs://{bucket}: snapshot ещё не загружен."
    return "\n".join([
        f"gs://{bucket}/questions/",
        f"max pack: {state.get('max_pack_id')}",
        f"строк в snapshot: {state.get('rows', '?')}",
        f"обновлено: {state.get('updated_at') or '-'}",
        f"последний прогон: {state.get('last_status') or '-'} "
        f"(запросов {state.get('last_fetched', 0)}, пакетов {state.get('last_written', 0)})",
    ])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    command = sys_argv(1)
    if command == "upload":
        print(upload_snapshot(Path(sys_argv(2))))
    elif command == "sync":
        result = run_incremental()
        print(result.message)
        if not result.ok:
            raise SystemExit(1)
    elif command == "materialize":
        print(materialize(Path(sys_argv(2))))
    elif command == "status":
        print(status_text())
    else:
        raise SystemExit("usage: questions_sync.py upload|sync|materialize|status ...")


def sys_argv(index: int) -> str:
    import sys

    if len(sys.argv) <= index:
        raise SystemExit("usage: questions_sync.py upload|sync|materialize|status ...")
    return sys.argv[index]


if __name__ == "__main__":
    main()

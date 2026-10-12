"""Tournament results from api.rating.chgk.info, stored next to the question corpus.

The working table is ``rating/snapshot.parquet`` (the local ``dataset.parquet``,
uploaded once) plus ``rating/parts/*.jsonl.gz``. A part replaces every row of
the tournaments it contains. Cron writes only that part: it does not download
the snapshot, and it does not use the 671 MB SQLite HTTP cache.

Rows join to the question corpus on ``question_id`` (``{tournament_id}_{n}``).
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from gae_platform import config
from questions_sync import Storage, open_storage

logger = logging.getLogger(__name__)

API = "https://api.rating.chgk.info"
SNAPSHOT_OBJECT = "rating/snapshot.parquet"
STATE_OBJECT = "state/rating.json"
PARTS_PREFIX = "rating/parts/"

DEFAULT_LOOKBACK_DAYS = 3
DEFAULT_MAX_TOURNAMENTS = 40

ONLINE_KEYWORDS = (
    "онлайн",
    "online",
    "дискорд",
    "discord",
    "zoom",
    "зум",
    "интернет",
    "skype",
    "скайп",
)


class RatingBusy(Exception):
    """The rating API asked us to back off."""


@dataclass
class SyncResult:
    ok: bool
    notify: bool
    written: int
    fetched: int
    message: str


def _today(now: datetime | None) -> date:
    current = now or datetime.now(timezone.utc)
    return current.astimezone(timezone.utc).date()


def _stamp(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return current.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _members(data: Any) -> list:
    if not data:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.get("hydra:member") or [])
    return []


def _is_aggregate(type_name: str, title: str) -> bool:
    combined = f"{type_name} {title}".lower()
    patterns = (
        "общий зачёт",
        "общий зачет",
        "сезон",
        "рейтинг",
        "сводная",
        "сводный",
        "гран-при",
        "этап кубка мира",
    )
    return any(pattern in combined for pattern in patterns)


def _format_name(type_name: str, title: str) -> str:
    combined = f"{type_name} {title}".lower()
    if any(word in combined for word in ONLINE_KEYWORDS):
        return "online"
    if any(word in combined for word in ("синхрон", "sync", "асинхрон", "async", "зеркало")):
        return "sync"
    return "offline"


def _participation(tournament_format: str, venue_name: str) -> str:
    if tournament_format == "online":
        return "online"
    lowered = (venue_name or "").lower()
    if lowered and any(word in lowered for word in ONLINE_KEYWORDS):
        return "online"
    return tournament_format


def records_from_results(tournament: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tournament_id = int(tournament["id"])
    tournament_format = str(tournament.get("format") or "offline")
    played_on = str(tournament.get("dateStart") or "")[:10]
    rows: list[dict[str, Any]] = []
    for result in results:
        team_obj = result.get("team") or {}
        team_id = team_obj.get("id") or result.get("idTeam")
        if not team_id:
            continue
        members = result.get("teamMembers") or []
        players = [
            int(member["player"]["id"])
            for member in members
            if isinstance(member, dict) and (member.get("player") or {}).get("id")
        ]
        if not players:
            continue
        mask = result.get("mask") or ""
        if not mask:
            tours = result.get("tours") or []
            mask = "".join(tour.get("mask") or "" for tour in tours if isinstance(tour, dict))
        if not mask:
            continue
        venue = ((result.get("synchRequest") or {}).get("venue") or {})
        venue_name = venue.get("name") if isinstance(venue, dict) else str(venue or "")
        part_format = _participation(tournament_format, venue_name)
        question_index = 0
        for char in mask:
            if char in ("X", "x", "?"):
                continue
            if char not in ("0", "1"):
                continue
            question_index += 1
            rows.append({
                "tournament_id": tournament_id,
                "tournament_type": part_format,
                "date": played_on,
                "team_id": int(team_id),
                "player_ids": players[:6],
                "question_id": f"{tournament_id}_{question_index}",
                "solved": int(char),
            })
    return rows


def _live_get(url: str, params: dict | None = None) -> Any:
    import requests

    response = requests.get(
        url,
        params=params,
        timeout=30,
        headers={"Accept": "application/json", "User-Agent": "ChGKRating/1.0"},
    )
    if response.status_code == 404:
        return None
    if response.status_code in (429, 500, 502, 503, 504):
        raise RatingBusy(f"HTTP {response.status_code} on {url}")
    response.raise_for_status()
    return response.json()


def _live_list(since: str, page: int) -> list[dict[str, Any]]:
    data = _live_get(
        f"{API}/tournaments",
        params={"page": page, "itemsPerPage": 50, "lastEditDate[after]": since},
    )
    raw_items = _members(data)
    found: list[dict[str, Any]] = []
    for item in raw_items:
        title = str(item.get("name") or "")
        raw_type = item.get("type") or {}
        type_name = raw_type.get("name") if isinstance(raw_type, dict) else str(raw_type)
        if _is_aggregate(str(type_name), title):
            continue
        if not item.get("id"):
            continue
        found.append({
            "id": int(item["id"]),
            "name": title,
            "format": _format_name(str(type_name), title),
            "dateStart": item.get("dateStart") or "",
        })
    return found, len(raw_items) >= 50


def _live_results(tournament_id: int) -> list[dict[str, Any]]:
    data = _live_get(
        f"{API}/tournaments/{tournament_id}/results",
        params={"includeMasksAndControversials": 1, "includeTeamMembers": 1},
    )
    return [item for item in _members(data) if isinstance(item, dict)]


def load_state(storage: Storage) -> dict[str, Any] | None:
    raw = storage.get_bytes(STATE_OBJECT)
    if not raw:
        return None
    data = json.loads(raw)
    return data if isinstance(data, dict) else None


def save_state(storage: Storage, state: dict[str, Any]) -> None:
    storage.put_bytes(STATE_OBJECT, json.dumps(state, ensure_ascii=False).encode(), "application/json")


def _write_part(storage: Storage, rows: list[dict[str, Any]], moment: datetime) -> str:
    name = PARTS_PREFIX + moment.strftime("%Y%m%dT%H%M%SZ") + ".jsonl.gz"
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False).encode())
            handle.write(b"\n")
    storage.put_bytes(name, buffer.getvalue(), "application/gzip")
    return name


def _settings(lookback: int | None, max_tournaments: int | None) -> tuple[int, int]:
    if lookback is None:
        lookback = int(os.environ.get("RATING_LOOKBACK_DAYS") or DEFAULT_LOOKBACK_DAYS)
    if max_tournaments is None:
        max_tournaments = int(os.environ.get("RATING_MAX_TOURNAMENTS") or DEFAULT_MAX_TOURNAMENTS)
    return lookback, max_tournaments


def run_incremental(
    *,
    storage: Storage | None = None,
    list_tournaments: Callable[[str, int], list[dict[str, Any]]] | None = None,
    fetch_results: Callable[[int], list[dict[str, Any]]] | None = None,
    sleep: Callable[[float], None] | None = None,
    lookback: int | None = None,
    max_tournaments: int | None = None,
    now: datetime | None = None,
) -> SyncResult:
    lookback, max_tournaments = _settings(lookback, max_tournaments)
    if storage is None:
        if not config.gcs_bucket():
            return SyncResult(True, False, 0, 0, "GCS_BUCKET не задан, синхронизация рейтинга пропущена.")
        storage = open_storage()
    state = load_state(storage)
    if not state or "last_sync" not in state:
        return SyncResult(
            False,
            True,
            0,
            0,
            "Результаты турнирного сайта ещё не загружены. Нужен upload-rating.",
        )

    listing = list_tournaments or _live_list  # type: ignore[assignment]
    results_of = fetch_results or _live_results
    pause = sleep or __import__("time").sleep
    interval = float(os.environ.get("RATING_INTERVAL_SECONDS") or "0.2")
    moment = now or datetime.now(timezone.utc)
    today = _today(moment)

    if state.get("pass_since"):
        since = str(state["pass_since"])
        done = {int(item) for item in state.get("done_ids") or []}
    else:
        since = (today - timedelta(days=lookback)).isoformat()
        done = set()

    rows: list[dict[str, Any]] = []
    replaced: list[int] = []
    fetched = 0
    hit_cap = False
    page = 1
    try:
        while not hit_cap:
            if fetched and interval > 0:
                pause(interval)
            listed = listing(since, page)
            more = False
            if isinstance(listed, tuple):
                batch, more = listed
            else:
                batch = listed
                more = len(batch) >= 50
            fetched += 1
            if not batch and not more:
                break
            for tournament in batch:
                tournament_id = int(tournament["id"])
                if tournament_id in done:
                    continue
                if len(replaced) >= max_tournaments:
                    hit_cap = True
                    break
                if fetched and interval > 0:
                    pause(interval)
                results = results_of(tournament_id)
                fetched += 1
                produced = records_from_results(tournament, results)
                if not produced:
                    logger.warning("Tournament %s produced no rows", tournament_id)
                    continue
                rows.extend(produced)
                replaced.append(tournament_id)
                done.add(tournament_id)
            if hit_cap or not more:
                break
            page += 1
        status = "partial" if hit_cap else "ok"
    except RatingBusy as exc:
        logger.warning("Rating sync stopped: %s", exc)
        _persist(storage, state, rows, replaced, done, since, today, fetched, "busy", moment, hit_cap=True)
        return SyncResult(
            False,
            True,
            len(replaced),
            fetched,
            f"Синхронизация рейтинга остановлена: {exc}. Турниров записано: {len(replaced)}.",
        )

    _persist(storage, state, rows, replaced, done, since, today, fetched, status, moment, hit_cap=hit_cap)
    message = (
        f"Рейтинг: с {since}, запросов {fetched}, турниров {len(replaced)}"
        + (", есть продолжение" if hit_cap else "")
        + "."
    )
    logger.info(message)
    return SyncResult(True, bool(replaced) or hit_cap, len(replaced), fetched, message)


def _persist(
    storage: Storage,
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    replaced: list[int],
    done: set[int],
    since: str,
    today: date,
    fetched: int,
    status: str,
    moment: datetime,
    *,
    hit_cap: bool,
) -> None:
    part = _write_part(storage, rows, moment) if rows else ""
    parts = list(state.get("parts") or [])
    if replaced:
        parts.append({"name": part, "tournament_ids": replaced})
    state.update({
        "updated_at": _stamp(moment),
        "last_status": status,
        "last_fetched": fetched,
        "last_written": len(replaced),
        "parts": parts,
        "snapshot": SNAPSHOT_OBJECT,
    })
    if hit_cap:
        state["pass_since"] = since
        state["done_ids"] = sorted(done)
    else:
        state["pass_since"] = ""
        state["done_ids"] = []
        state["last_sync"] = today.isoformat()
    save_state(storage, state)


def upload_snapshot(path: Path, *, storage: Storage | None = None, now: datetime | None = None) -> str:
    import pandas as pd

    frame = pd.read_parquet(path)
    if frame.empty or "tournament_id" not in frame.columns:
        raise RuntimeError(f"{path} has no tournament rows")
    if storage is None:
        storage = open_storage()
    for name in storage.list_prefix(PARTS_PREFIX):
        storage.delete(name)
    storage.put_bytes(SNAPSHOT_OBJECT, Path(path).read_bytes())
    moment = now or datetime.now(timezone.utc)
    state = {
        "last_sync": _today(moment).isoformat(),
        "pass_since": "",
        "done_ids": [],
        "parts": [],
        "updated_at": _stamp(moment),
        "last_status": "uploaded",
        "last_fetched": 0,
        "last_written": 0,
        "snapshot": SNAPSHOT_OBJECT,
        "rows": int(len(frame)),
        "tournaments": int(frame["tournament_id"].nunique()),
    }
    save_state(storage, state)
    bucket = config.gcs_bucket() or "?"
    return (
        f"Загружен рейтинг ({len(frame)} строк, {state['tournaments']} турниров) "
        f"в gs://{bucket}/{SNAPSHOT_OBJECT}"
    )


def _read_part(raw: bytes) -> list[dict[str, Any]]:
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as handle:
        payload = handle.read().decode()
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def _players(value: object) -> list[int]:
    if value is None or isinstance(value, float):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def materialize(dest: Path, *, storage: Storage | None = None) -> str:
    import pandas as pd

    if storage is None:
        storage = open_storage()
    state = load_state(storage) or {}
    snapshot = storage.get_bytes(SNAPSHOT_OBJECT)
    if snapshot:
        frame = pd.read_parquet(io.BytesIO(snapshot))
    else:
        frame = pd.DataFrame()
    for part in state.get("parts") or []:
        ids = {int(item) for item in part.get("tournament_ids") or []}
        if not frame.empty and ids and "tournament_id" in frame.columns:
            frame = frame[~frame["tournament_id"].isin(ids)]
        raw = storage.get_bytes(part.get("name") or "")
        if not raw:
            continue
        rows = _read_part(raw)
        if rows:
            frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    if frame.empty:
        raise RuntimeError("В бакете нет рейтинга")
    frame["question_id"] = frame["question_id"].astype(str)
    frame["tournament_id"] = frame["tournament_id"].astype(int)
    frame["player_ids"] = frame["player_ids"].apply(_players)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest, index=False)
    return f"Собрано {len(frame)} строк рейтинга, {frame['tournament_id'].nunique()} турниров в {dest}"


def status_text(storage: Storage | None = None) -> str:
    bucket = config.gcs_bucket()
    if storage is None:
        if not bucket:
            return "Результаты турнирного сайта лежат в том же Cloud Storage. GCS_BUCKET не задан."
        try:
            storage = open_storage(bucket)
        except Exception as exc:
            logger.exception("rating status failed")
            return f"Не удалось открыть бакет: {exc.__class__.__name__}"
    state = load_state(storage)
    if not state:
        return f"gs://{bucket}: рейтинг ещё не загружен."
    return "\n".join([
        f"gs://{bucket}/rating/",
        f"последняя полная синхронизация: {state.get('last_sync') or '-'}",
        f"строк в snapshot: {state.get('rows', '?')}, турниров: {state.get('tournaments', '?')}",
        f"обновлено: {state.get('updated_at') or '-'}",
        f"последний прогон: {state.get('last_status') or '-'} "
        f"(запросов {state.get('last_fetched', 0)}, турниров {state.get('last_written', 0)})",
        "незаконченный проход: " + (state.get("pass_since") or "нет"),
    ])


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) < 2:
        raise SystemExit("usage: rating_sync.py upload|sync|materialize|status ...")
    command = sys.argv[1]
    if command == "upload":
        print(upload_snapshot(Path(sys.argv[2])))
    elif command == "sync":
        result = run_incremental()
        print(result.message)
        if not result.ok:
            raise SystemExit(1)
    elif command == "materialize":
        print(materialize(Path(sys.argv[2])))
    elif command == "status":
        print(status_text())
    else:
        raise SystemExit("usage: rating_sync.py upload|sync|materialize|status ...")


if __name__ == "__main__":
    main()

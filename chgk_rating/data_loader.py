"""Asynchronous data ingestion, SQLite caching, and preprocessing for ChGK rating data."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from chgk_rating.config import (
    DEFAULT_API_BASE_URL,
    DEFAULT_CACHE_DB_PATH,
    DEFAULT_PARQUET_PATH,
    FORMAT_OFFLINE,
    FORMAT_ONLINE,
    FORMAT_SYNC,
    DataIngestConfig,
)

logger = logging.getLogger(__name__)


class ResponseCache:
    """Persistent SQLite cache for raw HTTP responses."""

    def __init__(self, db_path: Path = DEFAULT_CACHE_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_cache (
                    url TEXT PRIMARY KEY,
                    status_code INTEGER,
                    response_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingested_tournaments (
                    tournament_id INTEGER PRIMARY KEY,
                    title TEXT,
                    tournament_type TEXT,
                    date_start TEXT,
                    date_end TEXT,
                    raw_data TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sync_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get(self, url: str) -> Optional[Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT response_json FROM api_cache WHERE url = ?", (url,))
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    return None
        return None

    def invalidate(self, url: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM api_cache WHERE url = ?", (url,))
            conn.commit()

    def get_sync_state(self, key: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM sync_state WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None

    def set_sync_state(self, key: str, value: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sync_state (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (key, value),
            )
            conn.commit()

    def set(self, url: str, status_code: int, data: Any) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO api_cache (url, status_code, response_json, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (url, status_code, json.dumps(data, ensure_ascii=False)),
            )
            conn.commit()

    def save_tournament_meta(
        self,
        tournament_id: int,
        title: str,
        tournament_type: str,
        date_start: str,
        date_end: str,
        raw_data: Dict[str, Any],
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO ingested_tournaments
                (tournament_id, title, tournament_type, date_start, date_end, raw_data)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    tournament_id,
                    title,
                    tournament_type,
                    date_start,
                    date_end,
                    json.dumps(raw_data, ensure_ascii=False),
                ),
            )
            conn.commit()


class AsyncRateLimiter:
    """Token bucket / interval rate limiter for asynchronous HTTP requests."""

    def __init__(self, max_per_second: float = 10.0) -> None:
        self.interval = 1.0 / max(max_per_second, 1.0)
        self.lock = asyncio.Lock()
        self.last_request_time: float = 0.0

    async def acquire(self) -> None:
        async with self.lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.last_request_time
            if elapsed < self.interval:
                await asyncio.sleep(self.interval - elapsed)
            self.last_request_time = asyncio.get_event_loop().time()


def normalize_tournament_format(type_name: str, title: str) -> str:
    """Normalize ChGK tournament type to offline, sync, or online."""
    combined = f"{type_name} {title}".lower()
    if any(k in combined for k in ["онлайн", "online", "дискорд", "discord", "zoom"]):
        return FORMAT_ONLINE
    elif any(k in combined for k in ["синхрон", "sync", "асинхрон", "async", "зеркало"]):
        return FORMAT_SYNC
    else:
        return FORMAT_OFFLINE


def is_aggregate_tournament(type_name: str, title: str) -> bool:
    """Identify and filter out aggregate / 'общий зачёт' / season rankings."""
    combined = f"{type_name} {title}".lower()
    ignore_patterns = [
        "общий зачёт",
        "общий зачет",
        "сезон",
        "рейтинг",
        "сводная",
        "сводный",
        "гран-при",
        "этап кубка мира",
    ]
    return any(pattern in combined for pattern in ignore_patterns)


class ChGKDataIngester:
    """Fetches tournaments, team rosters, and question masks from api.rating.chgk.info."""

    def __init__(self, config: Optional[DataIngestConfig] = None) -> None:
        self.config = config or DataIngestConfig()
        self.cache = ResponseCache(self.config.cache_db_path)
        self.limiter = AsyncRateLimiter(self.config.max_requests_per_second)

    async def _get_json(
        self, client: httpx.AsyncClient, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        cached = self.cache.get(url)
        if cached is not None:
            return cached

        retries = 0
        backoff = 1.0
        while retries <= self.config.max_retries:
            await self.limiter.acquire()
            try:
                response = await client.get(url, params=params, timeout=self.config.timeout_seconds)
                if response.status_code == 200:
                    data = response.json()
                    self.cache.set(url, 200, data)
                    return data
                elif response.status_code in (429, 500, 502, 503, 504):
                    retries += 1
                    logger.warning(
                        "HTTP %d on %s, retrying in %.1fs (attempt %d/%d)",
                        response.status_code,
                        url,
                        backoff,
                        retries,
                        self.config.max_retries,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                elif response.status_code == 404:
                    self.cache.set(url, 404, None)
                    return None
                else:
                    response.raise_for_status()
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                retries += 1
                if retries > self.config.max_retries:
                    logger.error("Failed to fetch %s after %d retries: %s", url, retries, exc)
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2.0
        return None

    async def fetch_tournaments(
        self,
        client: httpx.AsyncClient,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        last_edit_after: Optional[str] = None,
        max_tournaments: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch list of valid non-aggregate tournaments starting from date or edited since date."""
        tournaments: List[Dict[str, Any]] = []
        page = 1
        if last_edit_after:
            query_suffix = f"&lastEditDate[after]={last_edit_after}"
        else:
            query_suffix = f"&dateStart[after]={start_date}"
            if end_date:
                query_suffix += f"&dateEnd[before]={end_date}"

        while True:
            if max_tournaments is not None and len(tournaments) >= max_tournaments:
                break
            url = f"{self.config.api_base_url}/tournaments?page={page}&itemsPerPage=50{query_suffix}"
            data = await self._get_json(client, url)
            if not data:
                break
            items = data if isinstance(data, list) else data.get("hydra:member", [])
            if not items:
                break

            for t in items:
                t_id = t.get("id")
                title = t.get("name", "")
                t_type = t.get("type", {})
                type_name = t_type.get("name", "") if isinstance(t_type, dict) else str(t_type)

                if is_aggregate_tournament(type_name, title):
                    continue

                fmt = normalize_tournament_format(type_name, title)
                date_start = t.get("dateStart", "")
                date_end = t.get("dateEnd", date_start)

                tournaments.append({
                    "id": t_id,
                    "name": title,
                    "format": fmt,
                    "dateStart": date_start,
                    "dateEnd": date_end,
                })
                self.cache.save_tournament_meta(t_id, title, fmt, date_start, date_end, t)
                if max_tournaments is not None and len(tournaments) >= max_tournaments:
                    break
            page += 1
        return tournaments

    async def fetch_tournament_rosters(
        self, client: httpx.AsyncClient, tournament_id: int
    ) -> Dict[int, List[int]]:
        """Fetch team rosters for a tournament: team_id -> list of player_ids."""
        url = f"{self.config.api_base_url}/tournaments/{tournament_id}/teams"
        data = await self._get_json(client, url)
        if not data:
            return {}

        items = data if isinstance(data, list) else data.get("hydra:member", [])
        rosters: Dict[int, List[int]] = {}
        for team_entry in items:
            team_obj = team_entry.get("team", {})
            team_id = team_obj.get("id") or team_entry.get("id")
            if not team_id:
                continue

            members = team_entry.get("teamMembers") or team_entry.get("members") or []
            player_ids: List[int] = []
            for m in members:
                player = m.get("player") or {}
                p_id = player.get("id") or m.get("idPlayer")
                if p_id:
                    player_ids.append(int(p_id))
            rosters[int(team_id)] = player_ids
        return rosters

    async def fetch_tournament_results(
        self, client: httpx.AsyncClient, tournament_id: int
    ) -> List[Dict[str, Any]]:
        """Fetch question results/masks with team members for all teams in a tournament."""
        url = f"{self.config.api_base_url}/tournaments/{tournament_id}/results?includeMasksAndControversials=1&includeTeamMembers=1"
        data = await self._get_json(client, url)
        if not data:
            return []
        items = data if isinstance(data, list) else data.get("hydra:member", [])
        return items

    async def ingest_dataset(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        max_tournaments: Optional[int] = None,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> pd.DataFrame:
        """Run complete async ingestion and compile into Parquet format."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        async with httpx.AsyncClient() as client:
            tournaments = await self.fetch_tournaments(
                client,
                start_date=start_date,
                end_date=end_date,
                max_tournaments=max_tournaments,
            )
            logger.info("Found %d valid tournaments to process", len(tournaments))
            total_t = len(tournaments)

            schema = pa.schema([
                ("tournament_id", pa.int64()),
                ("tournament_type", pa.string()),
                ("date", pa.string()),
                ("team_id", pa.int64()),
                ("player_ids", pa.list_(pa.int64())),
                ("question_id", pa.string()),
                ("solved", pa.int8()),
            ])

            self.config.parquet_path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(self.config.parquet_path, schema)

            sem = asyncio.Semaphore(10)
            completed_count = 0
            records_buffer: List[Dict[str, Any]] = []
            total_saved_records = 0

            async def process_tournament(idx: int, t: Dict[str, Any]) -> List[Dict[str, Any]]:
                nonlocal completed_count
                t_id = t["id"]
                t_fmt = t["format"]
                t_date = (t["dateStart"] or "2020-01-01")[:10]

                async with sem:
                    results = await self.fetch_tournament_results(client, t_id)

                t_records: List[Dict[str, Any]] = []
                rosters: Optional[Dict[int, List[int]]] = None

                for res in results:
                    team_obj = res.get("team", {})
                    team_id = team_obj.get("id") or res.get("idTeam")
                    if not team_id:
                        continue
                    team_id = int(team_id)

                    # Extract player IDs directly from teamMembers if available
                    members = res.get("teamMembers") or []
                    players = [
                        int(m["player"]["id"])
                        for m in members
                        if isinstance(m, dict) and m.get("player") and m["player"].get("id")
                    ]
                    if not players:
                        continue

                    mask_str = res.get("mask", "")
                    if not mask_str:
                        tours = res.get("tours", [])
                        mask_str = "".join(
                            tour.get("mask", "") for tour in tours if isinstance(tour, dict)
                        )

                    if not mask_str:
                        continue

                    # Parse question mask
                    q_idx = 0
                    for char in mask_str:
                        if char in ("X", "x", "?"):
                            continue
                        elif char in ("0", "1"):
                            q_idx += 1
                            question_id = f"{t_id}_{q_idx}"
                            t_records.append({
                                "tournament_id": t_id,
                                "tournament_type": t_fmt,
                                "date": t_date,
                                "team_id": team_id,
                                "player_ids": players[:6],
                                "question_id": question_id,
                                "solved": int(char),
                            })

                completed_count += 1
                if progress_cb and (completed_count % 10 == 0 or completed_count == total_t):
                    progress_cb(
                        f"Fetched tournament {t_id} ({t['name'][:25]}...)",
                        completed_count,
                        total_t,
                    )
                return t_records

            # Process tournaments in concurrent chunks
            chunk_size = 50
            for i in range(0, total_t, chunk_size):
                chunk = tournaments[i : i + chunk_size]
                tasks = [process_tournament(i + j + 1, t) for j, t in enumerate(chunk)]
                results_list = await asyncio.gather(*tasks)

                for r_list in results_list:
                    records_buffer.extend(r_list)

                # Write out chunk if buffer has accumulated sufficient rows
                if len(records_buffer) >= 20000:
                    batch_df = pd.DataFrame(records_buffer)
                    table = pa.Table.from_pandas(batch_df, schema=schema, preserve_index=False)
                    writer.write_table(table)
                    total_saved_records += len(records_buffer)
                    records_buffer.clear()

            # Flush remaining records
            if records_buffer:
                batch_df = pd.DataFrame(records_buffer)
                table = pa.Table.from_pandas(batch_df, schema=schema, preserve_index=False)
                writer.write_table(table)
                total_saved_records += len(records_buffer)
                records_buffer.clear()

            writer.close()
            logger.info("Saved %d total records into %s", total_saved_records, self.config.parquet_path)
            self.cache.set_sync_state("rating_last_sync", datetime.now().isoformat())

            if total_saved_records > 0:
                return pd.read_parquet(self.config.parquet_path)
            return pd.DataFrame()

    async def update_incremental(
        self,
        since_date: Optional[str] = None,
        lookback_days: int = 30,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> pd.DataFrame:
        """Incrementally update dataset with tournaments created or edited recently."""
        if since_date is None:
            last_sync = self.cache.get_sync_state("rating_last_sync")
            if last_sync:
                try:
                    dt = datetime.fromisoformat(last_sync[:10]) - timedelta(days=lookback_days)
                    since_date = dt.strftime("%Y-%m-%d")
                except ValueError:
                    since_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            else:
                since_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        logger.info("Checking for edited tournaments on rating site since %s...", since_date)

        async with httpx.AsyncClient() as client:
            updated_tournaments = await self.fetch_tournaments(
                client,
                last_edit_after=since_date,
            )

            if not updated_tournaments:
                logger.info("No tournaments were updated since %s.", since_date)
                self.cache.set_sync_state("rating_last_sync", datetime.now().isoformat())
                if self.config.parquet_path.exists():
                    return pd.read_parquet(self.config.parquet_path)
                return pd.DataFrame()

            logger.info("Found %d edited/new tournaments to sync.", len(updated_tournaments))
            updated_t_ids = {int(t["id"]) for t in updated_tournaments if t.get("id")}

            # Invalidate cached results for these updated tournaments so fresh results are fetched
            for t_id in updated_t_ids:
                res_url = f"{self.config.api_base_url}/tournaments/{t_id}/results?includeMasksAndControversials=1&includeTeamMembers=1"
                self.cache.invalidate(res_url)

            # Ingest records for the updated tournaments
            sem = asyncio.Semaphore(10)
            new_records: List[Dict[str, Any]] = []
            completed = 0
            total_t = len(updated_tournaments)

            async def process_one(t: Dict[str, Any]) -> List[Dict[str, Any]]:
                nonlocal completed
                t_id = int(t["id"])
                t_fmt = t.get("format") or normalize_tournament_format(
                    t.get("type", {}).get("name", "") if isinstance(t.get("type"), dict) else str(t.get("type", "")),
                    t.get("name", ""),
                )
                t_date = (t.get("dateStart") or "2020-01-01")[:10]
                async with sem:
                    results = await self.fetch_tournament_results(client, t_id)
                t_recs = []
                for res in results:
                    team_obj = res.get("team", {})
                    team_id = team_obj.get("id") or res.get("idTeam")
                    if not team_id:
                        continue
                    team_id = int(team_id)
                    members = res.get("teamMembers") or []
                    players = [
                        int(m["player"]["id"])
                        for m in members
                        if isinstance(m, dict) and m.get("player") and m["player"].get("id")
                    ]
                    if not players:
                        continue
                    mask_str = res.get("mask", "")
                    if not mask_str:
                        tours = res.get("tours", [])
                        mask_str = "".join(
                            tour.get("mask", "") for tour in tours if isinstance(tour, dict)
                        )
                    if not mask_str:
                        continue
                    q_idx = 0
                    for char in mask_str:
                        if char in ("X", "x", "?"):
                            continue
                        elif char in ("0", "1"):
                            q_idx += 1
                            t_recs.append({
                                "tournament_id": t_id,
                                "tournament_type": t_fmt,
                                "date": t_date,
                                "team_id": team_id,
                                "player_ids": players[:6],
                                "question_id": f"{t_id}_{q_idx}",
                                "solved": int(char),
                            })
                completed += 1
                if progress_cb and (completed % 10 == 0 or completed == total_t):
                    progress_cb(f"Updated tournament {t_id}", completed, total_t)
                return t_recs

            tasks = [process_one(t) for t in updated_tournaments]
            results_nested = await asyncio.gather(*tasks)
            for rlist in results_nested:
                new_records.extend(rlist)

            # Merge with existing parquet dataset
            if self.config.parquet_path.exists():
                existing_df = pd.read_parquet(self.config.parquet_path)
                kept_df = existing_df[~existing_df["tournament_id"].isin(updated_t_ids)]
                if new_records:
                    new_df = pd.DataFrame(new_records)
                    merged_df = pd.concat([kept_df, new_df], ignore_index=True)
                else:
                    merged_df = kept_df
            else:
                merged_df = pd.DataFrame(new_records)

            if not merged_df.empty:
                merged_df = merged_df.sort_values(by=["date", "tournament_id"]).reset_index(drop=True)
                merged_df.to_parquet(self.config.parquet_path, index=False)
                logger.info(
                    "Updated dataset at %s: total records now %d (updated %d tournaments)",
                    self.config.parquet_path,
                    len(merged_df),
                    len(updated_tournaments),
                )

            self.cache.set_sync_state("rating_last_sync", datetime.now().isoformat())
            return merged_df


def generate_synthetic_dataset(
    output_parquet_path: Path = DEFAULT_PARQUET_PATH,
    num_tournaments: int = 30,
    num_players: int = 300,
    num_teams: int = 50,
    questions_per_tournament: int = 36,
    seed: int = 42,
    start_year: int = 2022,
) -> pd.DataFrame:
    """Generate realistic synthetic ChGK dataset with temporal intervals and hiatuses for dynamic tracking."""
    random.seed(seed)
    formats = [FORMAT_OFFLINE, FORMAT_SYNC, FORMAT_ONLINE]
    format_weights = [0.4, 0.4, 0.2]

    # Pre-generate true latent player skills
    player_true_skills = {p: random.gauss(0.0, 1.0) for p in range(1, num_players + 1)}

    # Build stable teams of 6 players
    teams: Dict[int, List[int]] = {}
    player_pool = list(range(1, num_players + 1))
    for t_id in range(1, num_teams + 1):
        roster = random.sample(player_pool, 6)
        teams[t_id] = roster

    # Dedicated hiatus players (e.g., players 1-10 play early, then experience a 2-3 year gap)
    hiatus_players = set(range(1, 15))

    records: List[Dict[str, Any]] = []
    base_date = datetime(start_year, 1, 1)
    player_last_seen: Dict[int, datetime] = {}

    for t_idx in range(1, num_tournaments + 1):
        t_id = 1000 + t_idx
        t_fmt = random.choices(formats, weights=format_weights, k=1)[0]
        # Spread tournaments every ~30 to 45 days across multiple years
        curr_dt = base_date + timedelta(days=int(t_idx * 40))
        t_date = curr_dt.strftime("%Y-%m-%d")

        # Sample participating teams
        participating_teams = random.sample(list(teams.keys()), min(25, num_teams))

        for team_id in participating_teams:
            roster = list(teams[team_id])

            # In middle tournaments (e.g. t_idx between 10 and 24), hiatus players don't play
            if 10 <= t_idx <= 24:
                roster = [
                    random.choice([p for p in player_pool if p not in hiatus_players])
                    if p in hiatus_players else p
                    for p in roster
                ]
            elif random.random() < 0.15:
                # Occasional random substitution
                sub_idx = random.randint(0, 5)
                roster[sub_idx] = random.choice(player_pool)

            # Compute days since last game for each player in roster
            days_since_last: List[float] = []
            for p in roster:
                if p in player_last_seen:
                    gap = (curr_dt - player_last_seen[p]).total_seconds() / 86400.0
                else:
                    gap = 0.0
                days_since_last.append(round(gap, 1))
                player_last_seen[p] = curr_dt

            # Team knowledge pooling: max skill
            max_skill = max(player_true_skills[p] for p in roster)
            synergy = random.gauss(0.1, 0.2)
            anomaly = 0.4 if (t_fmt == FORMAT_ONLINE and random.random() < 0.3) else 0.0

            for q_num in range(1, questions_per_tournament + 1):
                q_id = f"{t_id}_{q_num}"
                q_difficulty = random.gauss(0.2, 1.2)

                prob = 1.0 / (1.0 + pow(2.7182818, -(max_skill - q_difficulty + synergy + anomaly)))
                solved = 1 if random.random() < prob else 0

                records.append({
                    "tournament_id": t_id,
                    "tournament_type": t_fmt,
                    "date": t_date,
                    "team_id": team_id,
                    "player_ids": roster,
                    "question_id": q_id,
                    "solved": solved,
                    "days_since_last_game": days_since_last,
                })

    df = pd.DataFrame(records)
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet_path, index=False)
    logger.info("Saved %d synthetic observations to %s", len(df), output_parquet_path)
    return df


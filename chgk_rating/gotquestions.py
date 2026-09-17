"""Polite, rate-limited, incremental ingestion for gotquestions.online question packages."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from chgk_rating.config import (
    DEFAULT_GQ_CACHE_DB_PATH,
    DEFAULT_GQ_PACKS_PARQUET,
    DEFAULT_GQ_QUESTIONS_PARQUET,
    GotQuestionsConfig,
)

from chgk_rating.data_loader import AsyncRateLimiter

logger = logging.getLogger(__name__)


def extract_pack_from_rsc(text: str) -> Optional[Dict[str, Any]]:
    """Extract the structured pack dict from a React Server Component (RSC) response stream."""
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


def _find_pack_in_obj(obj: Any) -> Optional[Dict[str, Any]]:
    """Recursively find 'pack' dict in nested JSON structures."""
    if isinstance(obj, dict):
        if "pack" in obj and isinstance(obj["pack"], dict) and "id" in obj["pack"]:
            return obj["pack"]
        for v in obj.values():
            res = _find_pack_in_obj(v)
            if res is not None:
                return res
    elif isinstance(obj, list):
        for item in obj:
            res = _find_pack_in_obj(item)
            if res is not None:
                return res
    return None


class GotQuestionsCache:
    """Persistent SQLite cache for gotquestions.online packages and sync state."""

    def __init__(self, db_path: Path = DEFAULT_GQ_CACHE_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gq_packs (
                    pack_id INTEGER PRIMARY KEY,
                    title TEXT,
                    t_ids TEXT,
                    raw_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gq_not_found (
                    pack_id INTEGER PRIMARY KEY,
                    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gq_sync_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get_pack(self, pack_id: int) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT raw_json FROM gq_packs WHERE pack_id = ?", (pack_id,))
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    return None
        return None

    def is_known_not_found(self, pack_id: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM gq_not_found WHERE pack_id = ?", (pack_id,))
            return cursor.fetchone() is not None

    def save_pack(self, pack_id: int, pack_dict: Dict[str, Any]) -> None:
        t_ids = pack_dict.get("tIds", [])
        title = pack_dict.get("title", "")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM gq_not_found WHERE pack_id = ?", (pack_id,))
            conn.execute(
                """
                INSERT OR REPLACE INTO gq_packs (pack_id, title, t_ids, raw_json, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (pack_id, title, json.dumps(t_ids), json.dumps(pack_dict, ensure_ascii=False)),
            )
            conn.commit()

    def mark_not_found(self, pack_id: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO gq_not_found (pack_id, checked_at) VALUES (?, CURRENT_TIMESTAMP)",
                (pack_id,),
            )
            conn.commit()

    def get_max_pack_id(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(pack_id) FROM gq_packs")
            row = cursor.fetchone()
            if row and row[0] is not None:
                return int(row[0])
        return 0

    def get_saved_pack_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM gq_packs")
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def get_all_packs(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT raw_json FROM gq_packs ORDER BY pack_id ASC")
            packs: List[Dict[str, Any]] = []
            for row in cursor.fetchall():
                if row[0]:
                    try:
                        packs.append(json.loads(row[0]))
                    except json.JSONDecodeError:
                        continue
        return packs


class GotQuestionsIngester:
    """Polite, rate-limited, incremental crawler for gotquestions.online packages."""

    def __init__(self, config: Optional[GotQuestionsConfig] = None) -> None:
        self.config = config or GotQuestionsConfig()
        self.cache = GotQuestionsCache(self.config.cache_db_path)
        self.rate_limiter = AsyncRateLimiter(max_per_second=self.config.max_requests_per_second)

    async def fetch_pack(
        self,
        client: httpx.AsyncClient,
        pack_id: int,
        use_cache: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Fetch and parse a single pack with polite rate limiting and SQLite caching."""
        if use_cache:
            cached = self.cache.get_pack(pack_id)
            if cached is not None:
                return cached
            if self.cache.is_known_not_found(pack_id):
                return None

        url = f"{self.config.base_url}/pack/{pack_id}"
        headers = {
            "RSC": "1",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) ChGKRating/1.0",
            "Accept": "*/*",
        }

        retries = 0
        backoff = 1.0
        while retries <= self.config.max_retries:
            await self.rate_limiter.acquire()
            try:
                resp = await client.get(
                    url,
                    headers=headers,
                    follow_redirects=True,
                    timeout=self.config.timeout_seconds,
                )

                if resp.status_code == 200:
                    if "NEXT_HTTP_ERROR_FALLBACK;404" in resp.text:
                        self.cache.mark_not_found(pack_id)
                        return None

                    pack = extract_pack_from_rsc(resp.text)
                    if pack is not None and pack.get("id") == pack_id:
                        self.cache.save_pack(pack_id, pack)
                        return pack
                    else:
                        self.cache.mark_not_found(pack_id)
                        return None
                elif resp.status_code == 404:
                    self.cache.mark_not_found(pack_id)
                    return None
                elif resp.status_code in (429, 500, 502, 503, 504):
                    retries += 1
                    logger.warning(
                        "HTTP %d on pack %d, waiting %.1fs backoff (retry %d/%d)",
                        resp.status_code,
                        pack_id,
                        backoff,
                        retries,
                        self.config.max_retries,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(60.0, backoff * 2.0)
                else:
                    return None
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                retries += 1
                if retries > self.config.max_retries:
                    logger.error("Network failure fetching pack %d: %s", pack_id, exc)
                    return None
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2.0)
        return None

    async def ingest_incremental(
        self,
        lookback_packs: int = 50,
        max_packs: Optional[int] = None,
        target_max_pack: Optional[int] = None,
        concurrency: int = 2,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> int:
        """Run incremental sync: rescan recent packs and discover all new packs until 404 stop."""
        max_known = self.cache.get_max_pack_id()
        start_id = 1
        if max_known > 0:
            start_id = max(1, max_known - lookback_packs)

        logger.info(
            "Starting incremental gotquestions sync from pack %d (highest saved: %d)",
            start_id,
            max_known,
        )

        consecutive_404s = 0
        current_id = start_id
        fetched_new = 0
        sem = asyncio.Semaphore(concurrency)

        async with httpx.AsyncClient() as client:
            while True:
                if max_packs is not None and fetched_new >= max_packs:
                    break
                if target_max_pack is not None and current_id > target_max_pack:
                    break
                if current_id > max_known and consecutive_404s >= self.config.consecutive_404_stop:
                    logger.info(
                        "Reached %d consecutive 404s at pack %d. Halting crawl.",
                        consecutive_404s,
                        current_id,
                    )
                    break

                chunk_size = min(10, concurrency * 5)
                if max_packs is not None:
                    chunk_size = min(chunk_size, max_packs - fetched_new)
                if target_max_pack is not None:
                    chunk_size = min(chunk_size, target_max_pack - current_id + 1)

                pids = list(range(current_id, current_id + chunk_size))
                if not pids:
                    break

                async def fetch_one(pid: int) -> Tuple[int, Optional[Dict[str, Any]]]:
                    async with sem:
                        is_rescan = pid <= max_known
                        p = await self.fetch_pack(client, pid, use_cache=is_rescan)
                        return pid, p

                tasks = [fetch_one(pid) for pid in pids]
                results = await asyncio.gather(*tasks)

                stopped_early = False
                for pid, pack in results:
                    if pack is not None:
                        consecutive_404s = 0
                        fetched_new += 1
                        t_ids = pack.get("tIds", [])
                        t_count = len(pack.get("tours", []))
                        if progress_cb and (fetched_new % 10 == 0 or fetched_new == 1):
                            progress_cb(
                                f"Pack #{pid}: '{pack.get('title', '')[:30]}' ({t_count} tours, tIds: {t_ids})",
                                pid,
                                max_known + fetched_new,
                            )
                    else:
                        if pid > max_known:
                            consecutive_404s += 1
                            if consecutive_404s >= self.config.consecutive_404_stop:
                                stopped_early = True
                                current_id = pid + 1
                                break

                # Periodically checkpoint export every 200 packs
                if fetched_new > 0 and fetched_new % 200 == 0:
                    self.export_to_parquet()

                if stopped_early:
                    logger.info(
                        "Reached %d consecutive 404s at pack %d. Halting crawl.",
                        consecutive_404s,
                        current_id,
                    )
                    break

                current_id = pids[-1] + 1

        self.export_to_parquet()
        logger.info("Incremental sync finished. Saved packs in cache: %d", self.cache.get_saved_pack_count())
        return fetched_new

    def export_to_parquet(
        self,
        questions_path: Optional[Path] = None,
        packs_path: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compile cached packs into Parquet tables with question-level and pack-level schemas."""
        q_path = questions_path or self.config.questions_parquet
        p_path = packs_path or self.config.packs_parquet

        packs = self.cache.get_all_packs()
        logger.info("Compiling %d cached gotquestions packs into Parquet...", len(packs))

        pack_records: List[Dict[str, Any]] = []
        question_records: List[Dict[str, Any]] = []

        for p in packs:
            pack_id = int(p.get("id", 0))
            title = str(p.get("title", ""))
            raw_t_ids = p.get("tIds", [])
            t_ids = [int(x) for x in raw_t_ids if str(x).isdigit()]
            pub_date = str(p.get("pubDate", p.get("startDate", "")))[:10]

            pack_records.append({
                "pack_id": pack_id,
                "title": title,
                "rating_tournament_ids": t_ids,
                "date": pub_date,
                "tours_count": len(p.get("tours", [])),
            })

            global_q_idx = 0
            for tour_idx, tour in enumerate(p.get("tours", []), 1):
                tour_id = int(tour.get("id", 0))
                tour_editors = tour.get("editors", [])
                editor_names = []
                editor_player_ids = []
                for e in tour_editors:
                    person = e.get("person", {})
                    if person:
                        name = person.get("name", "")
                        p_id = person.get("playerId")
                        if name:
                            editor_names.append(name)
                        if p_id:
                            editor_player_ids.append(int(p_id))

                for q in tour.get("questions", []):
                    global_q_idx += 1
                    q_num = int(q.get("number", global_q_idx))
                    q_text = str(q.get("text", "")).strip()
                    q_answer = str(q.get("answer", "")).strip()
                    q_comment = str(q.get("comment", "")).strip()

                    author_names = []
                    author_player_ids = []
                    for a in q.get("authors", []):
                        person = a.get("person", {})
                        if person:
                            name = person.get("name", "")
                            p_id = person.get("playerId")
                            if name:
                                author_names.append(name)
                            if p_id:
                                author_player_ids.append(int(p_id))

                    # If this pack is mapped to rating tournaments, record rows for each tournament
                    if t_ids:
                        for tid in t_ids:
                            question_records.append({
                                "question_id": f"{tid}_{global_q_idx}",
                                "rating_tournament_id": tid,
                                "pack_id": pack_id,
                                "tour_id": tour_id,
                                "tour_number": tour_idx,
                                "question_number": global_q_idx,
                                "text": q_text,
                                "answer": q_answer,
                                "comment": q_comment,
                                "author_names": author_names,
                                "author_player_ids": author_player_ids,
                                "editor_names": editor_names,
                                "editor_player_ids": editor_player_ids,
                            })
                    else:
                        question_records.append({
                            "question_id": f"gq_{pack_id}_{global_q_idx}",
                            "rating_tournament_id": 0,
                            "pack_id": pack_id,
                            "tour_id": tour_id,
                            "tour_number": tour_idx,
                            "question_number": global_q_idx,
                            "text": q_text,
                            "answer": q_answer,
                            "comment": q_comment,
                            "author_names": author_names,
                            "author_player_ids": author_player_ids,
                            "editor_names": editor_names,
                            "editor_player_ids": editor_player_ids,
                        })

        packs_df = pd.DataFrame(pack_records)
        questions_df = pd.DataFrame(question_records)

        p_path.parent.mkdir(parents=True, exist_ok=True)
        q_path.parent.mkdir(parents=True, exist_ok=True)

        if not packs_df.empty:
            packs_df.to_parquet(p_path, index=False)
            logger.info("Saved %d pack records to %s", len(packs_df), p_path)

        if not questions_df.empty:
            questions_df.to_parquet(q_path, index=False)
            logger.info("Saved %d question records to %s", len(questions_df), q_path)

        return questions_df, packs_df

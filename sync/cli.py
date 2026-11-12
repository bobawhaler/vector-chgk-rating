"""Local commands for the sync service.

Nightly incremental sync runs on App Engine. These commands upload a baseline,
run one pass on this machine, or rebuild the parquet files the rating and
questions projects read.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import click
import httpx
import pandas as pd

import questions_sync
import rating_sync
from tour_editors import credit_record

ROOT = Path(__file__).resolve().parent.parent
RATING_DATA = ROOT / "data"
QUESTIONS_DATA = ROOT / "questions" / "data"
DEFAULT_RATING = RATING_DATA / "dataset.parquet"
DEFAULT_CREDITS = RATING_DATA / "rating_credits.parquet"
DEFAULT_QUESTIONS = QUESTIONS_DATA / "gotquestions_questions.parquet"


@click.group()
def cli() -> None:
    """Daily ChGK downloads: rating.chgk.info results and gotquestions packs."""


@cli.command("upload-questions")
@click.option("--path", default=str(DEFAULT_QUESTIONS), help="Local questions parquet to upload")
def upload_questions_cmd(path: str) -> None:
    """Upload the questions parquet to Cloud Storage as the nightly-sync baseline."""
    click.echo(questions_sync.upload_snapshot(Path(path)))


@cli.command("sync-questions")
def sync_questions_cmd() -> None:
    """Run one incremental gotquestions sync against Cloud Storage."""
    result = questions_sync.run_incremental()
    click.echo(result.message)
    if not result.ok:
        raise SystemExit(1)


@cli.command("materialize-questions")
@click.option("--path", default=str(DEFAULT_QUESTIONS), help="Where to write the merged parquet")
def materialize_questions_cmd(path: str) -> None:
    """Download the Cloud Storage snapshot plus nightly parts into one local parquet."""
    click.echo(questions_sync.materialize(Path(path)))


@cli.command("upload-rating")
@click.option("--path", default=str(DEFAULT_RATING), help="Local rating parquet to upload")
def upload_rating_cmd(path: str) -> None:
    """Upload tournament results to Cloud Storage as the nightly-sync baseline."""
    click.echo(rating_sync.upload_snapshot(Path(path)))


@cli.command("sync-rating")
def sync_rating_cmd() -> None:
    """Run one incremental rating.chgk.info sync against Cloud Storage."""
    result = rating_sync.run_incremental()
    click.echo(result.message)
    if not result.ok:
        raise SystemExit(1)


@cli.command("materialize-rating")
@click.option("--path", default=str(DEFAULT_RATING), help="Where to write the merged rating parquet")
def materialize_rating_cmd(path: str) -> None:
    """Download rating snapshot plus nightly parts into one local parquet."""
    click.echo(rating_sync.materialize(Path(path)))


@cli.command("materialize-data")
def materialize_data_cmd() -> None:
    """Refresh local question and rating parquets from Cloud Storage."""
    click.echo(questions_sync.materialize(DEFAULT_QUESTIONS))
    click.echo(rating_sync.materialize(DEFAULT_RATING))
    click.echo(rating_sync.materialize_credits(DEFAULT_CREDITS))


@cli.command("backfill-credits")
@click.option("--path", default=str(DEFAULT_RATING), help="Local rating parquet whose tournaments to reload")
@click.option("--dest", default=str(DEFAULT_CREDITS), help="Where to write the local tour-editor table")
@click.option("--upload/--no-upload", default=True, help="Upload the table to Cloud Storage")
def backfill_credits_cmd(path: str, dest: str, upload: bool) -> None:
    """Fetch tour editors for every tournament already in the local rating table."""
    ids = sorted(int(item) for item in pd.read_parquet(path, columns=["tournament_id"])["tournament_id"].unique())
    records: list[dict] = []
    have: set[int] = set()
    dest_path = Path(dest)
    if dest_path.exists():
        previous = pd.read_parquet(dest_path)
        for row in previous.itertuples(index=False):
            blocks = row.blocks
            if isinstance(blocks, str):
                blocks = json.loads(blocks)
            package = row.package_editor_ids
            if hasattr(package, "tolist"):
                package = package.tolist()
            tournament_id = int(row.tournament_id)
            have.add(tournament_id)
            records.append({
                "tournament_id": tournament_id,
                "package_editor_ids": [int(item) for item in (package or [])],
                "blocks": list(blocks or []),
            })
    pending = [tournament_id for tournament_id in ids if tournament_id not in have]
    click.echo(f"Fetching tour editors for {len(pending)} tournaments, {len(have)} already saved")
    failed: list[int] = []
    done = {"n": 0}

    async def fetch_all() -> None:
        limits = httpx.Limits(max_connections=2)
        timeout = httpx.Timeout(30.0)
        headers = {"Accept": "application/json", "User-Agent": "ChGKRating/1.0"}
        semaphore = asyncio.Semaphore(2)

        async def one(client: httpx.AsyncClient, tournament_id: int) -> None:
            async with semaphore:
                url = f"https://api.rating.chgk.info/tournaments/{tournament_id}"
                card = None
                for attempt in range(6):
                    try:
                        response = await client.get(url)
                    except Exception:
                        await asyncio.sleep(min(30, 2 ** attempt))
                        continue
                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        wait = float(retry_after) if retry_after else min(30, 2 ** attempt)
                        await asyncio.sleep(wait)
                        continue
                    if response.status_code >= 500:
                        await asyncio.sleep(min(30, 2 ** attempt))
                        continue
                    if response.status_code == 404:
                        return
                    response.raise_for_status()
                    card = response.json()
                    break
                await asyncio.sleep(0.2)
                if isinstance(card, list):
                    card = card[0] if card else None
                if not isinstance(card, dict):
                    failed.append(tournament_id)
                    return
                record = credit_record(card)
                if record:
                    records.append(record)
                done["n"] += 1
                if done["n"] % 200 == 0 or done["n"] == len(pending):
                    click.echo(f"{done['n']}/{len(pending)}")

        async with httpx.AsyncClient(limits=limits, timeout=timeout, headers=headers) as client:
            await asyncio.gather(*(one(client, tournament_id) for tournament_id in pending))

    if pending:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        asyncio.run(fetch_all())
    records.sort(key=lambda row: int(row["tournament_id"]))
    click.echo(rating_sync.write_credits_table(records, Path(dest)))
    if failed:
        click.echo(f"Не удалось прочитать {len(failed)} турниров", err=True)
    if upload:
        click.echo(rating_sync.upload_credits(records))


@cli.command("status")
def status_cmd() -> None:
    """Show the Cloud Storage cursors for questions and rating."""
    click.echo(questions_sync.status_text())
    click.echo("")
    click.echo(rating_sync.status_text())


if __name__ == "__main__":
    cli()

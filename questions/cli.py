"""Command line for the question corpus."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
import pandas as pd

from corpus.analyze import link_coverage, people, search, summarize
from corpus.config import (
    DEFAULT_GQ_PACKS_PARQUET,
    DEFAULT_GQ_QUESTIONS_PARQUET,
    RATING_PARQUET,
    GotQuestionsConfig,
)
from corpus.gotquestions import GotQuestionsIngester


def _questions(path: str) -> pd.DataFrame:
    parquet = Path(path)
    if not parquet.exists():
        raise click.ClickException(f"Нет файла {parquet}. Соберите его: sync materialize-questions или fetch.")
    return pd.read_parquet(parquet)


@click.group()
def cli() -> None:
    """Разбор базы вопросов gotquestions."""


@cli.command("fetch")
@click.option("--lookback-packs", default=50, type=int, help="Recent packs to re-check")
@click.option("--max-packs", default=None, type=int, help="Max packs to fetch")
@click.option("--target-max-pack", default=None, type=int, help="Upper bound pack id")
@click.option("--rate-limit", default=1.0, type=float, help="Requests per second")
@click.option("--concurrency", default=1, type=int, help="Concurrent workers")
def fetch_cmd(
    lookback_packs: int,
    max_packs: int | None,
    target_max_pack: int | None,
    rate_limit: float,
    concurrency: int,
) -> None:
    """Polite local crawl. The nightly delta is the sync project, not this command."""
    config = GotQuestionsConfig(max_requests_per_second=rate_limit)
    ingester = GotQuestionsIngester(config)

    def progress(msg: str, current: int, total: int) -> None:
        click.echo(f"[{current}] {msg}")

    fetched = asyncio.run(
        ingester.ingest_incremental(
            lookback_packs=lookback_packs,
            max_packs=max_packs,
            target_max_pack=target_max_pack,
            concurrency=concurrency,
            progress_cb=progress,
        )
    )
    questions, packs = ingester.export_to_parquet()
    click.echo(f"Fetched {fetched} packs. {len(questions)} questions, {len(packs)} packages.")


@cli.command("summary")
@click.option("--questions", "questions_path", default=str(DEFAULT_GQ_QUESTIONS_PARQUET))
@click.option("--packs", "packs_path", default=str(DEFAULT_GQ_PACKS_PARQUET))
def summary_cmd(questions_path: str, packs_path: str) -> None:
    """How many questions, packages, authors, and editors are in the local parquet."""
    packs = pd.read_parquet(packs_path) if Path(packs_path).exists() else None
    click.echo(summarize(_questions(questions_path), packs))


@cli.command("link")
@click.option("--questions", "questions_path", default=str(DEFAULT_GQ_QUESTIONS_PARQUET))
@click.option("--rating", "rating_path", default=str(RATING_PARQUET))
def link_cmd(questions_path: str, rating_path: str) -> None:
    """Share of rating question ids that have a text in this corpus."""
    rating = Path(rating_path)
    if not rating.exists():
        raise click.ClickException(f"Нет рейтинга {rating}. Его собирает проект sync.")
    click.echo(link_coverage(pd.read_parquet(rating, columns=["question_id"]), _questions(questions_path)))


@cli.command("authors")
@click.option("--questions", "questions_path", default=str(DEFAULT_GQ_QUESTIONS_PARQUET))
@click.option("--limit", default=20, type=int)
def authors_cmd(questions_path: str, limit: int) -> None:
    """Authors named on the most question rows."""
    frame = people(_questions(questions_path), "author_names", limit)
    click.echo(frame.to_string(index=False))


@cli.command("editors")
@click.option("--questions", "questions_path", default=str(DEFAULT_GQ_QUESTIONS_PARQUET))
@click.option("--limit", default=20, type=int)
def editors_cmd(questions_path: str, limit: int) -> None:
    """Tour editors named on the most question rows."""
    frame = people(_questions(questions_path), "editor_names", limit)
    click.echo(frame.to_string(index=False))


@cli.command("search")
@click.argument("text")
@click.option("--questions", "questions_path", default=str(DEFAULT_GQ_QUESTIONS_PARQUET))
@click.option("--limit", default=20, type=int)
def search_cmd(text: str, questions_path: str, limit: int) -> None:
    """Find questions whose text contains TEXT."""
    frame = search(_questions(questions_path), text, limit)
    if frame.empty:
        click.echo("Ничего не найдено.")
        return
    for row in frame.itertuples(index=False):
        body = getattr(row, "text", "")
        click.echo(f"{row.question_id}\t{body[:180]}")


if __name__ == "__main__":
    cli()

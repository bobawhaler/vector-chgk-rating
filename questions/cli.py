"""Command line for the question corpus."""

from __future__ import annotations

import asyncio
from pathlib import Path

import json

import click
import pandas as pd

from corpus.analyze import link_coverage, people, search, summarize
from corpus.knowledge import KnowledgeError, about, format_card
from corpus.browse import serve
from corpus.catalog import corpus_meta, load_catalog
from corpus.config import (
    DEFAULT_GQ_PACKS_PARQUET,
    DEFAULT_GQ_QUESTIONS_PARQUET,
    RATING_PARQUET,
    GotQuestionsConfig,
)
from corpus.gotquestions import GotQuestionsIngester
from corpus.query import QueryError, QuestionFilter, run


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


@cli.command("about")
@click.argument("name")
@click.option("--force", is_flag=True, help="Rebuild the catalog first")
def about_cmd(name: str, force: bool) -> None:
    """What the corpus knows about one realia, and which editors use its neighbors."""
    try:
        card = about(_catalog(force), name)
    except KnowledgeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(format_card(card))


def _catalog(force: bool) -> pd.DataFrame:
    try:
        return load_catalog(progress=lambda message: click.echo(message, err=True), force=force)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc


def _joined(value: object) -> str:
    if not isinstance(value, list):
        return ""
    return ", ".join(str(item).strip() for item in value if str(item).strip())


def _card(row: dict[str, object]) -> str:
    title = str(row.get("pack_title") or "").strip() or f"Пакет {row.get('pack_id')}"
    start = str(row.get("start_date") or "")
    end = str(row.get("end_date") or "")
    when = start if not end or end == start else f"{start} — {end}"
    if not when:
        when = str(row.get("pub_date") or "")
    bits = [title]
    if when:
        bits.append(when)
    if row.get("tour_number"):
        bits.append(f"тур {row['tour_number']}")
    number = row.get("number_in_tour") or row.get("question_number")
    if number:
        bits.append(f"вопрос {number}")
    lines = [" · ".join(bits)]
    authors = _joined(row.get("author_names"))
    editors = _joined(row.get("editor_names"))
    pack_editors = _joined(row.get("pack_editor_names"))
    if authors:
        lines.append(f"Авторы: {authors}")
    if editors:
        lines.append(f"Редакторы: {editors}")
    elif pack_editors:
        lines.append(f"Редакторы пакета: {pack_editors}")
    lines.append("")
    lines.append(str(row.get("text") or "").strip())
    for label, key in (
        ("Ответ", "answer"),
        ("Зачёт", "zachet"),
        ("Незачёт", "nezachet"),
        ("Комментарий", "comment"),
        ("Источник", "source"),
    ):
        value = str(row.get(key) or "").strip()
        if value:
            lines.extend(["", f"{label}: {value}"])
    return "\n".join(lines).strip()


@cli.command("catalog")
@click.option("--force", is_flag=True, help="Rebuild even if the files have not changed")
def catalog_cmd(force: bool) -> None:
    """Join the question parquet with sources and dates from the local cache."""
    frame = _catalog(force)
    meta = corpus_meta(frame)
    click.echo(
        f"Вопросов: {meta['questions']}. Пакетов: {meta['packs']}. "
        f"С источником: {meta['with_source']}."
    )


@cli.command("query")
@click.option("--text", default="", help="Substring to find")
@click.option("--where", default="text", type=click.Choice(["text", "answer", "comment", "source", "all"]))
@click.option("--author", default="", help="Substring of an author name")
@click.option("--author-id", default=None, type=int, help="Rating player id of an author")
@click.option("--editor", default="", help="Substring of a tour or package editor name")
@click.option("--editor-id", default=None, type=int, help="Rating player id of an editor")
@click.option("--pack", default="", help="Substring of the package title")
@click.option("--pack-id", default=None, type=int)
@click.option("--tournament", "tournament_id", default=None, type=int, help="rating.chgk.info tournament id")
@click.option("--tour", default=None, type=int, help="Tour number inside the package")
@click.option("--number", default=None, type=int, help="Question number inside the tour")
@click.option("--since", default="", help="First play day, YYYY-MM-DD, inclusive")
@click.option("--until", default="", help="Last play day, YYYY-MM-DD, inclusive")
@click.option("--limit", default=20, type=click.IntRange(1, 200))
@click.option("--offset", default=0, type=int)
@click.option("--duplicates", is_flag=True, help="Keep a copy for every rating tournament")
@click.option("--json", "as_json", is_flag=True, help="Print the page as JSON")
@click.option("--force", is_flag=True, help="Rebuild the catalog first")
def query_cmd(
    text: str,
    where: str,
    author: str,
    author_id: int | None,
    editor: str,
    editor_id: int | None,
    pack: str,
    pack_id: int | None,
    tournament_id: int | None,
    tour: int | None,
    number: int | None,
    since: str,
    until: str,
    limit: int,
    offset: int,
    duplicates: bool,
    as_json: bool,
    force: bool,
) -> None:
    """Read question texts with answers, comments and sources."""
    try:
        spec = QuestionFilter(
            text=text,
            where=where,
            author=author,
            author_id=author_id,
            editor=editor,
            editor_id=editor_id,
            pack=pack,
            pack_id=pack_id,
            tournament_id=tournament_id,
            tour=tour,
            number=number,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
            unique=not duplicates,
        )
    except QueryError as exc:
        raise click.ClickException(str(exc)) from exc
    page = run(_catalog(force), spec)
    if as_json:
        click.echo(json.dumps({
            "total": page.total,
            "offset": page.offset,
            "limit": page.limit,
            "rows": page.rows,
        }, ensure_ascii=False))
        return
    if not page.rows:
        click.echo("Ничего не найдено.")
        return
    click.echo("\n\n".join(_card(row) for row in page.rows))
    click.echo(f"\nПоказано {len(page.rows)} из {page.total}.", err=True)


@cli.command("browse")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, type=int, show_default=True)
@click.option("--force", is_flag=True, help="Rebuild the catalog first")
def browse_cmd(host: str, port: int, force: bool) -> None:
    """Serve a local page for the same filters as query."""
    frame = _catalog(force)
    serve(frame, corpus_meta(frame), host, port)


if __name__ == "__main__":
    cli()

"""Commands for the chgk-rating control bot.

Nightly downloads and the Telegram webhook. Rating math lives in the
rating project. Question-text analysis lives in the questions project.
"""

from __future__ import annotations

from gae_platform.commands import CommandContext, command


@command("service")
def service_cmd(ctx: CommandContext) -> str:
    return (
        "Синхронизация ЧГК: App Engine F1 и вебхук бота.\n"
        "Вопросы и результаты турнирного сайта лежат в Cloud Storage.\n"
        "Ночью: вопросы в 03:30, рейтинг в 04:00 (Europe/Berlin).\n"
        "Локально: python cli.py materialize-data — parquet для rating и questions."
    )


@command("questions")
def questions_cmd(ctx: CommandContext) -> str:
    import questions_sync

    return questions_sync.status_text()


@command("rating")
def rating_cmd(ctx: CommandContext) -> str:
    import rating_sync

    return rating_sync.status_text()


@command("data")
def data_cmd(ctx: CommandContext) -> str:
    import questions_sync
    import rating_sync

    return questions_sync.status_text() + "\n\n" + rating_sync.status_text()


def on_cron() -> str | None:
    return None

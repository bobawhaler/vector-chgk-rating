"""Commands for the chgk-rating control bot.

The App Engine service is the free-tier management shell. Model training,
the parquet dataset, and the FastAPI rating API stay on a machine that can
hold PyTorch.
"""

from __future__ import annotations

from gae_platform.commands import CommandContext, command


@command("service")
def service_cmd(ctx: CommandContext) -> str:
    return (
        "chgk-rating: управляющий бот на App Engine F1.\n"
        "Вопросы и результаты турнирного сайта лежат в Cloud Storage.\n"
        "Ночью: вопросы в 03:30, рейтинг в 04:00 (Europe/Berlin).\n"
        "Локально после materialize-data: обучение и python cli.py run-api"
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

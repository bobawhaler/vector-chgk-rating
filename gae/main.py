"""Google App Engine entrypoint for the management Telegram bot."""

from __future__ import annotations

import logging
import os

from flask import Flask, request

import questions_sync
import rating_sync
from gae_platform import config, handlers, tasks, telegram_api

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

handlers.load_project_commands()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1_000_000


@app.get("/")
@app.get("/health")
def health():
    body, status = handlers.health()
    return body, status


@app.get("/botinfo")
def botinfo():
    body, status = handlers.botinfo()
    return body, status


@app.get("/setwebhook")
def setwebhook():
    body, status = handlers.set_webhook()
    return body, status


@app.post(config.webhook_path())
def command():
    body, status = handlers.webhook(request.get_data())
    return body, status


@app.post(config.process_task_path())
def process_task():
    body, status = handlers.process_task(request.get_data())
    return body, status


@app.get("/cron/tick")
def cron_tick():
    body, status = handlers.cron(request.headers)
    return body, status


def _cron_call(run, fail_message: str):
    if not handlers.cron_authorized(request.headers):
        return "forbidden", 403
    try:
        result = run()
    except Exception:
        logging.getLogger(__name__).exception("nightly sync failed")
        result = questions_sync.SyncResult(
            ok=False,
            notify=True,
            written=0,
            fetched=0,
            max_pack_id=0,
            message=fail_message,
        )
    # 200 even when the upstream site blocks us: a 500 would make App Engine retry immediately.
    if result.notify:
        for admin_id in sorted(config.admin_ids()):
            telegram_api.send_message(admin_id, result.message)
    return result.message, 200


@app.get("/cron/questions")
def cron_questions():
    return _cron_call(questions_sync.run_incremental, "Синхронизация вопросов упала. Подробности в логах.")


def _notify(message: str) -> None:
    for admin_id in sorted(config.admin_ids()):
        telegram_api.send_message(admin_id, message)


def _rating_batch(chain: int):
    try:
        result = rating_sync.run_incremental()
    except Exception:
        logging.getLogger(__name__).exception("nightly rating sync failed")
        message = "Синхронизация рейтинга упала. Подробности в логах."
        _notify(message)
        return message, 200
    if result.partial and chain + 1 < rating_sync.max_batches() and tasks.enqueue_rating_continue(chain + 1):
        return result.message, 200
    if result.partial or result.notify:
        _notify(result.message)
    return result.message, 200


@app.get("/cron/rating")
def cron_rating():
    if not handlers.cron_authorized(request.headers):
        return "forbidden", 403
    return _rating_batch(0)


@app.post(config.rating_continue_path())
def rating_continue():
    body = request.get_json(silent=True) or {}
    try:
        chain = int(body.get("chain") or 0)
    except (TypeError, ValueError):
        chain = 0
    return _rating_batch(chain)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="127.0.0.1", port=port, debug=True)

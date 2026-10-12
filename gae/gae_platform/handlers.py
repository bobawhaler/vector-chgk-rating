"""HTTP handlers shared by the Flask entrypoint. They do not import Flask."""

from __future__ import annotations

import json
import logging
import sys
from typing import Mapping

from gae_platform import config, tasks, telegram_api
from gae_platform.commands import COMMANDS, CommandContext, parse_command
from gae_platform.store import get_store

logger = logging.getLogger(__name__)

_project_commands_loaded = False


def load_project_commands() -> None:
    global _project_commands_loaded
    if _project_commands_loaded:
        return
    try:
        import project_commands  # noqa: F401
    except ImportError:
        logger.info("project_commands.py is not present")
    _project_commands_loaded = True


def health() -> tuple[str, int]:
    return "ok", 200


def botinfo() -> tuple[str, int]:
    try:
        info = telegram_api.get_me()
    except Exception:
        logger.exception("botinfo failed")
        return "failed", 500
    username = info.get("username") or "?"
    first_name = info.get("first_name") or ""
    return f"Bot: @{username} ({first_name})\nproject: {config.project_id() or '-'}", 200


def set_webhook() -> tuple[str, int]:
    try:
        telegram_api.set_webhook()
    except Exception:
        logger.exception("set_webhook failed")
        return "failed", 500
    return "webhook set", 200


def webhook(raw: bytes) -> tuple[str, int]:
    if not raw:
        return "", 200
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return "bad json", 400
    if not isinstance(body, dict):
        return "bad json", 400
    try:
        receive_update(body)
    except Exception:
        logger.exception("webhook failed")
    return "", 200


def process_task(raw: bytes) -> tuple[str, int]:
    if not raw:
        return "", 200
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return "bad json", 400
    if not isinstance(body, dict):
        return "bad json", 400
    try:
        process_update(body)
    except Exception:
        logger.exception("task processing failed")
    return "", 200


def cron_authorized(headers: Mapping[str, str]) -> bool:
    return _header(headers, "X-Appengine-Cron").lower() == "true"


def cron(headers: Mapping[str, str]) -> tuple[str, int]:
    if not cron_authorized(headers):
        return "forbidden", 403
    get_store().mark_cron()
    try:
        get_store().cleanup_updates()
    except Exception:
        logger.exception("update cleanup failed")
    try:
        note = _project_cron()
    except Exception:
        logger.exception("project cron hook failed")
        note = None
    if note:
        for admin_id in sorted(config.admin_ids()):
            telegram_api.send_message(admin_id, note)
    return "ok", 200


def receive_update(body: dict) -> None:
    if tasks.enqueue_update(body):
        return
    process_update(body)


def process_update(body: dict) -> None:
    update_id = body.get("update_id")
    if update_id is not None and not get_store().claim_update(int(update_id)):
        logger.info("Skipping duplicate update %s", update_id)
        return
    message = body.get("message") or body.get("edited_message")
    if not isinstance(message, dict):
        return
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return
    user_id = (message.get("from") or {}).get("id")
    parsed = parse_command(message.get("text") or "")
    if parsed is None:
        return
    name, args = parsed
    if not config.admin_ids():
        if name == "start":
            telegram_api.send_message(
                chat_id,
                f"Бот не настроен. Ваш chat id: {chat_id}. "
                "Укажите его в ADMIN_CHAT_ID и задеплойте снова.",
            )
        return
    if not config.is_admin(user_id, chat_id):
        return
    handler = COMMANDS.get(name)
    if handler is None:
        telegram_api.send_message(chat_id, "Неизвестная команда. /help")
        return
    ctx = CommandContext(
        chat_id=int(chat_id),
        user_id=int(user_id) if user_id is not None else None,
        text=message.get("text") or "",
        args=args,
    )
    try:
        reply = handler(ctx)
    except Exception:
        logger.exception("command %s failed", name)
        telegram_api.send_message(chat_id, "Команда завершилась с ошибкой. Подробности в логах.")
        return
    get_store().remember_command(name, int(chat_id))
    if reply:
        telegram_api.send_message(chat_id, reply)


def _project_cron() -> str | None:
    module = sys.modules.get("project_commands")
    if module is None:
        return None
    hook = getattr(module, "on_cron", None)
    if hook is None:
        return None
    note = hook()
    if note is None:
        return None
    return str(note)


def _header(headers: Mapping[str, str], name: str) -> str:
    target = name.lower()
    for key, value in headers.items():
        if str(key).lower() == target:
            return str(value)
    return ""

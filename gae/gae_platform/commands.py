"""Text-command registry for the management bot."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

COMMANDS: dict[str, Callable[["CommandContext"], str]] = {}


@dataclass
class CommandContext:
    chat_id: int
    user_id: int | None
    text: str
    args: str


def command(name: str) -> Callable:
    normalized = name.lstrip("/").lower()

    def decorate(fn: Callable[[CommandContext], str]) -> Callable[[CommandContext], str]:
        COMMANDS[normalized] = fn
        return fn

    return decorate


def parse_command(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    head, _, args = stripped.partition(" ")
    name = head[1:]
    if "@" in name:
        name = name.split("@", 1)[0]
    name = name.lower()
    if not name or not re.fullmatch(r"[a-z0-9_]+", name):
        return None
    return name, args.strip()


@command("start")
def start_cmd(ctx: CommandContext) -> str:
    from gae_platform.store import get_store

    get_store().remember_chat(ctx.chat_id)
    return "Бот на связи. /help — список команд."


@command("help")
def help_cmd(ctx: CommandContext) -> str:
    names = " ".join(f"/{name}" for name in sorted(COMMANDS))
    return f"Команды: {names}"


@command("ping")
def ping_cmd(ctx: CommandContext) -> str:
    return "pong"


@command("status")
def status_cmd(ctx: CommandContext) -> str:
    from gae_platform import config
    from gae_platform.store import get_store

    state = get_store().get_state()
    tasks = "on" if config.use_cloud_tasks() else "off"
    last_cron = state.get("last_cron") or "ещё не было"
    last_command = state.get("last_command") or "-"
    return "\n".join(
        [
            f"project: {config.project_id() or '-'}",
            f"region: {config.app_region()}",
            f"tasks: {config.cloud_tasks_queue()} ({tasks})",
            f"last cron: {last_cron}",
            f"last command: {last_command}",
        ]
    )

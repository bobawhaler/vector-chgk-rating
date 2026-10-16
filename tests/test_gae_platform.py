"""Management-bot handlers. No Google credentials and no Telegram network."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["ENV_YAML"] = "/nonexistent-gae-env.yaml"
os.environ["PROJECT_ID"] = "demo-project"
os.environ["TELEGRAM_API_TOKEN"] = "123:TEST"
os.environ["OBFUSCATION_TOKEN"] = "testtoken"
os.environ["ADMIN_CHAT_ID"] = "42"
os.environ["USE_CLOUD_TASKS"] = "false"
os.environ["USE_DATASTORE"] = "0"
os.environ["APP_REGION"] = "us-central1"
os.environ.pop("GAE_ENV", None)

_ROOT = Path(__file__).resolve().parents[1]
_SERVICE_ROOT = _ROOT
for _candidate in (_ROOT / "gae", _ROOT):
    if (_candidate / "gae_platform").is_dir():
        _SERVICE_ROOT = _candidate
        if str(_candidate) not in sys.path:
            sys.path.append(str(_candidate))
        break

from gae_platform import tasks, telegram_api
from gae_platform.commands import COMMANDS, CommandContext
from gae_platform.handlers import cron, load_project_commands, process_update, set_webhook, webhook
from gae_platform.store import reset_store

_SENT: list[tuple[object, str]] = []


def _capture(chat_id: object, text: str) -> bool:
    _SENT.append((chat_id, text))
    return True


def _update(update_id: int, chat_id: int, user_id: int, text: str) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "text": text,
            "chat": {"id": chat_id},
            "from": {"id": user_id},
        },
    }


def setup_function() -> None:
    reset_store()
    _SENT.clear()
    load_project_commands()
    telegram_api.send_message = _capture  # type: ignore[method-assign]


def test_admin_ping() -> None:
    process_update(_update(1, 42, 42, "/ping"))
    assert _SENT == [(42, "pong")]


def test_non_admin_is_silent() -> None:
    process_update(_update(2, 7, 7, "/ping"))
    assert _SENT == []


def test_duplicate_update_is_skipped() -> None:
    payload = _update(3, 42, 42, "/ping")
    process_update(payload)
    process_update(payload)
    assert _SENT == [(42, "pong")]


def test_unknown_command() -> None:
    process_update(_update(4, 42, 42, "/nope"))
    assert _SENT == [(42, "Неизвестная команда. /help")]


def test_webhook_bad_json() -> None:
    body, status = webhook(b"not-json")
    assert status == 400
    assert body == "bad json"


def test_webhook_defers_when_task_is_queued(monkeypatch) -> None:
    monkeypatch.setattr(tasks, "enqueue_update", lambda body: True)
    raw = json.dumps(_update(5, 42, 42, "/ping")).encode()
    body, status = webhook(raw)
    assert status == 200
    assert body == ""
    assert _SENT == []


def test_cron_requires_appengine_header() -> None:
    body, status = cron({})
    assert status == 403
    assert body == "forbidden"


def test_cron_records_heartbeat_and_status_sees_it() -> None:
    body, status = cron({"X-Appengine-Cron": "true"})
    assert status == 200
    assert body == "ok"
    reply = COMMANDS["status"](CommandContext(42, 42, "/status", ""))
    assert "demo-project" in reply
    assert "us-central1" in reply
    assert "ещё не было" not in reply


def test_placeholder_admin_is_unconfigured(monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_CHAT_ID", "<YOUR_ADMIN_CHAT_ID>")
    process_update(_update(8, 99, 99, "/start"))
    assert len(_SENT) == 1
    assert "99" in _SENT[0][1]


def test_unconfigured_start_returns_chat_id(monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_CHAT_ID", "")
    process_update(_update(6, 99, 99, "/start"))
    assert len(_SENT) == 1
    assert "99" in _SENT[0][1]
    _SENT.clear()
    process_update(_update(7, 99, 99, "/ping"))
    assert _SENT == []


def test_project_commands_reply_without_network() -> None:
    extras = set(COMMANDS) - {"start", "help", "ping", "status"}
    assert extras
    ctx = CommandContext(42, 42, "/cmd", "")
    for name in extras:
        reply = COMMANDS[name](ctx)
        assert isinstance(reply, str) and reply


def test_set_webhook_does_not_echo_the_url(monkeypatch) -> None:
    monkeypatch.setattr(telegram_api, "set_webhook", lambda: None)
    body, status = set_webhook()
    assert status == 200
    assert body == "webhook set"
    assert "testtoken" not in body


def test_on_cron_returns_nothing_or_text() -> None:
    import project_commands

    hook = getattr(project_commands, "on_cron", None)
    assert hook is not None
    result = hook()
    assert result is None or isinstance(result, str)


def test_flask_routes() -> None:
    import importlib.util

    import pytest

    pytest.importorskip("flask")
    spec = importlib.util.spec_from_file_location("gae_service_main", _SERVICE_ROOT / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    paths = {rule.rule for rule in module.app.url_map.iter_rules()}
    assert "/health" in paths
    assert "/setwebhook" in paths
    assert "/cron/tick" in paths
    assert "/cron/questions" in paths
    assert "/cron/rating" in paths
    assert "/rating-continuetesttoken" in paths
    assert "/commandtesttoken" in paths
    assert "/process_tasktesttoken" in paths

    client = module.app.test_client()
    assert client.get("/cron/tick").status_code == 403
    allowed = client.get("/cron/tick", headers={"X-Appengine-Cron": "true"})
    assert allowed.status_code == 200
    posted = client.post(
        "/commandtesttoken",
        data=json.dumps(_update(50, 42, 42, "/ping")),
        content_type="application/json",
    )
    assert posted.status_code == 200
    assert _SENT == [(42, "pong")]

"""Minimal Telegram Bot API client. The bot token is never logged."""

from __future__ import annotations

import logging

from gae_platform import config

logger = logging.getLogger(__name__)

_MAX_MESSAGE = 4000


def _api(method: str) -> str:
    token = config.telegram_token()
    if not token:
        raise RuntimeError("TELEGRAM_API_TOKEN is not set")
    return f"https://api.telegram.org/bot{token}/{method}"


def send_message(chat_id: int | str, text: str) -> bool:
    if not config.telegram_token():
        logger.error("TELEGRAM_API_TOKEN is not set")
        return False
    import requests

    response = requests.post(
        _api("sendMessage"),
        json={
            "chat_id": str(chat_id),
            "text": text[:_MAX_MESSAGE],
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    if not response.ok:
        logger.error("sendMessage failed: HTTP %s", response.status_code)
        return False
    return True


def get_me() -> dict:
    import requests

    response = requests.get(_api("getMe"), timeout=30)
    if not response.ok:
        raise RuntimeError(f"getMe HTTP {response.status_code}")
    return response.json().get("result") or {}


def get_webhook_info() -> dict:
    import requests

    response = requests.get(
        _api("getWebhookInfo"),
        headers={"Accept": "application/json"},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"getWebhookInfo HTTP {response.status_code}")
    return response.json()


def set_webhook() -> None:
    url = config.webhook_url()
    if not url:
        raise RuntimeError("PROJECT_ID and OBFUSCATION_TOKEN are required")
    current = get_webhook_info().get("result", {}).get("url")
    if current == url:
        logger.info("webhook already set for project %s", config.project_id())
        return
    import requests

    response = requests.post(
        _api("setWebhook"),
        json={"url": url, "drop_pending_updates": True},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"setWebhook HTTP {response.status_code}")
    logger.info("webhook set for project %s", config.project_id())

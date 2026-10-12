"""Environment for the App Engine management bot.

GAE injects variables from app.yaml (which includes env.yaml). Locally, keys
missing from the process environment are filled from env.yaml.
"""

from __future__ import annotations

import os
import re


def _env_yaml_path() -> str | None:
    explicit = os.environ.get("ENV_YAML")
    if explicit is not None:
        return explicit if explicit and os.path.isfile(explicit) else None
    service_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (
        os.path.join(service_root, "env.yaml"),
        os.path.join(os.getcwd(), "env.yaml"),
    ):
        if os.path.isfile(path):
            return path
    return None


def load_env_yaml() -> None:
    path = _env_yaml_path()
    if not path:
        return
    try:
        import yaml
    except ImportError:
        return
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    variables = data.get("env_variables") or {}
    if not isinstance(variables, dict):
        return
    for key, value in variables.items():
        if value is None:
            continue
        os.environ.setdefault(str(key), str(value))


load_env_yaml()


def _real(value: str) -> str:
    """Template placeholders are not configuration."""
    stripped = value.strip()
    if not stripped or stripped.startswith("<") or "YOUR_" in stripped:
        return ""
    return stripped


def _first(*values: str) -> str:
    for value in values:
        real = _real(value)
        if real:
            return real
    return ""


def project_id() -> str:
    return _first(
        os.environ.get("PROJECT_ID") or "",
        os.environ.get("GOOGLE_CLOUD_PROJECT") or "",
        os.environ.get("GCP_PROJECT") or "",
    )


def telegram_token() -> str:
    return _first(
        os.environ.get("TELEGRAM_API_TOKEN") or "",
        os.environ.get("TELEGRAM_BOT_TOKEN") or "",
    )


def obfuscation_token() -> str:
    return _first(
        os.environ.get("OBFUSCATION_TOKEN") or "",
        os.environ.get("WEBHOOK_SECRET") or "",
    )


def admin_ids() -> set[str]:
    raw = os.environ.get("ADMIN_CHAT_ID") or os.environ.get("CHAT_ID") or ""
    return {part for part in (_real(item) for item in re.split(r"[,\s]+", raw)) if part}


def is_admin(user_id: object, chat_id: object) -> bool:
    allowed = admin_ids()
    if not allowed:
        return False
    if user_id is not None and str(user_id) in allowed:
        return True
    if chat_id is not None and str(chat_id) in allowed:
        return True
    return False


def cloud_tasks_queue() -> str:
    return os.environ.get("CLOUD_TASKS_QUEUE") or "bot-tasks"


def app_region() -> str:
    return os.environ.get("APP_REGION") or os.environ.get("CLOUD_TASKS_LOCATION") or "us-central1"


def gcs_bucket() -> str:
    return _real(os.environ.get("GCS_BUCKET") or "")


def use_cloud_tasks() -> bool:
    return os.environ.get("USE_CLOUD_TASKS", "true").lower() in ("1", "true", "yes")


def use_datastore() -> bool:
    flag = os.environ.get("USE_DATASTORE", "").lower()
    if flag in ("0", "false", "no"):
        return False
    if flag in ("1", "true", "yes"):
        return True
    return os.environ.get("GAE_ENV", "") == "standard"


def webhook_path() -> str:
    return f"/command{obfuscation_token()}"


def process_task_path() -> str:
    return f"/process_task{obfuscation_token()}"


def webhook_url() -> str:
    project = project_id()
    if not project or not obfuscation_token():
        return ""
    return f"https://{project}.appspot.com{webhook_path()}"

"""Update dedup and bot heartbeat.

On App Engine Standard this is Cloud Datastore. Locally and in tests it is an
in-memory store so the process does not need Google credentials.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Protocol

from gae_platform import config

logger = logging.getLogger(__name__)

_EMPTY_STATE = {"last_cron": "", "last_command": "", "last_chat_id": ""}
_UPDATE_TTL = timedelta(days=7)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime | None = None) -> str:
    current = moment or _now()
    return current.replace(microsecond=0).isoformat().replace("+00:00", "Z")


class Store(Protocol):
    def claim_update(self, update_id: int) -> bool: ...

    def mark_cron(self) -> None: ...

    def remember_command(self, name: str, chat_id: int) -> None: ...

    def remember_chat(self, chat_id: int) -> None: ...

    def get_state(self) -> dict[str, str]: ...

    def cleanup_updates(self) -> None: ...


class MemoryStore:
    def __init__(self) -> None:
        self._updates: dict[str, datetime] = {}
        self._state = dict(_EMPTY_STATE)
        self._lock = threading.Lock()

    def claim_update(self, update_id: int) -> bool:
        with self._lock:
            key = str(update_id)
            if key in self._updates:
                return False
            self._updates[key] = _now()
            return True

    def mark_cron(self) -> None:
        with self._lock:
            self._state["last_cron"] = _stamp()

    def remember_command(self, name: str, chat_id: int) -> None:
        with self._lock:
            self._state["last_command"] = name
            self._state["last_chat_id"] = str(chat_id)

    def remember_chat(self, chat_id: int) -> None:
        with self._lock:
            self._state["last_chat_id"] = str(chat_id)

    def get_state(self) -> dict[str, str]:
        with self._lock:
            return dict(self._state)

    def cleanup_updates(self) -> None:
        cutoff = _now() - _UPDATE_TTL
        with self._lock:
            stale = [key for key, created in self._updates.items() if created < cutoff]
            for key in stale:
                del self._updates[key]


class DatastoreStore:
    def __init__(self) -> None:
        from google.cloud import datastore

        project = config.project_id() or None
        self._client = datastore.Client(project=project) if project else datastore.Client()

    def claim_update(self, update_id: int) -> bool:
        from google.cloud import datastore

        key = self._client.key("TelegramUpdate", str(update_id))
        with self._client.transaction():
            if self._client.get(key) is not None:
                return False
            entity = datastore.Entity(key=key)
            entity["created"] = _now()
            self._client.put(entity)
            return True

    def _update_state(self, **fields: str) -> None:
        from google.cloud import datastore

        key = self._client.key("BotState", "main")
        with self._client.transaction():
            entity = self._client.get(key)
            if entity is None:
                entity = datastore.Entity(key=key)
            for name, value in fields.items():
                entity[name] = value
            self._client.put(entity)

    def mark_cron(self) -> None:
        self._update_state(last_cron=_stamp())

    def remember_command(self, name: str, chat_id: int) -> None:
        self._update_state(last_command=name, last_chat_id=str(chat_id))

    def remember_chat(self, chat_id: int) -> None:
        self._update_state(last_chat_id=str(chat_id))

    def get_state(self) -> dict[str, str]:
        entity = self._client.get(self._client.key("BotState", "main"))
        if not entity:
            return dict(_EMPTY_STATE)
        return {
            "last_cron": str(entity.get("last_cron") or ""),
            "last_command": str(entity.get("last_command") or ""),
            "last_chat_id": str(entity.get("last_chat_id") or ""),
        }

    def cleanup_updates(self) -> None:
        from google.cloud.datastore.query import PropertyFilter

        cutoff = _now() - _UPDATE_TTL
        query = self._client.query(kind="TelegramUpdate")
        query.add_filter(filter=PropertyFilter("created", "<", cutoff))
        keys = [entity.key for entity in query.fetch(limit=100)]
        if keys:
            self._client.delete_multi(keys)


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        if config.use_datastore():
            logger.info("Using Cloud Datastore")
            _store = DatastoreStore()
        else:
            _store = MemoryStore()
    return _store


def reset_store() -> None:
    global _store
    _store = None

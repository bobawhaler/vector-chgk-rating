"""Interval limiter for the local gotquestions crawl."""

from __future__ import annotations

import asyncio


class AsyncRateLimiter:
    """Wait so that requests stay under ``max_per_second``."""

    def __init__(self, max_per_second: float = 10.0) -> None:
        self.interval = 1.0 / max(max_per_second, 1.0)
        self._lock: asyncio.Lock | None = None
        self.last_request_time: float = 0.0

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self) -> None:
        async with self.lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            elapsed = now - self.last_request_time
            if elapsed < self.interval:
                await asyncio.sleep(self.interval - elapsed)
            self.last_request_time = loop.time()

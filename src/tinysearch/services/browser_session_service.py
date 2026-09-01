"""In-process registry of short-lived browser sessions for the `browse` primitive.

Each session pairs one already-started `AsyncWebCrawler` (its own Chromium
process, via `site_crawl_service.create_browser_crawler`) with a fixed
crawl4ai `session_id`, so a caller can observe a rendered page and then act
on that *same* live page across several `browse()` calls without losing
state. Sessions expire on idle and are capped in count -- this is
deliberately not the persistent/authenticated-session work tracked as
#55/#59, just enough state to make click/type usable.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tinysearch.services.site_crawl_service import create_browser_crawler


class BrowserSessionError(Exception):
    """Base error for browser session lifecycle failures."""


class SessionExpiredError(BrowserSessionError):
    """The requested session_id is unknown or has expired."""


@dataclass
class BrowserSession:
    session_id: str
    crawler: Any
    last_used: float = field(default_factory=time.monotonic)
    current_url: str | None = None

    def touch(self) -> None:
        self.last_used = time.monotonic()


class BrowserSessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, BrowserSession] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        config: Mapping[str, Any],
        *,
        max_sessions: int,
        idle_seconds: float,
    ) -> BrowserSession:
        async with self._lock:
            await self._sweep_expired_locked(idle_seconds)
            if len(self._sessions) >= max_sessions:
                oldest = min(self._sessions.values(), key=lambda s: s.last_used)
                await self._close_locked(oldest)
            crawler = create_browser_crawler(config)
            await crawler.__aenter__()
            session = BrowserSession(session_id=uuid.uuid4().hex, crawler=crawler)
            self._sessions[session.session_id] = session
            return session

    async def get(self, session_id: str, *, idle_seconds: float) -> BrowserSession:
        async with self._lock:
            await self._sweep_expired_locked(idle_seconds)
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionExpiredError(
                    f"browser session {session_id!r} is unknown or has expired"
                )
            session.touch()
            return session

    async def close(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                await self._close_locked(session)

    async def _sweep_expired_locked(self, idle_seconds: float) -> None:
        now = time.monotonic()
        expired = [
            session
            for session in self._sessions.values()
            if now - session.last_used > idle_seconds
        ]
        for session in expired:
            await self._close_locked(session)

    async def _close_locked(self, session: BrowserSession) -> None:
        self._sessions.pop(session.session_id, None)
        try:
            await session.crawler.__aexit__(None, None, None)
        except Exception:
            pass


_registry = BrowserSessionRegistry()


def get_registry() -> BrowserSessionRegistry:
    return _registry

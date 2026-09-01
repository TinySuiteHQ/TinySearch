from __future__ import annotations

import unittest
from unittest.mock import patch

from tinysearch.services.browser_session_service import (
    BrowserSessionRegistry,
    SessionExpiredError,
)


class _FakeCrawler:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> "_FakeCrawler":
        self.entered = True
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.exited = True


def _patched_create_browser_crawler():
    return patch(
        "tinysearch.services.browser_session_service.create_browser_crawler",
        side_effect=lambda config: _FakeCrawler(),
    )


class BrowserSessionRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_returns_started_session(self) -> None:
        registry = BrowserSessionRegistry()
        with _patched_create_browser_crawler():
            session = await registry.create({}, max_sessions=4, idle_seconds=300)

        self.assertTrue(session.crawler.entered)
        self.assertTrue(session.session_id)

    async def test_get_returns_the_same_session_and_touches_it(self) -> None:
        registry = BrowserSessionRegistry()
        with _patched_create_browser_crawler():
            created = await registry.create({}, max_sessions=4, idle_seconds=300)
            fetched = await registry.get(created.session_id, idle_seconds=300)

        self.assertIs(fetched, created)

    async def test_get_unknown_session_raises(self) -> None:
        registry = BrowserSessionRegistry()
        with self.assertRaises(SessionExpiredError):
            await registry.get("does-not-exist", idle_seconds=300)

    async def test_idle_session_expires_and_is_closed(self) -> None:
        registry = BrowserSessionRegistry()
        with _patched_create_browser_crawler():
            session = await registry.create({}, max_sessions=4, idle_seconds=300)
            session.last_used -= 1000  # simulate elapsed idle time

            with self.assertRaises(SessionExpiredError):
                await registry.get(session.session_id, idle_seconds=300)

        self.assertTrue(session.crawler.exited)

    async def test_close_shuts_down_the_crawler(self) -> None:
        registry = BrowserSessionRegistry()
        with _patched_create_browser_crawler():
            session = await registry.create({}, max_sessions=4, idle_seconds=300)
            await registry.close(session.session_id)

        self.assertTrue(session.crawler.exited)
        with self.assertRaises(SessionExpiredError):
            await registry.get(session.session_id, idle_seconds=300)

    async def test_at_capacity_evicts_the_oldest_idle_session(self) -> None:
        registry = BrowserSessionRegistry()
        with _patched_create_browser_crawler():
            first = await registry.create({}, max_sessions=2, idle_seconds=300)
            first.last_used -= 10  # oldest
            await registry.create({}, max_sessions=2, idle_seconds=300)
            await registry.create({}, max_sessions=2, idle_seconds=300)

        self.assertTrue(first.crawler.exited)
        with self.assertRaises(SessionExpiredError):
            await registry.get(first.session_id, idle_seconds=300)


if __name__ == "__main__":
    unittest.main()

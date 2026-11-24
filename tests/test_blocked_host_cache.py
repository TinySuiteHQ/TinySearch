"""A host that refused automated access is not asked again for a short time.

MCP tool calls are stateless, so the memory is per host with a time limit,
not per conversation. It only ever *skips* a fetch: nothing here retries,
changes how a request looks, or otherwise tries to get past the refusal.
"""

from __future__ import annotations

import contextlib
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import tinysearch
from tinysearch.config import DEFAULT_CONFIG, normalize_config
from tinysearch.pipelines.scrape import run_scrape_pipeline
from tinysearch.services.blocked_host_cache import (
    SHARED_REFUSED_HOSTS,
    BlockedHostCache,
    HostRecentlyRefusedError,
)
from tinysearch.services.scrape_service import FetchUnavailableError


_LONG_TEXT = " ".join(f"Readable sentence number {n} about the product." for n in range(60))


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeSite:
    """Answers every page on a host with one status and records each fetch."""

    def __init__(self, status: int) -> None:
        self.status = status
        self.fetches: list[str] = []

    async def crawl(self, *, url, user_query, crawler=None):
        self.fetches.append(url)
        if self.status == 200:
            return {
                "final_url": url,
                "html": "<html><head><title>Product</title></head></html>",
                "markdown_raw": _LONG_TEXT,
                "metadata": {"title": "Product", "status": 200},
            }
        return {
            "final_url": url,
            "html": "<html><head><title>Just a moment...</title></head></html>",
            "markdown_raw": "Forbidden",
            "metadata": {"title": "Just a moment...", "status": self.status},
        }


def _fallback_recorder(page: dict | None = None):
    calls: list[str] = []

    async def _fallback(*, url, crawler):
        calls.append(url)
        if page is None:
            raise RuntimeError("fallback service down")
        return {**page, "final_url": url}

    _fallback.calls = calls
    return _fallback


def _safe(url, blocked_domains):
    return url


class BlockedHostCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.cache = BlockedHostCache(clock=self.clock)

    def test_refusal_is_remembered_for_every_page_on_the_host(self) -> None:
        self.cache.remember("https://shop.example/products/a", 403, ttl_seconds=600)
        refusal = self.cache.refusal_for("https://shop.example/collections/all")
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.status, 403)
        self.assertEqual(refusal.url, "https://shop.example/products/a")

    def test_host_match_ignores_case_port_and_leading_www(self) -> None:
        self.cache.remember("https://www.Shop.example/a", 429, ttl_seconds=600)
        self.assertIsNotNone(self.cache.refusal_for("https://shop.example:443/b"))

    def test_other_hosts_are_not_affected(self) -> None:
        self.cache.remember("https://shop.example/a", 403, ttl_seconds=600)
        self.assertIsNone(self.cache.refusal_for("https://other.example/a"))
        self.assertIsNone(self.cache.refusal_for("https://cdn.shop.example/a"))

    def test_ttl_expires_the_memory(self) -> None:
        self.cache.remember("https://shop.example/a", 403, ttl_seconds=600)
        self.clock.now += 599
        self.assertIsNotNone(self.cache.refusal_for("https://shop.example/b"))
        self.clock.now += 1
        self.assertIsNone(self.cache.refusal_for("https://shop.example/b"))

    def test_ttl_zero_remembers_nothing(self) -> None:
        self.cache.remember("https://shop.example/a", 403, ttl_seconds=0)
        self.assertIsNone(self.cache.refusal_for("https://shop.example/a"))

    def test_success_and_other_statuses_are_never_stored(self) -> None:
        for status in (200, 301, 404, 500, None):
            self.cache.remember("https://shop.example/a", status, ttl_seconds=600)
        self.assertIsNone(self.cache.refusal_for("https://shop.example/a"))

    def test_ttl_default_is_ten_minutes_and_configurable(self) -> None:
        self.assertEqual(DEFAULT_CONFIG["scrape_refused_host_ttl_seconds"], 600)
        self.assertEqual(
            normalize_config({"scrape_refused_host_ttl_seconds": "90"})[
                "scrape_refused_host_ttl_seconds"
            ],
            90.0,
        )
        with self.assertRaises(ValueError):
            normalize_config({"scrape_refused_host_ttl_seconds": -1})

    def test_refused_error_is_a_fetch_unavailable_error(self) -> None:
        self.assertTrue(issubclass(HostRecentlyRefusedError, FetchUnavailableError))


class PipelineRefusalMemoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.cache = BlockedHostCache(clock=self.clock)

    async def _scrape(self, url, site, *, fallback_fn=None, document_fn=None, **config):
        with patch("tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_safe):
            return await run_scrape_pipeline(
                url,
                "*",
                config={"blocked_domains": [], **config},
                crawl_fn=site.crawl,
                fallback_fn=fallback_fn,
                document_fn=document_fn,
                refused_hosts=self.cache,
            )

    async def test_second_page_on_refused_host_is_answered_without_a_fetch(self) -> None:
        site = FakeSite(403)
        with self.assertRaises(FetchUnavailableError) as first:
            await self._scrape("https://shop.example/products/a", site)
        self.assertNotIsInstance(first.exception, HostRecentlyRefusedError)
        self.assertEqual(len(site.fetches), 1)

        for url in ("https://shop.example/products/a", "https://shop.example/collections/x"):
            with self.assertRaises(HostRecentlyRefusedError) as again:
                await self._scrape(url, site)
            message = str(again.exception)
            self.assertIn("HTTP 403", message)
            self.assertIn("shop.example", message)
            self.assertIn("browser", message)
        self.assertEqual(len(site.fetches), 1)

    async def test_remembered_host_skips_the_fallback_fetcher_too(self) -> None:
        site = FakeSite(429)
        fallback = _fallback_recorder()
        with self.assertRaises(FetchUnavailableError):
            await self._scrape("https://shop.example/a", site, fallback_fn=fallback)
        self.assertEqual(fallback.calls, ["https://shop.example/a"])
        with self.assertRaises(HostRecentlyRefusedError):
            await self._scrape("https://shop.example/b", site, fallback_fn=fallback)
        self.assertEqual(fallback.calls, ["https://shop.example/a"])
        self.assertEqual(len(site.fetches), 1)

    async def test_remembered_host_skips_document_downloads(self) -> None:
        site = FakeSite(503)
        with self.assertRaises(FetchUnavailableError):
            await self._scrape("https://shop.example/a", site)
        downloads: list[str] = []

        def _document(url):
            downloads.append(url)
            return "text", "pdf"

        with self.assertRaises(HostRecentlyRefusedError):
            await self._scrape("https://shop.example/manual.pdf", site, document_fn=_document)
        self.assertEqual(downloads, [])

    async def test_ttl_from_config_expires_and_the_host_is_fetched_once_more(self) -> None:
        site = FakeSite(403)
        with self.assertRaises(FetchUnavailableError):
            await self._scrape("https://shop.example/a", site, scrape_refused_host_ttl_seconds=60)
        self.clock.now += 61
        with self.assertRaises(FetchUnavailableError) as after:
            await self._scrape("https://shop.example/b", site, scrape_refused_host_ttl_seconds=60)
        self.assertNotIsInstance(after.exception, HostRecentlyRefusedError)
        self.assertEqual(len(site.fetches), 2)

    async def test_ttl_zero_in_config_turns_the_memory_off(self) -> None:
        site = FakeSite(403)
        for url in ("https://shop.example/a", "https://shop.example/b"):
            with self.assertRaises(FetchUnavailableError) as raised:
                await self._scrape(url, site, scrape_refused_host_ttl_seconds=0)
            self.assertNotIsInstance(raised.exception, HostRecentlyRefusedError)
        self.assertEqual(len(site.fetches), 2)

    async def test_success_is_never_remembered(self) -> None:
        site = FakeSite(200)
        await self._scrape("https://shop.example/a", site)
        await self._scrape("https://shop.example/b", site)
        self.assertEqual(len(site.fetches), 2)
        self.assertIsNone(self.cache.refusal_for("https://shop.example/a"))

    async def test_success_through_the_fallback_is_not_remembered(self) -> None:
        site = FakeSite(403)
        readable = await FakeSite(200).crawl(url="https://shop.example/a", user_query=None)
        fallback = _fallback_recorder(readable)
        await self._scrape("https://shop.example/a", site, fallback_fn=fallback)
        self.assertIsNone(self.cache.refusal_for("https://shop.example/a"))

    async def test_without_a_cache_the_pipeline_remembers_nothing(self) -> None:
        site = FakeSite(403)
        with patch("tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_safe):
            for url in ("https://shop.example/a", "https://shop.example/b"):
                with self.assertRaises(FetchUnavailableError):
                    await run_scrape_pipeline(
                        url, "*", config={"blocked_domains": []}, crawl_fn=site.crawl
                    )
        self.assertEqual(len(site.fetches), 2)


class ScrapeUrlsToolRefusalMemoryTests(unittest.IsolatedAsyncioTestCase):
    """The public scrape_urls operation (behind the MCP tool) shares one memory."""

    def setUp(self) -> None:
        SHARED_REFUSED_HOSTS.clear()
        self.addCleanup(SHARED_REFUSED_HOSTS.clear)

    async def test_refusal_carries_across_separate_scrape_urls_calls(self) -> None:
        site = FakeSite(403)
        with patch(
            "tinysearch.pipelines.scrape.fetch_html_for_query", side_effect=site.crawl
        ), patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_safe
        ), patch("tinysearch.core._ensure_browser_bundle", new=AsyncMock()), patch(
            "tinysearch.core.create_browser_crawler",
            return_value=contextlib.nullcontext(None),
        ):
            first = await tinysearch.scrape_urls([{"url": "https://shop.example/products/a"}])
            second = await tinysearch.scrape_urls([{"url": "https://shop.example/collections/x"}])

        self.assertEqual(first["results"][0]["error"]["code"], "FetchUnavailableError")
        error = second["results"][0]["error"]
        self.assertEqual(error["code"], "HostRecentlyRefusedError")
        self.assertIn("browser", error["message"])
        self.assertEqual(site.fetches, ["https://shop.example/products/a"])


if __name__ == "__main__":
    unittest.main()


class BrowserNavigateRefusalMemoryTests(unittest.IsolatedAsyncioTestCase):
    """browser_navigate shares the scrape memory: a refused host is not opened
    in a browser either, and a refusal the browser sees is remembered for both."""

    def setUp(self) -> None:
        SHARED_REFUSED_HOSTS.clear()
        self.addCleanup(SHARED_REFUSED_HOSTS.clear)

    async def test_navigate_to_a_refused_host_starts_no_browser(self) -> None:
        from tinysearch.services import browser_tool_service as bt

        SHARED_REFUSED_HOSTS.remember(
            "https://www.trianglelab.net/products/x", 403, ttl_seconds=600
        )
        acquire = AsyncMock()
        with patch.object(bt, "acquire_session", acquire):
            with self.assertRaises(bt.BrowserToolError) as raised:
                await bt.call_tool(
                    "navigate", normalize_config(), url="https://trianglelab.net/other"
                )
        acquire.assert_not_awaited()
        message = str(raised.exception)
        self.assertIn("trianglelab.net refused automated access", message)
        self.assertIn("HTTP 403", message)
        self.assertIn("will not ask it again", message)

    async def test_ttl_zero_lets_navigate_through(self) -> None:
        from tinysearch.services import browser_tool_service as bt

        SHARED_REFUSED_HOSTS.remember("https://shop.example/a", 403, ttl_seconds=600)
        session = MagicMock()
        session.navigate = AsyncMock(return_value="- Page URL: https://shop.example/b")
        session.started = True
        with patch.object(bt, "acquire_session", AsyncMock(return_value=session)):
            result = await bt.call_tool(
                "navigate",
                normalize_config({"scrape_refused_host_ttl_seconds": 0}),
                url="https://shop.example/b",
            )
        self.assertIn("shop.example/b", result)

    async def test_other_browser_actions_are_not_gated(self) -> None:
        from tinysearch.services import browser_tool_service as bt

        SHARED_REFUSED_HOSTS.remember("https://shop.example/a", 403, ttl_seconds=600)
        session = MagicMock()
        session.look = AsyncMock(return_value="- body")
        session.started = True
        with patch.object(bt, "acquire_session", AsyncMock(return_value=session)):
            self.assertEqual(await bt.call_tool("look", normalize_config()), "- body")

    def _session(self, status: int | None):
        from tinysearch.services import browser_tool_service as bt

        session = bt.BrowserSession(normalize_config())
        page = MagicMock()
        page.is_closed.return_value = False
        page.url = "https://shop.example/a"
        page.title = AsyncMock(return_value="Just a moment...")
        page.aria_snapshot = AsyncMock(return_value="- body")
        response = None if status is None else MagicMock(status=status)
        page.goto = AsyncMock(return_value=response)
        session._page = page
        session._context = MagicMock()
        session.start = AsyncMock()
        return session

    async def test_a_refusal_seen_by_the_browser_is_remembered_for_scrape_too(self) -> None:
        session = self._session(403)
        view = await session.navigate("https://www.shop.example/a")
        # This call still shows what the browser got; the memory governs the next one.
        self.assertIn("Just a moment...", view)
        refusal = SHARED_REFUSED_HOSTS.refusal_for("https://shop.example/collections/x")
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.status, 403)

    async def test_success_and_missing_response_are_not_remembered(self) -> None:
        for status in (200, 404, None):
            session = self._session(status)
            await session.navigate("https://shop.example/a")
            self.assertIsNone(SHARED_REFUSED_HOSTS.refusal_for("https://shop.example/a"))

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from tinysearch.services.site_crawl_service import (
    BOILERPLATE_EXCLUDED_TAGS,
    BrowserCrawlerSession,
    _crawler_config_for_fit_markdown,
    _lightweight_browser_config,
    fetch_html_for_query,
)


class _RecordingBrowserConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class BrowserBackendConfigTests(unittest.TestCase):
    def test_default_uses_bundled_lightweight_browser(self) -> None:
        config = _lightweight_browser_config(_RecordingBrowserConfig)

        self.assertTrue(config.kwargs["light_mode"])
        self.assertTrue(config.kwargs["memory_saving_mode"])
        self.assertNotIn("cdp_url", config.kwargs)

    def test_external_cdp_preserves_backend_identity(self) -> None:
        config = _lightweight_browser_config(
            _RecordingBrowserConfig,
            {"browser_cdp_url": "http://browser:9222"},
        )

        self.assertEqual(config.kwargs["browser_mode"], "custom")
        self.assertEqual(config.kwargs["cdp_url"], "http://browser:9222")
        self.assertTrue(config.kwargs["cdp_cleanup_on_close"])
        self.assertEqual(config.kwargs["cdp_close_delay"], 0)
        self.assertIsNone(config.user_agent)
        self.assertEqual(config.browser_hint, "")


class CrawlerConfigBoilerplateExclusionTests(unittest.TestCase):
    """nav/header/footer/aside chrome must be excluded before markdown
    generation or content filtering runs in every fit_markdown_mode branch --
    a BM25 relevance threshold alone does not reliably filter this content
    out (verified empirically: identical leaked output across thresholds
    1.5-4.0 on a real site), so it has to be stripped at the DOM level."""

    def test_off_mode_excludes_boilerplate_tags(self) -> None:
        config = _crawler_config_for_fit_markdown(
            fit_markdown_mode="off",
            user_query=None,
            bm25_threshold=1.5,
            bm25_language="english",
            pruning_threshold=0.48,
        )
        self.assertEqual(config.excluded_tags, BOILERPLATE_EXCLUDED_TAGS)

    def test_bm25_mode_excludes_boilerplate_tags(self) -> None:
        config = _crawler_config_for_fit_markdown(
            fit_markdown_mode="bm25",
            user_query="what is a bloom filter used for",
            bm25_threshold=1.5,
            bm25_language="english",
            pruning_threshold=0.48,
        )
        self.assertEqual(config.excluded_tags, BOILERPLATE_EXCLUDED_TAGS)

    def test_bm25_mode_with_empty_query_falls_back_but_still_excludes_tags(self) -> None:
        config = _crawler_config_for_fit_markdown(
            fit_markdown_mode="bm25",
            user_query="   ",
            bm25_threshold=1.5,
            bm25_language="english",
            pruning_threshold=0.48,
        )
        self.assertEqual(config.excluded_tags, BOILERPLATE_EXCLUDED_TAGS)

    def test_pruning_mode_excludes_boilerplate_tags(self) -> None:
        config = _crawler_config_for_fit_markdown(
            fit_markdown_mode="pruning",
            user_query=None,
            bm25_threshold=1.5,
            bm25_language="english",
            pruning_threshold=0.48,
        )
        self.assertEqual(config.excluded_tags, BOILERPLATE_EXCLUDED_TAGS)


class BrowserCrawlerSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_reuses_started_crawler_until_shutdown(self) -> None:
        crawler = MagicMock()
        crawler.start = AsyncMock()
        crawler.close = AsyncMock()
        session = BrowserCrawlerSession()

        with patch(
            "tinysearch.services.site_crawl_service.create_browser_crawler",
            return_value=crawler,
        ) as create:
            async with session.lease({"browser_idle_shutdown_seconds": 10}):
                pass
            async with session.lease({"browser_idle_shutdown_seconds": 10}):
                pass
            await session.close()

        create.assert_called_once()
        crawler.start.assert_awaited_once()
        crawler.close.assert_awaited_once()

    async def test_fetch_uses_configured_fit_mode_and_threshold(self) -> None:
        run_config = object()
        crawler = MagicMock()
        crawler.arun = AsyncMock(
            return_value=SimpleNamespace(
                url="https://example.com",
                html="<p>evidence</p>",
                markdown=SimpleNamespace(
                    raw_markdown="evidence",
                    fit_markdown="filtered evidence",
                ),
                metadata={},
            )
        )
        with patch(
            "tinysearch.services.site_crawl_service._crawl4ai_stack",
            return_value=tuple(MagicMock() for _ in range(6)),
        ), patch(
            "tinysearch.services.site_crawl_service._crawler_config_for_fit_markdown",
            return_value=run_config,
        ) as build_config:
            result = await fetch_html_for_query(
                "https://example.com",
                "query",
                fit_markdown_mode="pruning",
                pruning_threshold=0.33,
                crawler=crawler,
            )

        build_config.assert_called_once_with(
            fit_markdown_mode="pruning",
            user_query="query",
            bm25_threshold=1.5,
            bm25_language="english",
            pruning_threshold=0.33,
        )
        crawler.arun.assert_awaited_once_with(
            url="https://example.com", config=run_config
        )
        self.assertEqual(result["markdown_fit"], "filtered evidence")

    async def test_closes_crawler_after_idle_period(self) -> None:
        crawler = MagicMock()
        crawler.start = AsyncMock()
        crawler.close = AsyncMock()
        session = BrowserCrawlerSession()

        with patch(
            "tinysearch.services.site_crawl_service.create_browser_crawler",
            return_value=crawler,
        ):
            async with session.lease({"browser_idle_shutdown_seconds": 0.01}):
                pass
            idle_task = session._idle_task
            assert idle_task is not None
            await asyncio.wait_for(idle_task, timeout=5)

        self.assertFalse(session.started)
        crawler.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()

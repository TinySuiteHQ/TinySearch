from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from tinysearch.services.site_crawl_service import (
    BrowserCrawlerSession,
    DirectPlaywrightCrawler,
    _accessibility_text,
    fetch_html_for_query,
)


class AccessibilityExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_prefers_ai_aria_snapshot(self) -> None:
        body = MagicMock()
        body.aria_snapshot = AsyncMock(return_value='- heading "Example" [level=1]')
        body.inner_text = AsyncMock(return_value="fallback")
        page = MagicMock()
        page.locator.return_value = body

        text = await _accessibility_text(page)

        self.assertEqual(text, '- heading "Example" [level=1]')
        body.inner_text.assert_not_awaited()

    async def test_falls_back_to_visible_text(self) -> None:
        body = MagicMock()
        body.aria_snapshot = AsyncMock(side_effect=TypeError("unsupported"))
        body.inner_text = AsyncMock(return_value="Visible article text")
        page = MagicMock()
        page.locator.return_value = body

        text = await _accessibility_text(page)

        self.assertEqual(text, "Visible article text")


class DirectPlaywrightCrawlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_arun_returns_rendered_html_final_url_and_accessibility_text(self) -> None:
        response = MagicMock(status=200)
        page = MagicMock()
        page.url = "https://example.com/final"
        page.goto = AsyncMock(return_value=response)
        page.content = AsyncMock(return_value="<html><body>evidence</body></html>")
        page.title = AsyncMock(return_value="Example")
        context = MagicMock()
        context.new_page = AsyncMock(return_value=page)
        context.close = AsyncMock()

        crawler = DirectPlaywrightCrawler()
        crawler._runtime = MagicMock()
        crawler._runtime.new_context = AsyncMock(return_value=context)

        with patch(
            "tinysearch.services.site_crawl_service._accessibility_text",
            new=AsyncMock(return_value="- paragraph: evidence"),
        ):
            result = await crawler.arun(url="https://example.com")

        self.assertEqual(result["url"], "https://example.com/final")
        self.assertEqual(result["markdown_raw"], "- paragraph: evidence")
        self.assertEqual(result["metadata"]["title"], "Example")
        self.assertEqual(result["metadata"]["status"], 200)
        context.close.assert_awaited_once()

    async def test_fetch_keeps_old_signature_but_does_not_prefilter(self) -> None:
        crawler = MagicMock()
        crawler.arun = AsyncMock(
            return_value={
                "url": "https://example.com",
                "redirected_url": "https://example.com/final",
                "html": "<p>evidence</p>",
                "markdown_raw": "- paragraph: evidence",
                "markdown_fit": "",
                "metadata": {"title": "Example"},
            }
        )

        result = await fetch_html_for_query(
            "https://example.com",
            "query",
            crawler=crawler,
        )

        crawler.arun.assert_awaited_once_with(
            url="https://example.com", config=None
        )
        self.assertEqual(result["final_url"], "https://example.com/final")
        self.assertEqual(result["markdown_raw"], "- paragraph: evidence")
        self.assertEqual(result["markdown_fit"], "")


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

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from tinysearch.config import TinySearchConfig
from tinysearch.services.site_crawl_service import DirectPlaywrightCrawler, _stylesheet_route


def _route(url: str, resource_type: str, *, navigation: bool = False) -> MagicMock:
    route = MagicMock()
    route.request.url = url
    route.request.resource_type = resource_type
    route.request.is_navigation_request.return_value = navigation
    route.fulfill = AsyncMock()
    route.continue_ = AsyncMock()
    route.abort = AsyncMock()
    return route


def _safety(url: str, blocked_domains) -> str:
    if "internal" in url or any(domain in url for domain in blocked_domains):
        raise ValueError("not fetchable")
    return url


class StylesheetRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_serves_html_once_loads_safe_stylesheets_and_aborts_the_rest(self) -> None:
        handler = _stylesheet_route(html="<p>page</p>", blocked_domains=["ads.example"])
        page = _route("https://site.example/a", "document", navigation=True)
        css = _route("https://cdn.example/site.css", "stylesheet")
        internal_css = _route("http://internal.lan/x.css", "stylesheet")
        blocked_css = _route("https://ads.example/x.css", "stylesheet")
        script = _route("https://cdn.example/app.js", "script")
        image = _route("https://cdn.example/a.png", "image")
        second_navigation = _route("https://site.example/elsewhere", "document", navigation=True)

        with patch("tinysearch.services.site_crawl_service.assert_url_is_fetchable", side_effect=_safety):
            for route in (page, css, internal_css, blocked_css, script, image, second_navigation):
                await handler(route)

        page.fulfill.assert_awaited_once()
        self.assertEqual(page.fulfill.await_args.kwargs["body"], "<p>page</p>")
        css.continue_.assert_awaited_once()
        for route in (internal_css, blocked_css, script, image, second_navigation):
            route.abort.assert_awaited_once()
            route.continue_.assert_not_awaited()


class RenderHtmlTests(unittest.IsolatedAsyncioTestCase):
    def _crawler(self, page: MagicMock) -> tuple[DirectPlaywrightCrawler, MagicMock]:
        context = MagicMock()
        context.route = AsyncMock()
        context.new_page = AsyncMock(return_value=page)
        context.close = AsyncMock()
        crawler = DirectPlaywrightCrawler()
        crawler._runtime = MagicMock()
        crawler._runtime.new_context = AsyncMock(return_value=context)
        return crawler, context

    def _page(self) -> MagicMock:
        page = MagicMock()
        page.goto = AsyncMock()
        page.set_content = AsyncMock()
        page.title = AsyncMock(return_value="T")
        return page

    async def test_renders_at_the_page_url_with_javascript_off(self) -> None:
        page = self._page()
        crawler, _ = self._crawler(page)
        with patch("tinysearch.services.site_crawl_service.assert_url_is_fetchable", side_effect=_safety), patch(
            "tinysearch.services.site_crawl_service._accessibility_text", new=AsyncMock(return_value="text")
        ):
            result = await crawler.render_html(html="<p>x</p>", url="https://site.example/a")
        self.assertEqual(result["markdown_raw"], "text")
        crawler._runtime.new_context.assert_awaited_once_with(java_script_enabled=False)
        self.assertEqual(page.goto.await_args.args[0], "https://site.example/a")
        page.set_content.assert_not_awaited()

    async def test_unfetchable_or_missing_url_renders_fully_offline(self) -> None:
        for url in (None, "http://internal.lan/a"):
            page = self._page()
            crawler, _ = self._crawler(page)
            with patch("tinysearch.services.site_crawl_service.assert_url_is_fetchable", side_effect=_safety), patch(
                "tinysearch.services.site_crawl_service._accessibility_text", new=AsyncMock(return_value="text")
            ):
                await crawler.render_html(html="<p>x</p>", url=url)
            page.goto.assert_not_awaited()
            page.set_content.assert_awaited_once()

    async def test_a_slow_stylesheet_does_not_fail_the_render(self) -> None:
        page = self._page()
        page.goto = AsyncMock(side_effect=TimeoutError("load timed out"))
        crawler, _ = self._crawler(page)
        with patch("tinysearch.services.site_crawl_service.assert_url_is_fetchable", side_effect=_safety), patch(
            "tinysearch.services.site_crawl_service._accessibility_text", new=AsyncMock(return_value="text")
        ):
            result = await crawler.render_html(html="<p>x</p>", url="https://site.example/a")
        self.assertEqual(result["markdown_raw"], "text")


class SettleTests(unittest.IsolatedAsyncioTestCase):
    async def test_waits_for_network_idle_up_to_the_configured_cap(self) -> None:
        page = MagicMock()
        page.wait_for_load_state = AsyncMock()
        await DirectPlaywrightCrawler({"scrape_settle_seconds": 1.5})._settle(page)
        page.wait_for_load_state.assert_awaited_once_with("networkidle", timeout=1500.0)

    async def test_zero_disables_the_wait_and_a_timeout_is_not_an_error(self) -> None:
        page = MagicMock()
        page.wait_for_load_state = AsyncMock(side_effect=TimeoutError("never idle"))
        await DirectPlaywrightCrawler({"scrape_settle_seconds": 0})._settle(page)
        page.wait_for_load_state.assert_not_awaited()
        await DirectPlaywrightCrawler({"scrape_settle_seconds": 1})._settle(page)

    def test_setting_defaults_to_two_and_a_half_and_rejects_negatives(self) -> None:
        self.assertEqual(TinySearchConfig()["scrape_settle_seconds"], 2.5)
        with self.assertRaises(ValueError):
            TinySearchConfig(scrape_settle_seconds=-1)


if __name__ == "__main__":
    unittest.main()

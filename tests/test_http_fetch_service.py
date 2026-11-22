from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from tinysearch.services import site_crawl_service
from tinysearch.services.http_fetch_service import (
    MAX_PLAIN_RESPONSE_BYTES,
    DocumentResponse,
    PlainFetchError,
    download_document,
    configured_proxies,
    fetch_plain,
    httpx_proxy_url,
    plain_response_needs_browser,
)
from tinysearch.services.site_crawl_service import DirectPlaywrightCrawler, document_type
from tinysearch.services.url_safety_service import BlockedUrlError

ARTICLE = "<html><body><article>" + " ".join(["evidence"] * 300) + "</article></body></html>"


def _allow_all(url: str, blocked_domains: object) -> str:
    return url


class FetchPlainTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_html_status_and_final_url(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/html; charset=utf-8"}, text="<p>hi</p>"
            )
        )
        with patch("tinysearch.services.http_fetch_service.assert_url_is_fetchable", side_effect=_allow_all):
            page = await fetch_plain("https://example.com/a", config={}, transport=transport)
        self.assertEqual(page["status"], 200)
        self.assertEqual(page["content_type"], "text/html")
        self.assertEqual(page["html"], "<p>hi</p>")
        self.assertEqual(page["final_url"], "https://example.com/a")

    async def test_follows_redirects_checking_every_hop(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/start":
                return httpx.Response(302, headers={"location": "/next"})
            return httpx.Response(200, headers={"content-type": "text/html"}, text="done")

        checked: list[str] = []

        def check(url: str, blocked_domains: object) -> str:
            checked.append(url)
            return url

        with patch("tinysearch.services.http_fetch_service.assert_url_is_fetchable", side_effect=check):
            page = await fetch_plain(
                "https://example.com/start", config={}, transport=httpx.MockTransport(handler)
            )
        self.assertEqual(checked, ["https://example.com/start", "https://example.com/next"])
        self.assertEqual(page["final_url"], "https://example.com/next")

    async def test_a_redirect_to_a_refused_address_is_never_requested(self) -> None:
        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(str(request.url))
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest"})

        def check(url: str, blocked_domains: object) -> str:
            if "169.254" in url:
                raise BlockedUrlError("private address")
            return url

        with patch("tinysearch.services.http_fetch_service.assert_url_is_fetchable", side_effect=check):
            with self.assertRaises(BlockedUrlError):
                await fetch_plain("https://example.com/", config={}, transport=httpx.MockTransport(handler))
        self.assertEqual(requested, ["https://example.com/"])

    async def test_too_many_redirects_fail(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(302, headers={"location": "/loop"}))
        with patch("tinysearch.services.http_fetch_service.assert_url_is_fetchable", side_effect=_allow_all):
            with self.assertRaises(PlainFetchError):
                await fetch_plain("https://example.com/loop", config={}, transport=transport)

    async def test_an_oversized_body_fails(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/html", "content-length": str(MAX_PLAIN_RESPONSE_BYTES + 1)},
                content=b"x",
            )
        )
        with patch("tinysearch.services.http_fetch_service.assert_url_is_fetchable", side_effect=_allow_all):
            with self.assertRaises(PlainFetchError):
                await fetch_plain("https://example.com/", config={}, transport=transport)

    async def test_decodes_with_the_meta_charset_when_the_header_has_none(self) -> None:
        body = '<meta charset="iso-8859-1"><p>café</p>'.encode("iso-8859-1")
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "text/html"}, content=body)
        )
        with patch("tinysearch.services.http_fetch_service.assert_url_is_fetchable", side_effect=_allow_all):
            page = await fetch_plain("https://example.com/", config={}, transport=transport)
        self.assertIn("café", page["html"])


class NeedsBrowserTests(unittest.TestCase):
    def test_a_readable_page_needs_nothing(self) -> None:
        self.assertIsNone(plain_response_needs_browser(200, "text/html", ARTICLE))

    def test_each_signal(self) -> None:
        cases = {
            "status_403": (403, "text/html", ARTICLE),
            "status_500": (500, "text/html", ARTICLE),
            "not_html": (200, "application/pdf", "%PDF"),
            "meta_refresh": (200, "text/html", '<meta http-equiv="refresh" content="0;url=/x">'),
            "javascript_app": (200, "text/html", '<body><div id="root"></div><script src="a.js"></script></body>'),
            "javascript_required": (
                200,
                "text/html",
                "<body><noscript>You need to enable JavaScript to run this app.</noscript></body>",
            ),
        }
        for expected, args in cases.items():
            with self.subTest(expected):
                self.assertEqual(plain_response_needs_browser(*args), expected)

    def test_app_signals_are_ignored_when_the_html_has_plenty_of_text(self) -> None:
        html = ARTICLE.replace(
            "</body>", '<div id="app"></div><noscript>Please enable JavaScript for comments.</noscript></body>'
        )
        self.assertIsNone(plain_response_needs_browser(200, "text/html", html))


class ProxyTests(unittest.TestCase):
    def test_rotation_list_and_credentials(self) -> None:
        config = {
            "browser_proxy_server": "proxy-a:8080, http://proxy-b:3128",
            "browser_proxy_username": "user@corp",
            "browser_proxy_password": "p:ss",
        }
        proxies = configured_proxies(config)
        self.assertEqual([p["server"] for p in proxies], ["proxy-a:8080", "http://proxy-b:3128"])
        self.assertEqual(
            httpx_proxy_url(proxies[0], "https://example.com/"),
            "http://user%40corp:p%3Ass@proxy-a:8080",
        )

    def test_bypass_and_socks(self) -> None:
        proxy = {"server": "proxy:8080", "bypass": ".internal.example, example.org"}
        self.assertIsNone(httpx_proxy_url(proxy, "https://docs.internal.example/"))
        self.assertIsNone(httpx_proxy_url(proxy, "https://www.example.org/"))
        self.assertEqual(httpx_proxy_url(proxy, "https://example.com/"), "http://proxy:8080")
        with self.assertRaises(PlainFetchError):
            httpx_proxy_url({"server": "socks5://proxy:1080"}, "https://example.com/")


class TieredFetchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        site_crawl_service._needs_browser.clear()
        self.browser_page = {
            "url": "https://example.com/a",
            "redirected_url": "https://example.com/a",
            "html": "<p>rendered</p>",
            "markdown_raw": "rendered",
            "metadata": {"title": "", "status": 200},
        }

    def _crawler(self, **config: object) -> DirectPlaywrightCrawler:
        crawler = DirectPlaywrightCrawler({"scrape_http_first": True, **config})
        crawler.arun = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda **_: {**self.browser_page, "metadata": dict(self.browser_page["metadata"])}
        )
        crawler.render_html = AsyncMock(  # type: ignore[method-assign]
            return_value={"markdown_raw": " ".join(["word"] * 100), "title": "T"}
        )
        return crawler

    def _plain(self, html: str = ARTICLE, status: int = 200) -> AsyncMock:
        return AsyncMock(
            return_value={
                "final_url": "https://example.com/a",
                "status": status,
                "content_type": "text/html",
                "html": html,
            }
        )

    async def test_a_readable_plain_response_never_loads_the_page(self) -> None:
        crawler = self._crawler()
        with patch("tinysearch.services.http_fetch_service.fetch_plain", self._plain()):
            page = await crawler.fetch(url="https://example.com/a")
        self.assertEqual(page["metadata"]["fetch_tier"], "http")
        crawler.render_html.assert_awaited_once()
        crawler.arun.assert_not_awaited()

    async def test_off_by_default(self) -> None:
        crawler = self._crawler(scrape_http_first=False)
        fetch_plain = self._plain()
        with patch("tinysearch.services.http_fetch_service.fetch_plain", fetch_plain):
            page = await crawler.fetch(url="https://example.com/a")
        fetch_plain.assert_not_awaited()
        self.assertEqual(page["metadata"]["fetch_tier"], "browser")
        self.assertNotIn("escalation_reason", page["metadata"])

    async def test_escalates_and_remembers_the_domain(self) -> None:
        crawler = self._crawler()
        fetch_plain = self._plain(html='<div id="root"></div>')
        with patch("tinysearch.services.http_fetch_service.fetch_plain", fetch_plain):
            first = await crawler.fetch(url="https://example.com/a")
            second = await crawler.fetch(url="https://example.com/b")
        self.assertEqual(first["metadata"]["escalation_reason"], "javascript_app")
        self.assertEqual(second["metadata"]["escalation_reason"], "domain_hint")
        self.assertEqual(fetch_plain.await_count, 1)

    async def test_a_missing_page_is_answered_once_and_does_not_mark_the_domain(self) -> None:
        crawler = self._crawler()
        fetch_plain = self._plain(status=404)
        with patch("tinysearch.services.http_fetch_service.fetch_plain", fetch_plain):
            first = await crawler.fetch(url="https://example.com/a")
            await crawler.fetch(url="https://example.com/b")
        self.assertEqual(first["metadata"]["status"], 404)
        self.assertEqual(fetch_plain.await_count, 2)
        crawler.arun.assert_not_awaited()

    async def test_a_429_is_answered_as_is_without_a_second_request(self) -> None:
        crawler = self._crawler()
        with patch("tinysearch.services.http_fetch_service.fetch_plain", self._plain(status=429)):
            page = await crawler.fetch(url="https://example.com/a")
        self.assertEqual(page["metadata"]["status"], 429)
        self.assertEqual(page["metadata"]["fetch_tier"], "http")
        crawler.arun.assert_not_awaited()
        crawler.render_html.assert_not_awaited()

    async def test_too_little_rendered_text_escalates(self) -> None:
        crawler = self._crawler()
        crawler.render_html = AsyncMock(return_value={"markdown_raw": "Loading", "title": ""})  # type: ignore[method-assign]
        with patch("tinysearch.services.http_fetch_service.fetch_plain", self._plain()):
            page = await crawler.fetch(url="https://example.com/a")
        self.assertEqual(page["metadata"]["escalation_reason"], "too_little_text")
        crawler.arun.assert_awaited_once()

    async def test_a_short_script_heavy_page_escalates(self) -> None:
        crawler = self._crawler()
        app_shell = "<html><body><nav>Home Docs</nav><script>" + "x" * 5000 + "</script></body></html>"
        with patch("tinysearch.services.http_fetch_service.fetch_plain", self._plain(html=app_shell)):
            page = await crawler.fetch(url="https://example.com/a")
        self.assertEqual(page["metadata"]["escalation_reason"], "script_built_page")

    async def test_a_long_page_with_scripts_stays_on_plain_http(self) -> None:
        crawler = self._crawler()
        crawler.render_html = AsyncMock(  # type: ignore[method-assign]
            return_value={"markdown_raw": " ".join(["word"] * 400), "title": "T"}
        )
        html = ARTICLE.replace("</body>", "<script>" + "x" * 20000 + "</script></body>")
        with patch("tinysearch.services.http_fetch_service.fetch_plain", self._plain(html=html)):
            page = await crawler.fetch(url="https://example.com/a")
        self.assertEqual(page["metadata"]["fetch_tier"], "http")

    async def test_a_failed_plain_fetch_escalates(self) -> None:
        crawler = self._crawler()
        failing = AsyncMock(side_effect=httpx.ConnectTimeout("slow"))
        with patch("tinysearch.services.http_fetch_service.fetch_plain", failing):
            page = await crawler.fetch(url="https://example.com/a")
        self.assertEqual(page["metadata"]["escalation_reason"], "plain_fetch_failed:ConnectTimeout")

    async def test_a_refused_hop_is_not_retried_in_the_browser(self) -> None:
        crawler = self._crawler()
        refused = AsyncMock(side_effect=BlockedUrlError("private"))
        with patch("tinysearch.services.http_fetch_service.fetch_plain", refused):
            with self.assertRaises(BlockedUrlError):
                await crawler.fetch(url="https://example.com/a")
        crawler.arun.assert_not_awaited()

    async def test_domain_hint_expires(self) -> None:
        site_crawl_service._remember_needs_browser("https://example.com/a")
        self.assertTrue(site_crawl_service._recently_needed_browser("https://example.com/x"))
        site_crawl_service._needs_browser["example.com"] = 0.0
        self.assertFalse(site_crawl_service._recently_needed_browser("https://example.com/x"))


class DocumentTests(unittest.IsolatedAsyncioTestCase):
    """PDFs and DOCX files are read as documents whatever their URL looks like."""

    async def test_plain_tier_hands_a_pdf_to_the_document_reader(self) -> None:
        crawler = DirectPlaywrightCrawler({"scrape_http_first": True})
        crawler.arun = AsyncMock()  # type: ignore[method-assign]
        plain = AsyncMock(return_value={
            "final_url": "https://example.com/paper/123", "status": 200,
            "content_type": "application/pdf", "html": "%PDF-1.7",
        })
        with patch("tinysearch.services.http_fetch_service.fetch_plain", plain):
            with self.assertRaises(DocumentResponse):
                await crawler.fetch(url="https://example.com/paper/123")
        crawler.arun.assert_not_awaited()

    async def test_a_browser_download_is_a_document(self) -> None:
        page = MagicMock()
        page.goto = AsyncMock(side_effect=Exception("Page.goto: Download is starting"))
        context = MagicMock(route=AsyncMock(), new_page=AsyncMock(return_value=page), close=AsyncMock())
        crawler = DirectPlaywrightCrawler()
        crawler._runtime = MagicMock(new_context=AsyncMock(return_value=context))
        with self.assertRaises(DocumentResponse):
            await crawler.arun(url="https://example.com/paper/123")
        context.close.assert_awaited_once()

    def test_document_type(self) -> None:
        self.assertEqual(document_type("https://e.com/a.pdf", "", b""), "pdf")
        self.assertEqual(document_type("https://e.com/a", "application/pdf", b""), "pdf")
        self.assertEqual(document_type("https://e.com/a", "application/octet-stream", b"%PDF-1.4 ..."), "pdf")
        self.assertEqual(document_type("https://e.com/a", "", b"PK\x03\x04....word/document.xml"), "docx")
        self.assertIsNone(document_type("https://e.com/a", "text/html", b"<html>"))


class DownloadDocumentTests(unittest.TestCase):
    def test_follows_checked_redirects(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/start":
                return httpx.Response(302, headers={"location": "/file"})
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.7")

        checked: list[str] = []
        with patch(
            "tinysearch.services.http_fetch_service.assert_url_is_fetchable",
            side_effect=lambda url, _b: checked.append(url) or url,
        ):
            result = download_document("https://example.com/start", config={}, timeout=5, transport=httpx.MockTransport(handler))
        self.assertEqual(checked, ["https://example.com/start", "https://example.com/file"])
        self.assertEqual((result["content_type"], result["body"]), ("application/pdf", b"%PDF-1.7"))

    def test_a_redirect_to_a_refused_address_is_never_requested(self) -> None:
        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(str(request.url))
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest"})

        def check(url: str, _blocked: object) -> str:
            if "169.254" in url:
                raise BlockedUrlError("private address")
            return url

        with patch("tinysearch.services.http_fetch_service.assert_url_is_fetchable", side_effect=check):
            with self.assertRaises(BlockedUrlError):
                download_document("https://example.com/a.pdf", config={}, timeout=5, transport=httpx.MockTransport(handler))
        self.assertEqual(requested, ["https://example.com/a.pdf"])

    def test_size_cap_and_error_status(self) -> None:
        big = httpx.MockTransport(lambda r: httpx.Response(200, content=b"x" * 2048))
        missing = httpx.MockTransport(lambda r: httpx.Response(404))
        with patch("tinysearch.services.http_fetch_service.assert_url_is_fetchable", side_effect=lambda u, _b: u):
            with self.assertRaises(PlainFetchError):
                download_document("https://example.com/a.pdf", config={}, timeout=5, max_bytes=1024, transport=big)
            with self.assertRaises(PlainFetchError):
                download_document("https://example.com/a.pdf", config={}, timeout=5, transport=missing)


if __name__ == "__main__":
    unittest.main()

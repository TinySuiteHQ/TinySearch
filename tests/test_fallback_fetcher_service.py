from __future__ import annotations

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from tinysearch.config import normalize_config
from tinysearch.services.fallback_fetcher_service import (
    FallbackFetchError,
    fetch_page_via_fallback,
    page_needs_fallback,
    request_fallback_html,
    visible_word_count,
)


class _FetcherHandler(BaseHTTPRequestHandler):
    # Set per test: (status, body bytes). The last request body lands in `seen`.
    reply: tuple[int, bytes] = (200, b"{}")
    seen: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        length = int(self.headers.get("Content-Length") or 0)
        type(self).seen.append(
            {
                "path": self.path,
                "content_type": self.headers.get("Content-Type"),
                "body": json.loads(self.rfile.read(length)),
            }
        )
        status, body = type(self).reply
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


class RequestFallbackHtmlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _FetcherHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.fetcher_url = f"http://127.0.0.1:{cls.server.server_port}/fetch"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        _FetcherHandler.seen = []

    def _reply(self, status: int, payload) -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        _FetcherHandler.reply = (status, body)

    def test_posts_contract_and_parses_answer(self) -> None:
        self._reply(
            200,
            {"html": "<p>hi</p>", "final_url": "https://example.com/y", "status": 200},
        )
        answer = request_fallback_html(self.fetcher_url, "https://example.com/x", 12.5)
        self.assertEqual(
            answer,
            {"html": "<p>hi</p>", "final_url": "https://example.com/y", "status": 200},
        )
        self.assertEqual(_FetcherHandler.seen[0]["path"], "/fetch")
        self.assertEqual(_FetcherHandler.seen[0]["content_type"], "application/json")
        self.assertEqual(
            _FetcherHandler.seen[0]["body"],
            {"url": "https://example.com/x", "timeout_seconds": 12.5},
        )

    def test_optional_fields_default(self) -> None:
        self._reply(200, {"html": "<p>hi</p>"})
        answer = request_fallback_html(self.fetcher_url, "https://example.com/x", 5)
        self.assertEqual(answer["final_url"], "https://example.com/x")
        self.assertIsNone(answer["status"])

    def test_non_2xx_is_a_failed_fallback(self) -> None:
        self._reply(502, {"error": "upstream"})
        with self.assertRaises(FallbackFetchError):
            request_fallback_html(self.fetcher_url, "https://example.com/x", 5)

    def test_invalid_json_is_a_failed_fallback(self) -> None:
        self._reply(200, b"not json")
        with self.assertRaises(FallbackFetchError):
            request_fallback_html(self.fetcher_url, "https://example.com/x", 5)

    def test_missing_html_is_a_failed_fallback(self) -> None:
        self._reply(200, {"final_url": "https://example.com/x"})
        with self.assertRaises(FallbackFetchError):
            request_fallback_html(self.fetcher_url, "https://example.com/x", 5)

    def test_unreachable_service_is_a_failed_fallback(self) -> None:
        with self.assertRaises(FallbackFetchError):
            request_fallback_html("http://127.0.0.1:9/fetch", "https://example.com/x", 1)


class FetchPageViaFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_renders_returned_html_on_the_supplied_crawler(self) -> None:
        rendered: list[str] = []

        class _Crawler:
            async def render_html(self, *, html: str) -> dict:
                rendered.append(html)
                return {"markdown_raw": "Rendered text", "title": "Rendered"}

        with patch(
            "tinysearch.services.fallback_fetcher_service.request_fallback_html",
            return_value={
                "html": "<p>Rendered text</p>",
                "final_url": "https://example.com/y",
                "status": 200,
            },
        ):
            page = await fetch_page_via_fallback(
                url="https://example.com/x",
                fetcher_url="http://fetcher:8080/",
                timeout_seconds=5,
                crawler=_Crawler(),
            )

        self.assertEqual(rendered, ["<p>Rendered text</p>"])
        self.assertEqual(
            page,
            {
                "final_url": "https://example.com/y",
                "html": "<p>Rendered text</p>",
                "markdown_raw": "Rendered text",
                "metadata": {"title": "Rendered", "status": 200},
            },
        )


class PageNeedsFallbackTests(unittest.TestCase):
    def test_trigger_statuses(self) -> None:
        for status in (403, 429, 503):
            page = {"markdown_raw": "Some text", "metadata": {"status": status}}
            self.assertTrue(page_needs_fallback(page), status)

    def test_empty_text_triggers(self) -> None:
        self.assertTrue(page_needs_fallback({"markdown_raw": "  ", "metadata": {}}))

    def test_readable_page_or_missing_status_does_not_trigger(self) -> None:
        for status in (200, 404, None):
            page = {"markdown_raw": "Some text", "metadata": {"status": status}}
            self.assertFalse(page_needs_fallback(page), status)

    def test_min_words_triggers_on_short_pages_only(self) -> None:
        short = {"markdown_raw": "one two three four", "metadata": {"status": 200}}
        enough = {"markdown_raw": "one two three four five", "metadata": {"status": 200}}
        self.assertTrue(page_needs_fallback(short, min_words=5))
        self.assertFalse(page_needs_fallback(enough, min_words=5))
        self.assertFalse(page_needs_fallback(short, min_words=0))


class VisibleWordCountTests(unittest.TestCase):
    def test_ignores_outline_markup(self) -> None:
        outline = (
            "- generic [active]:\n"
            '  - heading "Annual report 2025" [level=1]\n'
            "  - paragraph: Revenue grew 12 percent.\n"
            '  - link "Home":\n'
            "    - /url: /home/index\n"
            "  - text: noscript fallback text"
        )
        # Annual report 2025 / Revenue grew 12 percent / Home / noscript fallback text
        self.assertEqual(visible_word_count(outline), 11)

    def test_plain_text(self) -> None:
        self.assertEqual(visible_word_count("Hello, world: it's 2026!"), 5)
        self.assertEqual(visible_word_count("   "), 0)


class FallbackFetcherConfigTests(unittest.TestCase):
    def test_defaults_off(self) -> None:
        config = normalize_config({})
        self.assertEqual(config["scrape_fallback_fetcher_url"], "")
        self.assertEqual(config["scrape_fallback_fetcher_timeout_seconds"], 60.0)

    def test_accepts_http_url(self) -> None:
        config = normalize_config(
            {"scrape_fallback_fetcher_url": " http://127.0.0.1:8002/fetch "}
        )
        self.assertEqual(config["scrape_fallback_fetcher_url"], "http://127.0.0.1:8002/fetch")

    def test_rejects_other_schemes_and_missing_host(self) -> None:
        for url in ("ftp://fetcher/", "file:///tmp/x", "http://"):
            with self.assertRaises(ValueError, msg=url):
                normalize_config({"scrape_fallback_fetcher_url": url})

    def test_rejects_non_positive_timeout(self) -> None:
        with self.assertRaises(ValueError):
            normalize_config({"scrape_fallback_fetcher_timeout_seconds": 0})

    def test_min_words_default_and_validation(self) -> None:
        self.assertEqual(normalize_config({})["scrape_fallback_min_words"], 30)
        self.assertEqual(
            normalize_config({"scrape_fallback_min_words": 0})["scrape_fallback_min_words"], 0
        )
        with self.assertRaises(ValueError):
            normalize_config({"scrape_fallback_min_words": -1})

    def test_env_var_sets_url(self) -> None:
        import tempfile
        from pathlib import Path

        from tinysearch.services.tinysearch_config_service import load_tinysearch_config

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"TINYSEARCH_SCRAPE_FALLBACK_FETCHER_URL": "http://127.0.0.1:8002/fetch"},
        ):
            config = load_tinysearch_config(Path(directory) / "absent.json")
        self.assertEqual(config["scrape_fallback_fetcher_url"], "http://127.0.0.1:8002/fetch")


if __name__ == "__main__":
    unittest.main()

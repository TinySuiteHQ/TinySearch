from __future__ import annotations

import json
import socket
import unittest
import urllib.error
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from tinysearch.config import DEFAULT_CONFIG, normalize_config
from tinysearch.services import web_search_service
from tinysearch.services.web_search_service import (
    BRAVE_API_KEY_ENV_VAR,
    DEFAULT_SEARXNG_URL,
    SearchBackendBlocked,
    SearchBackendError,
    SearchBackendUnavailable,
    SearchResult,
    _brave_search,
    _ddgs_search,
    _dispatch_search,
    _searxng_search,
    _with_brave_fallback,
    is_blocked_domain,
    normalize_domain,
    search,
    search_with_metadata,
    search_batch_with_metadata,
)


class _FakeDDGSException(Exception):
    pass


class _FakeRatelimitException(_FakeDDGSException):
    pass


class _FakeTimeoutException(_FakeDDGSException):
    pass


class _FakeDDGS:
    """Stand-in for ddgs.DDGS used to unit-test `_ddgs_search` without network access."""

    calls: list[dict[str, Any]] = []
    _result: Any = []
    _raise: BaseException | None = None

    def __init__(self, timeout: Any = None) -> None:
        self.timeout = timeout

    def text(self, query: str, **kwargs: Any) -> Any:
        type(self).calls.append({"query": query, "timeout": self.timeout, **kwargs})
        if type(self)._raise is not None:
            raise type(self)._raise
        return type(self)._result


@contextmanager
def _patch_ddgs(result: Any = None, raise_: BaseException | None = None):
    _FakeDDGS.calls = []
    _FakeDDGS._result = result if result is not None else []
    _FakeDDGS._raise = raise_
    with patch.object(web_search_service, "_ddgs_cls", return_value=_FakeDDGS), patch.object(
        web_search_service,
        "_ddgs_exceptions",
        return_value=(_FakeDDGSException, _FakeRatelimitException, _FakeTimeoutException),
    ):
        yield _FakeDDGS


class _FakeHeaders:
    def __init__(self, content_type: str = "", charset: str | None = "utf-8") -> None:
        self._content_type = content_type
        self._charset = charset

    def get(self, key: str, default: str = "") -> str:
        if key.lower() == "content-type":
            return self._content_type
        return default

    def get_content_charset(self, default: str | None = None) -> str | None:
        return self._charset if self._charset is not None else default


class _FakeUrlopenResponse:
    def __init__(
        self,
        body: bytes | str,
        *,
        content_type: str = "application/json",
        charset: str = "utf-8",
    ) -> None:
        if isinstance(body, str):
            body = body.encode(charset)
        self._body = body
        self._headers = _FakeHeaders(content_type=content_type, charset=charset)

    def __enter__(self) -> "_FakeUrlopenResponse":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def read(self) -> bytes:
        return self._body

    @property
    def headers(self) -> _FakeHeaders:
        return self._headers


def _make_urlopen_returning(
    body: bytes | str,
    *,
    content_type: str = "application/json",
    charset: str = "utf-8",
):
    def fake(req: Any, timeout: Any = None) -> _FakeUrlopenResponse:
        return _FakeUrlopenResponse(body, content_type=content_type, charset=charset)

    return fake


def _make_urlopen_raising(exc: BaseException):
    def fake(req: Any, timeout: Any = None) -> Any:
        raise exc

    return fake


class BlockedDomainTests(unittest.TestCase):
    def test_normalize_domain_accepts_bare_domains_and_urls(self) -> None:
        self.assertEqual(normalize_domain("Example.COM"), "example.com")
        self.assertEqual(normalize_domain("www.example.com"), "example.com")
        self.assertEqual(normalize_domain("https://www.example.com/path"), "example.com")

    def test_blocked_domain_matches_exact_www_and_subdomains(self) -> None:
        blocked = ["example.com"]

        self.assertTrue(is_blocked_domain("https://example.com/page", blocked))
        self.assertTrue(is_blocked_domain("https://www.example.com/page", blocked))
        self.assertTrue(is_blocked_domain("https://news.example.com/page", blocked))

    def test_blocked_domain_does_not_match_sibling_domain(self) -> None:
        self.assertFalse(
            is_blocked_domain("https://badexample.com/page", ["example.com"])
        )

    def test_blocked_domain_accepts_url_style_entries(self) -> None:
        self.assertTrue(
            is_blocked_domain(
                "https://news.example.com/page",
                ["https://www.example.com/anything"],
            )
        )


class BatchSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_preserves_order_and_returns_sibling_failure(self) -> None:
        good = [SearchResult(1, "Good", "https://sec.gov/report", "body")]
        with patch.object(
            web_search_service,
            "_backend_attempt_plan",
            side_effect=lambda config, query, limit: [
                ("first", lambda: (_ for _ in ()).throw(SearchBackendBlocked("blocked")))
                if query == "bad"
                else ("first", lambda: good)
            ],
        ):
            responses = await search_batch_with_metadata(
                [{"query": "bad", "domains": []}, {"query": "good", "domains": ["SEC.GOV"]}],
                limit=10,
                config={"blocked_domains": []},
                concurrency=2,
            )
        self.assertEqual([item.status for item in responses], ["error", "ok"])
        self.assertEqual(responses[1].results[0].url, "https://sec.gov/report")
        self.assertEqual(responses[0].attempts[0]["state"], "blocked")

    async def test_domain_filter_falls_through_after_zero_usable_results(self) -> None:
        wrong = [SearchResult(1, "Wrong", "https://example.com/report", "body")]
        right = [SearchResult(1, "Right", "https://www.sec.gov/report", "body")]
        with patch.object(
            web_search_service,
            "_backend_attempt_plan",
            return_value=[("one", lambda: wrong), ("two", lambda: right)],
        ):
            response = (await search_batch_with_metadata(
                [{"query": "filing", "domains": ["sec.gov"]}], limit=10,
                config={"blocked_domains": []}, concurrency=1,
            ))[0]
        self.assertEqual([attempt["result_count"] for attempt in response.attempts], [0, 1])
        self.assertEqual(response.results[0].url, "https://www.sec.gov/report")


class SearXNGBackendTests(unittest.TestCase):
    def test_searxng_json_response_maps_to_search_results(self) -> None:
        payload = {
            "results": [
                {
                    "title": "Async tasks in Python",
                    "url": "https://example.com/python",
                    "content": "Coroutines and tasks.",
                },
                {
                    "title": "Bread baking",
                    "url": "https://example.com/bread",
                    "content": "Flour and yeast.",
                },
            ]
        }
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_returning(json.dumps(payload)),
        ):
            results = _searxng_search(
                "python async", 5, url="http://searxng:8080/search"
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Async tasks in Python")
        self.assertEqual(results[0].url, "https://example.com/python")
        self.assertEqual(results[0].text, "Coroutines and tasks.")
        self.assertEqual(results[0].result_id, 1)
        self.assertEqual(results[1].result_id, 2)

    def test_searxng_maps_upstream_published_date(self) -> None:
        payload = {
            "results": [{
                "title": "Dated",
                "url": "https://example.com/dated",
                "content": "Preview",
                "publishedDate": "2026-08-01T12:00:00+00:00",
            }]
        }
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_returning(json.dumps(payload)),
        ):
            results = _searxng_search("dated", 10, url="http://searxng:8080/search")
        self.assertEqual(results[0].published_at, "2026-08-01T12:00:00+00:00")

    def test_searxng_respects_limit(self) -> None:
        payload = {
            "results": [
                {"title": f"r{i}", "url": f"https://example.com/{i}", "content": ""}
                for i in range(10)
            ]
        }
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_returning(json.dumps(payload)),
        ):
            results = _searxng_search("q", 3, url="http://searxng:8080/search")

        self.assertEqual(len(results), 3)

    def test_metadata_reports_the_fallback_backend(self) -> None:
        fallback = [SearchResult(1, "Fallback", "https://example.com", "Preview")]
        with patch.object(
            web_search_service,
            "_searxng_search",
            side_effect=SearchBackendUnavailable("down"),
        ), patch.object(web_search_service, "_ddgs_search", return_value=fallback):
            response = search_with_metadata(
                "q",
                config={
                    "search_backend": "searxng",
                    "search_backend_fallback": True,
                },
            )
        self.assertEqual(response.backend, "duckduckgo")
        self.assertEqual(response.results, fallback)

    def test_searxng_html_response_raises_unavailable_with_actionable_message(self) -> None:
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_returning(
                "<html><body>not json</body></html>", content_type="text/html"
            ),
        ):
            with self.assertRaises(SearchBackendUnavailable) as cm:
                _searxng_search("q", 5, url="http://searxng:8080/search")

        message = str(cm.exception).lower()
        self.assertIn("json", message)
        self.assertIn("formats", message)

    def test_searxng_network_error_raises_unavailable(self) -> None:
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_raising(urllib.error.URLError("connection refused")),
        ):
            with self.assertRaises(SearchBackendUnavailable):
                _searxng_search("q", 5, url="http://searxng:8080/search")

    def test_searxng_http_500_raises_unavailable(self) -> None:
        http_error = urllib.error.HTTPError(
            "http://searxng:8080/search", 500, "Server Error", {}, None
        )
        with patch.object(
            web_search_service, "urlopen", new=_make_urlopen_raising(http_error)
        ):
            with self.assertRaises(SearchBackendUnavailable):
                _searxng_search("q", 5, url="http://searxng:8080/search")

    def test_searxng_timeout_raises_unavailable(self) -> None:
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_raising(socket.timeout("timed out")),
        ):
            with self.assertRaises(SearchBackendUnavailable):
                _searxng_search("q", 5, url="http://searxng:8080/search")

    def test_searxng_empty_results_list_is_not_an_error(self) -> None:
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_returning(json.dumps({"results": []})),
        ):
            results = _searxng_search("q", 5, url="http://searxng:8080/search")
        self.assertEqual(results, [])

    def test_searxng_empty_results_with_unresponsive_engines_raises_blocked(self) -> None:
        """A rate-limited/CAPTCHA'd instance answers 200 with zero results.

        Treated as success it is indistinguishable from a genuine no-match, so
        the caller's configured fallback never engages and the outage is
        reported to the user as "nothing found".
        """
        payload = json.dumps(
            {
                "results": [],
                "unresponsive_engines": [["google", "CAPTCHA"], ["bing", "timeout"]],
            }
        )
        with patch.object(
            web_search_service, "urlopen", new=_make_urlopen_returning(payload)
        ):
            with self.assertRaises(SearchBackendBlocked) as ctx:
                _searxng_search("q", 5, url="http://searxng:8080/search")
        message = str(ctx.exception)
        self.assertIn("google (CAPTCHA)", message)
        self.assertIn("bing (timeout)", message)

    def test_searxng_accepts_bare_string_unresponsive_engine_names(self) -> None:
        payload = json.dumps({"results": [], "unresponsive_engines": ["duckduckgo"]})
        with patch.object(
            web_search_service, "urlopen", new=_make_urlopen_returning(payload)
        ):
            with self.assertRaises(SearchBackendBlocked) as ctx:
                _searxng_search("q", 5, url="http://searxng:8080/search")
        self.assertIn("duckduckgo", str(ctx.exception))

    def test_searxng_partial_engine_failure_still_returns_results(self) -> None:
        """Degraded is not down: usable results outrank a partial engine failure."""
        payload = json.dumps(
            {
                "results": [
                    {"title": "Hit", "url": "https://example.com/", "content": "snippet"}
                ],
                "unresponsive_engines": [["google", "CAPTCHA"]],
            }
        )
        with patch.object(
            web_search_service, "urlopen", new=_make_urlopen_returning(payload)
        ):
            results = _searxng_search("q", 5, url="http://searxng:8080/search")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Hit")


class DdgsSearchTests(unittest.TestCase):
    def test_ddgs_maps_title_href_body_to_search_results(self) -> None:
        raw = [
            {"title": "Example Title", "href": "https://example.com/page", "body": "Example snippet."},
            {"title": "Second", "href": "https://example.com/2", "body": ""},
        ]
        with _patch_ddgs(result=raw) as fake_cls:
            results = _ddgs_search("anything", 5, region="us-en", backend="auto", timeout=20.0)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Example Title")
        self.assertEqual(results[0].url, "https://example.com/page")
        self.assertEqual(results[0].text, "Example snippet.")
        self.assertEqual(results[0].result_id, 1)
        call = fake_cls.calls[0]
        self.assertEqual(call["query"], "anything")
        self.assertEqual(call["timeout"], 20.0)
        self.assertEqual(call["region"], "us-en")
        self.assertEqual(call["safesearch"], "moderate")
        self.assertEqual(call["max_results"], 5)
        self.assertEqual(call["backend"], "auto")

    def test_ddgs_omits_region_when_not_provided(self) -> None:
        with _patch_ddgs(result=[]) as fake_cls:
            _ddgs_search("anything", 5, region=None, backend="auto", timeout=20.0)

        self.assertNotIn("region", fake_cls.calls[0])

    def test_ddgs_respects_limit(self) -> None:
        raw = [
            {"title": f"r{i}", "href": f"https://example.com/{i}", "body": ""}
            for i in range(10)
        ]
        with _patch_ddgs(result=raw):
            results = _ddgs_search("q", 3, backend="auto", timeout=20.0)
        self.assertEqual(len(results), 3)

    def test_ddgs_skips_malformed_individual_results(self) -> None:
        raw = [
            {"title": "", "href": "https://example.com/missing-title", "body": ""},
            {"title": "No URL", "href": "", "body": ""},
            {"title": "Good", "href": "https://example.com/good", "body": "ok"},
        ]
        with _patch_ddgs(result=raw):
            results = _ddgs_search("q", 5, backend="auto", timeout=20.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Good")

    def test_ddgs_fully_malformed_response_raises_unavailable(self) -> None:
        with _patch_ddgs(result={"not": "a list"}):
            with self.assertRaises(SearchBackendUnavailable):
                _ddgs_search("q", 5, backend="auto", timeout=20.0)

    def test_ddgs_empty_results_is_not_an_error(self) -> None:
        with _patch_ddgs(result=[]):
            results = _ddgs_search("q", 5, backend="auto", timeout=20.0)
        self.assertEqual(results, [])

    def test_ddgs_no_results_found_exception_returns_empty(self) -> None:
        with _patch_ddgs(raise_=_FakeDDGSException("No results found.")):
            results = _ddgs_search("q", 5, backend="auto", timeout=20.0)
        self.assertEqual(results, [])

    def test_ddgs_timeout_raises_unavailable(self) -> None:
        with _patch_ddgs(raise_=_FakeTimeoutException("timed out")):
            with self.assertRaises(SearchBackendUnavailable):
                _ddgs_search("q", 5, backend="auto", timeout=20.0)

    def test_ddgs_ratelimit_raises_blocked(self) -> None:
        with _patch_ddgs(raise_=_FakeRatelimitException("rate limited")):
            with self.assertRaises(SearchBackendBlocked):
                _ddgs_search("q", 5, backend="auto", timeout=20.0)

    def test_ddgs_generic_exception_raises_unavailable(self) -> None:
        with _patch_ddgs(raise_=_FakeDDGSException("boom")):
            with self.assertRaises(SearchBackendUnavailable):
                _ddgs_search("q", 5, backend="auto", timeout=20.0)


class BraveSearchTests(unittest.TestCase):
    def test_brave_maps_web_results_to_search_results(self) -> None:
        payload = {
            "web": {
                "results": [
                    {
                        "title": "Async tasks in Python",
                        "url": "https://example.com/python",
                        "description": "Coroutines and tasks.",
                    },
                    {"title": "No URL", "url": "", "description": "skip me"},
                ]
            }
        }
        with patch.object(
            web_search_service, "urlopen", new=_make_urlopen_returning(json.dumps(payload))
        ):
            results = _brave_search("python async", 5, api_key="secret-key")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Async tasks in Python")
        self.assertEqual(results[0].url, "https://example.com/python")
        self.assertEqual(results[0].text, "Coroutines and tasks.")

    def test_brave_sends_auth_header_and_params(self) -> None:
        captured: dict[str, Any] = {}

        def fake_urlopen(req: Any, timeout: Any = None) -> _FakeUrlopenResponse:
            captured["url"] = req.full_url
            captured["header"] = req.get_header("X-subscription-token")
            return _FakeUrlopenResponse(json.dumps({"web": {"results": []}}))

        with patch.object(web_search_service, "urlopen", new=fake_urlopen):
            _brave_search("python async", 5, api_key="secret-key")

        self.assertIn("q=python", captured["url"].replace("+", " "))
        self.assertEqual(captured["header"], "secret-key")

    def test_brave_empty_results_is_not_an_error(self) -> None:
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_returning(json.dumps({"web": {"results": []}})),
        ):
            results = _brave_search("q", 5, api_key="secret-key")
        self.assertEqual(results, [])

    def test_brave_401_raises_blocked(self) -> None:
        http_error = urllib.error.HTTPError(
            "https://api.search.brave.com/res/v1/web/search", 401, "Unauthorized", {}, None
        )
        with patch.object(web_search_service, "urlopen", new=_make_urlopen_raising(http_error)):
            with self.assertRaises(SearchBackendBlocked):
                _brave_search("q", 5, api_key="bad-key")

    def test_brave_network_error_raises_unavailable(self) -> None:
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_raising(urllib.error.URLError("no network")),
        ):
            with self.assertRaises(SearchBackendUnavailable):
                _brave_search("q", 5, api_key="secret-key")

    def test_brave_malformed_response_raises_unavailable(self) -> None:
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_returning("not json", content_type="text/html"),
        ):
            with self.assertRaises(SearchBackendUnavailable):
                _brave_search("q", 5, api_key="secret-key")

    def test_brave_key_never_appears_in_raised_exception_message(self) -> None:
        http_error = urllib.error.HTTPError(
            "https://api.search.brave.com/res/v1/web/search?token=super-secret-key",
            403,
            "Forbidden",
            {},
            None,
        )
        with patch.object(web_search_service, "urlopen", new=_make_urlopen_raising(http_error)):
            with self.assertRaises(SearchBackendBlocked) as cm:
                _brave_search("q", 5, api_key="super-secret-key")
        self.assertNotIn("super-secret-key", str(cm.exception))


class BraveFallbackTests(unittest.TestCase):
    def test_ddgs_error_with_key_calls_brave(self) -> None:
        def failing_primary() -> list[SearchResult]:
            raise SearchBackendUnavailable("ddgs down")

        with patch.dict("os.environ", {BRAVE_API_KEY_ENV_VAR: "secret-key"}), patch.object(
            web_search_service,
            "_brave_search",
            return_value=[SearchResult(1, "Brave", "https://brave.example/", "")],
        ) as brave_mock:
            results = _with_brave_fallback(failing_primary, "q", 5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Brave")
        brave_mock.assert_called_once_with("q", 5, api_key="secret-key")

    def test_ddgs_error_without_key_propagates_unchanged(self) -> None:
        original = SearchBackendUnavailable("ddgs down")

        def failing_primary() -> list[SearchResult]:
            raise original

        with patch.dict("os.environ", {}, clear=True), patch.object(
            web_search_service, "_brave_search"
        ) as brave_mock:
            with self.assertRaises(SearchBackendUnavailable) as cm:
                _with_brave_fallback(failing_primary, "q", 5)

        self.assertIs(cm.exception, original)
        brave_mock.assert_not_called()

    def test_ddgs_empty_with_key_calls_brave(self) -> None:
        def empty_primary() -> list[SearchResult]:
            return []

        with patch.dict("os.environ", {BRAVE_API_KEY_ENV_VAR: "secret-key"}), patch.object(
            web_search_service,
            "_brave_search",
            return_value=[SearchResult(1, "Brave", "https://brave.example/", "")],
        ) as brave_mock:
            results = _with_brave_fallback(empty_primary, "q", 5)

        self.assertEqual(len(results), 1)
        brave_mock.assert_called_once_with("q", 5, api_key="secret-key")

    def test_ddgs_empty_without_key_returns_empty_and_skips_brave(self) -> None:
        def empty_primary() -> list[SearchResult]:
            return []

        with patch.dict("os.environ", {}, clear=True), patch.object(
            web_search_service, "_brave_search"
        ) as brave_mock:
            results = _with_brave_fallback(empty_primary, "q", 5)

        self.assertEqual(results, [])
        brave_mock.assert_not_called()

    def test_both_fail_raises_single_error_without_leaking_key(self) -> None:
        def failing_primary() -> list[SearchResult]:
            raise SearchBackendUnavailable("ddgs down")

        with patch.dict(
            "os.environ", {BRAVE_API_KEY_ENV_VAR: "super-secret-key"}
        ), patch.object(
            web_search_service,
            "_brave_search",
            side_effect=SearchBackendUnavailable("brave down"),
        ):
            with self.assertRaises(SearchBackendUnavailable) as cm:
                _with_brave_fallback(failing_primary, "q", 5)

        self.assertNotIn("super-secret-key", str(cm.exception))

    def test_missing_key_never_invokes_brave_for_success_case(self) -> None:
        def successful_primary() -> list[SearchResult]:
            return [SearchResult(1, "ok", "https://example.com/", "")]

        with patch.dict("os.environ", {}, clear=True), patch.object(
            web_search_service, "_brave_search"
        ) as brave_mock:
            results = _with_brave_fallback(successful_primary, "q", 5)

        self.assertEqual(len(results), 1)
        brave_mock.assert_not_called()


class DispatcherTests(unittest.TestCase):
    _CONFIG_BASE: dict[str, Any] = {
        "search_backend": "searxng",
        "search_backend_url": "http://searxng:8080/search",
        "search_engines": [],
        "search_region": "",
        "search_backend_fallback": True,
    }

    def _config(self, **overrides: Any) -> dict[str, Any]:
        merged = dict(self._CONFIG_BASE)
        merged.update(overrides)
        return merged

    def test_default_backend_calls_searxng(self) -> None:
        searxng_calls: list[tuple[str, int]] = []
        ddgs_calls: list[tuple[str, int]] = []

        def fake_searxng(query: str, limit: int, **_: Any) -> list[SearchResult]:
            searxng_calls.append((query, limit))
            return [SearchResult(1, "ok", "https://example.com/", "")]

        def fake_ddgs(query: str, limit: int, **_: Any) -> list[SearchResult]:
            ddgs_calls.append((query, limit))
            return []

        with patch.object(web_search_service, "_searxng_search", new=fake_searxng), patch.object(
            web_search_service, "_ddgs_search", new=fake_ddgs
        ):
            results = _dispatch_search("q", 5, config=self._config())

        self.assertEqual(len(results), 1)
        self.assertEqual(len(searxng_calls), 1)
        self.assertEqual(ddgs_calls, [])

    def test_configured_searxng_timeout_reaches_backend(self) -> None:
        captured: dict[str, Any] = {}

        def fake_searxng(query: str, limit: int, **kwargs: Any) -> list[SearchResult]:
            captured.update(kwargs)
            return []

        with patch.object(web_search_service, "_searxng_search", new=fake_searxng):
            _dispatch_search(
                "q", 5, config=self._config(searxng_timeout_seconds=2.5)
            )

        self.assertEqual(captured["timeout"], 2.5)

    def test_searxng_failure_falls_back_to_ddgs_duckduckgo_when_enabled(self) -> None:
        def failing_searxng(*args: Any, **kwargs: Any) -> list[SearchResult]:
            raise SearchBackendUnavailable("searxng down")

        def fake_ddgs(query: str, limit: int, **kwargs: Any) -> list[SearchResult]:
            self.assertEqual(kwargs.get("backend"), "duckduckgo")
            return [SearchResult(1, "DDG", "https://duck.example/", "snippet")]

        with patch.object(
            web_search_service, "_searxng_search", new=failing_searxng
        ), patch.object(web_search_service, "_ddgs_search", new=fake_ddgs):
            results = _dispatch_search(
                "q", 5, config=self._config(search_backend_fallback=True)
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "DDG")

    def test_suspended_searxng_reaches_brave_through_ddgs(self) -> None:
        """The production path: SearXNG blocked, DDGS also blocked, Brave answers.

        The searxng branch previously had no Brave escape hatch at all, unlike
        the ddgs/duckduckgo branches, so a suspension storm that took out both
        scraped backends had nowhere left to go.
        """
        def blocked_searxng(*args: Any, **kwargs: Any) -> list[SearchResult]:
            raise SearchBackendBlocked("engines unresponsive: google (CAPTCHA)")

        def blocked_ddgs(*args: Any, **kwargs: Any) -> list[SearchResult]:
            raise SearchBackendBlocked("ddgs rate limited")

        with patch.dict("os.environ", {BRAVE_API_KEY_ENV_VAR: "secret-key"}), patch.object(
            web_search_service, "_searxng_search", new=blocked_searxng
        ), patch.object(
            web_search_service, "_ddgs_search", new=blocked_ddgs
        ), patch.object(
            web_search_service,
            "_brave_search",
            return_value=[SearchResult(1, "Brave", "https://brave.example/", "")],
        ):
            results = _dispatch_search(
                "q", 5, config=self._config(search_backend_fallback=True)
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Brave")

    def test_suspended_searxng_does_not_reach_brave_when_fallback_disabled(self) -> None:
        """Fallback disabled means SearXNG-only: no silent detour to Brave.

        Deployments that require every query to leave through their own
        instance would otherwise have that guarantee broken by the new hatch.
        """
        def blocked_searxng(*args: Any, **kwargs: Any) -> list[SearchResult]:
            raise SearchBackendBlocked("engines unresponsive: google (CAPTCHA)")

        with patch.dict("os.environ", {BRAVE_API_KEY_ENV_VAR: "secret-key"}), patch.object(
            web_search_service, "_searxng_search", new=blocked_searxng
        ), patch.object(web_search_service, "_brave_search") as brave_mock, patch.object(
            web_search_service, "_ddgs_search"
        ) as ddgs_mock:
            with self.assertRaises(SearchBackendError):
                _dispatch_search(
                    "q", 5, config=self._config(search_backend_fallback=False)
                )

        brave_mock.assert_not_called()
        ddgs_mock.assert_not_called()

    def test_searxng_failure_raises_when_fallback_disabled(self) -> None:
        def failing_searxng(*args: Any, **kwargs: Any) -> list[SearchResult]:
            raise SearchBackendUnavailable("searxng down")

        def fake_ddgs(query: str, limit: int, **_: Any) -> list[SearchResult]:
            self.fail("ddgs should not be called when fallback is disabled")
            return []

        with patch.object(
            web_search_service, "_searxng_search", new=failing_searxng
        ), patch.object(web_search_service, "_ddgs_search", new=fake_ddgs):
            with self.assertRaises(SearchBackendError):
                _dispatch_search(
                    "q", 5, config=self._config(search_backend_fallback=False)
                )

    def test_duckduckgo_backend_skips_searxng_and_delegates_to_ddgs(self) -> None:
        def fake_searxng(*args: Any, **kwargs: Any) -> list[SearchResult]:
            self.fail("SearXNG should not be called for the duckduckgo backend")
            return []

        def fake_ddgs(query: str, limit: int, **kwargs: Any) -> list[SearchResult]:
            self.assertEqual(kwargs.get("backend"), "duckduckgo")
            return [SearchResult(1, "DDG", "https://duck.example/", "snippet")]

        with patch.object(
            web_search_service, "_searxng_search", new=fake_searxng
        ), patch.object(web_search_service, "_ddgs_search", new=fake_ddgs):
            results = _dispatch_search(
                "q", 5, config=self._config(search_backend="duckduckgo")
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "DDG")

    def test_ddgs_backend_skips_searxng_and_passes_config(self) -> None:
        def fake_searxng(*args: Any, **kwargs: Any) -> list[SearchResult]:
            self.fail("SearXNG should not be called for the ddgs backend")
            return []

        captured: dict[str, Any] = {}

        def fake_ddgs(query: str, limit: int, **kwargs: Any) -> list[SearchResult]:
            captured.update(kwargs)
            return [SearchResult(1, "DDGS", "https://ddgs.example/", "")]

        with patch.object(
            web_search_service, "_searxng_search", new=fake_searxng
        ), patch.object(web_search_service, "_ddgs_search", new=fake_ddgs):
            results = _dispatch_search(
                "q",
                5,
                config=self._config(
                    search_backend="ddgs",
                    search_region="uk-en",
                    ddgs_backend="brave",
                    ddgs_timeout_seconds=15.0,
                ),
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(captured["region"], "uk-en")
        self.assertEqual(captured["backend"], "brave")
        self.assertEqual(captured["timeout"], 15.0)

    def test_auto_backend_falls_back_regardless_of_fallback_flag(self) -> None:
        def failing_searxng(*args: Any, **kwargs: Any) -> list[SearchResult]:
            raise SearchBackendUnavailable("nope")

        def fake_ddgs(query: str, limit: int, **kwargs: Any) -> list[SearchResult]:
            self.assertEqual(kwargs.get("backend"), "duckduckgo")
            return [SearchResult(1, "DDG", "https://duck.example/", "")]

        with patch.object(
            web_search_service, "_searxng_search", new=failing_searxng
        ), patch.object(web_search_service, "_ddgs_search", new=fake_ddgs):
            results = _dispatch_search(
                "q",
                5,
                config=self._config(
                    search_backend="auto", search_backend_fallback=False
                ),
            )

        self.assertEqual(len(results), 1)

    def test_invalid_backend_falls_back_to_default(self) -> None:
        def fake_searxng(*args: Any, **kwargs: Any) -> list[SearchResult]:
            return [SearchResult(1, "ok", "https://example.com/", "")]

        with patch.object(web_search_service, "_searxng_search", new=fake_searxng):
            results = _dispatch_search(
                "q", 5, config=self._config(search_backend="bogus")
            )

        self.assertEqual(len(results), 1)

    def test_search_default_uses_searxng(self) -> None:
        captured: dict[str, Any] = {}

        def fake_searxng(query: str, limit: int, *, url: str, **kwargs: Any) -> list[SearchResult]:
            captured["url"] = url
            captured["query"] = query
            return [SearchResult(1, "ok", "https://example.com/", "")]

        with patch.object(
            web_search_service, "_load_search_config", return_value=self._config()
        ), patch.object(web_search_service, "_searxng_search", new=fake_searxng):
            results = search("hello")

        self.assertEqual(len(results), 1)
        self.assertEqual(captured["url"], "http://searxng:8080/search")
        self.assertEqual(captured["query"], "hello")

    def test_default_searxng_url_constant_matches_issue(self) -> None:
        self.assertEqual(DEFAULT_SEARXNG_URL, "http://searxng:8080/search")


class ConfigCoercionTests(unittest.TestCase):
    def test_default_search_backend_is_ddgs(self) -> None:
        self.assertEqual(DEFAULT_CONFIG["search_backend"], "ddgs")
        self.assertEqual(
            DEFAULT_CONFIG["search_backend_url"],
            "http://searxng:8080/search",
        )
        self.assertTrue(DEFAULT_CONFIG["search_backend_fallback"])
        self.assertEqual(DEFAULT_CONFIG["searxng_timeout_seconds"], 8.0)
        self.assertEqual(DEFAULT_CONFIG["ddgs_timeout_seconds"], 20.0)
        self.assertEqual(DEFAULT_CONFIG["ddgs_backend"], "auto")

    def test_normalize_config_defaults_to_ddgs_backend(self) -> None:
        config = normalize_config({})
        self.assertEqual(config["search_backend"], "ddgs")
        self.assertEqual(config["ddgs_timeout_seconds"], 20.0)
        self.assertEqual(config["ddgs_backend"], "auto")

    def test_normalize_config_accepts_ddgs_overrides(self) -> None:
        config = normalize_config({"ddgs_timeout_seconds": "5", "ddgs_backend": "brave"})
        self.assertEqual(config["ddgs_timeout_seconds"], 5.0)
        self.assertEqual(config["ddgs_backend"], "brave")

    def test_normalize_config_accepts_searxng_timeout_override(self) -> None:
        config = normalize_config({"searxng_timeout_seconds": "2.5"})
        self.assertEqual(config["searxng_timeout_seconds"], 2.5)

    def test_normalize_config_rejects_nonpositive_timeouts(self) -> None:
        for key in (
            "searxng_timeout_seconds",
            "browser_idle_shutdown_seconds",
            "browser_action_timeout_seconds",
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                normalize_config({key: 0})

    def test_ddgs_backend_is_allowed(self) -> None:
        config = normalize_config({"search_backend": "ddgs"})
        self.assertEqual(config["search_backend"], "ddgs")

    def test_invalid_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_config({"search_backend": "yandex"})

    def test_country_aliases_to_region(self) -> None:
        config = normalize_config({"search_country": "us-en"})
        self.assertEqual(config["search_region"], "us-en")
        self.assertNotIn("search_country", config)

    def test_engines_comma_string_is_normalized_to_list(self) -> None:
        config = normalize_config({"search_engines": "google, bing , duckduckgo"})
        self.assertEqual(config["search_engines"], ["google", "bing", "duckduckgo"])

    def test_environment_does_not_override_core_config_coercion(self) -> None:
        with patch.dict("os.environ", {"SEARXNG_URL": "http://example.test/search"}):
            config = normalize_config({"search_backend_url": "http://other:8080/"})
        self.assertEqual(config["search_backend_url"], "http://other:8080/")

    def test_duckduckgo_backend_preserves_existing_behavior(self) -> None:
        config = normalize_config({"search_backend": "duckduckgo"})
        self.assertEqual(config["search_backend"], "duckduckgo")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from tinysearch import core
from tinysearch.services.browser_session_service import SessionExpiredError
from tinysearch.services.scrape_service import EmptyContentError, FetchFailedError


def _fake_session(session_id: str = "sess-1", current_url: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        session_id=session_id,
        crawler=object(),
        current_url=current_url,
        touch=MagicMock(),
    )


def _fake_scrape_result(url: str = "https://example.com/x") -> SimpleNamespace:
    return SimpleNamespace(
        url=url,
        title="Title",
        query="*",
        chunks=[{"chunk_id": "0", "text": "evidence", "tokens": 1}],
        content_tokens=1,
        truncated=False,
        retrieved_at="2026-01-01T00:00:00Z",
        metadata={},
        links=[],
    )


class BrowseValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_requires_url_or_session_id(self) -> None:
        with self.assertRaises(ValueError):
            await core.browse()

    async def test_rejects_document_urls(self) -> None:
        with self.assertRaises(ValueError):
            await core.browse("https://example.com/report.pdf")

    async def test_rejects_too_many_actions(self) -> None:
        actions = [{"action": "wait", "seconds": 1}] * 9
        with self.assertRaises(ValueError):
            await core.browse("https://example.com", actions)

    async def test_rejects_non_positive_max_tokens(self) -> None:
        with self.assertRaises(ValueError):
            await core.browse("https://example.com", max_tokens=0)


class BrowseSessionPlumbingTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_session_opens_and_navigates(self) -> None:
        session = _fake_session()
        registry = SimpleNamespace(
            create=AsyncMock(return_value=session),
            get=AsyncMock(),
            close=AsyncMock(),
        )
        with patch("tinysearch.core.get_registry", return_value=registry), patch(
            "tinysearch.core._ensure_browser_bundle", new=AsyncMock()
        ), patch(
            "tinysearch.core.run_scrape_pipeline",
            new=AsyncMock(return_value=_fake_scrape_result()),
        ) as run_mock:
            result = await core.browse("https://example.com")

        registry.create.assert_awaited_once()
        registry.get.assert_not_awaited()
        crawl_fn = run_mock.await_args.kwargs["crawl_fn"]
        self.assertTrue(crawl_fn.keywords["navigate"])
        self.assertEqual(crawl_fn.keywords["crawl4ai_session_id"], "sess-1")
        self.assertIs(run_mock.await_args.kwargs["crawler"], session.crawler)
        self.assertEqual(result["operation"], "browse")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["stats"]["session_id"], "sess-1")
        self.assertEqual(result["stats"]["actions_executed"], 0)
        self.assertEqual(session.current_url, "https://example.com/x")
        session.touch.assert_called_once()
        json.dumps(result)

    async def test_existing_session_reuses_current_url_and_does_not_navigate(self) -> None:
        session = _fake_session(current_url="https://example.com/prior")
        registry = SimpleNamespace(
            create=AsyncMock(),
            get=AsyncMock(return_value=session),
            close=AsyncMock(),
        )
        actions = [{"action": "click", "selector": "#more"}]
        with patch("tinysearch.core.get_registry", return_value=registry), patch(
            "tinysearch.core._ensure_browser_bundle", new=AsyncMock()
        ), patch(
            "tinysearch.core.run_scrape_pipeline",
            new=AsyncMock(return_value=_fake_scrape_result("https://example.com/prior")),
        ) as run_mock:
            result = await core.browse(session_id="sess-1", actions=actions)

        registry.create.assert_not_awaited()
        registry.get.assert_awaited_once_with("sess-1", idle_seconds=unittest.mock.ANY)
        self.assertEqual(run_mock.await_args.args[0], "https://example.com/prior")
        crawl_fn = run_mock.await_args.kwargs["crawl_fn"]
        self.assertFalse(crawl_fn.keywords["navigate"])
        self.assertEqual(result["stats"]["actions_executed"], 1)

    async def test_expired_session_raises(self) -> None:
        registry = SimpleNamespace(
            create=AsyncMock(),
            get=AsyncMock(side_effect=SessionExpiredError("gone")),
            close=AsyncMock(),
        )
        with patch("tinysearch.core.get_registry", return_value=registry), patch(
            "tinysearch.core._ensure_browser_bundle", new=AsyncMock()
        ):
            with self.assertRaises(SessionExpiredError):
                await core.browse(session_id="sess-1")

    async def test_fetch_failure_closes_session_and_reraises(self) -> None:
        session = _fake_session()
        registry = SimpleNamespace(
            create=AsyncMock(return_value=session),
            get=AsyncMock(),
            close=AsyncMock(),
        )
        with patch("tinysearch.core.get_registry", return_value=registry), patch(
            "tinysearch.core._ensure_browser_bundle", new=AsyncMock()
        ), patch(
            "tinysearch.core.run_scrape_pipeline",
            new=AsyncMock(side_effect=FetchFailedError("boom")),
        ):
            with self.assertRaises(FetchFailedError):
                await core.browse("https://example.com")

        registry.close.assert_awaited_once_with("sess-1")

    async def test_empty_content_error_keeps_session_open(self) -> None:
        session = _fake_session()
        registry = SimpleNamespace(
            create=AsyncMock(return_value=session),
            get=AsyncMock(),
            close=AsyncMock(),
        )
        with patch("tinysearch.core.get_registry", return_value=registry), patch(
            "tinysearch.core._ensure_browser_bundle", new=AsyncMock()
        ), patch(
            "tinysearch.core.run_scrape_pipeline",
            new=AsyncMock(side_effect=EmptyContentError("empty")),
        ):
            with self.assertRaises(EmptyContentError):
                await core.browse("https://example.com")

        registry.close.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()

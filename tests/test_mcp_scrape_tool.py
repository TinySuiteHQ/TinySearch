from __future__ import annotations

import unittest
from inspect import signature
from unittest.mock import AsyncMock, patch

from tinysearch.servers.mcp_server import scrape_url_tool
from tinysearch.services.scrape_service import (
    EmptyContentError,
    FetchFailedError,
    FetchTimeoutError,
    UnsupportedDocumentError,
)
from tinysearch.services.url_safety_service import BlockedUrlError, InvalidUrlError
from tinysearch.results import result_envelope


def _result() -> dict:
    return result_envelope(
        operation="scrape",
        status="ok",
        query="q",
        retrieved_at="2026-06-12T10:30:00Z",
        sources=[{
            "id": "1",
            "url": "https://example.com/x",
            "title": "Title",
            "metadata": {},
            "chunks": [{"id": "1", "text": "Evidence.", "tokens": 2, "rank": 1, "scores": {}}],
        }],
        stats={"content_tokens": 42, "truncated": False},
    )


def _fn(coro):
    """unwrap a FastMCP-decorated tool to call the underlying coroutine."""
    return getattr(coro, "fn", coro)


class ScrapeUrlToolTests(unittest.IsolatedAsyncioTestCase):
    def test_mcp_signature_exposes_output_format(self) -> None:
        self.assertEqual(
            list(signature(_fn(scrape_url_tool)).parameters),
            ["url", "query", "output_format"],
        )

    async def test_returns_answer_and_diagnostics(self) -> None:
        scrape_mock = AsyncMock(return_value=_result())
        with patch("tinysearch.core.scrape_url", scrape_mock):
            payload = await _fn(scrape_url_tool)(
                "https://example.com/x", "q"
            )

        self.assertIn("URL-GROUNDED ANSWER PROMPT", payload["answer"])
        self.assertEqual(payload["url"], "https://example.com/x")
        self.assertEqual(payload["title"], "Title")
        self.assertEqual(payload["content_tokens"], 42)
        self.assertGreater(payload["answer_tokens"], 0)
        self.assertFalse(payload["truncated"])
        self.assertEqual(payload["retrieved_at"], "2026-06-12T10:30:00Z")
        self.assertNotIn("metadata", payload)

    async def test_json_output_returns_structured_result(self) -> None:
        with patch("tinysearch.core.scrape_url", AsyncMock(return_value=_result())):
            payload = await _fn(scrape_url_tool)(
                "https://example.com/x", "q", "json"
            )
        self.assertEqual(payload["schema_version"], "1")

    async def test_default_max_tokens_passed_through(self) -> None:
        scrape_mock = AsyncMock(return_value=_result())
        with patch("tinysearch.core.scrape_url", scrape_mock):
            await _fn(scrape_url_tool)("https://example.com/x", "q")

        self.assertEqual(scrape_mock.await_args.kwargs["max_tokens"], 4000)

    async def _run_with_exc(self, exc: Exception) -> ValueError:
        scrape_mock = AsyncMock(side_effect=exc)
        with patch("tinysearch.core.scrape_url", scrape_mock):
            try:
                await _fn(scrape_url_tool)("https://example.com/x", "q")
            except ValueError as raised:
                return raised
        self.fail("expected ValueError")

    async def test_invalid_url_re_raises_with_code_prefix(self) -> None:
        raised = await self._run_with_exc(InvalidUrlError("bad scheme"))
        self.assertIn("invalid_url:", str(raised))

    async def test_blocked_url_re_raises_with_code_prefix(self) -> None:
        raised = await self._run_with_exc(BlockedUrlError("blocked"))
        self.assertIn("blocked_url:", str(raised))

    async def test_fetch_timeout_re_raises_with_code_prefix(self) -> None:
        raised = await self._run_with_exc(FetchTimeoutError("slow"))
        self.assertIn("fetch_timeout:", str(raised))

    async def test_fetch_failed_re_raises_with_code_prefix(self) -> None:
        raised = await self._run_with_exc(FetchFailedError("dead"))
        self.assertIn("fetch_failed:", str(raised))

    async def test_unsupported_document_re_raises_with_code_prefix(self) -> None:
        raised = await self._run_with_exc(UnsupportedDocumentError(".doc"))
        self.assertIn("unsupported_document:", str(raised))

    async def test_empty_content_re_raises_with_code_prefix(self) -> None:
        raised = await self._run_with_exc(EmptyContentError("empty"))
        self.assertIn("empty_content:", str(raised))


if __name__ == "__main__":
    unittest.main()

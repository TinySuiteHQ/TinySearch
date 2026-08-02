from __future__ import annotations

import unittest
from inspect import signature
from unittest.mock import AsyncMock, patch

from mcp import types

from tinysearch.servers.mcp_server import mcp, research, scrape_url_tool
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
    def test_research_signature_only_exposes_query(self) -> None:
        self.assertEqual(list(signature(_fn(research)).parameters), ["query"])

    def test_mcp_signature_only_exposes_url_and_query(self) -> None:
        self.assertEqual(
            list(signature(_fn(scrape_url_tool)).parameters),
            ["url", "query"],
        )

    async def test_returns_xml_prompt_directly_with_diagnostics(self) -> None:
        scrape_mock = AsyncMock(return_value=_result())
        with patch("tinysearch.core.scrape_url", scrape_mock):
            prompt = await _fn(scrape_url_tool)(
                "https://example.com/x", "q"
            )

        self.assertIsInstance(prompt, str)
        self.assertIn("<url_grounded_answer", prompt)
        self.assertIn('retrieved_at="2026-06-12T10:30:00Z"', prompt)
        self.assertIn('truncated="false"', prompt)
        self.assertIn('content_tokens="42"', prompt)
        self.assertIn("<url>\nhttps://example.com/x\n</url>", prompt)

    async def test_mcp_wire_text_is_xml_not_a_json_answer_envelope(self) -> None:
        with patch("tinysearch.core.scrape_url", AsyncMock(return_value=_result())):
            content, structured = await mcp._tool_manager.call_tool(
                "scrape_url",
                {"url": "https://example.com/x", "query": "q"},
                convert_result=True,
            )

        self.assertEqual(len(content), 1)
        self.assertIsInstance(content[0], types.TextContent)
        self.assertTrue(content[0].text.startswith("<url_grounded_answer"))
        self.assertFalse(content[0].text.startswith("{"))
        self.assertEqual(structured, {"result": content[0].text})

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

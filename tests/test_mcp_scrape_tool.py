from __future__ import annotations

import unittest
from inspect import signature
from unittest.mock import AsyncMock, patch
from xml.etree import ElementTree

from mcp import types

from tinysearch.servers.mcp_server import mcp, scrape_urls_tool
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


class ScrapeUrlsToolTests(unittest.IsolatedAsyncioTestCase):
    def test_mcp_exposes_only_batch_scrape_tool(self) -> None:
        self.assertEqual(
            {name for name in mcp._tool_manager._tools if not name.startswith("browser_")},
            {"get_current_datetime", "search", "scrape_urls"},
        )
        self.assertEqual(
            {name for name in mcp._tool_manager._tools if name.startswith("browser_")},
            {"browser_navigate", "browser_act"},
        )
        self.assertEqual(list(signature(_fn(scrape_urls_tool)).parameters), ["items"])

    async def test_batch_tool_preserves_omitted_and_focused_queries(self) -> None:
        batch_mock = AsyncMock(return_value={"operation": "scrape_batch", "results": []})
        config = {"scrape_max_tokens": 2000}
        with patch("tinysearch.core.scrape_urls", batch_mock), patch(
            "tinysearch.servers.mcp_server.load_tinysearch_config", return_value=config
        ):
            result = await _fn(scrape_urls_tool)(
                [{"url": "https://one.example"}, {"url": "https://two.example", "query": "pricing"}]
            )
        self.assertTrue(result.startswith("<url_grounded_answers "))
        self.assertNotIn('"operation"', result)
        self.assertEqual(batch_mock.await_args.args[0][0], {"url": "https://one.example"})
        self.assertEqual(batch_mock.await_args.kwargs["max_tokens"], 2000)
        self.assertIs(batch_mock.await_args.kwargs["config"], config)

    async def test_mcp_batch_wire_text_is_well_formed_xml(self) -> None:
        batch = {"operation": "scrape_batch", "results": [{"status": "ok", "result": _result()}]}
        with patch("tinysearch.core.scrape_urls", AsyncMock(return_value=batch)):
            content, structured = await mcp._tool_manager.call_tool(
                "scrape_urls", {"items": [{"url": "https://example.com/x", "query": "q"}]},
                convert_result=True,
            )

        self.assertEqual(len(content), 1)
        self.assertIsInstance(content[0], types.TextContent)
        root = ElementTree.fromstring(content[0].text)
        self.assertEqual(root.tag, "url_grounded_answers")
        self.assertEqual(root.findtext("pages/page/title", default="").strip(), "Title")
        self.assertEqual(structured, {"result": content[0].text})

    async def test_configured_max_tokens_passed_through(self) -> None:
        scrape_mock = AsyncMock(return_value={"operation": "scrape_batch", "results": []})
        config = {"scrape_max_tokens": 2000}
        with patch("tinysearch.core.scrape_urls", scrape_mock), patch(
            "tinysearch.servers.mcp_server.load_tinysearch_config", return_value=config
        ):
            await _fn(scrape_urls_tool)([{"url": "https://example.com/x", "query": "q"}])

        self.assertEqual(scrape_mock.await_args.kwargs["max_tokens"], 2000)
        self.assertIs(scrape_mock.await_args.kwargs["config"], config)


if __name__ == "__main__":
    unittest.main()

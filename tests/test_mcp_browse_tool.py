from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch
from xml.etree import ElementTree

from mcp import types

from tinysearch.results import result_envelope
from tinysearch.servers.mcp_server import browse_tool, mcp


def _fn(coro):
    """unwrap a FastMCP-decorated tool to call the underlying coroutine."""
    return getattr(coro, "fn", coro)


def _browse_result() -> dict:
    return result_envelope(
        operation="browse",
        status="ok",
        query="*",
        retrieved_at="2026-06-12T10:30:00Z",
        sources=[{
            "id": "1",
            "url": "https://example.com/x",
            "title": "Title",
            "metadata": {},
            "chunks": [{"id": "1", "text": "Evidence.", "tokens": 2, "rank": 1, "scores": {}}],
            "links": [],
        }],
        stats={
            "content_tokens": 42,
            "truncated": False,
            "session_id": "sess-1",
            "session_expires_in_seconds": 300,
            "actions_executed": 0,
        },
    )


class BrowseToolTests(unittest.IsolatedAsyncioTestCase):
    def test_mcp_exposes_browse_tool(self) -> None:
        self.assertIn("browse", mcp._tool_manager._tools)

    async def test_passes_url_actions_query_and_session_id_through(self) -> None:
        browse_mock = AsyncMock(return_value=_browse_result())
        config = {"scrape_max_tokens": 2000}
        actions = [{"action": "click", "selector": "#accept"}]
        with patch("tinysearch.core.browse", browse_mock), patch(
            "tinysearch.servers.mcp_server.load_tinysearch_config", return_value=config
        ):
            result = await _fn(browse_tool)(
                url="https://example.com",
                actions=actions,
                query="pricing",
                session_id="sess-1",
            )

        self.assertTrue(result.startswith("<browse_result "))
        self.assertIn('session_id="sess-1"', result)
        self.assertEqual(browse_mock.await_args.args, ("https://example.com", actions))
        self.assertEqual(browse_mock.await_args.kwargs["query"], "pricing")
        self.assertEqual(browse_mock.await_args.kwargs["session_id"], "sess-1")
        self.assertEqual(browse_mock.await_args.kwargs["max_tokens"], 2000)

    async def test_mcp_wire_text_is_well_formed_xml_and_surfaces_session_id(self) -> None:
        with patch("tinysearch.core.browse", AsyncMock(return_value=_browse_result())):
            content, structured = await mcp._tool_manager.call_tool(
                "browse", {"url": "https://example.com/x"}, convert_result=True,
            )

        self.assertEqual(len(content), 1)
        self.assertIsInstance(content[0], types.TextContent)
        root = ElementTree.fromstring(content[0].text)
        self.assertEqual(root.tag, "browse_result")
        self.assertEqual(root.get("session_id"), "sess-1")
        self.assertEqual(root.findtext("page/title", default="").strip(), "Title")
        self.assertEqual(structured, {"result": content[0].text})


if __name__ == "__main__":
    unittest.main()

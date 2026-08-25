from __future__ import annotations

from inspect import signature
import json
import unittest
from unittest.mock import AsyncMock, patch
from xml.etree import ElementTree

import tinysearch
from fastapi import HTTPException
from mcp import types
from tinysearch.prompts import to_prompt
from tinysearch.servers.fastapi_server import SearchRequest, search_endpoint
from tinysearch.servers.mcp_server import mcp, search_tool
from tinysearch.services.web_search_service import SearchResponse, SearchResult
from tinysearch.services.web_search_service import SearchBackendUnavailable


def _fn(coro):
    return getattr(coro, "fn", coro)


def _payload() -> dict:
    return {
        "schema_version": "1",
        "operation": "search_batch",
        "status": "ok",
        "items": [{
            "query": "python async", "domains": [], "status": "ok",
            "results": [{"rank": 1, "title": "Async tasks", "url": "https://example.com/async", "preview": "Coroutines and tasks.", "published_at": "2026-08-01T12:00:00+00:00"}],
            "backend_attempts": [{"backend": "searxng", "state": "responded", "result_count": 1}],
            "error": None, "stats": {"result_count": 1, "latency_ms": 1},
        }],
        "errors": [],
        "stats": {"search_item_count": 1, "backend_attempt_count": 1, "latency_ms": 1},
    }


class FastSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_python_search_is_raw_and_does_not_initialize_heavy_dependencies(self) -> None:
        response = [
            type("Response", (), {"domains": [], "status": "ok", "results": [SearchResult(1, "First", "https://first.example", "One", "2026-01-01")], "attempts": [], "error": None, "latency_ms": 1})()
        ]
        with patch("tinysearch.core.search_batch_with_metadata", new=AsyncMock(return_value=response)) as search, patch(
            "tinysearch.core._ensure_local_bundle_for_config", new=AsyncMock()
        ) as embeddings, patch("tinysearch.core._ensure_browser_bundle", new=AsyncMock()) as browser:
            result = await tinysearch.search([{"query": "discovery"}], config={"blocked_domains": ["blocked.example"]})

        self.assertEqual(search.await_args.kwargs["limit"], 10)
        self.assertEqual(result["items"][0]["results"][0]["title"], "First")
        self.assertEqual(result["items"][0]["results"][0]["published_at"], "2026-01-01")
        self.assertNotIn("retrieved_at", result)
        embeddings.assert_not_awaited()
        browser.assert_not_awaited()

    async def test_python_search_rejects_more_than_five_items(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 5"):
            await tinysearch.search([{"query": "q"}] * 6)

    async def test_fastapi_json_and_text_formats(self) -> None:
        with patch("tinysearch.servers.fastapi_server.core.search", new=AsyncMock(return_value=_payload())):
            json_result = await search_endpoint(SearchRequest(items=[{"query": "q"}]))
        self.assertEqual(json_result["items"][0]["status"], "ok")

    async def test_mcp_returns_xml(self) -> None:
        config = {"search_max_results": 7}
        with patch(
            "tinysearch.servers.mcp_server.core.search", new=AsyncMock(return_value=_payload())
        ) as search, patch("tinysearch.servers.mcp_server.load_tinysearch_config", return_value=config):
            answer = await _fn(search_tool)([{"query": "q"}])
        root = ElementTree.fromstring(answer)
        self.assertEqual(root.find("./item/query").text.strip(), "python async")
        self.assertNotIn("backend_attempts", answer)
        self.assertEqual(search.await_args.args[0], [{"query": "q"}])
        self.assertIs(search.await_args.kwargs["config"], config)

    async def test_mcp_search_wire_text_is_well_formed_xml(self) -> None:
        with patch(
            "tinysearch.servers.mcp_server.core.search", new=AsyncMock(return_value=_payload())
        ):
            content, structured = await mcp._tool_manager.call_tool(
            "search", {"items": [{"query": "q"}]}, convert_result=True
            )

        self.assertEqual(len(content), 1)
        self.assertIsInstance(content[0], types.TextContent)
        ElementTree.fromstring(content[0].text)
        self.assertIn("<search_results>", content[0].text)

    def test_mcp_search_exposes_only_items(self) -> None:
        self.assertEqual(list(signature(_fn(search_tool)).parameters), ["items"])

    def test_batch_payload_is_json_serializable(self) -> None:
        self.assertEqual(json.loads(json.dumps(_payload()))["operation"], "search_batch")

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
        "operation": "search",
        "status": "ok",
        "query": "python async",
        "backend": "searxng",
        "results": [{
            "rank": 1,
            "title": "Async tasks",
            "url": "https://example.com/async",
            "preview": "Coroutines and tasks.",
            "published_at": "2026-08-01T12:00:00+00:00",
        }],
        "errors": [],
        "stats": {"result_count": 1},
    }


class FastSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_python_search_is_raw_and_does_not_initialize_heavy_dependencies(self) -> None:
        response = SearchResponse(
            [
                SearchResult(1, "First", "https://first.example", "One", "2026-01-01"),
                SearchResult(2, "Blocked", "https://blocked.example", "Two"),
                SearchResult(3, "Third", "https://third.example", "Three"),
            ],
            "searxng",
        )
        with patch("tinysearch.core.search_with_metadata", return_value=response) as search, patch(
            "tinysearch.core._ensure_local_bundle_for_config", new=AsyncMock()
        ) as embeddings, patch("tinysearch.core._ensure_browser_bundle", new=AsyncMock()) as browser:
            result = await tinysearch.search(
                "  discovery  ", config={"blocked_domains": ["blocked.example"]}
            )

        self.assertEqual(search.call_args.args[:2], ("discovery", 10))
        self.assertEqual([item["title"] for item in result["results"]], ["First", "Third"])
        self.assertEqual(result["results"][0]["published_at"], "2026-01-01")
        self.assertNotIn("retrieved_at", result)
        embeddings.assert_not_awaited()
        browser.assert_not_awaited()

    async def test_python_search_rejects_out_of_range_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 50"):
            await tinysearch.search("q", limit=51)

    async def test_fastapi_json_and_text_formats(self) -> None:
        with patch("tinysearch.servers.fastapi_server.core.search", new=AsyncMock(return_value=_payload())):
            json_result = await search_endpoint(SearchRequest(query="q", output_format="json"))
            text_result = await search_endpoint(SearchRequest(query="q", output_format="prompt"))
        self.assertEqual(json_result["backend"], "searxng")
        self.assertIn("Published: 2026-08-01T12:00:00+00:00", text_result["answer"])

    async def test_mcp_returns_xml(self) -> None:
        config = {"search_max_results": 7}
        with patch(
            "tinysearch.servers.mcp_server.core.search", new=AsyncMock(return_value=_payload())
        ) as search, patch("tinysearch.servers.mcp_server.load_tinysearch_config", return_value=config):
            answer = await _fn(search_tool)("  q  ")
        self.assertTrue(answer.startswith("<search_results>"))
        self.assertEqual(ElementTree.fromstring(answer).findtext("query", default="").strip(), "python async")
        self.assertIn("<title>\nAsync tasks\n</title>", answer)
        self.assertIn("<search_preview>\nCoroutines and tasks.\n</search_preview>", answer)
        self.assertEqual(search.await_args.kwargs["limit"], 7)
        self.assertIs(search.await_args.kwargs["config"], config)

    async def test_mcp_search_wire_text_is_well_formed_xml(self) -> None:
        with patch(
            "tinysearch.servers.mcp_server.core.search", new=AsyncMock(return_value=_payload())
        ):
            content, structured = await mcp._tool_manager.call_tool(
                "search", {"query": "q"}, convert_result=True
            )

        self.assertEqual(len(content), 1)
        self.assertIsInstance(content[0], types.TextContent)
        root = ElementTree.fromstring(content[0].text)
        self.assertEqual(root.tag, "search_results")
        self.assertEqual(structured, {"result": content[0].text})

    def test_mcp_search_exposes_only_query(self) -> None:
        self.assertEqual(list(signature(_fn(search_tool)).parameters), ["query"])

    async def test_fastapi_maps_backend_failure_to_bad_gateway(self) -> None:
        with patch(
            "tinysearch.servers.fastapi_server.core.search",
            new=AsyncMock(side_effect=SearchBackendUnavailable("down")),
        ):
            with self.assertRaises(HTTPException) as raised:
                await search_endpoint(SearchRequest(query="q"))
        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.detail["code"], "search_backend_error")

    def test_plain_renderer_never_uses_grounded_xml(self) -> None:
        rendered = to_prompt(_payload())
        self.assertNotIn("<search_grounded_answer>", rendered)
        self.assertIn("URL: https://example.com/async", rendered)
        self.assertEqual(json.loads(json.dumps(_payload()))["operation"], "search")

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from tinysearch.servers.fastapi_server import ScrapeRequest, scrape_endpoint
from tinysearch.servers.mcp_server import scrape_url_tool
from tinysearch.results import result_envelope


def _fn(coro):
    return getattr(coro, "fn", coro)


def _shared_result() -> dict:
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
            "chunks": [{"id": "1", "text": "Shared.", "tokens": 1, "rank": 1, "scores": {}}],
        }],
        stats={"content_tokens": 10, "truncated": False},
    )


class ScrapeFastApiMcpParityTests(unittest.IsolatedAsyncioTestCase):
    """Both adapters delegate to the same `tinysearch.core.scrape_url`, so
    parity is structural rather than something each adapter must separately
    get right — these tests guard against that delegation drifting apart."""

    async def test_both_adapters_return_identical_answer(self) -> None:
        with patch(
            "tinysearch.core.scrape_url", AsyncMock(return_value=_shared_result())
        ):
            fastapi_payload = await scrape_endpoint(
                ScrapeRequest(url="https://example.com/x", query="q")
            )
            mcp_payload = await _fn(scrape_url_tool)("https://example.com/x", "q")

        self.assertEqual(fastapi_payload, mcp_payload)

    async def test_both_adapters_pass_the_same_url_and_query_to_core(self) -> None:
        core_mock = AsyncMock(return_value=_shared_result())
        with patch("tinysearch.core.scrape_url", core_mock):
            await scrape_endpoint(ScrapeRequest(url="https://example.com/x", query="q"))
            fastapi_args = core_mock.await_args.args

            await _fn(scrape_url_tool)("https://example.com/x", "q")
            mcp_args = core_mock.await_args.args

        self.assertEqual(fastapi_args, mcp_args)


if __name__ == "__main__":
    unittest.main()

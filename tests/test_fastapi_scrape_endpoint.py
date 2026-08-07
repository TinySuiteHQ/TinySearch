from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from tinysearch.servers.fastapi_server import (
    ScrapeBatchRequest,
    ScrapeRequest,
    scrape_batch_endpoint,
    scrape_endpoint,
)
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


class ScrapeRequestValidationTests(unittest.TestCase):
    def test_empty_or_omitted_query_selects_page_order_mode(self) -> None:
        self.assertIsNone(ScrapeRequest(url="https://example.com/x").query)
        self.assertEqual(ScrapeRequest(url="https://example.com/x", query="").query, "")

    def test_rejects_non_http_scheme(self) -> None:
        with self.assertRaises(ValidationError):
            ScrapeRequest(url="ftp://example.com/x", query="q")


class ScrapeEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_passes_independent_optional_queries_to_core(self) -> None:
        batch_mock = AsyncMock(return_value={"operation": "scrape_batch", "status": "ok", "results": []})
        with patch("tinysearch.core.scrape_urls", batch_mock):
            payload = await scrape_batch_endpoint(
                ScrapeBatchRequest(
                    items=[
                        {"url": "https://one.example"},
                        {"url": "https://two.example", "query": "find pricing"},
                    ]
                )
            )
        self.assertEqual(payload["operation"], "scrape_batch")
        self.assertEqual(batch_mock.await_args.args[0][0]["query"], None)
        self.assertEqual(batch_mock.await_args.args[0][1]["query"], "find pricing")

    async def test_returns_mcp_aligned_payload(self) -> None:
        scrape_mock = AsyncMock(return_value=_result())
        with patch("tinysearch.core.scrape_url", scrape_mock):
            payload = await scrape_endpoint(
                ScrapeRequest(url="https://example.com/x", query="q")
            )

        self.assertIn("<url_grounded_answer", payload["answer"])
        self.assertEqual(payload["url"], "https://example.com/x")
        self.assertEqual(payload["title"], "Title")
        self.assertEqual(payload["content_tokens"], 42)
        self.assertGreater(payload["answer_tokens"], 0)
        self.assertFalse(payload["truncated"])
        self.assertEqual(payload["retrieved_at"], "2026-06-12T10:30:00Z")
        self.assertNotIn("query", payload)
        self.assertNotIn("metadata", payload)

    async def test_json_output_returns_structured_result(self) -> None:
        with patch("tinysearch.core.scrape_url", AsyncMock(return_value=_result())):
            payload = await scrape_endpoint(
                ScrapeRequest(
                    url="https://example.com/x",
                    query="q",
                    output_format="json",
                )
            )
        self.assertEqual(payload["schema_version"], "1")
        self.assertEqual(payload["operation"], "scrape")


class ScrapeEndpointErrorMappingTests(unittest.IsolatedAsyncioTestCase):
    async def _run_with_exc(self, exc: Exception) -> HTTPException:
        scrape_mock = AsyncMock(side_effect=exc)
        with patch("tinysearch.core.scrape_url", scrape_mock):
            try:
                await scrape_endpoint(
                    ScrapeRequest(url="https://example.com/x", query="q")
                )
            except HTTPException as raised:
                return raised
        self.fail("expected HTTPException")

    async def test_invalid_url_maps_to_400(self) -> None:
        raised = await self._run_with_exc(InvalidUrlError("bad"))
        self.assertEqual(raised.status_code, 400)
        self.assertEqual(raised.detail["code"], "invalid_url")

    async def test_blocked_url_maps_to_403(self) -> None:
        raised = await self._run_with_exc(BlockedUrlError("nope"))
        self.assertEqual(raised.status_code, 403)
        self.assertEqual(raised.detail["code"], "blocked_url")

    async def test_fetch_timeout_maps_to_504(self) -> None:
        raised = await self._run_with_exc(FetchTimeoutError("slow"))
        self.assertEqual(raised.status_code, 504)
        self.assertEqual(raised.detail["code"], "fetch_timeout")

    async def test_fetch_failed_maps_to_502(self) -> None:
        raised = await self._run_with_exc(FetchFailedError("dead"))
        self.assertEqual(raised.status_code, 502)
        self.assertEqual(raised.detail["code"], "fetch_failed")

    async def test_unsupported_document_maps_to_415(self) -> None:
        raised = await self._run_with_exc(UnsupportedDocumentError(".doc"))
        self.assertEqual(raised.status_code, 415)
        self.assertEqual(raised.detail["code"], "unsupported_document")

    async def test_empty_content_maps_to_422(self) -> None:
        raised = await self._run_with_exc(EmptyContentError("nothing"))
        self.assertEqual(raised.status_code, 422)
        self.assertEqual(raised.detail["code"], "empty_content")


if __name__ == "__main__":
    unittest.main()

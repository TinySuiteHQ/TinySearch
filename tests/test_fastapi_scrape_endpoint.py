from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from tinysearch.config import TinySearchConfig

from tinysearch.servers.fastapi_server import (
    ScrapeBatchRequest,
    app,
    scrape_endpoint,
)


class ScrapeUrlsRequestValidationTests(unittest.TestCase):
    def test_empty_or_omitted_query_selects_page_order_mode(self) -> None:
        request = ScrapeBatchRequest(items=[{"url": "https://example.com/x"}])
        self.assertIsNone(request.items[0].query)
        self.assertEqual(
            ScrapeBatchRequest(items=[{"url": "https://example.com/x", "query": ""}]).items[0].query,
            "",
        )

    def test_rejects_non_http_scheme(self) -> None:
        with self.assertRaises(ValidationError):
            ScrapeBatchRequest(items=[{"url": "ftp://example.com/x", "query": "q"}])

    def test_requires_at_least_one_item_and_leaves_the_cap_to_config(self) -> None:
        with self.assertRaises(ValidationError):
            ScrapeBatchRequest(items=[])
        request = ScrapeBatchRequest(items=[{"url": f"https://{index}.example"} for index in range(6)])
        self.assertEqual(len(request.items), 6)

    def test_rejects_removed_output_format(self) -> None:
        with self.assertRaises(ValidationError):
            ScrapeBatchRequest(
                items=[{"url": "https://example.com"}],
                output_format="prompt",
            )

    def test_exposes_only_consolidated_scrape_route(self) -> None:
        paths = {route.path for route in app.routes}
        self.assertIn("/scrape", paths)
        self.assertNotIn("/scrape/batch", paths)


class ScrapeEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_more_items_than_scrape_max_items(self) -> None:
        batch_mock = AsyncMock()
        request = ScrapeBatchRequest(items=[{"url": f"https://{index}.example"} for index in range(6)])
        with patch("tinysearch.core.scrape_urls", batch_mock), patch(
            "tinysearch.servers.fastapi_server.load_tinysearch_config",
            return_value=TinySearchConfig(),
        ):
            with self.assertRaises(HTTPException) as raised:
                await scrape_endpoint(request)
        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail["code"], "too_many_items")
        batch_mock.assert_not_awaited()

    async def test_accepts_a_larger_batch_when_scrape_max_items_allows_it(self) -> None:
        batch_mock = AsyncMock(return_value={"operation": "scrape_batch", "status": "ok", "results": []})
        request = ScrapeBatchRequest(items=[{"url": f"https://{index}.example"} for index in range(6)])
        with patch("tinysearch.core.scrape_urls", batch_mock), patch(
            "tinysearch.servers.fastapi_server.load_tinysearch_config",
            return_value=TinySearchConfig(scrape_max_items=8),
        ), patch("tinysearch.servers.fastapi_server.record_quietly"):
            await scrape_endpoint(request)
        self.assertEqual(len(batch_mock.await_args.args[0]), 6)

    async def test_batch_passes_independent_optional_queries_to_core(self) -> None:
        batch_mock = AsyncMock(return_value={"operation": "scrape_batch", "status": "ok", "results": []})
        crawler_session = object()
        with patch("tinysearch.core.scrape_urls", batch_mock), patch(
            "tinysearch.servers.fastapi_server._crawler_session", crawler_session
        ):
            payload = await scrape_endpoint(
                ScrapeBatchRequest(
                    items=[
                        {"url": "https://one.example"},
                        {"url": "https://two.example", "query": "find pricing"},
                    ],
                    max_tokens=321,
                )
            )
        self.assertEqual(payload["operation"], "scrape_batch")
        self.assertEqual(batch_mock.await_args.args[0][0]["query"], None)
        self.assertEqual(batch_mock.await_args.args[0][1]["query"], "find pricing")
        self.assertEqual(batch_mock.await_args.kwargs["max_tokens"], 321)
        self.assertIs(
            batch_mock.await_args.kwargs["crawler_session"], crawler_session
        )


if __name__ == "__main__":
    unittest.main()

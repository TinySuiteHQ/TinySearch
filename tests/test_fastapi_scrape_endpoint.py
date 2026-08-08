from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

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

    def test_limits_items_to_one_through_five(self) -> None:
        with self.assertRaises(ValidationError):
            ScrapeBatchRequest(items=[])
        with self.assertRaises(ValidationError):
            ScrapeBatchRequest(items=[{"url": f"https://{index}.example"} for index in range(6)])

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
    async def test_batch_passes_independent_optional_queries_to_core(self) -> None:
        batch_mock = AsyncMock(return_value={"operation": "scrape_batch", "status": "ok", "results": []})
        with patch("tinysearch.core.scrape_urls", batch_mock):
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


if __name__ == "__main__":
    unittest.main()

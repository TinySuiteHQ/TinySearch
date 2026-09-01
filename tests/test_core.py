from __future__ import annotations

import contextlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import tinysearch
from tinysearch.config import TinySearchConfig
from tinysearch.core import (
    _ensure_browser_bundle,
    _ensure_local_bundle_for_config,
    scrape_urls,
)
from tinysearch.results import result_envelope


def _result_payload(query: str = "hello") -> dict:
    return result_envelope(
        operation="scrape",
        status="ok",
        query=query,
        sources=[],
    )


class CorePublicApiTests(unittest.IsolatedAsyncioTestCase):
    def test_public_api_exports_config_and_prompt_renderer(self) -> None:
        self.assertIs(tinysearch.get_current_datetime, tinysearch.core.get_current_datetime)
        self.assertTrue(callable(tinysearch.to_prompt))
        self.assertEqual(TinySearchConfig()["search_backend"], "ddgs")

    async def test_scrape_urls_returns_single_item_in_common_batch_shape(self) -> None:
        scrape_result = SimpleNamespace(
            url="https://example.com/x",
            title="Title",
            query="q",
            chunks=[{"chunk_id": "0", "text": "evidence", "tokens": 1}],
            content_tokens=1,
            truncated=False,
            retrieved_at="2026-01-01T00:00:00Z",
            metadata={"author": "A"},
            links=[],
        )
        with patch(
            "tinysearch.core.run_scrape_pipeline",
            new=AsyncMock(return_value=scrape_result),
        ), patch("tinysearch.core._ensure_local_bundle_for_config", new=AsyncMock()), patch(
            "tinysearch.core._ensure_browser_bundle", new=AsyncMock()
        ), patch(
            "tinysearch.core.create_browser_crawler",
            return_value=contextlib.nullcontext(None),
        ):
            result = await tinysearch.scrape_urls([{"url": "https://example.com/x", "query": "q"}])

        self.assertEqual(result["operation"], "scrape_batch")
        self.assertEqual(result["results"][0]["status"], "ok")
        self.assertEqual(result["results"][0]["result"]["sources"][0]["chunks"][0]["text"], "evidence")
        self.assertEqual(result["results"][0]["result"]["stats"]["content_tokens"], 1)
        json.dumps(result)

    def test_public_api_does_not_export_single_url_scraper(self) -> None:
        self.assertFalse(hasattr(tinysearch, "scrape_url"))

    async def test_scrape_urls_returns_independent_partial_outcomes(self) -> None:
        successful = _result_payload("*")
        successful["operation"] = "scrape"
        with patch(
            "tinysearch.core._scrape_url_with_config",
            side_effect=[successful, ValueError("bad URL")],
        ), patch("tinysearch.core._ensure_browser_bundle", new=AsyncMock()), patch(
            "tinysearch.core._ensure_local_bundle_for_config", new=AsyncMock()
        ) as embeddings, patch(
            "tinysearch.core.create_browser_crawler",
            return_value=contextlib.nullcontext(None),
        ):
            result = await scrape_urls(
                [{"url": "https://one.example"}, {"url": "https://two.example", "query": "*"}]
            )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["results"][0]["status"], "ok")
        self.assertEqual(result["results"][1]["error"]["message"], "bad URL")
        embeddings.assert_not_awaited()

    async def test_ensure_local_bundle_skips_non_onnx_backend(self) -> None:
        with patch(
            "tinysearch.services.onnx_bundle_service.ensure_onnx_bundle_sync"
        ) as ensure:
            await _ensure_local_bundle_for_config(
                {"embedding_backend": "openai_compatible", "embedding_model": "x"}
            )
        ensure.assert_not_called()

    async def test_ensure_local_bundle_downloads_for_onnx_backend(self) -> None:
        with patch(
            "tinysearch.services.onnx_bundle_service.ensure_onnx_bundle_sync"
        ) as ensure:
            await _ensure_local_bundle_for_config(
                {"embedding_backend": "onnx", "embedding_model": "fast"}
            )
        ensure.assert_called_once_with("fast")

    async def test_external_cdp_skips_bundled_chromium_install(self) -> None:
        with patch(
            "tinysearch.services.browser_bundle_service.ensure_chromium_sync"
        ) as ensure:
            await _ensure_browser_bundle({"browser_cdp_url": "http://browser:9222"})

        ensure.assert_not_called()

    async def test_default_browser_ensures_bundled_chromium(self) -> None:
        with patch(
            "tinysearch.services.browser_bundle_service.ensure_chromium_sync"
        ) as ensure:
            await _ensure_browser_bundle({"browser_cdp_url": ""})

        ensure.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

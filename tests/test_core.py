from __future__ import annotations

import asyncio
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

    async def test_scrape_urls_runs_at_most_scrape_max_concurrent_items_at_once(self) -> None:
        active = 0
        peak = 0

        async def scrape(url, query, **_kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return _result_payload(url)

        with patch("tinysearch.core._scrape_url_with_config", side_effect=scrape), patch(
            "tinysearch.core._ensure_browser_bundle", new=AsyncMock()
        ), patch(
            "tinysearch.core._ensure_local_bundle_for_config", new=AsyncMock()
        ), patch(
            "tinysearch.core.create_browser_crawler",
            return_value=contextlib.nullcontext(None),
        ):
            result = await scrape_urls(
                [{"url": f"https://{n}.example"} for n in range(5)],
                config={"scrape_max_concurrent_items": 2},
            )

        self.assertEqual(peak, 2)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            [item["result"]["query"] for item in result["results"]],
            [f"https://{n}.example" for n in range(5)],
        )

    async def test_scrape_urls_enforces_scrape_max_items(self) -> None:
        scrape = AsyncMock(return_value=_result_payload("*"))
        six = [{"url": f"https://{n}.example"} for n in range(6)]
        with patch("tinysearch.core._scrape_url_with_config", scrape), patch(
            "tinysearch.core._ensure_browser_bundle", new=AsyncMock()
        ), patch(
            "tinysearch.core._ensure_local_bundle_for_config", new=AsyncMock()
        ), patch(
            "tinysearch.core.create_browser_crawler",
            return_value=contextlib.nullcontext(None),
        ):
            with self.assertRaisesRegex(ValueError, "between 1 and 5"):
                await scrape_urls(six)
            result = await scrape_urls(six, config={"scrape_max_items": 6})

        self.assertEqual(len(result["results"]), 6)

    def test_scrape_max_items_must_be_positive(self) -> None:
        self.assertEqual(TinySearchConfig()["scrape_max_items"], 5)
        with self.assertRaises(ValueError):
            TinySearchConfig(scrape_max_items=0)

    def test_scrape_max_concurrent_items_must_be_positive(self) -> None:
        self.assertEqual(TinySearchConfig()["scrape_max_concurrent_items"], 5)
        with self.assertRaises(ValueError):
            TinySearchConfig(scrape_max_concurrent_items=0)

    async def test_duplicate_scrape_items_share_one_pipeline_run(self) -> None:
        successful = _result_payload("q")
        scrape = AsyncMock(return_value=successful)
        with patch("tinysearch.core._scrape_url_with_config", scrape), patch(
            "tinysearch.core._ensure_browser_bundle", new=AsyncMock()
        ), patch(
            "tinysearch.core._ensure_local_bundle_for_config", new=AsyncMock()
        ), patch(
            "tinysearch.core.create_browser_crawler",
            return_value=contextlib.nullcontext(None),
        ):
            result = await scrape_urls(
                [
                    {"url": "https://one.example", "query": "q"},
                    {"url": "https://one.example", "query": "q"},
                ]
            )

        scrape.assert_awaited_once()
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0], result["results"][1])

    async def test_equivalent_raw_scrape_items_share_one_pipeline_run(self) -> None:
        successful = _result_payload("*")
        scrape = AsyncMock(return_value=successful)
        with patch("tinysearch.core._scrape_url_with_config", scrape), patch(
            "tinysearch.core._ensure_browser_bundle", new=AsyncMock()
        ), patch(
            "tinysearch.core._ensure_local_bundle_for_config", new=AsyncMock()
        ), patch(
            "tinysearch.core.create_browser_crawler",
            return_value=contextlib.nullcontext(None),
        ):
            await scrape_urls(
                [
                    {"url": "https://one.example"},
                    {"url": "https://one.example", "query": "*"},
                ]
            )

        scrape.assert_awaited_once()

    async def test_server_crawler_session_is_leased_instead_of_recreated(self) -> None:
        successful = _result_payload("*")

        class FakeSession:
            def __init__(self) -> None:
                self.leases = 0

            @contextlib.asynccontextmanager
            async def lease(self, config):
                self.leases += 1
                yield "shared-crawler"

        session = FakeSession()
        scrape = AsyncMock(return_value=successful)
        with patch("tinysearch.core._scrape_url_with_config", scrape), patch(
            "tinysearch.core._ensure_browser_bundle", new=AsyncMock()
        ), patch(
            "tinysearch.core._ensure_local_bundle_for_config", new=AsyncMock()
        ), patch("tinysearch.core.create_browser_crawler") as create:
            await scrape_urls(
                [{"url": "https://one.example"}], crawler_session=session
            )

        create.assert_not_called()
        self.assertEqual(session.leases, 1)
        self.assertEqual(scrape.await_args.kwargs["crawler"], "shared-crawler")

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

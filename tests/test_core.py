from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import tinysearch
from tinysearch.config import DEFAULT_CONFIG, TinySearchConfig
from tinysearch.core import _ensure_local_bundle_for_config
from tinysearch.pipelines.research import ResearchResult
from tinysearch.results import result_envelope


def _research_payload(query: str = "hello") -> dict:
    return result_envelope(
        operation="research",
        status="ok",
        query=query,
        sources=[],
    )


class CorePublicApiTests(unittest.IsolatedAsyncioTestCase):
    def test_public_api_exports_config_and_prompt_renderer(self) -> None:
        self.assertIs(tinysearch.get_current_datetime, tinysearch.core.get_current_datetime)
        self.assertTrue(callable(tinysearch.to_prompt))
        self.assertEqual(TinySearchConfig()["search_backend"], "ddgs")

    async def test_research_returns_json_serializable_schema_v1_result(self) -> None:
        run = AsyncMock(return_value=ResearchResult(_research_payload()))
        with patch("tinysearch.core.run_research_pipeline", new=run), patch(
            "tinysearch.core._ensure_local_bundle_for_config", new=AsyncMock()
        ):
            result = await tinysearch.research("  hello  ")

        self.assertEqual(result["schema_version"], "1")
        self.assertEqual(result["operation"], "research")
        self.assertEqual(result["query"], "hello")
        json.dumps(result)
        self.assertEqual(run.await_args.args[0], "hello")

    async def test_research_explicit_config_controls_bound_search(self) -> None:
        run = AsyncMock(return_value=ResearchResult(_research_payload()))
        with patch("tinysearch.core.run_research_pipeline", new=run), patch(
            "tinysearch.core._ensure_local_bundle_for_config", new=AsyncMock()
        ):
            await tinysearch.research(
                "hello",
                config={"search_backend": "searxng", "search_top_k": 3},
            )

        self.assertEqual(run.await_args.kwargs["config"]["search_top_k"], 3)
        bound_search = run.await_args.kwargs["search_fn"]
        self.assertEqual(bound_search.keywords["config"]["search_backend"], "searxng")

    async def test_research_explicit_config_is_merged_onto_defaults(self) -> None:
        run = AsyncMock(return_value=ResearchResult(_research_payload()))
        with patch("tinysearch.core.run_research_pipeline", new=run), patch(
            "tinysearch.core._ensure_local_bundle_for_config", new=AsyncMock()
        ):
            await tinysearch.research("hello", config={"search_region": "uk-en"})

        self.assertEqual(
            run.await_args.kwargs["config"]["search_top_k"],
            DEFAULT_CONFIG["search_top_k"],
        )

    async def test_research_rejects_invalid_config(self) -> None:
        with self.assertRaises(ValueError):
            await tinysearch.research(
                "hello",
                config={"search_backend": "not-a-backend"},
            )

    async def test_scrape_url_returns_common_structured_shape(self) -> None:
        scrape_result = SimpleNamespace(
            url="https://example.com/x",
            title="Title",
            query="q",
            chunks=[{"chunk_id": "0", "text": "evidence", "tokens": 1}],
            content_tokens=1,
            truncated=False,
            retrieved_at="2026-01-01T00:00:00Z",
            metadata={"author": "A"},
        )
        with patch(
            "tinysearch.core.run_scrape_pipeline",
            new=AsyncMock(return_value=scrape_result),
        ), patch("tinysearch.core._ensure_local_bundle_for_config", new=AsyncMock()):
            result = await tinysearch.scrape_url("https://example.com/x", "q")

        self.assertEqual(result["schema_version"], "1")
        self.assertEqual(result["operation"], "scrape")
        self.assertEqual(result["sources"][0]["chunks"][0]["text"], "evidence")
        self.assertEqual(result["stats"]["content_tokens"], 1)
        json.dumps(result)

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


if __name__ == "__main__":
    unittest.main()

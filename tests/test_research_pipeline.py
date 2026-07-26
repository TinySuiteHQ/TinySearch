from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinysearch import to_prompt
from tinysearch.config import TinySearchConfig
from tinysearch.pipelines.research import run_research_pipeline
from tinysearch.services.web_search_service import (
    SearchBackendUnavailable,
    SearchResult,
)


def _fake_search(query: str, limit: int) -> list[SearchResult]:
    return [
        SearchResult(
            result_id=1,
            title="Python Async Search",
            url="https://example.com/python",
            text="Python asyncio search guide.",
        ),
        SearchResult(
            result_id=2,
            title="Bread Recipes",
            url="https://example.com/bread",
            text="Bread recipes use flour and yeast.",
        ),
    ][:limit]


async def _fake_crawl(**kwargs):
    url = kwargs["url"]
    pages = {
        "https://example.com/python": (
            "# Intro\n\n"
            "Python asyncio search uses async tasks.\n\n"
            "Bread recipes use flour and yeast."
        ),
        "https://example.com/bread": (
            "# Cooking\n\n"
            "Bread recipes use flour and yeast."
        ),
    }
    return {"url": url, "markdown": pages[url], "markdown_raw": pages[url], "html": "", "tokens_raw": 10}


async def _fake_embedder(inputs: list[str]) -> list[list[float]]:
    vectors = []
    for text in inputs:
        lowered = text.removeprefix("task: search result | query: ").removeprefix(
            "title: none | text: "
        ).lower()
        if "python" in lowered or "async" in lowered:
            vectors.append([1.0, 0.0, 0.0])
        elif "bread" in lowered:
            vectors.append([0.0, 1.0, 0.0])
        else:
            vectors.append([0.0, 0.0, 1.0])
    return vectors


def _answer(result) -> str:
    return to_prompt(result.to_dict(), today="2026-06-12")


def _config(**overrides) -> TinySearchConfig:
    return TinySearchConfig(**overrides)


class ResearchPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_returns_prompt_from_ranked_search_and_chunks(self) -> None:
        result = await run_research_pipeline(
            "python async search",
            config=_config(
                search_top_k=2,
                search_max_results_to_keep=1,
                chunk_max_results_to_keep=1,
                crawl_max_chunk_tokens=40,
                crawl_overlap_tokens=0,
            ),
            embedder=_fake_embedder,
            search_fn=_fake_search,
            crawl_fn=_fake_crawl,
        )

        self.assertIn("SEARCH-GROUNDED ANSWER PROMPT", _answer(result))
        self.assertIn("CRITICAL INSTRUCTIONS", _answer(result))
        self.assertIn("You are answering the QUESTION using only the text under RESULTS.", _answer(result))
        self.assertIn("TODAY", _answer(result))
        self.assertIn("First resolve any relative date in the QUESTION using TODAY.", _answer(result))
        self.assertIn("Use only facts directly supported by RESULTS.", _answer(result))
        self.assertIn("RESULTS", _answer(result))
        self.assertIn("RESULT 1", _answer(result))
        self.assertIn("TITLE 1\n======\nPython Async Search", _answer(result))
        self.assertIn("URL 1\n======\nhttps://example.com/python", _answer(result))
        self.assertIn(
            "SEARCH PREVIEW 1\n======\nPython asyncio search guide.",
            _answer(result),
        )
        self.assertIn("RELEVANT TEXT 1\n======", _answer(result))
        self.assertIn("----- RELEVANT CHUNK 1 -----", _answer(result))
        self.assertIn(
            "Python asyncio search uses async tasks.",
            _answer(result),
        )
        self.assertIn("python async search", _answer(result))
        self.assertEqual(_answer(result).count("\nQUESTION\n"), 2)
        self.assertEqual(_answer(result).count("\nTODAY\n"), 2)
        self.assertNotIn("START", _answer(result))
        self.assertNotIn("END", _answer(result))
        self.assertNotIn("Bread Recipes", _answer(result))

    async def test_pipeline_formats_empty_results_prompt(self) -> None:
        result = await run_research_pipeline(
            "no results",
            config=_config(search_top_k=2),
            search_fn=lambda query, limit: [],
            crawl_fn=_fake_crawl,
        )

        self.assertIn("RESULTS", _answer(result))
        self.assertIn("QUESTION", _answer(result))
        self.assertIn("no results", _answer(result))
        self.assertNotIn("RESULT 1", _answer(result))
        self.assertNotIn("RELEVANT TEXT 1", _answer(result))

    async def test_pipeline_filters_blocked_domains_before_crawling(self) -> None:
        crawled_urls: list[str] = []

        def search_with_blocked_result(query: str, limit: int) -> list[SearchResult]:
            return [
                SearchResult(
                    result_id=1,
                    title="Blocked Python",
                    url="https://blocked.example/python",
                    text="Python asyncio search guide.",
                ),
                SearchResult(
                    result_id=2,
                    title="Allowed Python",
                    url="https://allowed.example/python",
                    text="Python asyncio search guide.",
                ),
            ][:limit]

        async def recording_crawl(**kwargs):
            url = kwargs["url"]
            crawled_urls.append(url)
            return {
                "url": url,
                "markdown": "Python asyncio search uses async tasks.",
                "markdown_raw": "Python asyncio search uses async tasks.",
                "html": "",
                "tokens_raw": 10,
            }

        result = await run_research_pipeline(
            "python async search",
            config=_config(
                search_top_k=2,
                search_max_results_to_keep=2,
                chunk_max_results_to_keep=1,
                crawl_max_chunk_tokens=40,
                crawl_overlap_tokens=0,
                blocked_domains=["blocked.example"],
            ),
            embedder=_fake_embedder,
            search_fn=search_with_blocked_result,
            crawl_fn=recording_crawl,
        )

        self.assertEqual(crawled_urls, ["https://allowed.example/python"])
        self.assertIn("Allowed Python", _answer(result))
        self.assertNotIn("Blocked Python", _answer(result))
        self.assertNotIn("https://blocked.example/python", _answer(result))

    async def test_pipeline_ranks_chunks_in_one_global_pool(self) -> None:
        result = await run_research_pipeline(
            "python async search",
            config=_config(
                search_top_k=2,
                search_max_results_to_keep=2,
                chunk_max_results_to_keep=1,
                crawl_max_chunk_tokens=40,
                crawl_overlap_tokens=0,
            ),
            embedder=_fake_embedder,
            search_fn=_fake_search,
            crawl_fn=_fake_crawl,
        )

        self.assertIn("RESULT 1", _answer(result))
        self.assertIn("RESULT 2", _answer(result))
        self.assertIn("----- RELEVANT CHUNK 1 -----", _answer(result))
        self.assertIn("Python asyncio search uses async tasks.", _answer(result))
        self.assertEqual(_answer(result).count("----- RELEVANT CHUNK 1 -----"), 1)

    async def test_pipeline_uses_embedding_tokenizer_for_crawl_chunks(self) -> None:
        seen_encoding_names: list[str] = []

        async def recording_crawl(**kwargs):
            seen_encoding_names.append(kwargs["encoding_name"])
            return await _fake_crawl(**kwargs)

        with patch(
            "tinysearch.pipelines.research.resolve_embedding_tokenizer_name",
            return_value="embedding-tokenizer",
        ):
            await run_research_pipeline(
                "python async search",
                config=_config(
                    search_top_k=1,
                    search_max_results_to_keep=1,
                    chunk_max_results_to_keep=1,
                    crawl_max_chunk_tokens=40,
                    crawl_overlap_tokens=0,
                    encoding_name="embedding",
                ),
                embedder=_fake_embedder,
                search_fn=_fake_search,
                crawl_fn=recording_crawl,
            )

        self.assertEqual(seen_encoding_names, ["embedding-tokenizer"])

    async def test_pipeline_timeout_returns_graceful_result(self) -> None:
        import asyncio as _asyncio

        async def slow_crawl(**kwargs):
            await _asyncio.sleep(10)
            return await _fake_crawl(**kwargs)

        result = await run_research_pipeline(
            "python async search",
            config=_config(
                search_top_k=1,
                search_max_results_to_keep=1,
                chunk_max_results_to_keep=1,
                crawl_max_chunk_tokens=40,
                crawl_overlap_tokens=0,
                pipeline_timeout_seconds=0.1,
            ),
            embedder=_fake_embedder,
            search_fn=_fake_search,
            crawl_fn=slow_crawl,
        )

        self.assertIn("QUESTION", _answer(result))
        self.assertIn("python async search", _answer(result))
        self.assertNotIn("RESULT 1", _answer(result))

    async def test_pipeline_no_timeout_when_none(self) -> None:
        result = await run_research_pipeline(
            "python async search",
            config=_config(
                search_top_k=1,
                search_max_results_to_keep=1,
                chunk_max_results_to_keep=1,
                crawl_max_chunk_tokens=40,
                crawl_overlap_tokens=0,
                pipeline_timeout_seconds=None,
            ),
            embedder=_fake_embedder,
            search_fn=_fake_search,
            crawl_fn=_fake_crawl,
        )

        self.assertIn("RESULT 1", _answer(result))

    async def test_pipeline_rejects_bm25_only_configuration(self) -> None:
        with self.assertRaises(ValueError):
            await run_research_pipeline(
                "python async search",
                config=_config(search_dense_weight=0.0),
                embedder=_fake_embedder,
                search_fn=_fake_search,
                crawl_fn=_fake_crawl,
            )

    async def test_pipeline_reports_distinct_status_when_backend_fails(self) -> None:
        def failing_search(query: str, limit: int) -> list[SearchResult]:
            raise SearchBackendUnavailable("all backends down")

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            result = await run_research_pipeline(
                "python async search",
                config=_config(
                    search_top_k=1,
                    search_max_results_to_keep=1,
                    trace_path=str(trace_path),
                ),
                embedder=_fake_embedder,
                search_fn=failing_search,
                crawl_fn=_fake_crawl,
            )

            self.assertIn("QUESTION", _answer(result))
            self.assertNotIn("RESULT 1", _answer(result))
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(trace["status"], "search_backend_error")

    async def test_pipeline_propagates_embedding_failures(self) -> None:
        call_count = 0

        async def failing_embedder(inputs: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise RuntimeError("embedding model died")
            return await _fake_embedder(inputs)

        with self.assertRaises(RuntimeError):
            await run_research_pipeline(
                "python async search",
                config=_config(
                    search_top_k=1,
                    search_max_results_to_keep=1,
                ),
                embedder=failing_embedder,
                search_fn=_fake_search,
                crawl_fn=_fake_crawl,
            )


if __name__ == "__main__":
    unittest.main()

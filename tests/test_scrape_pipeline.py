from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch

from tinysearch import to_prompt
from tinysearch.pipelines.scrape import run_scrape_pipeline
from tinysearch.results import public_chunk, result_envelope
from tinysearch.services.scrape_service import (
    DEFAULT_SCRAPE_MAX_TOKENS,
    EmptyContentError,
    FetchFailedError,
    FetchTimeoutError,
    SCRAPE_ERROR_MAP,
    ScrapeResult,
    UnsupportedDocumentError,
)
from tinysearch.services.token_counter_service import token_count
from tinysearch.services.url_safety_service import BlockedUrlError, InvalidUrlError


TOKENIZER = "o200k_base"


async def _fake_embedder(inputs: list[str]) -> list[list[float]]:
    return [
        [1.0, 0.0] if "async" in text.lower() else [0.0, 1.0]
        for text in inputs
    ]


async def scrape_url(*args, **kwargs) -> ScrapeResult:
    kwargs.pop("tokenizer_name", None)
    kwargs.setdefault("embedder", _fake_embedder)
    return await run_scrape_pipeline(*args, **kwargs)


def _answer(result: ScrapeResult) -> str:
    payload = result_envelope(
        operation="scrape",
        status="ok",
        query=result.query,
        retrieved_at=result.retrieved_at,
        sources=[
            {
                "id": "1",
                "title": result.title,
                "url": result.url,
                "metadata": result.metadata or {},
                "chunks": [
                    public_chunk(chunk, rank=rank)
                    for rank, chunk in enumerate(result.chunks, start=1)
                ],
            }
        ],
        stats={
            "content_tokens": result.content_tokens,
            "truncated": result.truncated,
        },
    )
    return to_prompt(payload, today="2026-06-12")


def _config(**overrides) -> dict:
    base = {
        "blocked_domains": [],
        "pipeline_timeout_seconds": 120.0,
        "crawl_max_chunk_tokens": 500,
        "crawl_overlap_tokens": 80,
        "crawl_bm25_threshold": 1.5,
        "crawl_bm25_language": "english",
    }
    base.update(overrides)
    return base


def _fake_safe_url(url, blocked_domains):
    return url


async def _fake_html_page(*, url, user_query, bm25_threshold, bm25_language):
    return {
        "final_url": url,
        "html": "<html><head><title>Example Article</title></head><body></body></html>",
        "markdown_raw": (
            "# Section A\n\nPython asyncio guide explains async tasks.\n\n"
            "# Section B\n\nBread recipes use flour and yeast.\n\n"
            "# Section C\n\nAnother paragraph about async."
        ),
        "markdown_fit": (
            "# Section A\n\nPython asyncio guide explains async tasks.\n\n"
            "# Section C\n\nAnother paragraph about async."
        ),
        "metadata": {
            "title": "Example Article",
            "description": "A short description",
            "author": "Alice",
            "article:published_time": "2026-01-01T00:00:00Z",
        },
    }


async def _fake_html_redirected(*, url, user_query, bm25_threshold, bm25_language):
    page = await _fake_html_page(
        url=url, user_query=user_query, bm25_threshold=bm25_threshold, bm25_language=bm25_language
    )
    page["final_url"] = "https://redirected.example/x"
    return page


async def _fake_html_redirect_to_blocked(*, url, user_query, bm25_threshold, bm25_language):
    page = await _fake_html_page(
        url=url, user_query=user_query, bm25_threshold=bm25_threshold, bm25_language=bm25_language
    )
    page["final_url"] = "https://blocked.example/x"
    return page


async def _fake_html_empty(*, url, user_query, bm25_threshold, bm25_language):
    return {
        "final_url": url,
        "html": "",
        "markdown_raw": "",
        "markdown_fit": "",
        "metadata": {},
    }


def _fake_document(url: str) -> tuple[str, str]:
    return (
        "## Page 1\n\nPython asyncio guide explains async tasks.\n\n## Page 2\n\nMore content.",
        "pdf",
    )


def _fake_document_doc(url: str) -> tuple[str, str]:
    raise ValueError("legacy .doc files are not supported; use PDF or DOCX")


class ScrapeUrlHappyPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_hybrid_embedding_ranking(self) -> None:
        embedded_inputs: list[str] = []

        async def recording_embedder(inputs: list[str]) -> list[list[float]]:
            embedded_inputs.extend(inputs)
            return await _fake_embedder(inputs)

        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable",
            side_effect=_fake_safe_url,
        ):
            result = await run_scrape_pipeline(
                "https://example.com/article",
                "async",
                config=_config(),
                embedder=recording_embedder,
                crawl_fn=_fake_html_page,
            )

        self.assertGreaterEqual(len(embedded_inputs), 2)
        self.assertIn("dense_score", result.chunks[0])
        self.assertIn("bm25_score", result.chunks[0])
        self.assertIn("rrf_score", result.chunks[0])

    async def test_returns_grounded_prompt_and_token_counts(self) -> None:
        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_fake_safe_url
        ):
            result = await scrape_url(
                "https://example.com/article",
                "What does this page say about async?",
                config=_config(),
                tokenizer_name=TOKENIZER,
                crawl_fn=_fake_html_page,
            )

        self.assertIsInstance(result, ScrapeResult)
        self.assertIn("URL-GROUNDED ANSWER PROMPT", _answer(result))
        self.assertIn("https://example.com/article", _answer(result))
        self.assertIn("Example Article", _answer(result))
        self.assertIn("What does this page say about async?", _answer(result))
        self.assertEqual(result.url, "https://example.com/article")
        self.assertEqual(result.title, "Example Article")
        self.assertEqual(result.query, "What does this page say about async?")
        self.assertGreater(result.content_tokens, 0)
        self.assertGreater(token_count(_answer(result), TOKENIZER), 0)
        self.assertFalse(result.truncated)

    async def test_metadata_populated_when_include_metadata_true(self) -> None:
        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_fake_safe_url
        ):
            result = await scrape_url(
                "https://example.com/article",
                "q",
                config=_config(),
                tokenizer_name=TOKENIZER,
                crawl_fn=_fake_html_page,
            )

        self.assertEqual(result.metadata["description"], "A short description")
        self.assertEqual(result.metadata["author"], "Alice")
        self.assertEqual(result.metadata["published_date"], "2026-01-01T00:00:00Z")

    async def test_metadata_omitted_when_include_metadata_false(self) -> None:
        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_fake_safe_url
        ):
            result = await scrape_url(
                "https://example.com/article",
                "q",
                include_metadata=False,
                config=_config(),
                tokenizer_name=TOKENIZER,
                crawl_fn=_fake_html_page,
            )

        self.assertIsNone(result.metadata)
        self.assertNotIn("metadata", result.to_response(include_metadata=False))

    async def test_metadata_partial_fills_with_none(self) -> None:
        async def _fake_partial(*, url, user_query, bm25_threshold, bm25_language):
            return {
                "final_url": url,
                "html": "<html><head><title>T</title></head></html>",
                "markdown_raw": "Hello world content about async.",
                "markdown_fit": "Hello world content about async.",
                "metadata": {"title": "T", "description": "only this"},
            }

        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_fake_safe_url
        ):
            result = await scrape_url(
                "https://example.com/x",
                "q",
                config=_config(),
                tokenizer_name=TOKENIZER,
                crawl_fn=_fake_partial,
            )

        self.assertEqual(result.metadata["description"], "only this")
        self.assertIsNone(result.metadata["author"])
        self.assertIsNone(result.metadata["published_date"])

    async def test_retrieved_at_is_utc_iso_with_z_suffix(self) -> None:
        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_fake_safe_url
        ):
            result = await scrape_url(
                "https://example.com/x",
                "q",
                config=_config(),
                tokenizer_name=TOKENIZER,
                crawl_fn=_fake_html_page,
            )

        self.assertTrue(result.retrieved_at.endswith("Z"))
        parsed = datetime.strptime(result.retrieved_at, "%Y-%m-%dT%H:%M:%SZ")
        self.assertIsNotNone(parsed)

    async def test_preserves_original_query_wording(self) -> None:
        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_fake_safe_url
        ):
            result = await scrape_url(
                "https://example.com/x",
                "  What does THIS page say about 'Async/Await'?  ",
                config=_config(),
                tokenizer_name=TOKENIZER,
                crawl_fn=_fake_html_page,
            )

        self.assertEqual(result.query, "What does THIS page say about 'Async/Await'?")
        self.assertIn("What does THIS page say about 'Async/Await'?", _answer(result))


class ScrapeUrlBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_truncates_when_total_exceeds_max_tokens(self) -> None:
        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_fake_safe_url
        ):
            result = await scrape_url(
                "https://example.com/x",
                "async",
                max_tokens=15,
                config=_config(crawl_max_chunk_tokens=20, crawl_overlap_tokens=0),
                tokenizer_name=TOKENIZER,
                crawl_fn=_fake_html_page,
            )

        self.assertTrue(result.truncated)
        self.assertLessEqual(result.content_tokens, 15)
        self.assertGreater(result.content_tokens, 0)

    async def test_no_truncation_when_budget_covers_all_chunks(self) -> None:
        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_fake_safe_url
        ):
            result = await scrape_url(
                "https://example.com/x",
                "async",
                max_tokens=100_000,
                config=_config(),
                tokenizer_name=TOKENIZER,
                crawl_fn=_fake_html_page,
            )

        self.assertFalse(result.truncated)

    async def test_single_oversized_chunk_is_truncated_at_token_level(self) -> None:
        long_text = "Python asyncio. " * 200

        async def _fake_long(*, url, user_query, bm25_threshold, bm25_language):
            return {
                "final_url": url,
                "html": "",
                "markdown_raw": long_text,
                "markdown_fit": long_text,
                "metadata": {"title": "Long"},
            }

        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_fake_safe_url
        ):
            result = await scrape_url(
                "https://example.com/x",
                "async",
                max_tokens=20,
                config=_config(crawl_max_chunk_tokens=4000, crawl_overlap_tokens=0),
                tokenizer_name=TOKENIZER,
                crawl_fn=_fake_long,
            )

        self.assertTrue(result.truncated)
        self.assertEqual(result.content_tokens, 20)


class ScrapeUrlValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_query_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            await scrape_url(
                "https://example.com/x",
                "   ",
                config=_config(),
                tokenizer_name=TOKENIZER,
                crawl_fn=_fake_html_page,
            )

    async def test_invalid_url_propagates(self) -> None:
        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable",
            side_effect=InvalidUrlError("bad"),
        ):
            with self.assertRaises(InvalidUrlError):
                await scrape_url(
                    "ftp://example.com/x",
                    "q",
                    config=_config(),
                    tokenizer_name=TOKENIZER,
                    crawl_fn=_fake_html_page,
                )

    async def test_blocked_url_propagates(self) -> None:
        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable",
            side_effect=BlockedUrlError("nope"),
        ):
            with self.assertRaises(BlockedUrlError):
                await scrape_url(
                    "https://blocked.example/x",
                    "q",
                    config=_config(blocked_domains=["blocked.example"]),
                    tokenizer_name=TOKENIZER,
                    crawl_fn=_fake_html_page,
                )

    async def test_redirect_to_blocked_host_raises(self) -> None:
        calls = {"n": 0}

        def _safe(url, blocked_domains):
            calls["n"] += 1
            if calls["n"] == 1:
                return url
            raise BlockedUrlError("redirect blocked")

        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_safe
        ):
            with self.assertRaises(BlockedUrlError):
                await scrape_url(
                    "https://example.com/x",
                    "q",
                    config=_config(blocked_domains=["blocked.example"]),
                    tokenizer_name=TOKENIZER,
                    crawl_fn=_fake_html_redirect_to_blocked,
                )


class ScrapeUrlErrorMappingTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_markdown_raises_empty_content(self) -> None:
        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_fake_safe_url
        ):
            with self.assertRaises(EmptyContentError):
                await scrape_url(
                    "https://example.com/x",
                    "q",
                    config=_config(),
                    tokenizer_name=TOKENIZER,
                    crawl_fn=_fake_html_empty,
                )

    async def test_crawl_timeout_raises_fetch_timeout(self) -> None:
        import asyncio

        async def _slow(*, url, user_query, bm25_threshold, bm25_language):
            raise asyncio.TimeoutError("slow")

        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_fake_safe_url
        ):
            with self.assertRaises(FetchTimeoutError):
                await scrape_url(
                    "https://example.com/x",
                    "q",
                    config=_config(),
                    tokenizer_name=TOKENIZER,
                    crawl_fn=_slow,
                )

    async def test_crawl_generic_failure_raises_fetch_failed(self) -> None:
        async def _boom(*, url, user_query, bm25_threshold, bm25_language):
            raise RuntimeError("crawler died")

        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_fake_safe_url
        ):
            with self.assertRaises(FetchFailedError):
                await scrape_url(
                    "https://example.com/x",
                    "q",
                    config=_config(),
                    tokenizer_name=TOKENIZER,
                    crawl_fn=_boom,
                )

    async def test_legacy_doc_raises_unsupported_document(self) -> None:
        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_fake_safe_url
        ):
            with self.assertRaises(UnsupportedDocumentError):
                await scrape_url(
                    "https://example.com/file.doc",
                    "q",
                    config=_config(),
                    tokenizer_name=TOKENIZER,
                    document_fn=_fake_document_doc,
                )


class ScrapeUrlDocumentPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_pdf_path_returns_grounded_prompt_with_null_metadata(self) -> None:
        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_fake_safe_url
        ):
            result = await scrape_url(
                "https://example.com/file.pdf",
                "What does the document say about async?",
                config=_config(),
                tokenizer_name=TOKENIZER,
                document_fn=_fake_document,
            )

        self.assertIn("URL-GROUNDED ANSWER PROMPT", _answer(result))
        self.assertEqual(result.url, "https://example.com/file.pdf")
        self.assertEqual(result.title, "")
        self.assertEqual(
            result.metadata,
            {"description": None, "author": None, "published_date": None},
        )

    async def test_pdf_path_omits_metadata_when_include_metadata_false(self) -> None:
        with patch(
            "tinysearch.pipelines.scrape.assert_url_is_fetchable", side_effect=_fake_safe_url
        ):
            result = await scrape_url(
                "https://example.com/file.pdf",
                "q",
                include_metadata=False,
                config=_config(),
                tokenizer_name=TOKENIZER,
                document_fn=_fake_document,
            )

        self.assertIsNone(result.metadata)


class ScrapeErrorMapTests(unittest.TestCase):
    def test_maps_all_known_errors(self) -> None:
        codes = {value[0] for value in SCRAPE_ERROR_MAP.values()}
        self.assertEqual(
            codes,
            {
                "invalid_url",
                "blocked_url",
                "fetch_timeout",
                "fetch_failed",
                "unsupported_document",
                "empty_content",
            },
        )

    def test_default_max_tokens(self) -> None:
        self.assertEqual(DEFAULT_SCRAPE_MAX_TOKENS, 4000)


if __name__ == "__main__":
    unittest.main()

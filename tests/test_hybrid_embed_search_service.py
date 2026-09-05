from __future__ import annotations

import unittest

from tinysearch.services.hybrid_embed_search_service import (
    rank_chunks_hybrid,
    shared_embedding_semaphore,
)


async def _fake_embedder(inputs: list[str]) -> list[list[float]]:
    vectors = {
        "python async search": [1.0, 0.0, 0.0],
        "Python asyncio search uses async tasks.": [1.0, 0.0, 0.0],
        "Bread recipes use flour and yeast.": [0.0, 1.0, 0.0],
        "Search ranking can combine sparse and dense scores.": [0.8, 0.0, 0.2],
    }
    normalized = [
        text.removeprefix("task: search result | query: ").removeprefix(
            "title: none | text: "
        )
        for text in inputs
    ]
    return [vectors[text] for text in normalized]


class HybridEmbedSearchServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_limiter_is_shared_within_event_loop(self) -> None:
        first = shared_embedding_semaphore(3)
        second = shared_embedding_semaphore(3)

        self.assertIs(first, second)
        self.assertIsNot(first, shared_embedding_semaphore(2))

    async def test_rank_chunks_hybrid_combines_bm25_and_dense_scores(self) -> None:
        chunks = [
            {
                "chunk_id": "bread",
                "heading": "Cooking",
                "tokens": 7,
                "text": "Bread recipes use flour and yeast.",
            },
            {
                "chunk_id": "python",
                "heading": "Async",
                "tokens": 6,
                "text": "Python asyncio search uses async tasks.",
            },
            {
                "chunk_id": "ranking",
                "heading": "Search",
                "tokens": 8,
                "text": "Search ranking can combine sparse and dense scores.",
            },
        ]

        ranked = await rank_chunks_hybrid(
            "python async search",
            chunks,
            embedder=_fake_embedder,
        )

        self.assertEqual(
            [chunk["chunk_id"] for chunk in ranked],
            ["python", "ranking", "bread"],
        )
        self.assertEqual(ranked[0]["bm25_rank"], 1)
        self.assertEqual(ranked[0]["dense_rank"], 1)
        self.assertGreater(
            ranked[0]["hybrid_similarity"],
            ranked[1]["hybrid_similarity"],
        )
        self.assertIn("rrf_score", ranked[0])
        self.assertIn("rrf_similarity", ranked[0])

    async def test_ranked_chunks_carry_dense_embedding_for_reuse(self) -> None:
        # Semantic dedup downstream reuses these vectors instead of re-embedding.
        chunks = [
            {"chunk_id": "python", "text": "Python asyncio search uses async tasks."},
            {"chunk_id": "bread", "text": "Bread recipes use flour and yeast."},
        ]

        ranked = await rank_chunks_hybrid(
            "python async search",
            chunks,
            embedder=_fake_embedder,
        )

        by_id = {chunk["chunk_id"]: chunk for chunk in ranked}
        self.assertEqual(by_id["python"]["dense_embedding"], [1.0, 0.0, 0.0])
        self.assertEqual(by_id["bread"]["dense_embedding"], [0.0, 1.0, 0.0])

    async def test_rank_chunks_hybrid_applies_similarity_cutoff_and_top_k(self) -> None:
        chunks = [
            {"chunk_id": "python", "text": "Python asyncio search uses async tasks."},
            {
                "chunk_id": "ranking",
                "text": "Search ranking can combine sparse and dense scores.",
            },
            {"chunk_id": "bread", "text": "Bread recipes use flour and yeast."},
        ]

        ranked = await rank_chunks_hybrid(
            "python async search",
            chunks,
            embedder=_fake_embedder,
            rrf_similarity_cutoff=0.98,
            top_k=1,
        )

        self.assertEqual([chunk["chunk_id"] for chunk in ranked], ["python"])

    async def test_sparse_only_configuration_is_rejected(self) -> None:
        called = False

        async def embedder(inputs: list[str]) -> list[list[float]]:
            nonlocal called
            called = True
            return [[0.0] for _ in inputs]

        with self.assertRaises(ValueError):
            await rank_chunks_hybrid(
                "python async search",
                [
                    {"chunk_id": "bread", "text": "Bread recipes use flour and yeast."},
                    {"chunk_id": "python", "text": "Python asyncio search uses async tasks."},
                ],
                embedder=embedder,
                dense_weight=0.0,
            )

        self.assertFalse(called)

    async def test_dense_only_skips_bm25_weight(self) -> None:
        ranked = await rank_chunks_hybrid(
            "python async search",
            [
                {"chunk_id": "bread", "text": "Bread recipes use flour and yeast."},
                {"chunk_id": "python", "text": "Python asyncio search uses async tasks."},
            ],
            embedder=_fake_embedder,
            dense_weight=1.0,
        )

        self.assertEqual([chunk["chunk_id"] for chunk in ranked], ["python", "bread"])
        self.assertEqual(ranked[0]["rrf_similarity"], 1.0)

    async def test_weight_must_be_between_zero_and_one(self) -> None:
        with self.assertRaises(ValueError):
            await rank_chunks_hybrid(
                "python async search",
                [{"chunk_id": "python", "text": "Python asyncio search uses async tasks."}],
                embedder=_fake_embedder,
                dense_weight=1.1,
            )

    async def test_dense_inputs_use_query_and_document_prefixes(self) -> None:
        seen_inputs: list[str] = []

        async def embedder(inputs: list[str]) -> list[list[float]]:
            seen_inputs.extend(inputs)
            return await _fake_embedder(inputs)

        await rank_chunks_hybrid(
            "python async search",
            [{"chunk_id": "python", "text": "Python asyncio search uses async tasks."}],
            embedder=embedder,
        )

        self.assertEqual(
            seen_inputs,
            [
                "task: search result | query: python async search",
                "title: none | text: Python asyncio search uses async tasks.",
            ],
        )

    async def test_dense_document_embedding_sub_batches(self) -> None:
        chunks = [
            {"chunk_id": "bread", "text": "Bread recipes use flour and yeast."},
            {"chunk_id": "python", "text": "Python asyncio search uses async tasks."},
            {
                "chunk_id": "ranking",
                "text": "Search ranking can combine sparse and dense scores.",
            },
        ]
        call_sizes: list[int] = []

        async def embedder(inputs: list[str]) -> list[list[float]]:
            call_sizes.append(len(inputs))
            return await _fake_embedder(inputs)

        ranked = await rank_chunks_hybrid(
            "python async search",
            chunks,
            embedder=embedder,
            dense_document_embed_batch_size=1,
        )

        self.assertEqual(call_sizes, [1, 1, 1, 1])
        self.assertEqual(
            [chunk["chunk_id"] for chunk in ranked],
            ["python", "ranking", "bread"],
        )

    async def test_dense_document_embed_batch_size_zero_single_document_call(
        self,
    ) -> None:
        chunks = [
            {"chunk_id": "python", "text": "Python asyncio search uses async tasks."},
            {"chunk_id": "bread", "text": "Bread recipes use flour and yeast."},
        ]
        call_sizes: list[int] = []

        async def embedder(inputs: list[str]) -> list[list[float]]:
            call_sizes.append(len(inputs))
            return await _fake_embedder(inputs)

        ranked = await rank_chunks_hybrid(
            "python async search",
            chunks,
            embedder=embedder,
            dense_document_embed_batch_size=0,
        )

        self.assertEqual(call_sizes, [1, 2])
        self.assertEqual([chunk["chunk_id"] for chunk in ranked], ["python", "bread"])


if __name__ == "__main__":
    unittest.main()

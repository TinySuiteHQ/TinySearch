from __future__ import annotations

import unittest

from tinysearch.services.chunk_pool_selection_service import (
    dedupe_chunks_by_token_jaccard,
    jaccard_similarity_tokens,
    select_chunks_with_quota_and_fill,
)


class ChunkPoolSelectionServiceTests(unittest.TestCase):
    def test_jaccard_identical_nonempty(self) -> None:
        a = frozenset({"hello", "world"})
        self.assertAlmostEqual(jaccard_similarity_tokens(a, a), 1.0)

    def test_dedupe_drops_second_identical(self) -> None:
        chunks = [
            {"chunk_id": "a", "text": "same words here for overlap test alpha"},
            {"chunk_id": "b", "text": "same words here for overlap test alpha"},
        ]
        deduped = dedupe_chunks_by_token_jaccard(chunks, threshold=0.9)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["chunk_id"], "a")

    def test_dedupe_pass_through_when_disabled(self) -> None:
        chunks = [{"chunk_id": "a", "text": "foo"}, {"chunk_id": "b", "text": "foo bar"}]
        deduped = dedupe_chunks_by_token_jaccard(chunks, threshold=1.0)
        self.assertEqual(deduped, chunks)

    def test_quota_then_fill_reaches_limit(self) -> None:
        ranked = [
            {"chunk_id": "u1:a", "source_url": "https://one.example/", "text": "one aaa"},
            {"chunk_id": "u1:b", "source_url": "https://one.example/", "text": "one bbb"},
            {"chunk_id": "u1:c", "source_url": "https://one.example/", "text": "one ccc"},
            {"chunk_id": "u2:a", "source_url": "https://two.example/", "text": "two ddd distinct"},
            {"chunk_id": "u2:b", "source_url": "https://two.example/", "text": "two eee distinct"},
        ]
        picked = select_chunks_with_quota_and_fill(
            ranked,
            final_limit=4,
            max_per_source_url=1,
            dedupe_jaccard_threshold=1.0,
        )
        self.assertEqual(len(picked), 4)
        self.assertEqual(
            [c["chunk_id"] for c in picked[:2]],
            ["u1:a", "u2:a"],
        )
        urls = [c["source_url"] for c in picked]
        self.assertGreaterEqual(urls.count("https://two.example/"), 1)


if __name__ == "__main__":
    unittest.main()

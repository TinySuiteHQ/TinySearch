from __future__ import annotations

import unittest

from tinysearch.services.chunk_pool_selection_service import (
    cosine_similarity,
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


class SemanticDedupeTests(unittest.TestCase):
    # Two chunks that paraphrase the same fact with almost no shared tokens, plus a
    # clearly distinct one. Embeddings encode the "meaning" so the two paraphrases are
    # near-collinear while the distinct chunk points elsewhere.
    PARAPHRASE_A = {
        "chunk_id": "a",
        "source_url": "https://a.example/",
        "text": "The rollout begins next Monday for every enrolled account.",
        "dense_embedding": [1.0, 0.0, 0.0],
    }
    PARAPHRASE_B = {
        "chunk_id": "b",
        "source_url": "https://b.example/",
        "text": "Starting the first weekday of the week, all signed-up users get access.",
        # cosine(A, B) == 0.96 (0.96^2 + 0.28^2 == 1)
        "dense_embedding": [0.96, 0.28, 0.0],
    }
    DISTINCT = {
        "chunk_id": "c",
        "source_url": "https://c.example/",
        "text": "Pricing tiers remain unchanged through the end of the fiscal year.",
        "dense_embedding": [0.0, 0.0, 1.0],
    }

    def test_cosine_similarity_basics(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertEqual(cosine_similarity([], [1.0]), 0.0)
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)

    def test_semantic_dedup_drops_low_jaccard_paraphrase(self) -> None:
        rejections: list[dict] = []
        picked = select_chunks_with_quota_and_fill(
            [self.PARAPHRASE_A, self.PARAPHRASE_B, self.DISTINCT],
            final_limit=5,
            max_per_source_url=0,
            dedupe_jaccard_threshold=0.92,
            semantic_dedupe_enabled=True,
            semantic_dedupe_threshold=0.9,
            rejections=rejections,
        )
        ids = [c["chunk_id"] for c in picked]
        # Highest-ranked paraphrase kept, its near-duplicate dropped, distinct chunk kept.
        self.assertEqual(ids, ["a", "c"])
        self.assertEqual(len(rejections), 1)
        self.assertEqual(rejections[0]["reason"], "semantic_duplicate")
        self.assertEqual(rejections[0]["chunk_id"], "b")
        self.assertEqual(rejections[0]["matched_chunk_id"], "a")
        self.assertGreaterEqual(rejections[0]["similarity"], 0.9)

    def test_semantic_dedup_disabled_keeps_paraphrase(self) -> None:
        # Jaccard cannot see the paraphrase (low token overlap), so with semantic
        # dedup off both paraphrases survive.
        picked = select_chunks_with_quota_and_fill(
            [self.PARAPHRASE_A, self.PARAPHRASE_B, self.DISTINCT],
            final_limit=5,
            max_per_source_url=0,
            dedupe_jaccard_threshold=0.92,
            semantic_dedupe_enabled=False,
        )
        self.assertEqual([c["chunk_id"] for c in picked], ["a", "b", "c"])

    def test_semantic_threshold_controls_aggressiveness(self) -> None:
        # A high threshold treats the paraphrases as distinct.
        picked = select_chunks_with_quota_and_fill(
            [self.PARAPHRASE_A, self.PARAPHRASE_B, self.DISTINCT],
            final_limit=5,
            max_per_source_url=0,
            dedupe_jaccard_threshold=0.92,
            semantic_dedupe_enabled=True,
            semantic_dedupe_threshold=0.99,
        )
        self.assertEqual([c["chunk_id"] for c in picked], ["a", "b", "c"])

    def test_semantic_dedup_keeps_related_but_distinct_chunks(self) -> None:
        picked = select_chunks_with_quota_and_fill(
            [self.PARAPHRASE_A, self.DISTINCT],
            final_limit=5,
            max_per_source_url=0,
            dedupe_jaccard_threshold=0.92,
            semantic_dedupe_enabled=True,
            semantic_dedupe_threshold=0.9,
        )
        self.assertEqual([c["chunk_id"] for c in picked], ["a", "c"])

    def test_semantic_dedup_passes_through_chunks_without_embeddings(self) -> None:
        # Missing embeddings must never crash or over-drop; such chunks skip the
        # semantic stage and are governed by lexical dedup only.
        chunks = [
            {"chunk_id": "a", "source_url": "https://a/", "text": "alpha beta gamma"},
            {"chunk_id": "b", "source_url": "https://b/", "text": "delta epsilon zeta"},
        ]
        picked = select_chunks_with_quota_and_fill(
            chunks,
            final_limit=5,
            max_per_source_url=0,
            dedupe_jaccard_threshold=0.92,
            semantic_dedupe_enabled=True,
            semantic_dedupe_threshold=0.9,
        )
        self.assertEqual([c["chunk_id"] for c in picked], ["a", "b"])

    def test_semantic_dedup_cooperates_with_quota_and_fill(self) -> None:
        # u2:a is a semantic duplicate of u1:a but on a DIFFERENT source, so the
        # per-source quota does not catch it -- the semantic stage must. u1:c is on
        # the same source as u1:a: the quota skips it in the first pass but the fill
        # pass admits it. The rejected u2:a is revisited in the fill pass and must be
        # logged exactly once.
        ranked = [
            {"chunk_id": "u1:a", "source_url": "https://one/", "text": "one alpha",
             "dense_embedding": [1.0, 0.0, 0.0]},
            {"chunk_id": "u2:a", "source_url": "https://two/", "text": "two reworded",
             "dense_embedding": [0.999, 0.01, 0.0]},
            {"chunk_id": "u1:c", "source_url": "https://one/", "text": "one gamma distinct",
             "dense_embedding": [0.0, 1.0, 0.0]},
        ]
        rejections: list[dict] = []
        picked = select_chunks_with_quota_and_fill(
            ranked,
            final_limit=3,
            max_per_source_url=1,
            dedupe_jaccard_threshold=1.0,
            semantic_dedupe_enabled=True,
            semantic_dedupe_threshold=0.9,
            rejections=rejections,
        )
        self.assertEqual([c["chunk_id"] for c in picked], ["u1:a", "u1:c"])
        # The dropped chunk is revisited in the fill pass but logged only once.
        self.assertEqual([r["chunk_id"] for r in rejections], ["u2:a"])

    def test_selection_strips_nothing_but_reuses_embeddings(self) -> None:
        # The selection service reads embeddings but does not mutate the input dicts.
        chunk = dict(self.PARAPHRASE_A)
        select_chunks_with_quota_and_fill(
            [chunk],
            final_limit=1,
            max_per_source_url=0,
            dedupe_jaccard_threshold=0.92,
            semantic_dedupe_enabled=True,
            semantic_dedupe_threshold=0.9,
        )
        self.assertIn("dense_embedding", chunk)


if __name__ == "__main__":
    unittest.main()

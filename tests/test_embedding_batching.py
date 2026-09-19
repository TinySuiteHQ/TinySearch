from __future__ import annotations

import asyncio
import unittest

from tinysearch.services.embedding_service import create_batching_embedder


class BatchingEmbedderTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_calls_coalesce_into_one_underlying_batch(self) -> None:
        calls: list[list[str]] = []

        async def base(inputs: list[str]) -> list[list[float]]:
            calls.append(list(inputs))
            return [[float(len(text))] for text in inputs]

        embed = create_batching_embedder(base, flush_delay_seconds=0.001)
        first, second = await asyncio.gather(
            embed(["same query", "doc a"]),
            embed(["same query", "doc b"]),
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(set(calls[0]), {"same query", "doc a", "doc b"})
        self.assertEqual(first[0], second[0])

    async def test_cached_input_is_not_reembedded(self) -> None:
        calls: list[list[str]] = []

        async def base(inputs: list[str]) -> list[list[float]]:
            calls.append(list(inputs))
            return [[1.0] for _ in inputs]

        embed = create_batching_embedder(base, flush_delay_seconds=0)
        await embed(["repeat"])
        await embed(["repeat"])

        self.assertEqual(calls, [["repeat"]])

    async def test_duplicates_within_one_call_are_embedded_once(self) -> None:
        calls: list[list[str]] = []

        async def base(inputs: list[str]) -> list[list[float]]:
            calls.append(list(inputs))
            return [[2.0] for _ in inputs]

        embed = create_batching_embedder(base, flush_delay_seconds=0)
        vectors = await embed(["x", "x", "y"])

        self.assertEqual(calls, [["x", "y"]])
        self.assertEqual(vectors, [[2.0], [2.0], [2.0]])


if __name__ == "__main__":
    unittest.main()

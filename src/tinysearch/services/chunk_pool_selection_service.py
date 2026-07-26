"""
Post-processing for globally ranked retrieval chunks: Jaccard near-duplicate suppression,
per-source quotas, then optional filler slots (dedupe only) to reach a target top-K.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from services.hybrid_embed_search_service import tokenize_for_retrieval


def jaccard_similarity_tokens(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def _chunk_identity(chunk: dict[str, Any]) -> Any:
    if chunk.get("chunk_id") is not None:
        return chunk["chunk_id"]
    return id(chunk)


def _token_set(text: str) -> frozenset[str]:
    return frozenset(tokenize_for_retrieval(text))


def _max_jaccard_to_accepted(candidate: frozenset[str], accepted_sets: list[frozenset[str]]) -> float:
    if not candidate:
        return 0.0
    if not accepted_sets:
        return 0.0
    return max(
        jaccard_similarity_tokens(candidate, s)
        for s in accepted_sets
        if s
    )


def dedupe_chunks_by_token_jaccard(
    ranked_chunks: Sequence[dict[str, Any]],
    *,
    threshold: float,
    text_key: str = "text",
) -> list[dict[str, Any]]:
    """Keep chunks in order; drop any whose token Jaccard to an earlier kept chunk is >= threshold."""
    if threshold >= 1.0:
        return list(ranked_chunks)

    accepted: list[dict[str, Any]] = []
    accepted_sets: list[frozenset[str]] = []

    for chunk in ranked_chunks:
        text = str(chunk.get(text_key) or "").strip()

        tokens = _token_set(text)
        if not tokens:
            if any(str(c.get(text_key) or "").strip() == text for c in accepted):
                continue
            accepted.append(chunk)
            accepted_sets.append(frozenset())
            continue

        if _max_jaccard_to_accepted(tokens, accepted_sets) >= threshold:
            continue
        accepted.append(chunk)
        accepted_sets.append(tokens)

    return accepted


def select_chunks_with_quota_and_fill(
    ranked_chunks: Sequence[dict[str, Any]],
    *,
    final_limit: int,
    max_per_source_url: int,
    dedupe_jaccard_threshold: float,
    source_key: str = "source_url",
    text_key: str = "text",
) -> list[dict[str, Any]]:
    """
    Dedupe globally, enforce at most ``max_per_source_url`` chunks per ``source_url`` in a first pass,
    then fill toward ``final_limit`` from the remaining ranked candidates while skipping the
    per-source cap but still rejecting near-duplicates (Jaccard) against the running output.
    If ``max_per_source_url <= 0``, only dedupe and truncate to ``final_limit``.
    """
    limit = max(0, final_limit)
    if limit == 0:
        return []

    ranked = dedupe_chunks_by_token_jaccard(
        ranked_chunks,
        threshold=dedupe_jaccard_threshold,
        text_key=text_key,
    )

    dedupe_relaxed = dedupe_jaccard_threshold >= 1.0

    if max_per_source_url <= 0:
        return ranked[:limit]

    url_counts: dict[str, int] = {}
    chosen_ids: set[Any] = set()
    out: list[dict[str, Any]] = []
    accepted_sets: list[frozenset[str]] = []

    def accepts_dedupe(chunk: dict[str, Any]) -> bool:
        if dedupe_relaxed:
            return True
        text = str(chunk.get(text_key) or "").strip()
        tokens = _token_set(text)
        if not tokens:
            return not any(
                str(c.get(text_key) or "").strip() == text for c in out
            )
        return _max_jaccard_to_accepted(tokens, accepted_sets) < dedupe_jaccard_threshold

    def append_chunk(chunk: dict[str, Any]) -> None:
        cid = _chunk_identity(chunk)
        chosen_ids.add(cid)
        out.append(chunk)
        text = str(chunk.get(text_key) or "").strip()
        tokens = _token_set(text)
        accepted_sets.append(tokens)

    for chunk in ranked:
        if len(out) >= limit:
            break
        cid = _chunk_identity(chunk)
        if cid in chosen_ids:
            continue
        url = str(chunk.get(source_key) or "")
        if url_counts.get(url, 0) >= max_per_source_url:
            continue
        if not accepts_dedupe(chunk):
            continue
        url_counts[url] = url_counts.get(url, 0) + 1
        append_chunk(chunk)

    if len(out) < limit:
        for chunk in ranked:
            if len(out) >= limit:
                break
            cid = _chunk_identity(chunk)
            if cid in chosen_ids:
                continue
            if not accepts_dedupe(chunk):
                continue
            append_chunk(chunk)

    return out[:limit]

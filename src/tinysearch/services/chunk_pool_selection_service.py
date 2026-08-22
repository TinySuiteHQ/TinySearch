"""
Post-processing for globally ranked retrieval chunks: Jaccard near-duplicate suppression,
optional embedding-based semantic deduplication, per-source quotas, then optional filler
slots (dedupe only) to reach a target top-K.

Lexical Jaccard runs first as a cheap prefilter. When semantic deduplication is enabled it
runs as a second stage against the running selected output, rejecting candidates whose dense
embedding cosine similarity to an already-accepted chunk exceeds a configurable threshold.
This catches syndicated / paraphrased / reworded content that shares little literal token
overlap. Embeddings are reused from the ranking stage (chunk key ``dense_embedding``) so no
extra embedding calls are made here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from tinysearch.services.hybrid_embed_search_service import tokenize_for_retrieval


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


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two dense vectors; returns 0.0 for empty or zero-norm inputs."""
    if not a or not b:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _chunk_embedding(chunk: dict[str, Any], embedding_key: str) -> list[float] | None:
    raw = chunk.get(embedding_key)
    if not raw:
        return None
    try:
        return [float(v) for v in raw]
    except (TypeError, ValueError):
        return None


def _max_cosine_to_accepted(
    candidate: list[float] | None,
    accepted_embeddings: list[list[float] | None],
) -> tuple[float, int]:
    """Return the (best cosine, index) of ``candidate`` against accepted embeddings."""
    if candidate is None:
        return 0.0, -1
    best = 0.0
    best_idx = -1
    for idx, emb in enumerate(accepted_embeddings):
        if emb is None:
            continue
        sim = cosine_similarity(candidate, emb)
        if sim > best:
            best = sim
            best_idx = idx
    return best, best_idx


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
    semantic_dedupe_enabled: bool = False,
    semantic_dedupe_threshold: float = 0.92,
    source_key: str = "source_url",
    text_key: str = "text",
    embedding_key: str = "dense_embedding",
    rejections: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Dedupe globally, enforce at most ``max_per_source_url`` chunks per ``source_url`` in a first pass,
    then fill toward ``final_limit`` from the remaining ranked candidates while skipping the
    per-source cap but still rejecting near-duplicates against the running output.
    If ``max_per_source_url <= 0``, only dedupe and truncate to ``final_limit``.

    Lexical Jaccard dedup always runs first as a cheap prefilter. When ``semantic_dedupe_enabled``
    is true, a second stage rejects candidates whose ``embedding_key`` cosine similarity to an
    already-accepted chunk is ``>= semantic_dedupe_threshold``. Because candidates are visited in
    ranked order, the highest-ranked member of a duplicate group is the one retained. Semantic
    dedup reuses embeddings attached during ranking and never triggers new embedding calls; a
    candidate with no usable embedding is passed through the semantic stage unchanged.

    When ``rejections`` is provided, one diagnostic record is appended per dropped candidate:
    ``{"chunk_id", "reason", "similarity", "matched_chunk_id"}`` where ``reason`` is
    ``"semantic_duplicate"`` (lexical drops are handled by the standard Jaccard filter).
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
    semantic_active = semantic_dedupe_enabled and semantic_dedupe_threshold < 1.0

    accepted_sets: list[frozenset[str]] = []
    accepted_embeddings: list[list[float] | None] = []
    accepted_ids: list[Any] = []
    out: list[dict[str, Any]] = []

    def accepts_lexical(chunk: dict[str, Any]) -> bool:
        if dedupe_relaxed:
            return True
        text = str(chunk.get(text_key) or "").strip()
        tokens = _token_set(text)
        if not tokens:
            return not any(
                str(c.get(text_key) or "").strip() == text for c in out
            )
        return _max_jaccard_to_accepted(tokens, accepted_sets) < dedupe_jaccard_threshold

    semantic_rejected: set[Any] = set()

    def accepts_semantic(chunk: dict[str, Any]) -> bool:
        if not semantic_active:
            return True
        candidate = _chunk_embedding(chunk, embedding_key)
        if candidate is None:
            return True
        best, best_idx = _max_cosine_to_accepted(candidate, accepted_embeddings)
        if best >= semantic_dedupe_threshold:
            # A candidate can be revisited in the fill pass; record its rejection once.
            if rejections is not None and _chunk_identity(chunk) not in semantic_rejected:
                matched_id = accepted_ids[best_idx] if best_idx >= 0 else None
                rejections.append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "reason": "semantic_duplicate",
                        "similarity": float(best),
                        "matched_chunk_id": matched_id,
                    }
                )
            semantic_rejected.add(_chunk_identity(chunk))
            return False
        return True

    def append_chunk(chunk: dict[str, Any]) -> None:
        out.append(chunk)
        text = str(chunk.get(text_key) or "").strip()
        accepted_sets.append(_token_set(text))
        accepted_embeddings.append(_chunk_embedding(chunk, embedding_key))
        accepted_ids.append(chunk.get("chunk_id"))

    if max_per_source_url <= 0:
        for chunk in ranked:
            if len(out) >= limit:
                break
            if not accepts_semantic(chunk):
                continue
            append_chunk(chunk)
        return out[:limit]

    url_counts: dict[str, int] = {}
    chosen_ids: set[Any] = set()

    for chunk in ranked:
        if len(out) >= limit:
            break
        cid = _chunk_identity(chunk)
        if cid in chosen_ids:
            continue
        url = str(chunk.get(source_key) or "")
        if url_counts.get(url, 0) >= max_per_source_url:
            continue
        if not accepts_lexical(chunk):
            continue
        if not accepts_semantic(chunk):
            continue
        url_counts[url] = url_counts.get(url, 0) + 1
        chosen_ids.add(cid)
        append_chunk(chunk)

    if len(out) < limit:
        for chunk in ranked:
            if len(out) >= limit:
                break
            cid = _chunk_identity(chunk)
            if cid in chosen_ids:
                continue
            if not accepts_lexical(chunk):
                continue
            if not accepts_semantic(chunk):
                continue
            chosen_ids.add(cid)
            append_chunk(chunk)

    return out[:limit]

"""Stable public result helpers for JSON-serializable TinySearch evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = "1"

# Scores are a relative ranking signal, not a precision-sensitive measurement;
# full float64 precision (e.g. 0.6379576852606178) only inflates the token
# count of every result payload for digits no caller acts on.
_SCORE_DECIMALS = 4


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round_score(value: float) -> float:
    return round(float(value), _SCORE_DECIMALS)


def public_chunk(chunk: dict[str, Any], *, rank: int) -> dict[str, Any]:
    return {
        "id": str(chunk.get("chunk_id") or rank),
        "text": str(chunk.get("text") or "").strip(),
        "tokens": int(chunk.get("tokens") or 0),
        "rank": rank,
        "scores": {
            "rrf": _round_score(
                chunk.get("rrf_similarity")
                or chunk.get("hybrid_similarity")
                or 0.0
            ),
            "dense": _round_score(chunk.get("dense_score") or 0.0),
            "bm25": _round_score(chunk.get("bm25_score") or chunk.get("score") or 0.0),
        },
    }


def public_link(link: dict[str, Any], *, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "url": str(link.get("url") or "").strip(),
        "text": str(link.get("text") or "").strip(),
        "score": (
            _round_score(link["score"]) if link.get("score") is not None else None
        ),
    }


def result_envelope(
    *,
    operation: str,
    status: str,
    query: str,
    sources: list[dict[str, Any]],
    errors: list[dict[str, str]] | None = None,
    stats: dict[str, Any] | None = None,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "operation": operation,
        "status": status,
        "query": query,
        "retrieved_at": retrieved_at or utc_timestamp(),
        "sources": sources,
        "errors": errors or [],
        "stats": stats or {},
    }

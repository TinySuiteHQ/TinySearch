"""Stable public result helpers for JSON-serializable TinySearch evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = "1"


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def public_chunk(chunk: dict[str, Any], *, rank: int) -> dict[str, Any]:
    return {
        "id": str(chunk.get("chunk_id") or rank),
        "text": str(chunk.get("text") or "").strip(),
        "tokens": int(chunk.get("tokens") or 0),
        "rank": rank,
        "scores": {
            "rrf": float(
                chunk.get("rrf_similarity")
                or chunk.get("hybrid_similarity")
                or 0.0
            ),
            "dense": float(chunk.get("dense_score") or 0.0),
            "bm25": float(chunk.get("bm25_score") or chunk.get("score") or 0.0),
        },
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

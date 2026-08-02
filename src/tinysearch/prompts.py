"""Pure prompt rendering for structured TinySearch results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tinysearch.results import SCHEMA_VERSION
from tinysearch.services.grounded_prompt_service import (
    format_search_grounded_prompt,
    format_url_grounded_prompt,
)


def _legacy_source(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "title": str(source.get("title") or ""),
        "url": str(source.get("url") or ""),
        "snippet": str(source.get("snippet") or ""),
        "ranked_chunks": [
            {"text": str(chunk.get("text") or "")}
            for chunk in source.get("chunks", [])
            if isinstance(chunk, Mapping)
        ],
    }


def to_prompt(result: Mapping[str, Any], *, today: str | None = None) -> str:
    """Render a schema-v1 research or scrape result as a grounded LLM prompt."""
    version = str(result.get("schema_version") or "")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported TinySearch result schema_version {version!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    operation = str(result.get("operation") or "")
    question = str(result.get("query") or "")
    raw_sources = result.get("sources")
    sources = (
        [_legacy_source(source) for source in raw_sources if isinstance(source, Mapping)]
        if isinstance(raw_sources, list)
        else []
    )
    if operation == "research":
        return format_search_grounded_prompt(
            question=question,
            results=sources,
            today=today,
            retrieved_at=str(result.get("retrieved_at") or "") or None,
        )
    if operation == "scrape":
        source = sources[0] if sources else {
            "title": "",
            "url": "",
            "ranked_chunks": [],
        }
        raw_stats = result.get("stats")
        stats = raw_stats if isinstance(raw_stats, Mapping) else {}
        truncated = stats.get("truncated")
        content_tokens = stats.get("content_tokens")
        return format_url_grounded_prompt(
            question=question,
            url=source["url"],
            title=source["title"],
            ranked_chunks=source["ranked_chunks"],
            today=today,
            retrieved_at=str(result.get("retrieved_at") or "") or None,
            truncated=truncated if isinstance(truncated, bool) else None,
            content_tokens=(
                int(content_tokens) if isinstance(content_tokens, int) else None
            ),
        )
    raise ValueError(f"unsupported TinySearch result operation {operation!r}")

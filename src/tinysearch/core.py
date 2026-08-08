"""The four operations TinySearch exposes, independent of transport.

`servers/mcp_server.py` and `servers/fastapi_server.py` are thin adapters
around these functions: they add transport-specific request/response schemas,
logging, and exception-to-transport-error mapping, but the actual work
(config loading, ensuring the embedding bundle is ready, running the
pipeline) lives here exactly once.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

from tinysearch.config import ConfigInput, resolve_config
from tinysearch.pipelines.research import run_research_pipeline
from tinysearch.pipelines.scrape import run_scrape_pipeline
from tinysearch.results import public_chunk, result_envelope
from tinysearch.services.current_datetime_service import current_datetime_payload
from tinysearch.services.embedding_service import normalize_embedding_backend
from tinysearch.services.scrape_service import DEFAULT_SCRAPE_MAX_TOKENS
from tinysearch.services.tinysearch_config_service import normalize_query
from tinysearch.services.web_search_service import (
    filter_blocked_search_results,
    search as web_search,
    search_with_metadata,
)

get_current_datetime = current_datetime_payload


def _resolve_config(config: ConfigInput | None) -> dict[str, Any]:
    """Resolve the config to use for one call.

    `config=None` uses canonical package defaults without reading the
    filesystem or environment. Programmatic callers can pass a
    `TinySearchConfig` or partial mapping, which is validated and merged
    onto those defaults. Server adapters resolve their ambient config before
    calling this module.
    """
    return resolve_config(config).to_dict()


async def _ensure_local_bundle_for_config(config: dict[str, Any]) -> None:
    if normalize_embedding_backend(str(config["embedding_backend"])) != "onnx":
        return
    from tinysearch.services.onnx_bundle_service import ensure_onnx_bundle_sync

    await asyncio.to_thread(ensure_onnx_bundle_sync, str(config["embedding_model"]))


async def _ensure_browser_bundle() -> None:
    from tinysearch.services.browser_bundle_service import ensure_chromium_sync

    await asyncio.to_thread(ensure_chromium_sync)


async def search(
    query: str,
    *,
    limit: int = 10,
    config: ConfigInput | None = None,
) -> dict[str, Any]:
    """Return raw, backend-ordered discovery results without deep research work."""
    query = normalize_query(query)
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    resolved_config = _resolve_config(config)
    response = search_with_metadata(query, limit, config=resolved_config)
    results = filter_blocked_search_results(
        [
            result
            for result in response.results
            if result.url.lower().startswith(("http://", "https://"))
        ],
        resolved_config["blocked_domains"],
    )
    return {
        "schema_version": "1",
        "operation": "search",
        "status": "ok",
        "query": query,
        "backend": response.backend,
        "results": [
            {
                "rank": rank,
                "title": result.title,
                "url": result.url,
                "preview": result.text,
                "published_at": result.published_at,
            }
            for rank, result in enumerate(results, start=1)
        ],
        "errors": [],
        "stats": {"result_count": len(results)},
    }


async def research(query: str, *, config: ConfigInput | None = None) -> dict[str, Any]:
    """Discover relevant URLs, crawl and rank them, and return structured evidence.

    `config`, if given, overrides the on-disk/env-driven config for this call
    only (see `_resolve_config`), pass a dict instead of pointing
    `TINYSEARCH_CONFIG_PATH` at a file when calling this as a library.
    """
    query = normalize_query(query)
    resolved_config = _resolve_config(config)
    await _ensure_local_bundle_for_config(resolved_config)
    await _ensure_browser_bundle()
    result = await run_research_pipeline(
        query,
        config=resolved_config,
        search_fn=partial(web_search, config=resolved_config),
    )
    return result.to_dict()


async def _scrape_url_with_config(
    url: str,
    query: str | None,
    *,
    max_tokens: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    scrape_result = await run_scrape_pipeline(
        url,
        query,
        max_tokens=max_tokens,
        include_metadata=True,
        config=config,
    )
    source = {
        "id": "1",
        "title": scrape_result.title,
        "url": scrape_result.url,
        "metadata": scrape_result.metadata or {},
        "chunks": [
            public_chunk(chunk, rank=rank)
            for rank, chunk in enumerate(scrape_result.chunks, start=1)
        ],
    }
    return result_envelope(
        operation="scrape",
        status="ok",
        query=scrape_result.query,
        retrieved_at=scrape_result.retrieved_at,
        sources=[source],
        stats={
            "content_tokens": scrape_result.content_tokens,
            "truncated": scrape_result.truncated,
        },
    )


async def scrape_urls(
    items: Sequence[Mapping[str, Any]],
    *,
    max_tokens: int = DEFAULT_SCRAPE_MAX_TOKENS,
    config: ConfigInput | None = None,
) -> dict[str, Any]:
    """Scrape up to five URL/query pairs concurrently.

    Each item needs ``url`` and may omit ``query`` (or set it to ``'*'``) for
    first-token page-order extraction. Repeat a URL in separate items when it
    needs distinct focused queries; the five-item cap bounds total work.
    """
    if not 1 <= len(items) <= 5:
        raise ValueError("items must contain between 1 and 5 URL/query pairs")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    normalized: list[tuple[str, str | None]] = []
    for item in items:
        url = item.get("url")
        query = item.get("query")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("every batch item requires a non-empty url")
        if query is not None and not isinstance(query, str):
            raise ValueError("batch item query must be a string, null, or omitted")
        normalized.append((url, query))

    resolved = _resolve_config(config)
    if any((query or "").strip() not in {"", "*"} for _, query in normalized):
        await _ensure_local_bundle_for_config(resolved)
    await _ensure_browser_bundle()
    settled = await asyncio.gather(
        *(
            _scrape_url_with_config(
                url,
                query,
                max_tokens=max_tokens,
                config=resolved,
            )
            for url, query in normalized
        ),
        return_exceptions=True,
    )
    results: list[dict[str, Any]] = []
    for (url, query), outcome in zip(normalized, settled, strict=True):
        if isinstance(outcome, Exception):
            results.append(
                {
                    "url": url,
                    "query": (query or "*").strip() or "*",
                    "status": "error",
                    "error": {"code": type(outcome).__name__, "message": str(outcome)},
                }
            )
        else:
            results.append({"status": "ok", "result": outcome})
    return {
        "schema_version": "1",
        "operation": "scrape_batch",
        "status": "partial" if any(item["status"] == "error" for item in results) else "ok",
        "results": results,
    }

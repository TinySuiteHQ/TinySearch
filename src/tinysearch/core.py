"""The four operations TinySearch exposes, independent of transport.

`servers/mcp_server.py` and `servers/fastapi_server.py` are thin adapters
around these functions: they add transport-specific request/response schemas,
logging, and exception-to-transport-error mapping, but the actual work
(config loading, ensuring the embedding bundle is ready, running the
pipeline) lives here exactly once.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

from tinysearch.config import ConfigInput, resolve_config
from tinysearch.pipelines.research import run_research_pipeline
from tinysearch.pipelines.scrape import run_scrape_pipeline
from tinysearch.results import public_chunk, public_link, result_envelope
from tinysearch.services.current_datetime_service import current_datetime_payload
from tinysearch.services.embedding_service import normalize_embedding_backend
from tinysearch.services.scrape_service import DEFAULT_SCRAPE_MAX_TOKENS
from tinysearch.services.site_crawl_service import create_browser_crawler, is_document_url
from tinysearch.services.tinysearch_config_service import normalize_query
from tinysearch.services.web_search_service import (
    search as web_search,
    search_batch_with_metadata,
)
from tinysearch.telemetry import span_scope

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


async def _ensure_browser_bundle(config: Mapping[str, Any]) -> None:
    if str(config.get("browser_cdp_url") or "").strip():
        return
    from tinysearch.services.browser_bundle_service import ensure_chromium_sync

    await asyncio.to_thread(ensure_chromium_sync)


def coerce_search_items(
    items: Any = None,
    *,
    query: str | None = None,
    domains: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Normalize the accepted search input shapes into a list of ``{"query", "domains"}``.

    The preferred shape is a batch ``items`` list. For backward compatibility a
    single item mapping, a bare query string, or a top-level ``query`` (with an
    optional ``domains`` list) are also accepted and wrapped into a one-item
    list. This is the single place both server adapters and the Python API funnel
    through, so all callers share identical rules.
    """
    if items is not None:
        if isinstance(items, str):
            raw_items: list[Any] = [{"query": items}]
        elif isinstance(items, Mapping):
            raw_items = [items]
        elif isinstance(items, Sequence):
            raw_items = list(items)
        else:
            raise ValueError("items must be a list of search items")
    elif query is not None:
        raw_items = [{"query": query, "domains": list(domains) if domains else []}]
    else:
        raise ValueError("provide items (a list of 1 to 5 search items) or a query")

    if not 1 <= len(raw_items) <= 5:
        raise ValueError("items must contain between 1 and 5 search items")

    normalized: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, str):
            item = {"query": item}
        if not isinstance(item, Mapping):
            raise ValueError("each search item must be a query string or an object")
        item_domains = item.get("domains", [])
        if item_domains is None:
            item_domains = []
        normalized.append({"query": item.get("query"), "domains": list(item_domains)})
    return normalized


async def search(
    items: Any = None,
    *,
    query: str | None = None,
    domains: Sequence[str] | None = None,
    config: ConfigInput | None = None,
) -> dict[str, Any]:
    """Return independent, backend-ordered discovery results for 1 to 5 items.

    Accepts the preferred batch ``items`` list and, for backward compatibility,
    a single ``query`` (with optional ``domains``); see ``coerce_search_items``.
    """
    normalized = coerce_search_items(items, query=query, domains=domains)
    with span_scope(
        "tinysearch.search",
        attributes={"tinysearch.batch.item.count": len(normalized)},
        operation="search",
        record_operation_metric=True,
    ) as telemetry:
        resolved_config = _resolve_config(config)
        started = time.monotonic()
        responses = await search_batch_with_metadata(
            normalized,
            limit=int(resolved_config["search_max_results"]),
            config=resolved_config,
            concurrency=int(resolved_config["search_max_concurrent_items"]),
        )
        output_items = []
        for item, response in zip(normalized, responses, strict=True):
            output_items.append({
                "query": item["query"] if isinstance(item["query"], str) else "",
                "domains": response.domains,
                "status": response.status,
                "results": [
                    {"rank": rank, "title": result.title, "url": result.url,
                     "preview": result.text, "published_at": result.published_at}
                    for rank, result in enumerate(response.results, start=1)
                ],
                "backend_attempts": response.attempts,
                "error": response.error,
                "stats": {"result_count": len(response.results), "latency_ms": response.latency_ms},
            })
        status = "partial" if any(item["status"] == "error" for item in output_items) else "ok"
        result_count = sum(len(response.results) for response in responses)
        telemetry.complete(
            outcome=status,
            result_count=result_count,
            error_type="partial_failure" if status == "partial" else None,
            attributes={
                "tinysearch.result.count": result_count,
                "tinysearch.search.backend.attempt.count": sum(
                    len(item["backend_attempts"]) for item in output_items
                ),
            },
        )
        return {
            "schema_version": "1",
            "operation": "search_batch",
            "status": status,
            "items": output_items,
            "errors": [],
            "stats": {
                "search_item_count": len(output_items),
                "backend_attempt_count": sum(len(item["backend_attempts"]) for item in output_items),
                "latency_ms": round((time.monotonic() - started) * 1000),
            },
        }


async def research(query: str, *, config: ConfigInput | None = None) -> dict[str, Any]:
    """Discover relevant URLs, crawl and rank them, and return structured evidence.

    `config`, if given, overrides the on-disk/env-driven config for this call
    only (see `_resolve_config`), pass a dict instead of pointing
    `TINYSEARCH_CONFIG_PATH` at a file when calling this as a library.
    """
    with span_scope(
        "tinysearch.research",
        operation="research",
        record_operation_metric=True,
    ) as telemetry:
        query = normalize_query(query)
        resolved_config = _resolve_config(config)
        await _ensure_local_bundle_for_config(resolved_config)
        await _ensure_browser_bundle(resolved_config)
        result = (
            await run_research_pipeline(
                query,
                config=resolved_config,
                search_fn=partial(web_search, config=resolved_config),
            )
        ).to_dict()
        status = str(result.get("status") or "ok")
        source_count = len(result.get("sources") or [])
        telemetry.complete(
            outcome=status,
            result_count=source_count,
            error_type=status if status in {"timeout", "search_backend_error", "partial"} else None,
            attributes={"tinysearch.source.count": source_count},
        )
        return result


async def _scrape_url_with_config(
    url: str,
    query: str | None,
    *,
    max_tokens: int,
    config: dict[str, Any],
    crawler: Any | None,
) -> dict[str, Any]:
    with span_scope(
        "tinysearch.scrape.item",
        attributes={"tinysearch.browser.used": not is_document_url(url)},
        operation="scrape_item",
    ) as telemetry:
        started = time.monotonic()
        scrape_result = await run_scrape_pipeline(
            url,
            query,
            max_tokens=max_tokens,
            include_metadata=True,
            config=config,
            crawler=crawler,
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
            "links": [
                public_link(link, rank=rank)
                for rank, link in enumerate(scrape_result.links, start=1)
            ],
        }
        telemetry.complete(
            result_count=1,
            attributes={
                "tinysearch.chunk.count": len(scrape_result.chunks),
                "tinysearch.content.token.count": scrape_result.content_tokens,
                "tinysearch.related_link.token.count": getattr(scrape_result, "link_tokens", 0),
            },
        )
        return result_envelope(
            operation="scrape",
            status="ok",
            query=scrape_result.query,
            retrieved_at=scrape_result.retrieved_at,
            sources=[source],
            stats={
                "requested_url": url,
                "content_tokens": scrape_result.content_tokens,
                "related_link_tokens": getattr(scrape_result, "link_tokens", 0),
                "truncated": scrape_result.truncated,
                "browser_used": not is_document_url(url),
                "latency_ms": round((time.monotonic() - started) * 1000),
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
    with span_scope(
        "tinysearch.scrape",
        attributes={"tinysearch.batch.item.count": len(items)},
        operation="scrape",
        record_operation_metric=True,
    ) as telemetry:
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
        await _ensure_browser_bundle(resolved)

        needs_browser = any(not is_document_url(url) for url, _ in normalized)
        crawler_ctx = (
            create_browser_crawler(resolved)
            if needs_browser
            else contextlib.nullcontext(None)
        )
        async with crawler_ctx as shared_crawler:
            settled = await asyncio.gather(
                *(
                    _scrape_url_with_config(
                        url,
                        query,
                        max_tokens=max_tokens,
                        config=resolved,
                        crawler=shared_crawler,
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
        status = "partial" if any(item["status"] == "error" for item in results) else "ok"
        success_count = sum(1 for item in results if item["status"] == "ok")
        telemetry.complete(
            outcome=status,
            result_count=success_count,
            error_type="partial_failure" if status == "partial" else None,
            attributes={"tinysearch.result.count": success_count},
        )
        return {
            "schema_version": "1",
            "operation": "scrape_batch",
            "status": status,
            "results": results,
        }

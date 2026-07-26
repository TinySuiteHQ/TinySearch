"""The three operations TinySearch exposes, independent of transport.

`servers/mcp_server.py` and `servers/fastapi_server.py` are thin adapters
around these functions: they add transport-specific request/response schemas,
logging, and exception-to-transport-error mapping, but the actual work
(config loading, ensuring the embedding bundle is ready, running the
pipeline) lives here exactly once.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

from tinysearch.config import ConfigInput, resolve_config
from tinysearch.pipelines.agentic_research import agentic_run
from tinysearch.results import public_chunk, result_envelope
from tinysearch.services.current_datetime_service import current_datetime_payload
from tinysearch.services.embedding_service import normalize_embedding_backend
from tinysearch.services.research_config_service import (
    config_trace_path,
    normalize_research_query,
    research_run_kwargs,
    research_tokenizer_name,
)
from tinysearch.services.scrape_service import DEFAULT_SCRAPE_MAX_TOKENS
from tinysearch.services.scrape_service import scrape_url as _scrape_url_service
from tinysearch.services.web_search_service import search

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


async def research(query: str, *, config: ConfigInput | None = None) -> dict[str, Any]:
    """Discover relevant URLs, crawl and rank them, and return a grounded answer prompt.

    `config`, if given, overrides the on-disk/env-driven config for this call
    only (see `_resolve_config`) — pass a dict instead of pointing
    `TINYSEARCH_CONFIG_PATH` at a file when calling this as a library.
    """
    query = normalize_research_query(query)
    resolved_config = _resolve_config(config)
    await _ensure_local_bundle_for_config(resolved_config)
    result = await agentic_run(
        query,
        trace_path=config_trace_path(resolved_config),
        search_fn=partial(search, config=resolved_config),
        **research_run_kwargs(resolved_config),
    )
    return result.to_dict()


async def scrape_url(
    url: str,
    query: str,
    *,
    max_tokens: int = DEFAULT_SCRAPE_MAX_TOKENS,
    config: ConfigInput | None = None,
) -> dict[str, Any]:
    """Crawl a specific URL and return a grounded answer prompt ranked against query.

    `config`, if given, overrides the on-disk/env-driven config for this call
    only (see `_resolve_config`).
    """
    config = _resolve_config(config)
    await _ensure_local_bundle_for_config(config)
    tokenizer = research_tokenizer_name(config)
    scrape_result = await _scrape_url_service(
        url,
        query,
        max_tokens=max_tokens,
        include_metadata=True,
        config=config,
        tokenizer_name=tokenizer,
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

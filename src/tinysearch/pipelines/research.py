"""Search, crawl, and hybrid-rank web evidence for the research operation."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from tinysearch.config import normalize_config
from tinysearch.services.embedding_service import (
    create_embedder,
    resolve_local_embedding_model_spec,
    resolve_embedding_tokenizer_name,
)
from tinysearch.services.chunk_pool_selection_service import select_chunks_with_quota_and_fill
from tinysearch.services.hybrid_embed_search_service import EmbeddingFn, rank_chunks_hybrid
from tinysearch.results import public_chunk, result_envelope
from tinysearch.services.site_crawl_service import crawl
from tinysearch.services.text_chunking_service import chunk_text, truncate_text_to_max_tokens
from tinysearch.services.web_search_service import (
    SearchBackendError,
    SearchResult,
    filter_blocked_search_results,
    search,
)


ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]
SearchFn = Callable[[str, int], Sequence[SearchResult]]
CrawlFn = Callable[..., Awaitable[dict[str, Any]]]

_HTTP_URL = re.compile(r"^https?://", re.IGNORECASE)


@dataclass(frozen=True)
class ResearchResult:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload


def _research_log(msg: str) -> None:
    print(f"[research] {msg}", file=sys.stderr, flush=True)


def _write_trace(trace_path: str | Path | None, payload: dict[str, Any]) -> None:
    if not trace_path:
        return
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _research_log(f"saved trace JSON to {str(path)!r}")


def _is_http_url(url: str) -> bool:
    return bool(url and _HTTP_URL.match(url.strip()))


def _domain_from_url(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").removeprefix("www.")
    except ValueError:
        return ""


def _search_result_doc(result: SearchResult) -> str:
    title = result.title.strip()
    url = result.url.strip()
    domain = _domain_from_url(url)
    snippet = result.text.strip()
    return f"""
Title: {title}
URL: {url}
Domain: {domain}
Snippet: {snippet}
""".strip()


def _search_chunk(result: SearchResult) -> dict[str, Any]:
    return {
        "result_id": result.result_id,
        "title": result.title,
        "url": result.url,
        "domain": _domain_from_url(result.url),
        "snippet": result.text,
        "text": _search_result_doc(result),
    }


async def _rank(
    *,
    query: str,
    chunks: Sequence[dict[str, Any]],
    dense_weight: float,
    rrf_similarity_cutoff: float,
    max_results: int,
    embedder: EmbeddingFn | None,
    semaphore: asyncio.Semaphore | None,
    timeout_seconds: float,
    timeout_retries: int,
    dense_query_prefix: str,
    dense_document_prefix: str,
    dense_document_embed_batch_size: int | None,
) -> list[dict[str, Any]]:
    return await rank_chunks_hybrid(
        query,
        chunks,
        embedder=embedder,
        top_k=max_results,
        rrf_similarity_cutoff=rrf_similarity_cutoff,
        dense_weight=dense_weight,
        dense_query_prefix=dense_query_prefix,
        dense_document_prefix=dense_document_prefix,
        dense_document_embed_batch_size=dense_document_embed_batch_size,
        semaphore=semaphore,
        timeout_seconds=timeout_seconds,
        max_timeout_retries=timeout_retries,
    )


async def run_research_pipeline(
    query: str,
    *,
    config: Mapping[str, Any],
    progress_callback: ProgressCallback | None = None,
    embedder: EmbeddingFn | None = None,
    search_fn: SearchFn = search,
    crawl_fn: CrawlFn = crawl,
) -> ResearchResult:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    resolved = normalize_config(config)
    search_top_k = resolved["search_top_k"]
    search_rrf_cutoff = resolved["search_rrf_cutoff"]
    search_dense_weight = resolved["search_dense_weight"]
    search_max_results_to_keep = resolved["search_max_results_to_keep"]
    chunk_rrf_cutoff = resolved["chunk_rrf_cutoff"]
    chunk_dense_weight = resolved["chunk_dense_weight"]
    chunk_max_results_to_keep = resolved["chunk_max_results_to_keep"]
    max_concurrent_crawls = resolved["max_concurrent_crawls"]
    max_concurrent_embedding_calls = resolved["max_concurrent_embedding_calls"]
    embedding_timeout_seconds = resolved["embedding_timeout_seconds"]
    embedding_timeout_retries = resolved["embedding_timeout_retries"]
    crawl_max_chunk_tokens = resolved["crawl_max_chunk_tokens"]
    crawl_overlap_tokens = resolved["crawl_overlap_tokens"]
    crawl_max_page_tokens = resolved["crawl_max_page_tokens"]
    crawl_fit_markdown_mode = resolved["crawl_fit_markdown_mode"]
    crawl_fit_min_chars = resolved["crawl_fit_min_chars"]
    crawl_bm25_threshold = resolved["crawl_bm25_threshold"]
    crawl_bm25_language = resolved["crawl_bm25_language"]
    crawl_pruning_threshold = resolved["crawl_pruning_threshold"]
    chunk_rank_oversample = resolved["chunk_rank_oversample"]
    chunk_dedupe_jaccard_threshold = resolved["chunk_dedupe_jaccard_threshold"]
    chunk_max_per_source_url = resolved["chunk_max_per_source_url"]
    encoding_name = resolved["encoding_name"]
    embedding_backend = resolved["embedding_backend"]
    embedding_model = resolved["embedding_model"]
    embedding_openai_env_file = resolved["embedding_openai_env_file"]
    dense_query_prefix = resolved["dense_query_prefix"]
    dense_document_prefix = resolved["dense_document_prefix"]
    dense_document_embed_batch_size = resolved["dense_document_embed_batch_size"]
    blocked_domains = resolved["blocked_domains"]
    trace_path = resolved["trace_path"]
    pipeline_timeout_seconds = resolved["pipeline_timeout_seconds"]

    embedding_model = str(embedding_model).strip()
    env_file = str(embedding_openai_env_file).strip()
    if search_dense_weight <= 0.0 or chunk_dense_weight <= 0.0:
        raise ValueError(
            "dense embeddings are required; search_dense_weight and "
            "chunk_dense_weight must both be greater than 0"
        )
    local_model_spec = (
        resolve_local_embedding_model_spec(embedding_model)
        if embedding_backend == "onnx"
        else None
    )
    started_at = datetime.now(UTC).isoformat()
    trace: dict[str, Any] = {
        "query": query,
        "started_at": started_at,
        "finished_at": None,
        "status": "running",
        "config": {
            "search_top_k": search_top_k,
            "search_rrf_cutoff": search_rrf_cutoff,
            "search_dense_weight": search_dense_weight,
            "search_max_results_to_keep": search_max_results_to_keep,
            "chunk_rrf_cutoff": chunk_rrf_cutoff,
            "chunk_dense_weight": chunk_dense_weight,
            "chunk_max_results_to_keep": chunk_max_results_to_keep,
            "max_concurrent_crawls": max_concurrent_crawls,
            "max_concurrent_embedding_calls": max_concurrent_embedding_calls,
            "embedding_timeout_seconds": embedding_timeout_seconds,
            "embedding_timeout_retries": embedding_timeout_retries,
            "crawl_max_chunk_tokens": crawl_max_chunk_tokens,
            "crawl_overlap_tokens": crawl_overlap_tokens,
            "crawl_max_page_tokens": crawl_max_page_tokens,
            "crawl_fit_markdown_mode": crawl_fit_markdown_mode,
            "crawl_fit_min_chars": crawl_fit_min_chars,
            "crawl_bm25_threshold": crawl_bm25_threshold,
            "crawl_bm25_language": crawl_bm25_language,
            "crawl_pruning_threshold": crawl_pruning_threshold,
            "chunk_rank_oversample": chunk_rank_oversample,
            "chunk_dedupe_jaccard_threshold": chunk_dedupe_jaccard_threshold,
            "chunk_max_per_source_url": chunk_max_per_source_url,
            "encoding_name": encoding_name or "embedding",
            "tokenizer_name": None,
            "embedding_backend": embedding_backend,
            "embedding_model": embedding_model,
            "embedding_model_repo_id": (
                local_model_spec.repo_id if local_model_spec is not None else None
            ),
            "embedding_model_local_dir": (
                str(local_model_spec.local_dir) if local_model_spec is not None else None
            ),
            "embedding_openai_env_file": env_file,
            "dense_query_prefix": dense_query_prefix,
            "dense_document_prefix": dense_document_prefix,
            "dense_document_embed_batch_size": dense_document_embed_batch_size,
            "blocked_domains": list(blocked_domains or []),
        },
        "web_search": [],
        "ranked_search_results": [],
        "crawl_results": [],
        "ranked_chunk_pool": [],
        "final_result": None,
        "crawl_errors": [],
    }

    async def emit(event: str, **payload: Any) -> None:
        if progress_callback is not None:
            await progress_callback(event, payload)

    def finish(
        status: str,
        sources: list[dict[str, Any]],
        crawl_errors: Sequence[str],
        stats: dict[str, Any],
        *,
        error_code: str = "crawl_failed",
    ) -> ResearchResult:
        errors = [
            {"code": error_code, "message": error}
            for error in crawl_errors
        ]
        payload = result_envelope(
            operation="research",
            status=status,
            query=query,
            sources=sources,
            errors=errors,
            stats=stats,
        )
        trace["status"] = status
        trace["finished_at"] = datetime.now(UTC).isoformat()
        trace["final_result"] = payload
        trace["crawl_errors"] = list(crawl_errors)
        _write_trace(trace_path, trace)
        return ResearchResult(payload=payload)

    try:
        async with asyncio.timeout(pipeline_timeout_seconds):
            _research_log(f"start query={query!r}")
            await emit("start", query=query)
            await emit("search_start", query=query, search_top_k=search_top_k)
            _research_log(f"search start top_k={search_top_k}")
            try:
                raw_results = search_fn(query, max(1, search_top_k))
            except SearchBackendError as exc:
                _research_log(f"search backend error: {exc}")
                await emit("search_backend_error", error=str(exc))
                return finish(
                    "search_backend_error",
                    [],
                    [str(exc)],
                    {
                        "search_results": 0,
                        "sources_crawled": 0,
                        "chunks_considered": 0,
                        "chunks_selected": 0,
                    },
                    error_code="search_backend_error",
                )
            results = [result for result in raw_results if _is_http_url(result.url)]
            results = filter_blocked_search_results(results, blocked_domains or [])
            _research_log(f"search done results={len(results)}")
            trace["web_search"] = [asdict(result) for result in results]
            await emit("search_results", results_count=len(results))

            if not results:
                return finish(
                    "no_results",
                    [],
                    [],
                    {
                        "search_results": 0,
                        "sources_crawled": 0,
                        "chunks_considered": 0,
                        "chunks_selected": 0,
                    },
                )

            tokenizer_name = (
                str(encoding_name).strip()
                if encoding_name is not None and str(encoding_name).strip().lower() != "embedding"
                else resolve_embedding_tokenizer_name(
                    backend=embedding_backend,
                    embedding_model=embedding_model,
                    openai_env_file=env_file if embedding_backend == "openai_compatible" else None,
                )
            )
            trace["config"]["tokenizer_name"] = tokenizer_name

            if embedder is None:
                embedder = create_embedder(
                    backend=embedding_backend,
                    embedding_model=embedding_model,
                    openai_env_file=env_file if embedding_backend == "openai_compatible" else None,
                )
            embedding_semaphore = asyncio.Semaphore(max(1, max_concurrent_embedding_calls))

            search_chunks = [_search_chunk(result) for result in results]
            await emit("search_embed_ranking", snippets=len(search_chunks))
            _research_log(f"search rank start snippets={len(search_chunks)}")
            ranked_search_chunks = await _rank(
                query=query,
                chunks=search_chunks,
                dense_weight=search_dense_weight,
                rrf_similarity_cutoff=search_rrf_cutoff,
                max_results=search_max_results_to_keep,
                embedder=embedder,
                semaphore=embedding_semaphore,
                timeout_seconds=embedding_timeout_seconds,
                timeout_retries=embedding_timeout_retries,
                dense_query_prefix=dense_query_prefix,
                dense_document_prefix=dense_document_prefix,
                dense_document_embed_batch_size=dense_document_embed_batch_size,
            )
            _research_log(f"search rank done kept={len(ranked_search_chunks)}")
            trace["ranked_search_results"] = ranked_search_chunks
            await emit("search_ranked", kept_results=len(ranked_search_chunks))

            crawl_semaphore = asyncio.Semaphore(max(1, max_concurrent_crawls))

            async def crawl_result(search_doc: dict[str, Any]) -> dict[str, Any]:
                url = str(search_doc["url"])
                async with crawl_semaphore:
                    await emit("crawl_start", url=url)
                    try:
                        crawled = await crawl_fn(
                            url=url,
                            encoding_name=tokenizer_name,
                            user_query=query,
                            fit_markdown_mode=crawl_fit_markdown_mode,
                            fit_min_chars=crawl_fit_min_chars,
                            bm25_threshold=crawl_bm25_threshold,
                            bm25_language=crawl_bm25_language,
                            pruning_threshold=crawl_pruning_threshold,
                        )
                    except Exception as exc:
                        error = f"{url}: {exc}"
                        await emit("crawl_error", url=url, error=str(exc))
                        return {
                            **search_doc,
                            "ranked_chunks": [],
                            "chunks_total": 0,
                            "crawl_error": error,
                        }
                    markdown = str(
                        crawled.get("markdown") or crawled.get("markdown_raw") or ""
                    ).strip()
                    markdown = truncate_text_to_max_tokens(
                        markdown,
                        crawl_max_page_tokens,
                        tokenizer_name,
                    )
                    chunks = chunk_text(
                        markdown,
                        max_chunk_tokens=crawl_max_chunk_tokens,
                        overlap_tokens=crawl_overlap_tokens,
                        encoding_name=tokenizer_name,
                    )
                    source_chunks = [
                        {
                            **chunk,
                            "source_url": url,
                            "source_title": str(search_doc["title"]),
                            "source_result_id": search_doc["result_id"],
                            "source_chunk_id": chunk.get("chunk_id"),
                            "chunk_id": f"{search_doc['result_id']}:{chunk.get('chunk_id')}",
                        }
                        for chunk in chunks
                    ]
                    await emit("crawl_done", url=url, chunks=len(chunks), kept_chunks=0)
                    return {
                        **search_doc,
                        "chunks": source_chunks,
                        "ranked_chunks": [],
                        "chunks_total": len(chunks),
                        "crawl_error": None,
                    }

            crawled_results = await asyncio.gather(
                *(crawl_result(search_doc) for search_doc in ranked_search_chunks)
            )
            chunk_pool = [
                chunk
                for result in crawled_results
                for chunk in result.get("chunks", [])
                if not result.get("crawl_error")
            ]
            oversample = max(1, chunk_rank_oversample)
            chunk_rank_pool_cap = max(
                1,
                min(len(chunk_pool), chunk_max_results_to_keep * oversample),
            )
            await emit("chunk_embed_ranking", chunks=len(chunk_pool), rank_pool_cap=chunk_rank_pool_cap)
            ranked_wide = await _rank(
                query=query,
                chunks=chunk_pool,
                dense_weight=chunk_dense_weight,
                rrf_similarity_cutoff=chunk_rrf_cutoff,
                max_results=chunk_rank_pool_cap,
                embedder=embedder,
                semaphore=embedding_semaphore,
                timeout_seconds=embedding_timeout_seconds,
                timeout_retries=embedding_timeout_retries,
                dense_query_prefix=dense_query_prefix,
                dense_document_prefix=dense_document_prefix,
                dense_document_embed_batch_size=dense_document_embed_batch_size,
            )
            ranked_chunk_pool = select_chunks_with_quota_and_fill(
                ranked_wide,
                final_limit=chunk_max_results_to_keep,
                max_per_source_url=chunk_max_per_source_url,
                dedupe_jaccard_threshold=chunk_dedupe_jaccard_threshold,
            )
            chunks_by_url: dict[str, list[dict[str, Any]]] = {}
            for chunk in ranked_chunk_pool:
                chunks_by_url.setdefault(str(chunk.get("source_url") or ""), []).append(chunk)
            for result in crawled_results:
                result["ranked_chunks"] = chunks_by_url.get(str(result["url"]), [])
            trace["crawl_results"] = crawled_results
            trace["ranked_chunk_pool"] = ranked_chunk_pool
            crawl_errors = [
                str(result["crawl_error"])
                for result in crawled_results
                if result.get("crawl_error")
            ]
            await emit(
                "pages_indexed",
                urls_read=len(ranked_search_chunks),
                chunks_extracted=len(chunk_pool),
                chunks_selected=len(ranked_chunk_pool),
                crawl_errors_count=len(crawl_errors),
            )
            await emit("done", results_count=len(crawled_results), crawl_errors_count=len(crawl_errors))
            _research_log(f"done results={len(crawled_results)} crawl_errors={len(crawl_errors)}")
            rank_by_chunk_id = {
                str(chunk.get("chunk_id") or ""): rank
                for rank, chunk in enumerate(ranked_chunk_pool, start=1)
            }
            public_sources = [
                {
                    "id": str(result.get("result_id") or ordinal),
                    "title": str(result.get("title") or ""),
                    "url": str(result.get("url") or ""),
                    "snippet": str(result.get("snippet") or ""),
                    "chunks": [
                        public_chunk(
                            chunk,
                            rank=rank_by_chunk_id.get(
                                str(chunk.get("chunk_id") or ""),
                                chunk_ordinal,
                            ),
                        )
                        for chunk_ordinal, chunk in enumerate(
                            result.get("ranked_chunks") or [],
                            start=1,
                        )
                    ],
                }
                for ordinal, result in enumerate(crawled_results, start=1)
            ]
            return finish(
                "partial" if crawl_errors else "ok",
                public_sources,
                crawl_errors,
                {
                    "search_results": len(results),
                    "sources_crawled": sum(
                        1 for result in crawled_results if not result.get("crawl_error")
                    ),
                    "chunks_considered": len(chunk_pool),
                    "chunks_selected": len(ranked_chunk_pool),
                },
            )
    except TimeoutError:
        _research_log(f"timeout query={query!r} limit_s={pipeline_timeout_seconds}")
        return finish(
            "timeout",
            [],
            [],
            {
                "search_results": 0,
                "sources_crawled": 0,
                "chunks_considered": 0,
                "chunks_selected": 0,
            },
        )

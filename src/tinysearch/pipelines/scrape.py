"""Fetch, chunk, and hybrid-rank evidence from one known URL."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from functools import partial
from typing import Any

from tinysearch.config import normalize_config
from tinysearch.services.embedding_service import create_embedder
from rank_bm25 import BM25Okapi

from tinysearch.services.hybrid_embed_search_service import (
    EmbeddingFn,
    rank_chunks_hybrid,
    tokenize_for_retrieval,
)
from tinysearch.services.link_extraction_service import (
    extract_links_from_html,
    sanitize_and_dedupe_links,
)
from tinysearch.services.tinysearch_config_service import tokenizer_name_for_config
from tinysearch.services.scrape_service import (
    DEFAULT_SCRAPE_MAX_TOKENS,
    DocumentExtractFn,
    EmptyContentError,
    HtmlCrawlFn,
    ScrapeResult,
    UnsupportedDocumentError,
    extract_document_with_timeout,
    extract_metadata,
    extract_title,
    fetch_html_with_timeout,
    select_chunks_under_budget,
    utc_iso8601_z,
)
from tinysearch.services.token_counter_service import decode_tokens, encode_tokens
from tinysearch.services.site_crawl_service import (
    extract_document_text,
    fetch_html_for_query,
    is_document_url,
    url_path_suffix,
)
from tinysearch.services.text_chunking_service import chunk_text
from tinysearch.services.url_safety_service import assert_url_is_fetchable


# Bounds embedding cost on link-heavy pages; ranking only ever returns
# `max_links` of these, but every candidate in the pool gets embedded.
_MAX_LINK_CANDIDATES = 100


def _prefilter_links_by_bm25(
    candidate_links: list[dict[str, str]],
    query: str,
    limit: int,
) -> list[dict[str, str]]:
    """Pick the `limit` most lexically relevant links before the expensive rerank.

    A link-dense page (a news homepage, say) can carry hundreds of anchors;
    truncating to the first `limit` in DOM order would silently drop a
    genuinely relevant link that happens to sit later on the page, before it
    ever gets a chance to be scored. A cheap BM25 pass over the full
    candidate set keeps embedding cost bounded without that blind spot.
    """
    if len(candidate_links) <= limit:
        return candidate_links
    query_tokens = tokenize_for_retrieval(query)
    if not query_tokens:
        return candidate_links[:limit]
    corpus = [
        tokenize_for_retrieval(f"{link['text']} {link['context']}")
        for link in candidate_links
    ]
    if not any(corpus):
        # BM25Okapi divides by average document length; an all-empty corpus
        # (e.g. image-only nav links with no text or context) makes that
        # zero. Nothing to lexically rank in that case, so keep DOM order.
        return candidate_links[:limit]
    scores = BM25Okapi(corpus).get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(candidate_links)), key=lambda idx: scores[idx], reverse=True
    )
    return [candidate_links[idx] for idx in ranked_indices[:limit]]


async def _select_top_links(
    candidate_links: list[dict[str, str]],
    *,
    query: str,
    max_links: int,
    embedder: EmbeddingFn | None,
    resolved: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Rank candidate links against `query`, or fall back to page order.

    Reuses the same hybrid BM25 + dense retrieval as content-chunk ranking
    (see `hybrid_embed_search_service.rank_chunks_hybrid`), scoring each
    link's text plus nearby page context. Falls back to unranked page-order
    when there is no real query (raw page-order requests) to rank against.
    """
    if not candidate_links or max_links <= 0:
        return []
    if not query or embedder is None:
        return [
            {"url": link["url"], "text": link["text"] or link["url"], "score": None}
            for link in candidate_links[:max_links]
        ]
    pool = _prefilter_links_by_bm25(candidate_links, query, _MAX_LINK_CANDIDATES)
    link_chunks = [
        {
            "url": link["url"],
            "link_text": link["text"],
            "text": f"{link['text']} {link['context']}".strip(),
        }
        for link in pool
    ]
    ranked = await rank_chunks_hybrid(
        query,
        link_chunks,
        embedder=embedder,
        top_k=max_links,
        rrf_similarity_cutoff=resolved["chunk_rrf_cutoff"],
        dense_weight=resolved["chunk_dense_weight"],
        dense_query_prefix=resolved["dense_query_prefix"],
        dense_document_prefix=resolved["dense_document_prefix"],
        dense_document_embed_batch_size=resolved["dense_document_embed_batch_size"],
        semaphore=asyncio.Semaphore(
            max(1, resolved["max_concurrent_embedding_calls"])
        ),
        timeout_seconds=resolved["embedding_timeout_seconds"],
        max_timeout_retries=resolved["embedding_timeout_retries"],
    )
    return [
        {
            "url": item["url"],
            "text": item.get("link_text") or item["url"],
            "score": float(item.get("rrf_similarity") or 0.0),
        }
        for item in ranked
    ]


async def run_scrape_pipeline(
    url: str,
    query: str | None,
    *,
    config: Mapping[str, Any],
    max_tokens: int = DEFAULT_SCRAPE_MAX_TOKENS,
    include_metadata: bool = True,
    embedder: EmbeddingFn | None = None,
    crawl_fn: HtmlCrawlFn | None = None,
    document_fn: DocumentExtractFn | None = None,
    crawler: Any | None = None,
) -> ScrapeResult:
    """Extract a URL in page order, or rank chunks only for a supplied query.

    Omitted, blank, and ``'*'`` queries select raw page-order extraction; any
    other non-empty query enables the existing focused chunk-ranking path.

    Pass `crawler` (an already-started AsyncWebCrawler, see
    site_crawl_service.create_browser_crawler()) to reuse one browser across
    several pipeline calls instead of launching a fresh one per call.
    """
    cleaned_query = (query or "").strip()
    raw_page_order = cleaned_query in {"", "*"}
    public_query = "*" if raw_page_order else cleaned_query
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    resolved = normalize_config(config)
    blocked_domains = resolved["blocked_domains"]
    safe_url = assert_url_is_fetchable(url, blocked_domains)
    fetch_timeout_seconds = float(resolved["pipeline_timeout_seconds"] or 120.0)
    tokenizer_name = tokenizer_name_for_config(resolved)

    is_document = is_document_url(safe_url)
    final_url = safe_url
    markdown = ""
    html = ""
    crawl_metadata: dict[str, Any] = {}
    candidate_links: list[dict[str, str]] = []

    if is_document:
        suffix = url_path_suffix(safe_url)
        if suffix == "doc":
            raise UnsupportedDocumentError(
                "legacy .doc files are not supported; use PDF or DOCX"
            )
        if document_fn is None:
            # Bound the blocking download's own socket timeout by the pipeline
            # budget: asyncio.timeout() below can only cancel the *awaiting*
            # task, not the underlying to_thread() download, so an unbounded
            # per-request timeout would keep that thread (and its socket) alive
            # well past the point callers were told the fetch had timed out.
            document_fn = partial(
                extract_document_text, timeout_seconds=min(fetch_timeout_seconds, 30.0)
            )
        markdown, _document_type = await extract_document_with_timeout(
            url=safe_url,
            timeout_seconds=fetch_timeout_seconds,
            document_fn=document_fn,
        )
    else:
        crawl_fn = crawl_fn or fetch_html_for_query
        page = await fetch_html_with_timeout(
            url=safe_url,
            query=None if raw_page_order else cleaned_query,
            bm25_threshold=resolved["crawl_bm25_threshold"],
            bm25_language=resolved["crawl_bm25_language"],
            timeout_seconds=fetch_timeout_seconds,
            crawl_fn=crawl_fn,
            crawler=crawler,
        )
        final_url = str(page.get("final_url") or safe_url)
        html = str(page.get("html") or "")
        crawl_metadata = page.get("metadata") or {}
        if final_url != safe_url:
            final_url = assert_url_is_fetchable(final_url, blocked_domains)
        candidate_links = sanitize_and_dedupe_links(
            extract_links_from_html(html, final_url),
            base_url=final_url,
            blocked_domains=blocked_domains,
        )
        markdown_raw = str(page.get("markdown_raw") or "")
        markdown_fit = str(page.get("markdown_fit") or "")
        markdown = markdown_raw if raw_page_order else markdown_fit or markdown_raw

    if not markdown or not markdown.strip():
        raise EmptyContentError(f"no readable content extracted from {final_url}")

    if raw_page_order:
        tokens = encode_tokens(markdown, tokenizer_name)
        selected_tokens = tokens[:max_tokens]
        content = decode_tokens(selected_tokens, tokenizer_name)
        if not content.strip():
            raise EmptyContentError(f"no content fit the max_tokens budget for {final_url}")
        title = "" if is_document else extract_title(crawl_metadata, html)
        metadata: dict[str, str | None] | None
        if not include_metadata:
            metadata = None
        elif is_document:
            metadata = {"description": None, "author": None, "published_date": None}
        else:
            metadata = extract_metadata(crawl_metadata, html)
        links = await _select_top_links(
            candidate_links,
            query="",
            max_links=int(resolved["scrape_max_links"]),
            embedder=None,
            resolved=resolved,
        )
        return ScrapeResult(
            url=final_url,
            title=title,
            query=public_query,
            chunks=[{"chunk_id": "1", "text": content, "tokens": len(selected_tokens)}],
            content_tokens=len(selected_tokens),
            truncated=len(tokens) > max_tokens,
            retrieved_at=utc_iso8601_z(),
            metadata=metadata,
            links=links,
        )

    chunks = chunk_text(
        text=markdown,
        max_chunk_tokens=resolved["crawl_max_chunk_tokens"],
        overlap_tokens=resolved["crawl_overlap_tokens"],
        encoding_name=tokenizer_name,
    )
    if not chunks:
        raise EmptyContentError(f"no chunks produced from {final_url}")

    if embedder is None:
        embedder = create_embedder(
            backend=resolved["embedding_backend"],
            embedding_model=resolved["embedding_model"],
            openai_env_file=(
                resolved["embedding_openai_env_file"]
                if resolved["embedding_backend"] == "openai_compatible"
                else None
            ),
        )
    links = await _select_top_links(
        candidate_links,
        query=cleaned_query,
        max_links=int(resolved["scrape_max_links"]),
        embedder=embedder,
        resolved=resolved,
    )
    ranked = await rank_chunks_hybrid(
        cleaned_query,
        chunks,
        embedder=embedder,
        top_k=len(chunks),
        rrf_similarity_cutoff=resolved["chunk_rrf_cutoff"],
        dense_weight=resolved["chunk_dense_weight"],
        dense_query_prefix=resolved["dense_query_prefix"],
        dense_document_prefix=resolved["dense_document_prefix"],
        dense_document_embed_batch_size=resolved[
            "dense_document_embed_batch_size"
        ],
        semaphore=asyncio.Semaphore(
            max(1, resolved["max_concurrent_embedding_calls"])
        ),
        timeout_seconds=resolved["embedding_timeout_seconds"],
        max_timeout_retries=resolved["embedding_timeout_retries"],
    )
    if not ranked:
        raise EmptyContentError(f"no chunks ranked for {final_url}")

    selected, content_tokens, truncated = select_chunks_under_budget(
        ranked, max_tokens, tokenizer_name
    )
    if not selected:
        raise EmptyContentError(f"no chunk fit the max_tokens budget for {final_url}")

    title = "" if is_document else extract_title(crawl_metadata, html)
    metadata: dict[str, str | None] | None
    if not include_metadata:
        metadata = None
    elif is_document:
        metadata = {"description": None, "author": None, "published_date": None}
    else:
        metadata = extract_metadata(crawl_metadata, html)

    return ScrapeResult(
        url=final_url,
        title=title,
        query=public_query,
        chunks=selected,
        content_tokens=content_tokens,
        truncated=truncated,
        retrieved_at=utc_iso8601_z(),
        metadata=metadata,
        links=links,
    )

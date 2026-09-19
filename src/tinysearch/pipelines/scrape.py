"""Fetch, chunk, and hybrid-rank evidence from one known URL."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from tinysearch.config import normalize_config
from tinysearch.services.embedding_service import create_embedder
from rank_bm25 import BM25Okapi

from tinysearch.services.hybrid_embed_search_service import (
    EmbeddingFn,
    rank_chunks_hybrid,
    shared_embedding_semaphore,
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
from tinysearch.services.text_chunking_service import (
    chunk_text,
    truncate_text_to_max_tokens,
)
from tinysearch.services.url_safety_service import assert_url_is_fetchable
from tinysearch.telemetry import span_scope


async def _select_top_links(
    candidate_links: list[dict[str, str]],
    *,
    query: str,
    max_links: int,
) -> list[dict[str, Any]]:
    """Rank related links with BM25 only; raw requests keep page order."""
    if not candidate_links or max_links <= 0:
        return []
    if not query:
        return [
            {"url": link["url"], "text": link["text"] or link["url"], "score": None}
            for link in candidate_links[:max_links]
        ]

    query_tokens = tokenize_for_retrieval(query)
    corpus = [
        tokenize_for_retrieval(f"{link['text']} {link['context']}")
        for link in candidate_links
    ]
    if not query_tokens or not any(corpus):
        return [
            {
                "url": link["url"],
                "text": link["text"] or link["url"],
                "score": 0.0,
            }
            for link in candidate_links[:max_links]
        ]

    scores = BM25Okapi(corpus).get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(candidate_links)),
        key=lambda idx: float(scores[idx]),
        reverse=True,
    )[:max_links]
    return [
        {
            "url": candidate_links[idx]["url"],
            "text": candidate_links[idx]["text"] or candidate_links[idx]["url"],
            "score": float(scores[idx]),
        }
        for idx in ranked_indices
    ]

def _links_under_budget(
    links: list[dict[str, Any]], *, max_tokens: int, tokenizer_name: str
) -> tuple[list[dict[str, Any]], int]:
    """Keep complete related links while their independent payload budget permits."""
    selected: list[dict[str, Any]] = []
    total = 0
    for link in links:
        # Count exactly the public fields the caller receives, with a small,
        # stable label cost so budget behavior does not depend on JSON spacing.
        text = f"URL: {link['url']}\nText: {link['text']}"
        cost = len(encode_tokens(text, tokenizer_name))
        if total + cost > max_tokens:
            continue
        selected.append(link)
        total += cost
    return selected, total


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

    Pass `crawler` (an already-started direct Playwright crawler, see
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
        with span_scope(
            "tinysearch.fetch",
            attributes={
                "tinysearch.browser.used": False,
                "tinysearch.document.type": suffix or "document",
            },
            operation="fetch",
        ) as fetch_telemetry:
            markdown, _document_type = await extract_document_with_timeout(
                url=safe_url,
                timeout_seconds=fetch_timeout_seconds,
                document_fn=document_fn,
            )
            fetch_telemetry.complete()
    else:
        if crawl_fn is None:
            crawl_fn = fetch_html_for_query
        with span_scope(
            "tinysearch.fetch",
            attributes={"tinysearch.browser.used": True, "tinysearch.document.type": "html"},
            operation="fetch",
        ) as fetch_telemetry:
            page = await fetch_html_with_timeout(
                url=safe_url,
                query=None if raw_page_order else cleaned_query,
                timeout_seconds=fetch_timeout_seconds,
                crawl_fn=crawl_fn,
                crawler=crawler,
            )
            fetch_telemetry.complete()
        final_url = str(page.get("final_url") or safe_url)
        html = str(page.get("html") or "")
        crawl_metadata = page.get("metadata") or {}
        if final_url != safe_url:
            final_url = assert_url_is_fetchable(final_url, blocked_domains)
        with span_scope("tinysearch.extract", operation="extract") as extract_telemetry:
            if int(resolved["scrape_max_links"]) > 0:
                candidate_links = sanitize_and_dedupe_links(
                    extract_links_from_html(html, final_url),
                    base_url=final_url,
                    blocked_domains=blocked_domains,
                )
            extract_telemetry.complete(
                result_count=len(candidate_links),
                attributes={"tinysearch.link.candidate.count": len(candidate_links)},
            )
        markdown = str(page.get("markdown_raw") or "")

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
        )
        links, link_tokens = _links_under_budget(
            links,
            max_tokens=int(resolved["scrape_max_link_tokens"]),
            tokenizer_name=tokenizer_name,
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
            link_tokens=link_tokens,
        )

    page_token_limit = int(resolved["crawl_max_page_tokens"])
    if page_token_limit > 0:
        markdown = truncate_text_to_max_tokens(
            markdown,
            page_token_limit,
            tokenizer_name,
        )

    with span_scope("tinysearch.extract", operation="extract") as extract_telemetry:
        chunks = chunk_text(
            text=markdown,
            max_chunk_tokens=resolved["crawl_max_chunk_tokens"],
            overlap_tokens=resolved["crawl_overlap_tokens"],
            encoding_name=tokenizer_name,
        )
        extract_telemetry.complete(
            result_count=len(chunks),
            attributes={"tinysearch.chunk.count": len(chunks)},
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
    )
    links, link_tokens = _links_under_budget(
        links,
        max_tokens=int(resolved["scrape_max_link_tokens"]),
        tokenizer_name=tokenizer_name,
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
        semaphore=(
            None
            if getattr(embedder, "manages_concurrency", False)
            else shared_embedding_semaphore(
                resolved["max_concurrent_embedding_calls"]
            )
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
        link_tokens=link_tokens,
    )

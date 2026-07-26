"""Single-URL scrape orchestrator backing /scrape and the MCP scrape_url tool.

Implements upstream issue #10: validate the URL, fetch it through the existing
Crawl4AI path or the bundled PDF/DOCX extractor, chunk and rank the extracted
markdown against the caller's query, select chunks under a token budget, and
return a grounded answer prompt plus token accounting.
"""

from __future__ import annotations

import asyncio
import socket
import urllib.error
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

from services.grounded_prompt_service import format_url_grounded_prompt
from services.site_crawl_service import (
    _extract_document_text,
    _is_document_url,
    _url_path_suffix,
    fetch_html_for_query,
    rank_chunks_bm25,
)
from services.text_chunking_service import chunk_text
from services.token_counter_service import (
    decode_tokens,
    encode_tokens,
    token_count,
)
from services.url_safety_service import (
    BlockedUrlError,
    InvalidUrlError,
    assert_url_is_fetchable,
)


DEFAULT_SCRAPE_MAX_TOKENS = 4000


class ScrapeError(Exception):
    """Base error for /scrape failures other than URL safety."""


class FetchTimeoutError(ScrapeError):
    """Fetching the URL exceeded the pipeline timeout."""


class FetchFailedError(ScrapeError):
    """Fetching the URL failed for a non-timeout reason."""


class UnsupportedDocumentError(ScrapeError):
    """The URL points to a document format the scrape pipeline cannot read."""


class EmptyContentError(ScrapeError):
    """The page produced no usable text after extraction and chunking."""


SCRAPE_ERROR_MAP: dict[type, tuple[str, int]] = {
    InvalidUrlError: ("invalid_url", 400),
    BlockedUrlError: ("blocked_url", 403),
    FetchTimeoutError: ("fetch_timeout", 504),
    FetchFailedError: ("fetch_failed", 502),
    UnsupportedDocumentError: ("unsupported_document", 415),
    EmptyContentError: ("empty_content", 422),
}


@dataclass(frozen=True)
class ScrapeResult:
    answer: str
    url: str
    title: str
    query: str
    content_tokens: int
    answer_tokens: int
    truncated: bool
    retrieved_at: str
    metadata: dict[str, str | None] | None = None

    def to_response(self, *, include_metadata: bool) -> dict[str, Any]:
        payload = asdict(self)
        if not include_metadata:
            payload.pop("metadata", None)
        return payload


class _HtmlMetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered == "title":
            self._in_title = True
            return
        if lowered != "meta":
            return
        attr_map = {key.lower(): (value or "") for key, value in attrs}
        key = (attr_map.get("name") or attr_map.get("property") or "").strip().lower()
        value = attr_map.get("content", "").strip()
        if key and value:
            self.meta.setdefault(key, value)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)


def _parse_html_meta_and_title(html: str) -> tuple[dict[str, str], str]:
    parser = _HtmlMetaParser()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:
        return {}, ""
    title = "".join(parser.title_parts).strip()
    return parser.meta, title


def _scan_html_meta(html: str) -> dict[str, str]:
    meta, _ = _parse_html_meta_and_title(html)
    return meta


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_title(crawl_metadata: dict[str, Any], html: str) -> str:
    title = _coerce_str(crawl_metadata.get("title"))
    if title:
        return title
    _, html_title = _parse_html_meta_and_title(html)
    return html_title


def _extract_metadata(
    crawl_metadata: dict[str, Any], html: str
) -> dict[str, str | None]:
    html_meta = _scan_html_meta(html)

    def _pick(*keys: str) -> str | None:
        for key in keys:
            val = _coerce_str(crawl_metadata.get(key))
            if val:
                return val
            val = html_meta.get(key.lower(), "").strip()
            if val:
                return val
        return None

    return {
        "description": _pick("description", "og:description"),
        "author": _pick("author", "article:author"),
        "published_date": _pick(
            "article:published_time",
            "og:article:published_time",
            "date",
            "datePublished",
        ),
    }


def _utc_iso8601_z(now: datetime | None = None) -> str:
    moment = now or datetime.now(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _select_chunks_under_budget(
    ranked: list[dict[str, Any]],
    max_tokens: int,
    tokenizer: str,
) -> tuple[list[dict[str, Any]], int, bool]:
    selected: list[dict[str, Any]] = []
    total = 0
    truncated = False
    for chunk in ranked:
        chunk_tokens = int(chunk.get("tokens") or 0)
        if total + chunk_tokens > max_tokens:
            truncated = True
            break
        selected.append(chunk)
        total += chunk_tokens
    if not selected and ranked:
        first = ranked[0]
        text = str(first.get("text") or "")
        tokens = encode_tokens(text, tokenizer)
        if len(tokens) > max_tokens:
            truncated_text = decode_tokens(tokens[:max_tokens], tokenizer)
            selected.append({**first, "text": truncated_text, "tokens": max_tokens})
            total = max_tokens
            truncated = True
    return selected, total, truncated


HtmlCrawlFn = Callable[..., Awaitable[dict[str, Any]]]
DocumentExtractFn = Callable[[str], tuple[str, str]]


async def _fetch_html_with_timeout(
    *,
    url: str,
    query: str,
    bm25_threshold: float,
    bm25_language: str,
    timeout_seconds: float,
    crawl_fn: HtmlCrawlFn,
) -> dict[str, Any]:
    try:
        async with asyncio.timeout(timeout_seconds):
            return await crawl_fn(
                url=url,
                user_query=query,
                bm25_threshold=bm25_threshold,
                bm25_language=bm25_language,
            )
    except (asyncio.TimeoutError, TimeoutError) as exc:
        raise FetchTimeoutError(f"fetch timed out after {timeout_seconds}s") from exc
    except (InvalidUrlError, BlockedUrlError):
        raise
    except Exception as exc:  # noqa: BLE001 - mapped to a stable user-facing code
        raise FetchFailedError(f"fetch failed: {exc}") from exc


async def _extract_document_with_timeout(
    *,
    url: str,
    timeout_seconds: float,
    document_fn: DocumentExtractFn,
) -> tuple[str, str]:
    try:
        async with asyncio.timeout(timeout_seconds):
            return await asyncio.to_thread(document_fn, url)
    except (asyncio.TimeoutError, TimeoutError) as exc:
        raise FetchTimeoutError(f"fetch timed out after {timeout_seconds}s") from exc
    except ValueError as exc:
        raise UnsupportedDocumentError(str(exc)) from exc
    except (urllib.error.URLError, socket.timeout) as exc:
        if isinstance(exc, socket.timeout):
            raise FetchTimeoutError(f"download timed out: {exc}") from exc
        raise FetchFailedError(f"download failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise FetchFailedError(f"document extraction failed: {exc}") from exc


async def scrape_url(
    url: str,
    query: str,
    *,
    max_tokens: int = DEFAULT_SCRAPE_MAX_TOKENS,
    include_metadata: bool = True,
    config: dict[str, Any],
    tokenizer_name: str,
    crawl_fn: HtmlCrawlFn | None = None,
    document_fn: DocumentExtractFn | None = None,
) -> ScrapeResult:
    """Inspect a single URL and return a grounded answer prompt for `query`.

    `config` carries the research-config values we need (blocked_domains,
    crawl/chunk parameters, pipeline_timeout_seconds). `tokenizer_name` is
    resolved by the caller via `research_tokenizer_name(config)` so we do not
    have to import the embedding stack here.
    """
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise ValueError("query must not be empty")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    blocked_domains = config.get("blocked_domains") or []
    safe_url = assert_url_is_fetchable(url, blocked_domains)

    timeout_seconds = float(config.get("pipeline_timeout_seconds") or 120.0)
    max_chunk_tokens = int(config.get("crawl_max_chunk_tokens") or 500)
    overlap_tokens = int(config.get("crawl_overlap_tokens") or 80)
    bm25_threshold = float(config.get("crawl_bm25_threshold") or 1.5)
    bm25_language = str(config.get("crawl_bm25_language") or "english")

    is_document = _is_document_url(safe_url)
    final_url = safe_url
    markdown = ""
    html = ""
    crawl_metadata: dict[str, Any] = {}

    if is_document:
        suffix = _url_path_suffix(safe_url)
        if suffix == "doc":
            raise UnsupportedDocumentError(
                "legacy .doc files are not supported; use PDF or DOCX"
            )
        document_fn = document_fn or _extract_document_text
        markdown, _document_type = await _extract_document_with_timeout(
            url=safe_url,
            timeout_seconds=timeout_seconds,
            document_fn=document_fn,
        )
    else:
        crawl_fn = crawl_fn or fetch_html_for_query
        page = await _fetch_html_with_timeout(
            url=safe_url,
            query=cleaned_query,
            bm25_threshold=bm25_threshold,
            bm25_language=bm25_language,
            timeout_seconds=timeout_seconds,
            crawl_fn=crawl_fn,
        )
        final_url = str(page.get("final_url") or safe_url)
        html = str(page.get("html") or "")
        crawl_metadata = page.get("metadata") or {}
        if final_url != safe_url:
            final_url = assert_url_is_fetchable(final_url, blocked_domains)
        markdown_raw = str(page.get("markdown_raw") or "")
        markdown_fit = str(page.get("markdown_fit") or "")
        markdown = markdown_fit or markdown_raw

    if not markdown or not markdown.strip():
        raise EmptyContentError(f"no readable content extracted from {final_url}")

    chunks = chunk_text(
        text=markdown,
        max_chunk_tokens=max_chunk_tokens,
        overlap_tokens=overlap_tokens,
        encoding_name=tokenizer_name,
    )
    if not chunks:
        raise EmptyContentError(f"no chunks produced from {final_url}")

    ranked = rank_chunks_bm25(query=cleaned_query, chunks=chunks, top_k=len(chunks))
    if not ranked:
        ranked = chunks

    selected, content_tokens, truncated = _select_chunks_under_budget(
        ranked, max_tokens, tokenizer_name
    )
    if not selected:
        raise EmptyContentError(f"no chunk fit the max_tokens budget for {final_url}")

    title = "" if is_document else _extract_title(crawl_metadata, html)
    metadata: dict[str, str | None] | None
    if not include_metadata:
        metadata = None
    elif is_document:
        metadata = {"description": None, "author": None, "published_date": None}
    else:
        metadata = _extract_metadata(crawl_metadata, html)

    answer = format_url_grounded_prompt(
        question=cleaned_query,
        url=final_url,
        title=title,
        ranked_chunks=selected,
    )
    answer_tokens = token_count(answer, tokenizer_name)

    return ScrapeResult(
        answer=answer,
        url=final_url,
        title=title,
        query=cleaned_query,
        content_tokens=content_tokens,
        answer_tokens=answer_tokens,
        truncated=truncated,
        retrieved_at=_utc_iso8601_z(),
        metadata=metadata,
    )

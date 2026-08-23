"""Shared result types, errors, and extraction helpers for URL scraping."""

from __future__ import annotations

import asyncio
import socket
import urllib.error
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any

from tinysearch.services.token_counter_service import (
    decode_tokens,
    encode_tokens,
)
from tinysearch.services.url_safety_service import (
    BlockedUrlError,
    InvalidUrlError,
)


DEFAULT_SCRAPE_MAX_TOKENS = 4000
DEFAULT_SCRAPE_MAX_LINKS = 8
DEFAULT_SCRAPE_MAX_LINK_TOKENS = 500


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
    url: str
    title: str
    query: str
    chunks: list[dict[str, Any]]
    content_tokens: int
    truncated: bool
    retrieved_at: str
    metadata: dict[str, str | None] | None = None
    links: list[dict[str, Any]] = field(default_factory=list)
    link_tokens: int = 0

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


def scan_html_meta(html: str) -> dict[str, str]:
    meta, _ = _parse_html_meta_and_title(html)
    return meta


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def extract_title(crawl_metadata: dict[str, Any], html: str) -> str:
    title = _coerce_str(crawl_metadata.get("title"))
    if title:
        return title
    _, html_title = _parse_html_meta_and_title(html)
    return html_title


def extract_metadata(
    crawl_metadata: dict[str, Any], html: str
) -> dict[str, str | None]:
    html_meta = scan_html_meta(html)

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


def utc_iso8601_z(now: datetime | None = None) -> str:
    moment = now or datetime.now(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def select_chunks_under_budget(
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


async def fetch_html_with_timeout(
    *,
    url: str,
    query: str | None,
    bm25_threshold: float,
    bm25_language: str,
    timeout_seconds: float,
    crawl_fn: HtmlCrawlFn,
    crawler: Any | None = None,
) -> dict[str, Any]:
    try:
        async with asyncio.timeout(timeout_seconds):
            kwargs: dict[str, Any] = {
                "url": url,
                "user_query": query,
                "bm25_threshold": bm25_threshold,
                "bm25_language": bm25_language,
            }
            if crawler is not None:
                kwargs["crawler"] = crawler
            return await crawl_fn(**kwargs)
    except (asyncio.TimeoutError, TimeoutError) as exc:
        raise FetchTimeoutError(f"fetch timed out after {timeout_seconds}s") from exc
    except (InvalidUrlError, BlockedUrlError):
        raise
    except Exception as exc:  # noqa: BLE001 - mapped to a stable user-facing code
        raise FetchFailedError(f"fetch failed: {exc}") from exc


async def extract_document_with_timeout(
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

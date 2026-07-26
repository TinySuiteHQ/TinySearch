from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import sys
import urllib.error
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


@lru_cache(maxsize=1)
def _async_web_crawler_cls() -> Any:
    from crawl4ai import AsyncWebCrawler

    return AsyncWebCrawler


@lru_cache(maxsize=1)
def _ddgs_cls() -> Any:
    from ddgs import DDGS

    return DDGS


@lru_cache(maxsize=1)
def _ddgs_exceptions() -> tuple[type[Exception], type[Exception], type[Exception]]:
    from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

    return DDGSException, RatelimitException, TimeoutException


def _ensure_utf8_stdio() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # py311+
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


@dataclass(frozen=True)
class SearchResult:
    result_id: int
    title: str
    url: str
    text: str


class SearchBackendError(Exception):
    """Base error raised when a search backend fails to produce results."""


class SearchBackendUnavailable(SearchBackendError):
    """Network, timeout, non-200, or non-JSON response from a backend."""


class SearchBackendBlocked(SearchBackendError):
    """Backend rejected the request (HTTP 403/429 or CAPTCHA/challenge page)."""


ALLOWED_SEARCH_BACKENDS: frozenset[str] = frozenset(
    {"searxng", "duckduckgo", "auto", "ddgs"}
)
DEFAULT_SEARXNG_URL = "http://searxng:8080/search"
DEFAULT_DDGS_REGION = "us-en"
DEFAULT_DDGS_BACKEND = "auto"
_DEFAULT_SEARXNG_TIMEOUT = 8.0
_DEFAULT_DDGS_TIMEOUT = 20.0
_DEFAULT_BRAVE_TIMEOUT = 10.0
BRAVE_API_KEY_ENV_VAR = "BRAVE_SEARCH_API_KEY"
_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


def normalize_domain(value: str) -> str:
    raw = value.strip().lower()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw or raw.startswith("//") else f"//{raw}")
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host or any(char.isspace() for char in host):
        return ""
    return host.removeprefix("www.")


def is_blocked_domain(url: str, blocked_domains: Iterable[str]) -> bool:
    host = normalize_domain(url)
    if not host:
        return False
    for blocked in blocked_domains:
        blocked_host = normalize_domain(blocked)
        if not blocked_host:
            continue
        if host == blocked_host or host.endswith(f".{blocked_host}"):
            return True
    return False


def filter_blocked_search_results(
    search_results: list[SearchResult],
    blocked_domains: Iterable[str],
) -> list[SearchResult]:
    return [
        result
        for result in search_results
        if not is_blocked_domain(result.url, blocked_domains)
    ]


def _extract_links_from_html(html: str) -> list[str]:
    # Very small/fast extractor; good enough for basic crawling.
    return list(dict.fromkeys(re.findall(r'href="([^"]+)"', html, flags=re.IGNORECASE)))


async def crawl(url: str) -> dict:
    """
    Crawl a page with crawl4ai and return basic extracted content.

    Returns a dict with: url, markdown, html, links.
    """
    _ensure_utf8_stdio()
    AsyncWebCrawler = _async_web_crawler_cls()
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    html = getattr(result, "html", "") or ""
    markdown = getattr(result, "markdown", "") or ""
    links = _extract_links_from_html(html) if html else []
    return {"url": url, "markdown": markdown, "html": html, "links": links}


def _ddgs_search(
    query: str,
    limit: int,
    *,
    region: str | None = None,
    backend: str = DEFAULT_DDGS_BACKEND,
    timeout: float = _DEFAULT_DDGS_TIMEOUT,
) -> list[SearchResult]:
    """Query the ddgs package's automatic text-search backend selection."""
    DDGS = _ddgs_cls()
    DDGSException, RatelimitException, TimeoutException = _ddgs_exceptions()

    kwargs: dict[str, Any] = {
        "safesearch": "moderate",
        "max_results": limit,
        "backend": backend or DEFAULT_DDGS_BACKEND,
    }
    if region:
        kwargs["region"] = region

    try:
        raw_results = DDGS(timeout=timeout).text(query, **kwargs)
    except TimeoutException as exc:
        raise SearchBackendUnavailable(f"ddgs timed out: {exc}") from exc
    except RatelimitException as exc:
        raise SearchBackendBlocked(f"ddgs rate limited the request: {exc}") from exc
    except DDGSException as exc:
        if str(exc) == "No results found.":
            return []
        raise SearchBackendUnavailable(f"ddgs request failed: {exc}") from exc

    if not isinstance(raw_results, list):
        raise SearchBackendUnavailable("ddgs returned a malformed response")

    out: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        target = str(item.get("href") or "").strip()
        text_field = str(item.get("body") or "").strip()
        if not title or not target:
            continue
        out.append(
            SearchResult(
                result_id=len(out) + 1,
                title=title,
                url=target,
                text=text_field,
            )
        )
        if len(out) >= limit:
            break

    return out


def _read_brave_api_key() -> str | None:
    return os.environ.get(BRAVE_API_KEY_ENV_VAR, "").strip() or None


def _brave_search(
    query: str,
    limit: int,
    *,
    api_key: str,
    timeout: float = _DEFAULT_BRAVE_TIMEOUT,
) -> list[SearchResult]:
    """Query Brave's official Web Search API."""
    full_url = f"{_BRAVE_SEARCH_URL}?{urlencode([('q', query), ('count', str(limit))])}"
    req = Request(
        full_url,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        },
    )

    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 429):
            raise SearchBackendBlocked(
                f"Brave refused the request (HTTP {exc.code})"
            ) from exc
        raise SearchBackendUnavailable(f"Brave returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
        raise SearchBackendUnavailable("Brave unreachable") from exc

    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise SearchBackendUnavailable("Brave did not return JSON") from exc

    if not isinstance(payload, dict):
        raise SearchBackendUnavailable("Brave JSON payload was not an object")

    web = payload.get("web")
    raw_results = web.get("results") if isinstance(web, dict) else []
    if raw_results is None:
        raw_results = []
    if not isinstance(raw_results, list):
        raise SearchBackendUnavailable("Brave 'web.results' field was not a list")

    out: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        target = str(item.get("url") or "").strip()
        text_field = str(item.get("description") or "").strip()
        if not title or not target:
            continue
        out.append(
            SearchResult(
                result_id=len(out) + 1,
                title=title,
                url=target,
                text=text_field,
            )
        )
        if len(out) >= limit:
            break

    return out


def _with_brave_fallback(
    primary: Callable[[], list[SearchResult]],
    query: str,
    limit: int,
) -> list[SearchResult]:
    """Wrap a DDGS-backed search call with a keyed Brave fallback.

    Brave is only ever consulted when BRAVE_SEARCH_API_KEY is present, and
    only when the primary (DDGS) call errors or returns no results.
    """
    api_key = _read_brave_api_key()

    try:
        results = primary()
    except SearchBackendError as primary_exc:
        if not api_key:
            raise
        try:
            return _brave_search(query, limit, api_key=api_key)
        except SearchBackendError as brave_exc:
            raise SearchBackendUnavailable(
                f"ddgs failed ({type(primary_exc).__name__}); "
                f"brave failed ({type(brave_exc).__name__})"
            ) from primary_exc

    if results or not api_key:
        return results

    return _brave_search(query, limit, api_key=api_key)


def _normalize_engines(engines: Any) -> str:
    if engines is None:
        return ""
    if isinstance(engines, str):
        parts = [part.strip() for part in engines.split(",")]
    elif isinstance(engines, Sequence):
        parts = [str(part).strip() for part in engines]
    else:
        parts = [str(engines).strip()]
    return ",".join(part for part in parts if part)


def _searxng_search(
    query: str,
    limit: int,
    *,
    url: str,
    engines: Any = None,
    region: str | None = None,
    timeout: float = _DEFAULT_SEARXNG_TIMEOUT,
) -> list[SearchResult]:
    """Query a SearXNG-compatible JSON endpoint."""
    if not url or not url.strip():
        raise SearchBackendUnavailable("SearXNG search_backend_url is empty")

    params: list[tuple[str, str]] = [
        ("q", query),
        ("format", "json"),
        ("pageno", "1"),
    ]
    engines_str = _normalize_engines(engines)
    if engines_str:
        params.append(("engines", engines_str))
    if region:
        params.append(("language", str(region)))

    full_url = f"{url}?{urlencode(params)}"
    req = Request(
        full_url,
        headers={
            "User-Agent": "TinySearch/0.1 (+searxng)",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            content_type = resp.headers.get("Content-Type", "") or ""
    except urllib.error.HTTPError as exc:
        raise SearchBackendUnavailable(
            f"SearXNG returned HTTP {exc.code} from {url}"
        ) from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
        raise SearchBackendUnavailable(
            f"SearXNG unreachable at {url}: {exc}"
        ) from exc

    text = raw.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SearchBackendUnavailable(
            "SearXNG did not return JSON "
            f"(content-type={content_type!r}). "
            "Enable JSON output by adding 'json' to search.formats in searxng settings.yml."
        ) from exc

    if not isinstance(payload, dict):
        raise SearchBackendUnavailable("SearXNG JSON payload was not an object")

    raw_results = payload.get("results") or []
    if not isinstance(raw_results, list):
        raise SearchBackendUnavailable("SearXNG 'results' field was not a list")

    out: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        target = str(item.get("url") or "").strip()
        text_field = str(item.get("content") or "").strip()
        if not title or not target:
            continue
        out.append(
            SearchResult(
                result_id=len(out) + 1,
                title=title,
                url=target,
                text=text_field,
            )
        )
        if len(out) >= limit:
            break

    return out


def _load_search_config() -> dict[str, Any]:
    # Lazy import to avoid a circular dependency with research_config_service.
    from services.research_config_service import load_research_config

    return load_research_config()


def _dispatch_search(
    query: str,
    limit: int,
    *,
    config: dict[str, Any],
) -> list[SearchResult]:
    backend = str(config.get("search_backend") or "searxng").strip().lower()
    if backend not in ALLOWED_SEARCH_BACKENDS:
        backend = "searxng"
    url = str(config.get("search_backend_url") or DEFAULT_SEARXNG_URL)
    engines = config.get("search_engines")
    region = (
        config.get("search_region")
        or config.get("search_country")
        or ""
    )
    fallback_enabled = bool(config.get("search_backend_fallback", True))
    ddgs_backend = str(config.get("ddgs_backend") or DEFAULT_DDGS_BACKEND)
    ddgs_timeout = float(config.get("ddgs_timeout_seconds") or _DEFAULT_DDGS_TIMEOUT)

    if backend == "ddgs":
        return _with_brave_fallback(
            lambda: _ddgs_search(
                query,
                limit,
                region=str(region) or None,
                backend=ddgs_backend,
                timeout=ddgs_timeout,
            ),
            query,
            limit,
        )

    if backend == "duckduckgo":
        return _with_brave_fallback(
            lambda: _ddgs_search(
                query, limit, region=str(region) or None, backend="duckduckgo", timeout=ddgs_timeout
            ),
            query,
            limit,
        )

    if backend == "auto":
        try:
            return _searxng_search(
                query, limit, url=url, engines=engines, region=str(region) or None
            )
        except SearchBackendError:
            return _ddgs_search(
                query, limit, region=str(region) or None, backend="duckduckgo", timeout=ddgs_timeout
            )

    # backend == "searxng"
    try:
        return _searxng_search(
            query, limit, url=url, engines=engines, region=str(region) or None
        )
    except SearchBackendError:
        if fallback_enabled:
            return _ddgs_search(
                query, limit, region=str(region) or None, backend="duckduckgo", timeout=ddgs_timeout
            )
        raise


def search(query: str, limit: int = 10) -> list[SearchResult]:
    """
    Run a web search using the configured backend.

    Returns items shaped like:
      Title:
      URL:
      Text:
    """
    config = _load_search_config()
    return _dispatch_search(query, limit, config=config)


def search_to_markdown(search_results: list[SearchResult]) -> str:
    markdown = ""
    for result in search_results:
        markdown += f"## {result.result_id}. {result.title}\n"
        markdown += f"URL: {result.url}\n"
        markdown += f"Text: {result.text}\n\n"
    return markdown

def search_markdown(query: str, limit: int = 10) -> str:
    search_results = search(query, limit)
    return search_to_markdown(search_results)

if __name__ == "__main__":
    _ensure_utf8_stdio()
    print(search_markdown("What is model context protocol?", limit=10))

    crawl_result = asyncio.run(crawl("https://example.com"))
    print(crawl_result.get("markdown", "")[:500])

from __future__ import annotations

import asyncio
import json
import os
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
    published_at: str | None = None


@dataclass(frozen=True)
class SearchResponse:
    """Backend results plus the backend that ultimately supplied them."""

    results: list[SearchResult]
    backend: str


class BackendResults(list[SearchResult]):
    """List-compatible backend results with a compact partial-outage signal."""

    def __init__(
        self, values: Iterable[SearchResult], *, degraded: bool = False, reason: str = ""
    ) -> None:
        super().__init__(values)
        self.degraded = degraded
        self.reason = reason


class SearchBackendError(Exception):
    """Base error raised when a search backend fails to produce results."""


class SearchBackendUnavailable(SearchBackendError):
    """Network, timeout, non-200, or non-JSON response from a backend."""


class SearchBackendBlocked(SearchBackendError):
    """Backend rejected the request (HTTP 403/429 or CAPTCHA/challenge page)."""


def normalize_domains(values: Sequence[str]) -> list[str]:
    """Validate a positive domain allowlist without broadening malformed input."""
    normalized = [normalize_domain(value) for value in values]
    if any(not value for value in normalized):
        raise ValueError("search item domains must contain valid domains")
    return list(dict.fromkeys(normalized))


def matches_domain(url: str, domains: Sequence[str]) -> bool:
    if not domains:
        return True
    host = normalize_domain(url)
    return bool(host) and any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _restricted_query(query: str, domains: Sequence[str]) -> str:
    """Use the portable `site:` operator where ordinary web backends support it."""
    if not domains:
        return query
    clauses = [f"site:{domain}" for domain in domains]
    return f"{query} ({' OR '.join(clauses)})"


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
        published_at = _published_at(item)
        if not title or not target:
            continue
        out.append(
            SearchResult(
                result_id=len(out) + 1,
                title=title,
                url=target,
                text=text_field,
                published_at=published_at,
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
        published_at = _published_at(item)
        if not title or not target:
            continue
        out.append(
            SearchResult(
                result_id=len(out) + 1,
                title=title,
                url=target,
                text=text_field,
                published_at=published_at,
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


def _with_brave_fallback_with_metadata(
    primary: Callable[[], list[SearchResult]],
    query: str,
    limit: int,
    *,
    primary_backend: str,
) -> SearchResponse:
    """Metadata-preserving variant of the legacy list-returning fallback."""
    api_key = _read_brave_api_key()
    try:
        results = primary()
    except SearchBackendError as primary_exc:
        if not api_key:
            raise
        try:
            return SearchResponse(_brave_search(query, limit, api_key=api_key), "brave")
        except SearchBackendError as brave_exc:
            raise SearchBackendUnavailable(
                f"{primary_backend} failed ({type(primary_exc).__name__}); "
                f"brave failed ({type(brave_exc).__name__})"
            ) from primary_exc
    if results or not api_key:
        return SearchResponse(results, primary_backend)
    return SearchResponse(_brave_search(query, limit, api_key=api_key), "brave")


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


def _published_at(item: dict[str, Any]) -> str | None:
    """Return an upstream result date without deriving or fetching one."""
    for key in ("publishedDate", "published_at", "date", "published", "pubdate"):
        value = item.get(key)
        if value is None:
            continue
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        value = str(value).strip()
        if value:
            return value
    return None


def _unresponsive_engine_names(payload: dict[str, Any]) -> list[str]:
    """Extract engine names from SearXNG's `unresponsive_engines` field.

    SearXNG reports these as `[["google", "CAPTCHA"], ...]` engine/reason
    pairs, but the shape has varied across versions and a bare list of names
    also appears, so both are accepted.
    """
    raw = payload.get("unresponsive_engines")
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for entry in raw:
        if isinstance(entry, str):
            name = entry.strip()
        elif isinstance(entry, (list, tuple)) and entry:
            engine = str(entry[0]).strip()
            reason = str(entry[1]).strip() if len(entry) > 1 else ""
            name = f"{engine} ({reason})" if reason else engine
        else:
            continue
        if name:
            names.append(name)
    return names


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
        published_at = _published_at(item)
        if not title or not target:
            continue
        out.append(
            SearchResult(
                result_id=len(out) + 1,
                title=title,
                url=target,
                text=text_field,
                published_at=published_at,
            )
        )
        if len(out) >= limit:
            break

    unresponsive = _unresponsive_engine_names(payload)
    if not out:
        if unresponsive:
            # SearXNG answers 200 with an empty result set when its engines are
            # rate-limited, CAPTCHA-challenged, or suspended; the failure shows
            # up in `unresponsive_engines`, not in the status code. Returned as
            # a successful empty list it is indistinguishable from a genuine
            # no-match, so callers report "nothing found" while the whole search
            # layer is down and the configured fallback never engages.
            raise SearchBackendBlocked(
                "SearXNG returned no results; engines unresponsive: "
                + ", ".join(unresponsive)
            )

    return BackendResults(
        out,
        degraded=bool(unresponsive),
        reason=("unresponsive engines: " + ", ".join(unresponsive)) if unresponsive else "",
    )


def _load_search_config() -> dict[str, Any]:
    # Lazy import to avoid a circular dependency with tinysearch_config_service.
    from tinysearch.services.tinysearch_config_service import load_tinysearch_config

    return load_tinysearch_config()


def _dispatch_search_with_metadata(
    query: str,
    limit: int,
    *,
    config: dict[str, Any],
) -> SearchResponse:
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
        return _with_brave_fallback_with_metadata(
            lambda: _ddgs_search(
                query,
                limit,
                region=str(region) or None,
                backend=ddgs_backend,
                timeout=ddgs_timeout,
            ),
            query,
            limit,
            primary_backend="ddgs",
        )

    if backend == "duckduckgo":
        return _with_brave_fallback_with_metadata(
            lambda: _ddgs_search(
                query, limit, region=str(region) or None, backend="duckduckgo", timeout=ddgs_timeout
            ),
            query,
            limit,
            primary_backend="duckduckgo",
        )

    if backend == "auto":
        try:
            return SearchResponse(_searxng_search(
                query, limit, url=url, engines=engines, region=str(region) or None
            ), "searxng")
        except SearchBackendError:
            return SearchResponse(_ddgs_search(
                query, limit, region=str(region) or None, backend="duckduckgo", timeout=ddgs_timeout
            ), "duckduckgo")

    # backend == "searxng"
    try:
        return SearchResponse(_searxng_search(
            query, limit, url=url, engines=engines, region=str(region) or None
        ), "searxng")
    except SearchBackendError:
        if not fallback_enabled:
            # Fallback disabled means SearXNG-only by policy: a deployment that
            # requires every query to leave through its own instance must not be
            # silently redirected to DDGS or Brave.
            raise
        return _with_brave_fallback_with_metadata(
            lambda: _ddgs_search(
                query,
                limit,
                region=str(region) or None,
                backend="duckduckgo",
                timeout=ddgs_timeout,
            ),
            query,
            limit,
            primary_backend="duckduckgo",
        )


def _dispatch_search(
    query: str,
    limit: int,
    *,
    config: dict[str, Any],
) -> list[SearchResult]:
    """Compatibility wrapper for the existing research pipeline."""
    return _dispatch_search_with_metadata(query, limit, config=config).results


def search(
    query: str,
    limit: int = 10,
    *,
    config: dict[str, Any] | None = None,
) -> list[SearchResult]:
    """
    Run a web search using the configured backend.

    Returns items shaped like:
      Title:
      URL:
      Text:
    """
    resolved_config = _load_search_config() if config is None else config
    return _dispatch_search(query, limit, config=resolved_config)


def search_with_metadata(
    query: str,
    limit: int = 10,
    *,
    config: dict[str, Any] | None = None,
) -> SearchResponse:
    """Run a search and report the backend that supplied its results."""
    resolved_config = _load_search_config() if config is None else config
    return _dispatch_search_with_metadata(query, limit, config=resolved_config)


def _attempt_state(exc: SearchBackendError) -> str:
    if isinstance(exc, SearchBackendBlocked):
        return "blocked"
    message = str(exc).lower()
    if any(marker in message for marker in ("malformed", "did not return json", "was not a list", "was not an object")):
        return "invalid_response"
    return "unavailable"


def _backend_attempt_plan(
    config: dict[str, Any], query: str, limit: int
) -> list[tuple[str, Callable[[], list[SearchResult]]]]:
    """Build the existing configured route as explicit, one-shot backend attempts."""
    backend = str(config.get("search_backend") or "searxng").strip().lower()
    url = str(config.get("search_backend_url") or DEFAULT_SEARXNG_URL)
    engines = config.get("search_engines")
    region = str(config.get("search_region") or config.get("search_country") or "") or None
    ddgs_backend = str(config.get("ddgs_backend") or DEFAULT_DDGS_BACKEND)
    ddgs_timeout = float(config.get("ddgs_timeout_seconds") or _DEFAULT_DDGS_TIMEOUT)
    searxng = lambda: _searxng_search(query, limit, url=url, engines=engines, region=region)
    ddgs = lambda: _ddgs_search(query, limit, region=region, backend=ddgs_backend, timeout=ddgs_timeout)
    duckduckgo = lambda: _ddgs_search(query, limit, region=region, backend="duckduckgo", timeout=ddgs_timeout)
    brave_key = _read_brave_api_key()
    brave = (lambda: _brave_search(query, limit, api_key=brave_key)) if brave_key else None

    if backend == "ddgs":
        return [("ddgs", ddgs)] + ([("brave", brave)] if brave else [])
    if backend == "duckduckgo":
        return [("duckduckgo", duckduckgo)] + ([("brave", brave)] if brave else [])
    if backend == "auto":
        return [("searxng", searxng), ("duckduckgo", duckduckgo)]
    attempts: list[tuple[str, Callable[[], list[SearchResult]]]] = [("searxng", searxng)]
    if bool(config.get("search_backend_fallback", True)):
        attempts.append(("duckduckgo", duckduckgo))
        if brave:
            attempts.append(("brave", brave))
    return attempts


@dataclass(frozen=True)
class BatchSearchResponse:
    results: list[SearchResult]
    domains: list[str]
    status: str
    attempts: list[dict[str, Any]]
    error: dict[str, str] | None
    latency_ms: int


async def search_batch_with_metadata(
    items: Sequence[dict[str, Any]],
    *,
    limit: int,
    config: dict[str, Any],
    concurrency: int,
) -> list[BatchSearchResponse]:
    """Run isolated search items concurrently while retaining caller order."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_item(item: dict[str, Any]) -> BatchSearchResponse:
        async with semaphore:
            started = asyncio.get_running_loop().time()
            try:
                query = str(item["query"]).strip() if isinstance(item.get("query"), str) else ""
                if not query:
                    raise ValueError("every search item requires a non-empty query string")
                raw_domains = item.get("domains", [])
                if not isinstance(raw_domains, list) or not all(isinstance(domain, str) for domain in raw_domains):
                    raise ValueError("search item domains must be a list of strings")
                domains = normalize_domains(raw_domains)
            except ValueError as exc:
                return BatchSearchResponse([], [], "error", [], {"code": "invalid_request", "message": str(exc)}, round((asyncio.get_running_loop().time() - started) * 1000))
            attempts: list[dict[str, Any]] = []
            constrained_query = _restricted_query(query, domains)
            for backend, call in _backend_attempt_plan(config, constrained_query, limit):
                attempt_started = asyncio.get_running_loop().time()
                try:
                    raw_results = await asyncio.to_thread(call)
                    filtered = [
                        result for result in raw_results
                        if result.url.lower().startswith(("http://", "https://"))
                        and not is_blocked_domain(result.url, config["blocked_domains"])
                        and matches_domain(result.url, domains)
                    ]
                    state = "degraded" if getattr(raw_results, "degraded", False) else "responded"
                    # SearXNG can return useful results while individual engines are
                    # degraded. Its legacy list API does not expose that detail, so
                    # retain the compact observable distinction when it does.
                    attempt: dict[str, Any] = {
                        "backend": backend,
                        "state": state,
                        "result_count": len(filtered),
                        "latency_ms": round((asyncio.get_running_loop().time() - attempt_started) * 1000),
                    }
                    if state == "degraded" and getattr(raw_results, "reason", ""):
                        attempt["reason"] = raw_results.reason
                    attempts.append(attempt)
                    if filtered:
                        return BatchSearchResponse(filtered, domains, "ok", attempts, None, round((asyncio.get_running_loop().time() - started) * 1000))
                except SearchBackendError as exc:
                    attempts.append({
                        "backend": backend,
                        "state": _attempt_state(exc),
                        "result_count": 0,
                        "reason": str(exc),
                        "latency_ms": round((asyncio.get_running_loop().time() - attempt_started) * 1000),
                    })
            if attempts and all(attempt["state"] != "responded" for attempt in attempts):
                last = attempts[-1]
                return BatchSearchResponse([], domains, "error", attempts, {"code": last["state"], "message": str(last.get("reason") or "all search backends failed")}, round((asyncio.get_running_loop().time() - started) * 1000))
            return BatchSearchResponse([], domains, "ok", attempts, None, round((asyncio.get_running_loop().time() - started) * 1000))

    return await asyncio.gather(*(run_item(item) for item in items))


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

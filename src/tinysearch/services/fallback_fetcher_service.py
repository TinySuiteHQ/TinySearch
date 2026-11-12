"""Optional second fetch path for pages the built-in browser could not read.

When the direct render answers HTTP 403, 429, or 503, or yields no readable
text, the scrape pipeline can ask an operator-run fetch service for the page
instead. TinySearch defines the contract; what runs behind it -- a separate
rendering service, a hosted fetch API, or anything else that speaks it -- is
the operator's choice.

Contract: ``POST <scrape_fallback_fetcher_url>`` with the JSON body
``{"url": "<page url>", "timeout_seconds": <float>}``. A 2xx answer carries
``{"html": "<page html>", "final_url"?: "<url after redirects>",
"status"?: <upstream HTTP status>}``. Any other answer is a failed fallback.

The returned HTML is rendered offline (JavaScript off, network blocked), so
the fetch service can supply page content but never code that runs here.
"""

from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from typing import Any

from tinysearch.services.site_crawl_service import render_html_text


# Forbidden, too many requests, temporarily unavailable: answers where a second
# fetch path can still return the page, unlike "not found" or "gone".
FALLBACK_TRIGGER_STATUSES = frozenset({403, 429, 503})

# A rendered page is a few MB at most; anything far beyond that is not one.
_MAX_FALLBACK_RESPONSE_BYTES = 20 * 1024 * 1024
# Headroom on top of the service's own deadline for the HTTP round trip.
_TRANSPORT_GRACE_SECONDS = 5.0

FallbackFetchFn = Callable[..., Awaitable[dict[str, Any]]]


class FallbackFetchError(Exception):
    """The fallback fetch service could not supply the page."""


def page_status(page: dict[str, Any]) -> int | None:
    metadata = page.get("metadata") or {}
    try:
        return int(metadata.get("status"))
    except (TypeError, ValueError):
        return None


# Page text is an accessibility outline ("- heading "Title" [level=1]",
# "- paragraph: text", "- /url: /path"). Its role prefixes, attribute markers,
# and property lines are structure, not words a reader sees.
_OUTLINE_PROPERTY_LINE_RE = re.compile(r"^\s*-\s+/\w+:.*$", re.MULTILINE)
_OUTLINE_ROLE_PREFIX_RE = re.compile(r"^\s*-\s+[a-z]+\b:?", re.MULTILINE)
_OUTLINE_ATTRIBUTE_RE = re.compile(r"\[[a-z-]+(?:=[^\]]*)?\]")
_WORD_RE = re.compile(r"[^\W_]+")


def visible_word_count(text: str) -> int:
    """Count the words a reader would see, ignoring outline markup."""
    text = _OUTLINE_PROPERTY_LINE_RE.sub(" ", text)
    text = _OUTLINE_ROLE_PREFIX_RE.sub(" ", text)
    text = _OUTLINE_ATTRIBUTE_RE.sub(" ", text)
    return len(_WORD_RE.findall(text))


def page_needs_fallback(page: dict[str, Any], *, min_words: int = 0) -> bool:
    """True when the page answered a trigger status or has too little text.

    `min_words` = 0 accepts any non-empty page; a positive value also treats
    pages with fewer visible words as unreadable (e.g. a short holding page
    shown before the real content).
    """
    if page_status(page) in FALLBACK_TRIGGER_STATUSES:
        return True
    text = str(page.get("markdown_raw") or "")
    if not text.strip():
        return True
    return min_words > 0 and visible_word_count(text) < min_words


def request_fallback_html(
    fetcher_url: str, url: str, timeout_seconds: float
) -> dict[str, Any]:
    """Blocking call to the fetch service; returns {html, final_url, status}."""
    body = json.dumps({"url": url, "timeout_seconds": timeout_seconds}).encode()
    request = urllib.request.Request(
        fetcher_url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout_seconds + _TRANSPORT_GRACE_SECONDS
        ) as response:
            raw = response.read(_MAX_FALLBACK_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise FallbackFetchError(f"fallback fetcher answered HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise FallbackFetchError(f"fallback fetcher unreachable: {exc}") from exc
    if len(raw) > _MAX_FALLBACK_RESPONSE_BYTES:
        raise FallbackFetchError("fallback fetcher response too large")
    try:
        answer = json.loads(raw)
    except ValueError as exc:
        raise FallbackFetchError("fallback fetcher returned invalid JSON") from exc
    if not isinstance(answer, dict) or not isinstance(answer.get("html"), str):
        raise FallbackFetchError("fallback fetcher response has no html string")

    final_url = answer.get("final_url")
    status = answer.get("status")
    return {
        "html": answer["html"],
        "final_url": final_url if isinstance(final_url, str) and final_url else url,
        "status": status if isinstance(status, int) else None,
    }


async def fetch_page_via_fallback(
    *,
    url: str,
    fetcher_url: str,
    timeout_seconds: float,
    crawler: Any | None = None,
) -> dict[str, Any]:
    """Fetch through the service, then extract text like a direct render."""
    answer = await asyncio.to_thread(
        request_fallback_html, fetcher_url, url, timeout_seconds
    )
    rendered = await render_html_text(answer["html"], crawler=crawler)
    return {
        "final_url": answer["final_url"],
        "html": answer["html"],
        "markdown_raw": str(rendered.get("markdown_raw") or ""),
        "metadata": {"title": rendered.get("title") or "", "status": answer["status"]},
    }

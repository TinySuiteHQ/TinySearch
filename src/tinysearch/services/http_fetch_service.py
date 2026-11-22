"""Plain HTTP fetch: the first, cheapest tier of a scrape.

Most pages carry their text in the HTML the server sends, so they can be read
without loading the page in a browser and running its scripts. This module
fetches that HTML over plain HTTP and decides whether it is usable as it is,
or whether the page needs a full browser load (see
``DirectPlaywrightCrawler.fetch``).

Every redirect hop is checked with ``assert_url_is_fetchable`` before it is
requested, so a redirect cannot reach a private address or a blocked domain.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import httpx

from tinysearch.services.url_safety_service import assert_url_is_fetchable

MAX_REDIRECTS = 5
PLAIN_FETCH_TIMEOUT_SECONDS = 15.0
# An HTML page is rarely over a few MB; anything far larger is not one to read.
MAX_PLAIN_RESPONSE_BYTES = 10 * 1024 * 1024
# Below this many words of static text, signs of a JavaScript-built page count.
_STATIC_TEXT_WORDS_FOR_APP_SIGNALS = 200

def _user_agent() -> str:
    """Identifies TinySearch by name and version, with a link to what it is."""
    try:
        version = importlib.metadata.version("tinysuite-search")
    except importlib.metadata.PackageNotFoundError:
        version = "dev"
    return f"TinySearch/{version} (+https://github.com/TinySuiteHQ/TinySearch)"


_REQUEST_HEADERS = {
    "User-Agent": _user_agent(),
    # Matches the browser contexts' `locale`, so both tiers ask for the same page.
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
}
_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")


class PlainFetchError(Exception):
    """The plain fetch could not produce a response worth reading."""


class DocumentResponse(Exception):
    """The URL answered with a document (PDF, DOCX), not a web page.

    Raised by the fetch tiers so the scrape reads the URL as a document,
    whatever its path looks like.
    """


# Documents the scrape can read, by the Content-Type a server sends for them.
DOCUMENT_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024


# ---------- proxies ----------


def configured_proxies(config: Mapping[str, Any]) -> list[dict[str, str]]:
    """The configured proxies, in Playwright's proxy shape, for both fetch tiers."""
    servers = [
        entry.strip()
        for entry in str(config.get("browser_proxy_server") or "").split(",")
        if entry.strip()
    ]
    username = str(config.get("browser_proxy_username") or "").strip()
    password = str(config.get("browser_proxy_password") or "").strip()
    bypass = str(config.get("browser_proxy_bypass") or "").strip()
    proxies: list[dict[str, str]] = []
    for server in servers:
        proxy: dict[str, str] = {"server": server}
        if username:
            proxy["username"] = username
        if password:
            proxy["password"] = password
        if bypass:
            proxy["bypass"] = bypass
        proxies.append(proxy)
    return proxies


def _bypasses_proxy(host: str, bypass: str) -> bool:
    """Playwright's bypass list: '.example.com' matches subdomains, 'example.com' the host and its subdomains."""
    host = host.lower().rstrip(".")
    for entry in (part.strip().lower() for part in bypass.split(",")):
        if not entry:
            continue
        if entry == "*":
            return True
        if entry.startswith("."):
            if host.endswith(entry) or host == entry[1:]:
                return True
        elif host == entry or host.endswith(f".{entry}"):
            return True
    return False


def httpx_proxy_url(proxy: Mapping[str, str], url: str) -> str | None:
    """Turn a Playwright-shaped proxy into an httpx proxy URL, or None to go direct.

    Raises PlainFetchError for SOCKS proxies: httpx needs an extra package for
    them, so those pages go to the browser tier, which supports them.
    """
    bypass = proxy.get("bypass") or ""
    host = urlparse(url).hostname or ""
    if bypass and _bypasses_proxy(host, bypass):
        return None
    server = proxy["server"]
    if "://" not in server:
        server = f"http://{server}"
    parsed = urlparse(server)
    if parsed.scheme not in {"http", "https"}:
        raise PlainFetchError(f"{parsed.scheme} proxies are only supported by the browser")
    username = proxy.get("username")
    if username:
        password = proxy.get("password") or ""
        auth = f"{quote(username, safe='')}:{quote(password, safe='')}@"
        server = f"{parsed.scheme}://{auth}{parsed.netloc}"
    return server


_proxy_cursor = 0


def _next_proxy_url(config: Mapping[str, Any], url: str) -> str | None:
    global _proxy_cursor
    proxies = configured_proxies(config)
    if not proxies:
        return None
    proxy = proxies[_proxy_cursor % len(proxies)]
    _proxy_cursor += 1
    return httpx_proxy_url(proxy, url)


# ---------- fetching ----------


_META_CHARSET_RE = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?([\w.:-]+)""", re.IGNORECASE)


def _decode(body: bytes, header_charset: str | None) -> str:
    for candidate in (header_charset, _meta_charset(body), "utf-8"):
        if not candidate:
            continue
        try:
            return body.decode(candidate, errors="replace")
        except LookupError:
            continue
    return body.decode("utf-8", errors="replace")


def _meta_charset(body: bytes) -> str | None:
    match = _META_CHARSET_RE.search(body[:4096])
    return match.group(1).decode("ascii", errors="ignore") if match else None


async def _read_capped(response: httpx.Response) -> bytes:
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_PLAIN_RESPONSE_BYTES:
        raise PlainFetchError("response too large for a plain fetch")
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > MAX_PLAIN_RESPONSE_BYTES:
            raise PlainFetchError("response too large for a plain fetch")
        chunks.append(chunk)
    return b"".join(chunks)


async def fetch_plain(
    url: str,
    *,
    config: Mapping[str, Any],
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """GET ``url`` without a browser, following redirects one checked hop at a time.

    Returns ``{"final_url", "status", "content_type", "html"}``. Raises the
    URL-safety errors as they are (a blocked hop must not be retried by the
    browser), and PlainFetchError or an httpx error when the fetch fails.
    ``transport`` is for tests.
    """
    blocked_domains = list(config.get("blocked_domains") or [])
    timeout = min(
        float(config.get("pipeline_timeout_seconds") or 120.0), PLAIN_FETCH_TIMEOUT_SECONDS
    )
    current = url
    for _hop in range(MAX_REDIRECTS + 1):
        current = await asyncio.to_thread(assert_url_is_fetchable, current, blocked_domains)
        client_options: dict[str, Any] = {
            "follow_redirects": False,
            "timeout": timeout,
            "headers": _REQUEST_HEADERS,
        }
        if transport is not None:
            client_options["transport"] = transport
        else:
            proxy = _next_proxy_url(config, current)
            if proxy:
                client_options["proxy"] = proxy
        async with httpx.AsyncClient(**client_options) as client:
            async with client.stream("GET", current) as response:
                location = response.headers.get("location")
                if response.is_redirect and location:
                    current = urljoin(current, location)
                    continue
                body = await _read_capped(response)
                content_type = response.headers.get("content-type", "")
                return {
                    "final_url": str(response.url),
                    "status": response.status_code,
                    "content_type": content_type.split(";")[0].strip().lower(),
                    "html": _decode(body, response.charset_encoding),
                }
    raise PlainFetchError(f"more than {MAX_REDIRECTS} redirects")


def download_document(
    url: str,
    *,
    config: Mapping[str, Any],
    timeout: float,
    max_bytes: int = MAX_DOCUMENT_BYTES,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Download a document's bytes, checking every redirect hop like a page fetch.

    Blocking (document extraction runs in a worker thread). Returns
    ``{"final_url", "content_type", "body"}``; raises the URL-safety errors as
    they are, and PlainFetchError for an error status, too many redirects or
    a body over ``max_bytes``. ``transport`` is for tests.
    """
    blocked_domains = list(config.get("blocked_domains") or [])
    current = url
    for _hop in range(MAX_REDIRECTS + 1):
        current = assert_url_is_fetchable(current, blocked_domains)
        client_options: dict[str, Any] = {
            "follow_redirects": False,
            "timeout": timeout,
            "headers": {**_REQUEST_HEADERS, "Accept": "*/*"},
        }
        if transport is not None:
            client_options["transport"] = transport
        else:
            proxy = _next_proxy_url(config, current)
            if proxy:
                client_options["proxy"] = proxy
        with httpx.Client(**client_options) as client:
            with client.stream("GET", current) as response:
                location = response.headers.get("location")
                if response.is_redirect and location:
                    current = urljoin(current, location)
                    continue
                if response.status_code >= 400:
                    raise PlainFetchError(f"{current} answered HTTP {response.status_code}")
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise PlainFetchError("document too large to read")
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise PlainFetchError("document too large to read")
                    chunks.append(chunk)
                content_type = response.headers.get("content-type", "")
                return {
                    "final_url": str(response.url),
                    "content_type": content_type.split(";")[0].strip().lower(),
                    "body": b"".join(chunks),
                }
    raise PlainFetchError(f"more than {MAX_REDIRECTS} redirects")


# ---------- is the plain response usable? ----------


_META_REFRESH_RE = re.compile(r"""<meta[^>]+http-equiv\s*=\s*["']?refresh""", re.IGNORECASE)
_EMPTY_APP_ROOT_RE = re.compile(
    r"""<div\b[^>]*\bid\s*=\s*["']?(?:root|app|__next|__nuxt|svelte|main-app)["']?[^>]*>\s*</div>""",
    re.IGNORECASE,
)
_NOSCRIPT_RE = re.compile(r"<noscript\b[^>]*>(.*?)</noscript>", re.IGNORECASE | re.DOTALL)
_JS_REQUIRED_RE = re.compile(
    r"(?:enable|turn on|requires?|need)\b[^<]{0,40}javascript|javascript[^<]{0,40}(?:required|disabled|enable)",
    re.IGNORECASE,
)
_NON_TEXT_BLOCK_RE = re.compile(
    r"<(script|style|noscript|template|svg)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
_TAG_RE = re.compile(r"<[^>]+>")
_WORD_RE = re.compile(r"[^\W_]+")


_SCRIPT_BLOCK_RE = re.compile(r"<script\b[^>]*>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL)


def script_share(html: str) -> float:
    """How much of the HTML is inline script, from 0 to 1."""
    if not html:
        return 0.0
    return sum(len(block) for block in _SCRIPT_BLOCK_RE.findall(html)) / len(html)


def static_text_words(html: str) -> int:
    """Roughly how many words the HTML shows without running anything."""
    text = _NON_TEXT_BLOCK_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", text)
    return len(_WORD_RE.findall(text))


def plain_response_needs_browser(status: int, content_type: str, html: str) -> str | None:
    """Why this plain response can't be read as it is, or None when it can.

    Every signal is deterministic. A false "needs the browser" only costs time;
    the page is still read, by the full page load.
    """
    if not 200 <= status < 300:
        return f"status_{status}"
    if content_type and content_type not in _HTML_CONTENT_TYPES:
        return "not_html"
    if _META_REFRESH_RE.search(html):
        return "meta_refresh"
    # A server-rendered page can carry these too (an app shell around real
    # text, a "JavaScript required for comments" note), so they only count
    # when the HTML itself has little to read.
    if static_text_words(html) < _STATIC_TEXT_WORDS_FOR_APP_SIGNALS:
        if _EMPTY_APP_ROOT_RE.search(html):
            return "javascript_app"
        for block in _NOSCRIPT_RE.findall(html):
            if _JS_REQUIRED_RE.search(_TAG_RE.sub(" ", block)):
                return "javascript_required"
    return None

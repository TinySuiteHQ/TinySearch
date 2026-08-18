"""Discover and rank next-hop links found on an already-fetched page.

Links are captured from the page's raw HTML before markdown link-filtering
strips navigation context, resolved against the page URL, reduced to safe
HTTP(S) targets, and canonicalized/deduplicated. Ranking reuses the same
hybrid BM25 + dense retrieval used for content chunks (see
``hybrid_embed_search_service.rank_chunks_hybrid``), scored against the link
text plus nearby page context. This surfaces candidates for a caller to
follow next; it never fetches them.
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

from tinysearch.services.url_safety_service import (
    BlockedUrlError,
    InvalidUrlError,
    enforce_blocked_domains,
    validate_public_url,
)


_SKIP_TEXT_TAGS = frozenset({"script", "style", "noscript", "template", "svg"})
_CONTEXT_WINDOW_CHARS = 240
_MAX_LINK_TEXT_CHARS = 160


class _LinkParser(HTMLParser):
    """Collects anchor href/text pairs plus the text immediately before each link."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[dict[str, str]] = []
        self._skip_depth = 0
        self._anchor_stack: list[dict[str, Any]] = []
        self._context_buffer: list[str] = []

    def _context_text(self) -> str:
        return "".join(self._context_buffer)[-_CONTEXT_WINDOW_CHARS:]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in _SKIP_TEXT_TAGS:
            self._skip_depth += 1
            return
        if lowered == "a" and self._skip_depth == 0:
            attr_map = {key.lower(): (value or "") for key, value in attrs}
            href = attr_map.get("href", "").strip()
            self._anchor_stack.append(
                {"href": href, "text_parts": [], "context": self._context_text()}
            )

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() in _SKIP_TEXT_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in _SKIP_TEXT_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if lowered == "a" and self._anchor_stack:
            anchor = self._anchor_stack.pop()
            text = "".join(anchor["text_parts"])
            if anchor["href"]:
                self.links.append(
                    {
                        "href": anchor["href"],
                        "text": text,
                        "context": anchor["context"],
                    }
                )
            if text:
                self._context_buffer.append(text)
                self._context_buffer = [self._context_text()]

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._anchor_stack:
            self._anchor_stack[-1]["text_parts"].append(data)
            return
        self._context_buffer.append(data)
        self._context_buffer = [self._context_text()]


def extract_links_from_html(html: str, base_url: str) -> list[dict[str, str]]:
    """Return raw ``{href, text, context}`` dicts for every anchor in ``html``.

    ``href`` values are left unresolved; ``base_url`` is accepted for a
    consistent signature with the sanitizing step but is not used here.
    """
    del base_url
    parser = _LinkParser()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:
        return []
    return parser.links


def _clip_link_text(text: str, *, max_chars: int = _MAX_LINK_TEXT_CHARS) -> str:
    """Cap a link's visible text to a label-sized string, not a full card blob.

    Anchors that wrap an entire card (heading + date + preview paragraph, a
    common blog/listing pattern) yield very long inner text; callers expect a
    short label like the rest of the candidates, not a paragraph.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].rstrip()
    return f"{truncated}…" if truncated else f"{text[:max_chars].rstrip()}…"


def _canonicalize(resolved_url: str) -> str | None:
    parsed = urlsplit(resolved_url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return None
    if not parsed.hostname:
        return None
    path = parsed.path or "/"
    return urlunsplit((scheme, parsed.netloc.lower(), path, parsed.query, ""))


def sanitize_and_dedupe_links(
    raw_links: list[dict[str, str]],
    *,
    base_url: str,
    blocked_domains: list[str],
) -> list[dict[str, str]]:
    """Resolve, filter, canonicalize and deduplicate links found on ``base_url``.

    Rejects same-page fragments, non-HTTP(S) schemes, and blocked domains.
    Performs no DNS resolution: these are candidates handed back to the
    caller to follow, not URLs this pipeline fetches itself.
    """
    base_canonical = _canonicalize(base_url)
    seen: set[str] = set()
    output: list[dict[str, str]] = []
    for link in raw_links:
        href = (link.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        resolved = urljoin(base_url, href)
        canonical = _canonicalize(resolved)
        if not canonical or canonical == base_canonical:
            continue
        try:
            validate_public_url(canonical)
            enforce_blocked_domains(canonical, blocked_domains)
        except (InvalidUrlError, BlockedUrlError):
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        output.append(
            {
                "url": canonical,
                "text": _clip_link_text(" ".join((link.get("text") or "").split())),
                "context": " ".join((link.get("context") or "").split()),
            }
        )
    return output

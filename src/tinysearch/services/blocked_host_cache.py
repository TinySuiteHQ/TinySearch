"""Remember, for a short time, hosts that refused automated access.

When a site answers HTTP 403, 429, or 503 and no readable copy is available,
asking it again a minute later (the same page, or another page on the same
site) costs a full fetch and gets the same answer. Callers of the scrape tool
are stateless, so TinySearch cannot tell which requests belong to one task;
the memory is therefore per host, with a time limit.

The memory only ever *skips* a fetch. It never retries, changes how a request
looks, or tries any other way past the refusal: a refused host is answered
from here until the entry expires, and is then fetched normally, once.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from urllib.parse import urlsplit

from tinysearch.services.scrape_service import FetchUnavailableError


# Same statuses that make the scrape pipeline report FetchUnavailableError.
REFUSAL_STATUSES = frozenset({403, 429, 503})


class HostRecentlyRefusedError(FetchUnavailableError):
    """The host refused automated access a moment ago, so it was not asked again."""


@dataclass(frozen=True)
class HostRefusal:
    host: str
    status: int
    url: str
    refused_at: float
    expires_at: float


def host_key(url: str) -> str:
    """The site a URL belongs to: lowercase hostname, without a leading ``www.``."""
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


class BlockedHostCache:
    """Process-local, time-limited memory of hosts that refused a fetch."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._entries: dict[str, HostRefusal] = {}
        self._lock = Lock()

    def remember(self, url: str, status: int | None, *, ttl_seconds: float) -> None:
        """Record a refusal. Anything but 403/429/503, or a TTL of 0, is ignored."""
        if status not in REFUSAL_STATUSES or ttl_seconds <= 0:
            return
        host = host_key(url)
        if not host:
            return
        now = self._clock()
        with self._lock:
            self._entries[host] = HostRefusal(
                host=host,
                status=int(status),
                url=url,
                refused_at=now,
                expires_at=now + ttl_seconds,
            )

    def refusal_for(self, url: str) -> HostRefusal | None:
        """The live refusal for this URL's host, or None (expired entries are dropped)."""
        host = host_key(url)
        now = self._clock()
        with self._lock:
            entry = self._entries.get(host)
            if entry is None:
                return None
            if now >= entry.expires_at:
                del self._entries[host]
                return None
            return entry

    def refused_error(self, url: str, refusal: HostRefusal) -> HostRecentlyRefusedError:
        now = self._clock()
        ago = max(0, round(now - refusal.refused_at))
        left = max(1, round(refusal.expires_at - now))
        return HostRecentlyRefusedError(
            f"{url} was not fetched: {refusal.host} refused automated access "
            f"{ago}s ago (HTTP {refusal.status} for {refusal.url}), and TinySearch "
            f"will not ask it again for {left}s. Other pages on this site will be "
            "refused the same way; a person can open the page in their own browser."
        )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# The one memory shared by every scrape_urls and browser_navigate call in this process.
SHARED_REFUSED_HOSTS = BlockedHostCache()

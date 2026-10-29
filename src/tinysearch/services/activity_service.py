"""Opt-in record of what the servers did, for an operator or a client UI.

Off by default, and nothing is written while it is off. ``TINYSEARCH_ACTIVITY``:

- ``off``: nothing (default). OpenTelemetry stays payload-free either way.
- ``aggregate``: per operation, outcome, latency and error code, plus how often
  each *domain* turned up in search results ("discovered") or was actually
  fetched. No queries, no URLs, nothing tied to a caller.
- ``history``: aggregate, plus each caller's own queries and URLs, readable
  only by that caller. A caller is a trusted identity (see
  management_service.current_principal) or the local stdio user; activity
  from anyone else is kept as aggregate only.
- ``content``: history, plus the text each tool call returned to the caller:
  exactly what the model was shown, never whole pages. Its size is whatever
  the tools' own limits (``scrape_max_tokens``, ``browser_response_char_budget``)
  let through, so nothing is cut that the model saw.

Rows older than ``TINYSEARCH_ACTIVITY_RETENTION_DAYS`` (default 7) and beyond
``TINYSEARCH_ACTIVITY_MAX_EVENTS`` (default 5000) are deleted as new ones
arrive. Callers are stored as salted hashes, never as the raw identity.
The store is one stdlib SQLite file (``TINYSEARCH_ACTIVITY_DB``).
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import platformdirs

from tinysearch.paths import private_file
from tinysearch.services.web_search_service import normalize_domain

MODES = ("off", "aggregate", "history", "content")
# The modes that keep each caller's own rows (queries, URLs, and in content
# mode what they were shown).
DETAILED_MODES = ("history", "content")
_PURGE_EVERY_SECONDS = 60.0
_MAX_QUERY_CHARS = 500
_MAX_URL_CHARS = 2000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  principal TEXT,
  operation TEXT NOT NULL,
  outcome TEXT NOT NULL,
  latency_ms INTEGER,
  item_count INTEGER,
  error_code TEXT,
  query TEXT,
  content TEXT
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS events_principal ON events(principal, ts);
CREATE TABLE IF NOT EXISTS urls (
  event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
  ts TEXT NOT NULL,
  principal TEXT,
  url TEXT,
  domain TEXT NOT NULL,
  source TEXT NOT NULL,
  outcome TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS urls_domain ON urls(domain);
CREATE INDEX IF NOT EXISTS urls_principal ON urls(principal, ts);
CREATE INDEX IF NOT EXISTS urls_event ON urls(event_id);
"""


def activity_mode() -> str:
    mode = os.environ.get("TINYSEARCH_ACTIVITY", "").strip().lower() or "off"
    if mode not in MODES:
        raise ValueError(f"TINYSEARCH_ACTIVITY must be one of {', '.join(MODES)}")
    return mode


def _bounded_int(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def retention_days() -> int:
    return _bounded_int("TINYSEARCH_ACTIVITY_RETENTION_DAYS", 7, 1, 365)


def max_events() -> int:
    return _bounded_int("TINYSEARCH_ACTIVITY_MAX_EVENTS", 5000, 100, 1_000_000)


def activity_db_path() -> Path:
    raw = os.environ.get("TINYSEARCH_ACTIVITY_DB", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(platformdirs.user_data_dir("tinysearch")) / "activity.db"


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


class ActivityStore:
    """One SQLite file, one connection, one lock: writes are small and rare."""

    def __init__(self, path: Path) -> None:
        self.path = path
        # Queries, URLs and what the model was shown: owner-only.
        private_file(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            # A store from before content mode has no column for it.
            columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(events)")}
            if "content" not in columns:
                self._conn.execute("ALTER TABLE events ADD COLUMN content TEXT")
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'salt'").fetchone()
            if row is None:
                self._salt = secrets.token_hex(16)
                self._conn.execute("INSERT INTO meta (key, value) VALUES ('salt', ?)", (self._salt,))
                self._conn.commit()
            else:
                self._salt = str(row["value"])
        self._last_purge = 0.0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def principal_key(self, principal: str | None) -> str | None:
        """What is stored for a caller: a salted hash, never the identity."""
        if not principal:
            return None
        return hashlib.sha256(f"{self._salt}:{principal}".encode()).hexdigest()[:32]

    # ---------- writing ----------

    def record(
        self,
        *,
        operation: str,
        outcome: str,
        latency_ms: int | None,
        item_count: int,
        error_code: str | None = None,
        queries: Sequence[str] = (),
        discovered: Iterable[str] = (),
        fetched: Iterable[tuple[str, str]] = (),
        shown: str | None = None,
        principal: str | None,
        mode: str,
    ) -> None:
        if mode == "off":
            return
        # Raw queries and URLs only in a detailed mode, and only for a caller
        # who can later read (and delete) them; everything else is counts.
        detailed = mode in DETAILED_MODES and principal is not None
        key = self.principal_key(principal) if detailed else None
        ts = _iso(_now())
        query = "\n".join(q for q in queries if q)[:_MAX_QUERY_CHARS] if detailed else None
        content = shown if detailed and mode == "content" else None
        url_rows: list[tuple[str | None, str, str, str]] = []
        for url in discovered:
            domain = normalize_domain(url)
            if domain:
                url_rows.append((url[:_MAX_URL_CHARS] if detailed else None, domain, "discovered", "ok"))
        for url, result in fetched:
            domain = normalize_domain(url)
            if domain:
                url_rows.append((url[:_MAX_URL_CHARS] if detailed else None, domain, "fetched", result))
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO events (ts, principal, operation, outcome, latency_ms, item_count, error_code, query, content)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, key, operation, outcome, latency_ms, item_count, error_code, query, content),
            )
            event_id = cursor.lastrowid
            self._conn.executemany(
                "INSERT INTO urls (event_id, ts, principal, url, domain, source, outcome)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(event_id, ts, key, url, domain, source, result) for url, domain, source, result in url_rows],
            )
            self._conn.commit()
        if time.monotonic() - self._last_purge >= _PURGE_EVERY_SECONDS:
            self.purge()

    def purge(self, *, days: int | None = None, keep: int | None = None) -> None:
        cutoff = _iso(_now() - timedelta(days=days or retention_days()))
        keep = keep or max_events()
        with self._lock:
            self._conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
            self._conn.execute(
                "DELETE FROM events WHERE id <= (SELECT id FROM events ORDER BY id DESC LIMIT 1 OFFSET ?)",
                (keep,),
            )
            self._conn.commit()
        self._last_purge = time.monotonic()

    def clear(self, principal: str | None = None) -> int:
        """Delete one caller's history, or everything when principal is None."""
        with self._lock:
            if principal is None:
                deleted = self._conn.execute("DELETE FROM events").rowcount
            else:
                deleted = self._conn.execute(
                    "DELETE FROM events WHERE principal = ?", (self.principal_key(principal),)
                ).rowcount
            self._conn.commit()
        return int(deleted)

    # ---------- reading ----------

    def _rows(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    def summary(self) -> dict[str, Any]:
        since_day = _iso(_now() - timedelta(days=1))
        rows = self._rows(
            """
            SELECT operation,
                   COUNT(*) AS total,
                   SUM(ts >= ?) AS last_day,
                   SUM(outcome NOT IN ('ok')) AS failed,
                   AVG(latency_ms) AS avg_latency_ms,
                   MIN(ts) AS oldest
            FROM events GROUP BY operation
            """,
            (since_day,),
        )
        return {row["operation"]: row for row in rows}

    def domains(self, *, search: str | None, source: str | None, offset: int, limit: int) -> dict[str, Any]:
        where, params = ["1=1"], []
        if search:
            where.append("domain LIKE ? ESCAPE '!'")
            params.append("%" + search.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%")
        if source in {"discovered", "fetched"}:
            where.append("source = ?")
            params.append(source)
        clause = " AND ".join(where)
        rows = self._rows(
            f"""
            SELECT domain,
                   SUM(source = 'discovered') AS discovered,
                   SUM(source = 'fetched' AND outcome = 'ok') AS fetched,
                   SUM(source = 'fetched' AND outcome != 'ok') AS failed,
                   MAX(ts) AS last_seen
            FROM urls WHERE {clause}
            GROUP BY domain
            ORDER BY (SUM(source = 'discovered') + SUM(source = 'fetched')) DESC, domain
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        )
        total = self._rows(f"SELECT COUNT(DISTINCT domain) AS n FROM urls WHERE {clause}", params)[0]["n"]
        return {"rows": rows, "total": int(total or 0)}

    def events_for(self, principal: str, *, search: str | None, offset: int, limit: int) -> dict[str, Any]:
        key = self.principal_key(principal)
        where, params = ["principal = ?"], [key]
        if search:
            where.append("query LIKE ? ESCAPE '!'")
            params.append("%" + search.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%")
        clause = " AND ".join(where)
        rows = self._rows(
            f"SELECT id, ts, operation, outcome, query, item_count, latency_ms, error_code,"
            f" LENGTH(content) AS shown_chars"
            f" FROM events WHERE {clause} ORDER BY id DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        total = self._rows(f"SELECT COUNT(*) AS n FROM events WHERE {clause}", params)[0]["n"]
        return {"rows": rows, "total": int(total or 0)}

    def event_for(self, principal: str, event_id: int) -> dict[str, Any] | None:
        """One of this caller's own events with its URLs and what it showed; None for anyone else's."""
        key = self.principal_key(principal)
        rows = self._rows(
            "SELECT id, ts, operation, outcome, query, item_count, latency_ms, error_code, content"
            " FROM events WHERE id = ? AND principal = ?",
            (event_id, key),
        )
        if not rows:
            return None
        event = rows[0]
        event["urls"] = self._rows(
            "SELECT url, source, outcome FROM urls WHERE event_id = ? AND principal = ? AND url IS NOT NULL"
            " ORDER BY rowid",
            (event_id, key),
        )
        return event

    def urls_for(self, principal: str, *, source: str | None, search: str | None, offset: int, limit: int) -> dict[str, Any]:
        key = self.principal_key(principal)
        where, params = ["principal = ?", "url IS NOT NULL"], [key]
        if source in {"discovered", "fetched"}:
            where.append("source = ?")
            params.append(source)
        if search:
            where.append("url LIKE ? ESCAPE '!'")
            params.append("%" + search.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%")
        clause = " AND ".join(where)
        rows = self._rows(
            f"SELECT ts, url, domain, source, outcome FROM urls WHERE {clause}"
            " ORDER BY ts DESC, rowid DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        total = self._rows(f"SELECT COUNT(*) AS n FROM urls WHERE {clause}", params)[0]["n"]
        return {"rows": rows, "total": int(total or 0)}


_store: ActivityStore | None = None
_store_lock = threading.Lock()


def get_store() -> ActivityStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = ActivityStore(activity_db_path())
        return _store


def reset_store() -> None:
    """Close the process-wide store (tests, shutdown)."""
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
        _store = None


# ---------- what an operation's result says happened ----------


def record_search(result: Mapping[str, Any], *, principal: str | None, latency_ms: int,
                  shown: str | None = None) -> None:
    mode = activity_mode()
    if mode == "off":
        return
    items = result.get("items") or []
    get_store().record(
        operation="search",
        outcome=str(result.get("status") or "ok"),
        latency_ms=latency_ms,
        item_count=len(items),
        error_code=next((str(i["error"].get("code") if isinstance(i.get("error"), Mapping) else "search_error")
                         for i in items if i.get("status") == "error"), None),
        queries=[str(i.get("query") or "") for i in items],
        discovered=[str(r.get("url") or "") for i in items for r in (i.get("results") or [])],
        shown=shown,
        principal=principal,
        mode=mode,
    )


def record_scrape(result: Mapping[str, Any], *, principal: str | None, latency_ms: int,
                  shown: str | None = None) -> None:
    mode = activity_mode()
    if mode == "off":
        return
    fetched: list[tuple[str, str]] = []
    queries: list[str] = []
    error_code = None
    for item in result.get("results") or []:
        if item.get("status") == "ok":
            envelope = item.get("result") or {}
            source = (envelope.get("sources") or [{}])[0]
            fetched.append((str(source.get("url") or envelope.get("stats", {}).get("requested_url") or ""), "ok"))
            queries.append(str(envelope.get("query") or ""))
        else:
            code = str((item.get("error") or {}).get("code") or "error")
            error_code = error_code or code
            fetched.append((str(item.get("url") or ""), code))
            queries.append(str(item.get("query") or ""))
    get_store().record(
        operation="scrape",
        outcome=str(result.get("status") or "ok"),
        latency_ms=latency_ms,
        item_count=len(fetched),
        error_code=error_code,
        queries=[q for q in queries if q and q != "*"],
        fetched=fetched,
        shown=shown,
        principal=principal,
        mode=mode,
    )


def record_browser(action: str, *, url: str | None, ok: bool, error_code: str | None,
                   principal: str | None, latency_ms: int, shown: str | None = None) -> None:
    mode = activity_mode()
    if mode == "off":
        return
    get_store().record(
        operation=f"browser_{action}",
        outcome="ok" if ok else "error",
        latency_ms=latency_ms,
        item_count=1,
        error_code=error_code,
        fetched=[(url, "ok" if ok else (error_code or "error"))] if url else [],
        shown=shown,
        principal=principal,
        mode=mode,
    )


_warned_failure = False


def record_quietly(kind: str, *args: Any, **kwargs: Any) -> None:
    """Record without ever failing the operation it describes."""
    global _warned_failure
    recorders = {"search": record_search, "scrape": record_scrape, "browser": record_browser}
    try:
        recorders[kind](*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - bookkeeping must not break a search
        if not _warned_failure:
            _warned_failure = True
            print(f"[tinysearch] activity not recorded: {exc}", file=sys.stderr, flush=True)

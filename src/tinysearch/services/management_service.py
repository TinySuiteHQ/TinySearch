"""Management views for a client UI (the ``tinysuite.manage/1`` contract).

A client finds a resource with the ``application/vnd.tinysuite.manage+json``
media type in ``resources/list``, reads it as a manifest, and then reads the
views it names, over the same MCP session as the tools.

Off unless ``TINYSEARCH_MANAGEMENT=on``. Who may read what:

- A local stdio client is the operator: every view.
- Over HTTP, instance views (overview, domains, rules) need
  ``Authorization: Bearer <TINYSEARCH_MANAGEMENT_TOKEN>``. Without a token they
  are not offered at all.
- Over HTTP, a caller's own history (activity, pages) needs a trusted
  identity: ``TINYSEARCH_TRUSTED_USER_HEADER`` sent by a peer inside
  ``TINYSEARCH_TRUSTED_PROXY_CIDRS`` together with the proxy's shared
  ``X-TinySearch-Proxy-Token``, checked on every message, never taken from
  tool arguments. The address alone does not prove the proxy sent it:
  anything else on that network (another container, a local program when
  the server listens on loopback) could otherwise name any user.

Turning management on over HTTP with neither a token nor a trusted identity
refuses to start, rather than serving views to whoever can reach the port.
"""

from __future__ import annotations

import hmac
import importlib.metadata
import ipaddress
import os
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl, unquote

from tinysearch.services import activity_service as activity

MANIFEST_URI = "tinysearch://manage/manifest"
MANIFEST_MIME = "application/vnd.tinysuite.manage+json"
CONTRACT = "tinysuite.manage/1"
VIEW_URI = "tinysearch://manage/views/{view}/{{params}}"
ACTIVITY_ITEM_URI = "tinysearch://manage/items/activity/{id}"
LOCAL_PRINCIPAL = "local"

_ON = {"on", "1", "true", "yes"}
_OFF = {"", "off", "0", "false", "no"}
_MIN_TOKEN_CHARS = 24
_MIN_PROXY_TOKEN_CHARS = 32
PROXY_TOKEN_HEADER = "X-TinySearch-Proxy-Token"
_PAGE_MAX = 100
_SEARCH_MAX = 200
_IDENTITY_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")
_HEADER_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")


class ManagementAccessError(PermissionError):
    """The caller may not read this management view."""


def management_enabled() -> bool:
    raw = os.environ.get("TINYSEARCH_MANAGEMENT", "").strip().lower()
    if raw in _ON:
        return True
    if raw in _OFF:
        return False
    raise ValueError("TINYSEARCH_MANAGEMENT must be 'on' or 'off'")


def management_token() -> str | None:
    token = os.environ.get("TINYSEARCH_MANAGEMENT_TOKEN", "").strip()
    if token and len(token) < _MIN_TOKEN_CHARS:
        raise ValueError(
            f"TINYSEARCH_MANAGEMENT_TOKEN must be at least {_MIN_TOKEN_CHARS} characters"
        )
    return token or None


def _trusted_identity() -> tuple[str, tuple[Any, ...], str] | None:
    header = os.environ.get("TINYSEARCH_TRUSTED_USER_HEADER", "").strip()
    cidrs = os.environ.get("TINYSEARCH_TRUSTED_PROXY_CIDRS", "").strip()
    token = os.environ.get("TINYSEARCH_TRUSTED_PROXY_TOKEN", "").strip()
    if not header and not cidrs:
        return None
    if not _HEADER_RE.fullmatch(header):
        raise ValueError("TINYSEARCH_TRUSTED_USER_HEADER must be an HTTP header name")
    if header.lower() == PROXY_TOKEN_HEADER.lower():
        raise ValueError(f"TINYSEARCH_TRUSTED_USER_HEADER cannot be {PROXY_TOKEN_HEADER}")
    try:
        networks = tuple(
            ipaddress.ip_network(part.strip(), strict=False)
            for part in cidrs.split(",") if part.strip()
        )
    except ValueError as exc:
        raise ValueError("TINYSEARCH_TRUSTED_PROXY_CIDRS is invalid") from exc
    if not networks:
        raise ValueError(
            "TINYSEARCH_TRUSTED_USER_HEADER needs TINYSEARCH_TRUSTED_PROXY_CIDRS: "
            "the header is only believed from those peers"
        )
    if len(token) < _MIN_PROXY_TOKEN_CHARS:
        raise ValueError(
            f"TINYSEARCH_TRUSTED_USER_HEADER needs TINYSEARCH_TRUSTED_PROXY_TOKEN "
            f"(at least {_MIN_PROXY_TOKEN_CHARS} characters): the proxy sends it as "
            f"{PROXY_TOKEN_HEADER} on every request"
        )
    return header, networks, token


def multi_tenant() -> bool:
    """Whether callers are told apart (a trusted identity is configured)."""
    return _trusted_identity() is not None


def check_management_startup(transport: str) -> None:
    """Validate the settings now, and fail closed for unauthenticated HTTP."""
    activity.activity_mode()
    activity.retention_days()
    activity.max_events()
    identity = _trusted_identity()
    if not management_enabled():
        return
    if transport != "stdio" and not management_token() and identity is None:
        raise ValueError(
            "TINYSEARCH_MANAGEMENT=on over HTTP needs TINYSEARCH_MANAGEMENT_TOKEN "
            "or a trusted identity (TINYSEARCH_TRUSTED_USER_HEADER and _PROXY_CIDRS)"
        )


def _mcp_request() -> tuple[bool, Mapping[str, Any] | None]:
    """(inside an MCP message, that message's ASGI scope or None for stdio)."""
    try:
        from mcp.server.lowlevel.server import request_ctx
    except ImportError:
        return False, None
    context = request_ctx.get(None)
    if context is None:
        return False, None
    scope = getattr(getattr(context, "request", None), "scope", None)
    return True, scope if isinstance(scope, Mapping) else None


def principal_from_scope(scope: Mapping[str, Any]) -> str | None:
    """A trusted caller identity from one HTTP request, or None."""
    identity = _trusted_identity()
    if identity is None:
        return None
    header, networks, token = identity
    client = scope.get("client")
    try:
        peer = ipaddress.ip_address(client[0] if client else "")
    except ValueError:
        return None
    if not any(peer in network for network in networks):
        return None

    def sent(name: str) -> list[bytes]:
        wanted = name.lower().encode("ascii")
        return [bytes(v) for k, v in scope.get("headers", []) if bytes(k).lower() == wanted]

    tokens = sent(PROXY_TOKEN_HEADER)
    if len(tokens) != 1 or not hmac.compare_digest(tokens[0].strip(), token.encode()):
        return None
    values = sent(header)
    if len(values) != 1:
        return None
    try:
        value = values[0].decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    return value if _IDENTITY_RE.fullmatch(value) else None


def _stdio_transport() -> bool:
    return (os.environ.get("MCP_TRANSPORT", "stdio").strip() or "stdio") == "stdio"


def _local_operator(inside: bool, scope: Mapping[str, Any] | None) -> bool:
    # A message with no HTTP request is the local user only on a server
    # started for stdio. Decided by configuration, so an HTTP message that
    # somehow lost its request is never mistaken for the operator.
    return inside and scope is None and _stdio_transport()


def current_principal() -> str | None:
    """Whose activity this MCP call is: the stdio user, a trusted identity, or nobody."""
    inside, scope = _mcp_request()
    if _local_operator(inside, scope):
        return LOCAL_PRINCIPAL
    return principal_from_scope(scope) if scope is not None else None


def _token_presented(scope: Mapping[str, Any]) -> bool:
    token = management_token()
    if not token:
        return False
    for key, value in scope.get("headers", []):
        if bytes(key).lower() == b"authorization":
            text = bytes(value).decode("latin-1").strip()
            if text[:7].lower() == "bearer " and hmac.compare_digest(token, text[7:].strip()):
                return True
    return False


def _instance_allowed() -> bool:
    inside, scope = _mcp_request()
    return _local_operator(inside, scope) or (scope is not None and _token_presented(scope))


def authorize(view_scope: str) -> None:
    if not management_enabled():
        raise ManagementAccessError("management views are disabled on this server")
    if view_scope == "instance":
        if not _instance_allowed():
            raise ManagementAccessError("instance views need the management token")
        return
    if current_principal() is None:
        raise ManagementAccessError("this view needs a trusted caller identity")


def history_tool_enabled() -> bool:
    """Whether callers may clear their own history (a tool, so only when it means something)."""
    return management_enabled() and activity.activity_mode() in activity.DETAILED_MODES


def _version() -> str:
    try:
        return importlib.metadata.version("tinysuite-search")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


# ---------- manifest ----------


def manifest() -> dict[str, Any]:
    if not management_enabled():
        raise ManagementAccessError("management views are disabled on this server")
    views: list[dict[str, Any]] = []
    if _instance_allowed():
        views += [
            {"id": "overview", "title": "Overview", "kind": "stats", "scope": "instance",
             "resource": VIEW_URI.format(view="overview")},
            {"id": "domains", "title": "Domains", "kind": "table", "scope": "instance",
             "resource": VIEW_URI.format(view="domains"), "pageSize": 50,
             "params": [
                 {"name": "q", "label": "Domain", "type": "text", "maxLength": _SEARCH_MAX},
                 {"name": "source", "label": "Seen as", "type": "enum", "values": ["discovered", "fetched"]},
             ],
             "rowKey": "domain",
             "columns": [
                 {"key": "domain", "label": "Domain", "format": "text"},
                 {"key": "discovered", "label": "In results", "format": "number"},
                 {"key": "fetched", "label": "Fetched", "format": "number"},
                 {"key": "failed", "label": "Failed fetches", "format": "number"},
                 {"key": "last_seen", "label": "Last seen", "format": "datetime"},
             ]},
            {"id": "rules", "title": "Blocking rules", "kind": "table", "scope": "instance",
             "resource": VIEW_URI.format(view="rules"), "pageSize": 100, "rowKey": "rule",
             "columns": [
                 {"key": "rule", "label": "Rule", "format": "text"},
                 {"key": "applies_to", "label": "Applies to", "format": "text"},
                 {"key": "source", "label": "Set by", "format": "text"},
             ]},
        ]
    if current_principal() is not None:
        clear = [{
            "id": "clear", "label": "Clear my history", "tool": "clear_search_history",
            "target": "view", "args": {}, "destructive": True,
            "confirm": "Delete your whole search history on this server?",
        }] if history_tool_enabled() else []
        shown = ([{"key": "shown_chars", "label": "Shown (chars)", "format": "number"}]
                 if activity.activity_mode() == "content" else [])
        views += [
            {"id": "activity", "title": "My searches", "kind": "table", "scope": "user",
             "resource": VIEW_URI.format(view="activity"), "pageSize": 50,
             "params": [{"name": "q", "label": "Search", "type": "text", "maxLength": _SEARCH_MAX}],
             "rowKey": "id",
             "detail": {"resource": ACTIVITY_ITEM_URI},
             "columns": [
                 {"key": "ts", "label": "When", "format": "datetime"},
                 {"key": "operation", "label": "What", "format": "text"},
                 {"key": "query", "label": "Query", "format": "text"},
                 {"key": "outcome", "label": "Outcome", "format": "text"},
                 {"key": "latency_ms", "label": "ms", "format": "number"},
                 *shown,
             ],
             "actions": clear},
            {"id": "pages", "title": "My pages", "kind": "table", "scope": "user",
             "resource": VIEW_URI.format(view="pages"), "pageSize": 50,
             "params": [
                 {"name": "q", "label": "URL", "type": "text", "maxLength": _SEARCH_MAX},
                 {"name": "source", "label": "Seen as", "type": "enum", "values": ["discovered", "fetched"]},
             ],
             "rowKey": "url",
             "columns": [
                 {"key": "ts", "label": "When", "format": "datetime"},
                 {"key": "url", "label": "URL", "format": "text"},
                 {"key": "source", "label": "Seen as", "format": "text"},
                 {"key": "outcome", "label": "Outcome", "format": "text"},
             ],
             "actions": clear},
        ]
    return {
        "contract": CONTRACT,
        "id": "tinysearch",
        "title": "TinySearch",
        "description": "Web search and page fetching: what was searched, which sites, what failed.",
        "icon": "search",
        "version": _version(),
        "views": views,
    }


# ---------- views ----------


def decode_params(raw: str, allowed: set[str]) -> dict[str, str]:
    if raw in {"", "-"}:
        return {}
    params: dict[str, str] = {}
    for key, value in parse_qsl(unquote(raw), keep_blank_values=True, max_num_fields=16):
        if key not in allowed:
            raise ValueError(f"unknown view parameter {key!r}")
        params[key] = value
    return params


def _page(params: Mapping[str, str]) -> tuple[int, int]:
    def number(key: str, default: int, low: int, high: int) -> int:
        try:
            return max(low, min(high, int(params.get(key) or default)))
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer") from exc
    return number("offset", 0, 0, 10_000_000), number("limit", 50, 1, _PAGE_MAX)


def _table(result: Mapping[str, Any], offset: int, limit: int, notice: str | None = None) -> dict[str, Any]:
    rows = list(result.get("rows") or [])
    total = int(result.get("total") or 0)
    out: dict[str, Any] = {"rows": rows, "total": total, "offset": offset, "limit": limit,
                           "hasMore": offset + len(rows) < total}
    if notice:
        out["notice"] = notice
    return out


def _collection_notice(needed: str) -> str | None:
    mode = activity.activity_mode()
    if mode == "off":
        return f"Activity collection is off on this server (TINYSEARCH_ACTIVITY), so there is no {needed} to show."
    if needed == "history" and mode not in activity.DETAILED_MODES:
        return "This server keeps aggregate counts only (TINYSEARCH_ACTIVITY=aggregate), not individual searches."
    return None


def overview_view(raw: str, config: Mapping[str, Any]) -> dict[str, Any]:
    authorize("instance")
    decode_params(raw, set())
    mode = activity.activity_mode()
    items: list[dict[str, Any]] = [
        {"label": "Version", "value": _version()},
        {"label": "Search backend", "value": str(config.get("search_backend") or "")},
        {"label": "Browser tools", "value": str(config.get("browser_backend") or "")},
        {"label": "Activity collection", "value": mode},
    ]
    notice = _collection_notice("activity")
    if mode != "off":
        items.append({"label": "Kept for", "value": activity.retention_days(), "unit": "days"})
        summary = activity.get_store().summary()
        for operation, label in (("search", "Searches"), ("scrape", "Page fetches")):
            row = summary.get(operation) or {}
            items += [
                {"label": f"{label}, last 24h", "value": int(row.get("last_day") or 0)},
                {"label": f"{label}, not ok", "value": int(row.get("failed") or 0)},
                {"label": f"{label}, average", "value": round(float(row.get("avg_latency_ms") or 0)), "unit": "ms"},
            ]
        browser = [row for op, row in summary.items() if op.startswith("browser_")]
        items.append({"label": "Browser actions", "value": sum(int(r.get("total") or 0) for r in browser)})
        oldest = min((str(r["oldest"]) for r in summary.values() if r.get("oldest")), default=None)
        items.append({"label": "Since", "value": oldest, "format": "datetime"})
    out: dict[str, Any] = {"items": items}
    if notice:
        out["notice"] = notice
    return out


def domains_view(raw: str) -> dict[str, Any]:
    authorize("instance")
    params = decode_params(raw, {"q", "source", "offset", "limit"})
    offset, limit = _page(params)
    notice = _collection_notice("activity")
    if activity.activity_mode() == "off":
        return _table({}, offset, limit, notice)
    result = activity.get_store().domains(
        search=(params.get("q") or "")[:_SEARCH_MAX] or None,
        source=params.get("source") or None, offset=offset, limit=limit,
    )
    return _table(result, offset, limit)


def rules_view(raw: str, config: Mapping[str, Any]) -> dict[str, Any]:
    authorize("instance")
    decode_params(raw, {"offset", "limit"})
    rows = [
        {"rule": domain, "applies_to": "search results, fetches and links (subdomains too)",
         "source": "blocked_domains"}
        for domain in config.get("blocked_domains") or []
    ]
    # Always on, not configurable: shown so an operator knows why a fetch failed.
    rows += [
        {"rule": "Private, loopback and link-local addresses", "applies_to": "fetches and redirects",
         "source": "built in"},
        {"rule": "Credentials in URLs, non-http(s) schemes", "applies_to": "fetches", "source": "built in"},
    ]
    notice = None
    if not config.get("blocked_domains"):
        notice = ("No domains are blocked. A search item's own domains narrow that one "
                  "search; they are not a blocklist.")
    return _table({"rows": rows, "total": len(rows)}, 0, len(rows), notice)


def activity_view(raw: str) -> dict[str, Any]:
    authorize("user")
    params = decode_params(raw, {"q", "offset", "limit"})
    offset, limit = _page(params)
    notice = _collection_notice("history")
    if activity.activity_mode() not in activity.DETAILED_MODES:
        return _table({}, offset, limit, notice)
    result = activity.get_store().events_for(
        current_principal() or "", search=(params.get("q") or "")[:_SEARCH_MAX] or None,
        offset=offset, limit=limit,
    )
    return _table(result, offset, limit)


def pages_view(raw: str) -> dict[str, Any]:
    authorize("user")
    params = decode_params(raw, {"q", "source", "offset", "limit"})
    offset, limit = _page(params)
    notice = _collection_notice("history")
    if activity.activity_mode() not in activity.DETAILED_MODES:
        return _table({}, offset, limit, notice)
    result = activity.get_store().urls_for(
        current_principal() or "", source=params.get("source") or None,
        search=(params.get("q") or "")[:_SEARCH_MAX] or None, offset=offset, limit=limit,
    )
    return _table(result, offset, limit)


def activity_item(raw_id: str) -> dict[str, Any]:
    """One of the caller's own events: what ran, on which URLs, and what it showed."""
    authorize("user")
    try:
        event_id = int(unquote(raw_id))
    except ValueError as exc:
        raise ValueError("an activity id is a number") from exc
    event = (activity.get_store().event_for(current_principal() or "", event_id)
             if activity.activity_mode() in activity.DETAILED_MODES else None)
    # Someone else's id reads exactly like one that does not exist.
    if event is None:
        raise ManagementAccessError("no such activity on record for this caller")
    urls = "\n".join(f"{u['url']}  ({u['source']}, {u['outcome']})" for u in event.pop("urls"))
    content = event.pop("content")
    if content is None:
        content = ("Not kept: this server records what was shown only with TINYSEARCH_ACTIVITY=content."
                   if activity.activity_mode() != "content" else "Nothing was shown (the call failed).")
    fields = [
        {"label": "When", "value": event["ts"], "format": "datetime"},
        {"label": "What", "value": event["operation"]},
        {"label": "Query", "value": event["query"], "format": "multiline"},
        {"label": "Outcome", "value": event["outcome"]},
        {"label": "Error", "value": event["error_code"]},
        {"label": "URLs", "value": urls or None, "format": "multiline"},
        {"label": "What the model was shown", "value": content, "format": "multiline"},
    ]
    return {"fields": fields, "row": event}


def clear_own_history() -> dict[str, Any]:
    principal = current_principal()
    if principal is None:
        raise ManagementAccessError("clearing history needs a trusted caller identity")
    return {"deleted_events": activity.get_store().clear(principal)}

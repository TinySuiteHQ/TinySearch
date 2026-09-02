from __future__ import annotations

import asyncio
import faulthandler
import os
import sys
import time
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Settings as FastMCPSettings
from pydantic import Field
from starlette.datastructures import Headers
from starlette.routing import BaseRoute, Mount, Route

from tinysearch import core
from tinysearch.services.tinysearch_config_service import load_tinysearch_config
from tinysearch.telemetry import configure_from_environment, shutdown as shutdown_telemetry

# MCP 1.x defines its generic Settings model before FastMCP, leaving the
# Settings.lifespan annotation as an unresolved forward reference. Rebuild it
# after the module is fully imported so pydantic-settings does not warn while
# loading environment-backed settings.
FastMCPSettings.model_rebuild()


def _mcp_host() -> str:
    return os.environ.get("MCP_HOST", "127.0.0.1").strip() or "127.0.0.1"


def _mcp_port() -> int:
    raw = os.environ.get("MCP_PORT", "8000").strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError("MCP_PORT must be an integer") from exc


def _mcp_cors_origins() -> list[str]:
    """Origins allowed for browser MCP clients (e.g. llama.cpp web UI). Default: all."""
    raw = os.environ.get("MCP_CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _streamable_http_cors_middleware() -> list[Any]:
    from mcp.server.streamable_http import (
        MCP_PROTOCOL_VERSION_HEADER,
        MCP_SESSION_ID_HEADER,
    )
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware

    return [
        Middleware(
            CORSMiddleware,
            allow_origins=_mcp_cors_origins(),
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=[
                "accept",
                "content-type",
                MCP_SESSION_ID_HEADER,
                MCP_PROTOCOL_VERSION_HEADER,
            ],
            expose_headers=[MCP_SESSION_ID_HEADER, MCP_PROTOCOL_VERSION_HEADER],
        )
    ]


class _StreamablePathLegacySseBridge:
    """Starlette ``Route`` wraps async *functions* as request handlers; raw ASGI must be a non-function callable."""

    def __init__(self, streamable_asgi: Any, sse_starlette: Any, sse_path: str) -> None:
        self._streamable = streamable_asgi
        self._sse = sse_starlette
        self._sse_path = sse_path

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._streamable(scope, receive, send)
            return
        if scope.get("method", "GET").upper() == "GET":
            headers = Headers(scope=scope)
            if not (headers.get("mcp-session-id") or "").strip():
                sse_scope = dict(scope)
                sse_scope["path"] = self._sse_path
                sse_scope["raw_path"] = self._sse_path.encode("ascii")
                await self._sse(sse_scope, receive, send)
                return
        await self._streamable(scope, receive, send)


def _route_identity(route: BaseRoute) -> tuple[Any, ...]:
    if isinstance(route, Route):
        methods = route.methods
        key_methods: tuple[str, ...] = (
            tuple(sorted(methods)) if methods is not None else ("*",)
        )
        return ("Route", route.path, key_methods)
    if isinstance(route, Mount):
        return ("Mount", route.path)
    return ("other", type(route).__name__, id(route))


async def _run_streamable_http_combined_async() -> None:
    """Streamable HTTP on /mcp plus SSE on /mcp/sse; sessionless GET /mcp → SSE for strict-URL clients."""

    import uvicorn
    from starlette.applications import Starlette

    stream_app = mcp.streamable_http_app()
    sse_starlette = mcp.sse_app()
    mcp_path = mcp.settings.streamable_http_path
    sse_path = mcp.settings.sse_path

    streamable_asgi: Any = None
    bridged_stream_routes: list[BaseRoute] = []
    for r in stream_app.routes:
        if isinstance(r, Route) and r.path == mcp_path:
            streamable_asgi = r.endpoint
            bridged_stream_routes.append(
                Route(
                    mcp_path,
                    endpoint=_StreamablePathLegacySseBridge(
                        streamable_asgi, sse_starlette, sse_path
                    ),
                    methods=r.methods,
                )
            )
        else:
            bridged_stream_routes.append(r)

    if streamable_asgi is None:
        raise RuntimeError(f"No Route found for Streamable HTTP path {mcp_path!r}")

    primary_keys = {_route_identity(r) for r in bridged_stream_routes}
    extra_sse = [
        r for r in sse_starlette.routes if _route_identity(r) not in primary_keys
    ]
    app = Starlette(
        debug=mcp.settings.debug,
        routes=bridged_stream_routes + extra_sse,
        middleware=_streamable_http_cors_middleware() + stream_app.user_middleware,
        lifespan=stream_app.router.lifespan_context,
    )
    config = uvicorn.Config(
        app,
        host=mcp.settings.host,
        port=mcp.settings.port,
        log_level=mcp.settings.log_level.lower(),
    )
    await uvicorn.Server(config).serve()


MCP_INSTRUCTIONS = """
This MCP server exposes four tools:

1. get_current_datetime()
2. search(items)
3. scrape_urls(items)
4. browse(url, actions, query, session_id, control_revision)

Before calling search for time-sensitive questions, or if you need to add
year/month/day context to a query, call get_current_datetime() first to orient
on the current date and time (UTC).

Formulate queries for effective retrieval. You may rewrite, clarify terminology,
correct spelling, expand abbreviations, add relevant temporal context, translate,
or narrow the request when useful. Preserve important names, constraints,
qualifiers, negations, and the user's underlying intent.

Use search first for fast top-level discovery. Use one item for a simple
lookup; combine independent subquestions or source strategies in one call.
Use domains only for hard source restrictions. It returns backend-ordered
titles, URLs, previews, dates, and compact backend outcomes; it does not
embed, rerank, crawl, or ground the results.

Use scrape_urls after URLs are already known. Pass one to five independent
URL/query pairs. Omit an item's query or pass `*` to receive the configured
2,000-token budget of clean Markdown in page order; supply a focused query
only when relevant chunks should be selected. Each page also returns a
bounded list of related_links -- links found on that page, ranked against
the query -- so you can decide which page to open next. It does not crawl
them automatically.

Use browse only when scrape_urls cannot reach the needed information because
it requires clicking, typing, scrolling, or waiting for content to appear
(e.g. dismissing a banner, submitting a search box, paging through results).
It is an observe-then-act primitive, not a one-shot batch call: call it
first with just a url and no actions to open the page and see it rendered;
its response includes session_id and control_revision. Then call it again with that
session_id, control_revision, and one or more ref-based actions to act
on the *same* live page and see the updated result -- omit url on this
follow-up call. The session stays open, idle, for a few minutes so you can
keep interacting; it is not a persistent or authenticated session, and it
does not batch multiple URLs.
""".strip()


def _log(message: str) -> None:
    print(f"[tinysearch] {message}", file=sys.stderr, flush=True)


def _enable_traceback_dump() -> None:
    raw = os.environ.get("TINYSEARCH_DUMP_TRACEBACK_AFTER", "").strip()
    if not raw:
        return
    try:
        delay = max(1.0, float(raw))
    except ValueError:
        delay = 30.0
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(delay, repeat=True, file=sys.stderr)


mcp = FastMCP(
    "tinysearch",
    instructions=MCP_INSTRUCTIONS,
    host=_mcp_host(),
    port=_mcp_port(),
    sse_path="/mcp/sse",
    message_path="/mcp/messages/",
)


@mcp.tool(
    name="get_current_datetime",
    title="Get Current Datetime",
    description=(
        "Return the current date and time in UTC. Call this first for "
        "time-sensitive questions, relative dates such as latest, this year, "
        "or last month, or before adding year/month/day context to a search "
        "query."
    ),
)
async def get_current_datetime_tool() -> str:
    _log("get_current_datetime called")
    from tinysearch.services.grounded_prompt_service import format_current_datetime

    return format_current_datetime(**core.get_current_datetime())


@mcp.tool(
    name="search",
    title="Search",
    description=(
        "Fast discovery for one to five independent items. Use one item for a simple "
        "lookup, multiple only for independent subquestions, and domains for hard "
        "source restrictions. Does not crawl or rerank."
    ),
)
async def search_tool(
    items: Annotated[
        list[dict[str, Any]] | None,
        Field(description="One to five items, each with query and optional positive domains."),
    ] = None,
    query: Annotated[
        str | None,
        Field(description="Deprecated single-query compatibility; prefer items."),
    ] = None,
    domains: Annotated[
        list[str] | None,
        Field(description="Deprecated; positive domains for the single-query compatibility form."),
    ] = None,
) -> str:
    started = time.monotonic()
    config = load_tinysearch_config()
    if items is None and query is not None:
        _log("search called with deprecated single-query shape; prefer items")
    item_count = len(items) if items is not None else (1 if query is not None else 0)
    _log(f"search called items={item_count}")
    result = await core.search(items, query=query, domains=domains, config=config)
    _log(
        f"search returning items={result['stats']['search_item_count']} "
        f"attempts={result['stats']['backend_attempt_count']} elapsed={time.monotonic() - started:.2f}s"
    )
    from tinysearch.services.grounded_prompt_service import format_search_batch_results

    return format_search_batch_results(items=result["items"])


@mcp.tool(
    name="scrape_urls",
    title="Scrape URLs",
    description=(
        "Inspect one to five URL/query pairs. Each item has url and optional query; "
        "omit query or use '*' for page-order content, or provide a focused query "
        "describing needed evidence. Related links are navigation candidates; batch "
        "URLs only after selecting them. Returns each item's success or error independently."
    ),
)
async def scrape_urls_tool(
    items: Annotated[
        list[dict[str, str | None]],
        Field(description="One to five items, each with url and optional query."),
    ],
) -> str:
    config = load_tinysearch_config()
    max_tokens = config["scrape_max_tokens"]
    _log(f"scrape_urls called items={len(items)} max_tokens={max_tokens}")
    result = await core.scrape_urls(
        items,
        max_tokens=max_tokens,
        config=config,
    )
    from tinysearch.services.grounded_prompt_service import format_url_grounded_answers

    return format_url_grounded_answers(results=result["results"])


async def _browser(name: str, **arguments: Any) -> str:
    """Run one browser tool against the shared session."""
    from tinysearch.services.browser_tool_service import call_tool

    _log(f"browser_{name} called")
    return await call_tool(name, load_tinysearch_config(), **arguments)


@mcp.tool(
    name="browser_navigate",
    title="Browser Navigate",
    description=(
        "Open an exact URL in a live browser and return its accessibility snapshot. "
        "Use only after scrape_urls returned thin, empty, or clearly incomplete "
        "content for this URL, or when the page needs a small read-only interaction "
        "to reveal already-requested information. Not a discovery tool."
    ),
)
async def browser_navigate_tool(
    url: Annotated[str, Field(description="Exact URL to open.")],
) -> str:
    return await _browser("navigate", url=url)


@mcp.tool(
    name="browser_find",
    title="Browser Find",
    description=(
        "Locate text or an element on the current page. This is the normal way to "
        "find a target: it returns matching accessibility nodes and nearby context "
        "far more cheaply than a full snapshot. Reuse the returned ref for the next "
        "interaction instead of looking at the page again."
    ),
)
async def browser_find_tool(
    text: Annotated[str | None, Field(description="Case-insensitive substring to find.")] = None,
    regex: Annotated[str | None, Field(description="Regular expression to find. Provide text or regex, not both.")] = None,
) -> str:
    return await _browser("find", text=text, regex=regex)


@mcp.tool(
    name="browser_snapshot",
    title="Browser Snapshot",
    description=(
        "Capture the page's accessibility tree. Use only when a targeted find cannot "
        "explain the page or produce a usable target. Prefer a small depth: it returns "
        "a shallower but still valid tree, and a full-page snapshot can be fifty times "
        "larger. Never snapshot merely to search for text find could locate."
    ),
)
async def browser_snapshot_tool(
    depth: Annotated[int | None, Field(description="Maximum tree depth. Omit for the configured default.")] = None,
) -> str:
    return await _browser("snapshot", depth=depth)


@mcp.tool(
    name="browser_click",
    title="Browser Click",
    description=(
        "Click the element with this ref, taken from a prior find or snapshot. "
        "Read-only interactions that reveal already-public content (pagination, "
        "expanding a section, accepting a cookie banner) need no confirmation; any "
        "real-world side effect does."
    ),
)
async def browser_click_tool(
    target: Annotated[str, Field(description="Element ref from a snapshot, such as 'e42'.")],
) -> str:
    return await _browser("click", target=target)


@mcp.tool(
    name="browser_type",
    title="Browser Type",
    description=(
        "Type text into the element with this ref. Do not enter credentials or other "
        "sensitive data, and confirm before submitting anything with a side effect."
    ),
)
async def browser_type_tool(
    target: Annotated[str, Field(description="Element ref from a snapshot, such as 'e42'.")],
    text: Annotated[str, Field(description="Text to type.")],
    submit: Annotated[bool, Field(description="Press Enter after typing.")] = False,
) -> str:
    return await _browser("type", target=target, text=text, submit=submit)


@mcp.tool(
    name="browser_wait_for",
    title="Browser Wait For",
    description=(
        "Wait for text to appear or disappear, or for a fixed delay, when a page "
        "needs time to render after an interaction."
    ),
)
async def browser_wait_for_tool(
    time: Annotated[float | None, Field(description="Seconds to wait.")] = None,
    text: Annotated[str | None, Field(description="Text to wait for.")] = None,
    text_gone: Annotated[str | None, Field(description="Text to wait to disappear.")] = None,
) -> str:
    return await _browser("wait_for", time_seconds=time, text=text, text_gone=text_gone)


@mcp.tool(
    name="browser_take_screenshot",
    title="Browser Take Screenshot",
    description=(
        "Save a screenshot to a file and return its path, for when a visual check "
        "genuinely matters. The image is never inlined into the conversation. Do not "
        "use it to choose an interaction target; use find instead."
    ),
)
async def browser_take_screenshot_tool(
    full_page: Annotated[bool, Field(description="Capture the full scrollable page.")] = False,
) -> str:
    return await _browser("take_screenshot", full_page=full_page)


@mcp.tool(
    name="browser_tabs",
    title="Browser Tabs",
    description="List, select, open, or close browser tabs.",
)
async def browser_tabs_tool(
    action: Annotated[str, Field(description="One of: list, new, select, close.")] = "list",
    index: Annotated[int | None, Field(description="Tab index for select and close.")] = None,
) -> str:
    return await _browser("tabs", action=action, index=index)


@mcp.tool(
    name="browser_close",
    title="Browser Close",
    description=(
        "Close the browser session. Call this when the current research task is "
        "complete; do not leave a session open for speculative exploration."
    ),
)
async def browser_close_tool() -> str:
    return await _browser("close")


def unregister_browser_tools_if_disabled() -> list[str]:
    """Drop the browser tools from the schema when the backend is off.

    They are registered by decorator at import time, so disabling has to
    remove them rather than skip registration. A tool a client cannot see is
    a tool it cannot be talked into calling, which is the same reason the
    code-execution tools have no implementation here at all.
    """
    from tinysearch.services.browser_tool_service import browser_backend_enabled

    if browser_backend_enabled(load_tinysearch_config()):
        return []
    removed = [name for name in mcp._tool_manager._tools if name.startswith("browser_")]
    for name in removed:
        del mcp._tool_manager._tools[name]
    _log(f"browser_backend is 'off'; removed {len(removed)} browser tools")
    return removed


def main() -> None:
    _enable_traceback_dump()
    configure_from_environment()
    unregister_browser_tools_if_disabled()
    transport = os.environ.get("MCP_TRANSPORT", "stdio").strip() or "stdio"
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise ValueError(
            "MCP_TRANSPORT must be one of: stdio, sse, streamable-http "
            "(default stdio for IDE-spawned MCP; set env only for standalone HTTP/SSE)"
        )
    try:
        if transport == "streamable-http":
            import anyio

            anyio.run(_run_streamable_http_combined_async)
        else:
            mcp.run(transport=transport)
    finally:
        import anyio

        anyio.run(core.close_browser_sessions)
        shutdown_telemetry()


if __name__ == "__main__":
    main()

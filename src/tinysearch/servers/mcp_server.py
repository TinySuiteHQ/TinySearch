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


async def register_browser_tools() -> list[str]:
    """Register the allowlisted Playwright tools using the child's own schemas.

    Argument schemas are fetched from the running `@playwright/mcp` child and
    re-exported verbatim rather than restated in Python signatures, so an
    argument rename on a version bump is inherited instead of silently
    diverging. Registration therefore requires briefly starting the child.

    Tools outside `exposed_tool_names()` -- notably `browser_evaluate` and
    `browser_run_code_unsafe` -- are never registered at all. A page cannot
    talk a model into calling a tool that is absent from the schema.
    """
    from mcp.server.fastmcp.tools.base import Tool
    from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata
    from pydantic import ConfigDict

    from tinysearch.services import playwright_mcp_service as pw

    config = load_tinysearch_config()
    if not pw.browser_backend_enabled(config):
        _log("browser_backend is 'off'; playwright browser tools not registered")
        return []

    class _PassthroughArgs(ArgModelBase):
        """Forward validated-upstream arguments through without re-modelling them."""

        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

        def model_dump_one_level(self) -> dict[str, Any]:
            return dict(self.__pydantic_extra__ or {})

    schemas = await pw.fetch_tool_schemas(config)
    registered: list[str] = []
    for upstream_name, schema in schemas.items():
        public_name = pw.public_tool_name(upstream_name)

        async def handler(_upstream: str = upstream_name, **arguments: Any) -> str:
            _log(f"{pw.public_tool_name(_upstream)} called")
            return await pw.get_client(load_tinysearch_config()).call(_upstream, arguments)

        mcp._tool_manager._tools[public_name] = Tool(
            fn=handler,
            name=public_name,
            title=public_name.replace("_", " ").title(),
            description=pw.tool_description(upstream_name),
            parameters=schema,
            fn_metadata=FuncMetadata(arg_model=_PassthroughArgs),
            is_async=True,
            context_kwarg=None,
            annotations=None,
        )
        registered.append(public_name)

    _log(f"registered {len(registered)} playwright browser tools")
    return registered


def _register_browser_tools_blocking() -> None:
    """Register browser tools before the transport starts serving.

    Failure is non-fatal: TinySearch's search and scrape tools must keep
    working on a host with no Node.js runtime installed.
    """
    import anyio

    try:
        anyio.run(register_browser_tools)
    except Exception as exc:  # noqa: BLE001 - browser backend is optional
        _log(f"playwright browser tools unavailable: {exc}")


def main() -> None:
    _enable_traceback_dump()
    configure_from_environment()
    _register_browser_tools_blocking()
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

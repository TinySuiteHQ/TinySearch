from __future__ import annotations

import asyncio
import faulthandler
import os
import sys
import time
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from starlette.datastructures import Headers
from starlette.routing import BaseRoute, Mount, Route

from tinysearch import core
from tinysearch.services.tinysearch_config_service import (
    load_tinysearch_config,
    tokenizer_name_for_config,
)
from tinysearch.services.token_counter_service import token_count


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
2. search(query)
3. research(query)
4. scrape_urls(items)

Before calling research for time-sensitive questions, or if you need to add
year/month/day context to a query, call get_current_datetime() first to orient
on the current date and time (UTC).

Formulate queries for effective retrieval. You may rewrite, clarify terminology,
correct spelling, expand abbreviations, add relevant temporal context, translate,
or narrow the request when useful. Preserve important names, constraints,
qualifiers, negations, and the user's underlying intent.

Use search first for fast top-level discovery. It returns the configured
backend's results in backend order with titles, URLs, previews, and upstream
publication dates when available. It does not embed, rerank, crawl, or ground
the results.

The research tool is deprecated. Use search instead.

Use scrape_urls after URLs are already known. Pass one to five independent
URL/query pairs. Omit an item's query or pass `*` to receive the configured
2,000-token budget of clean Markdown in page order; supply a focused query
only when relevant chunks should be selected. Each page also returns a
bounded list of related_links -- links found on that page, ranked against
the query -- so you can decide which page to open next. It does not crawl
them automatically.
""".strip()


def _answer_tokens(answer: str) -> int:
    return token_count(answer, encoding_name=tokenizer_name_for_config())


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
        "or last month, or before adding year/month/day context to a research "
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
        "Fast top-level discovery. Return backend-ordered web results with titles, "
        "URLs, previews, and upstream dates when available. Does not crawl or rerank."
    ),
)
async def search_tool(
    query: Annotated[
        str,
        Field(
            description=(
                "A search query describing the information to find. Refine it for "
                "effective retrieval while preserving names, constraints, and intent."
            )
        ),
    ],
) -> str:
    started = time.monotonic()
    config = load_tinysearch_config()
    limit = config["search_max_results"]
    _log(f"search called query={query!r} limit={limit}")
    try:
        result = await core.search(
            query,
            limit=limit,
            config=config,
        )
    except Exception as exc:
        elapsed = time.monotonic() - started
        _log(f"search failed elapsed={elapsed:.2f}s error={exc!r}")
        raise ValueError(f"search_backend_error: {exc}") from exc
    from tinysearch.services.grounded_prompt_service import format_search_results

    answer = format_search_results(query=result["query"], results=result["results"])
    _log(
        f"search returning results={result['stats']['result_count']} "
        f"backend={result['backend']!r} elapsed={time.monotonic() - started:.2f}s"
    )
    return answer


@mcp.tool(
    name="research",
    title="Research (Deprecated)",
    description=(
        "Deprecated: use search instead. This legacy tool discovers relevant URLs, "
        "crawls ranked pages, and returns a search-grounded XML answer prompt."
    ),
)
async def research(
    query: Annotated[
        str,
        Field(
            description=(
                "A query describing the information to find. "
                "Rewrite or refine it as needed for effective retrieval while "
                "preserving important names, constraints, qualifiers, and intent."
            )
        ),
    ],
) -> str:
    started = time.monotonic()
    _log(f"research called query={query!r}")
    try:
        config = load_tinysearch_config()
        result = await core.research(query, config=config)
        elapsed = time.monotonic() - started
        from tinysearch.prompts import to_prompt

        answer = to_prompt(result)
        _log(
            "research returning "
            f"answer_tokens={_answer_tokens(answer)} "
            f"elapsed={elapsed:.2f}s"
        )
        return answer
    except Exception as exc:
        elapsed = time.monotonic() - started
        _log(f"research failed elapsed={elapsed:.2f}s error={exc!r}")
        raise


@mcp.tool(
    name="scrape_urls",
    title="Scrape URLs",
    description=(
        "Inspect one to five URL/query pairs. Each item has url and optional query; "
        "omit query or use '*' for page-order content, or provide a focused query "
        "to rank chunks. Returns each item's success or error independently."
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


def main() -> None:
    _enable_traceback_dump()
    transport = os.environ.get("MCP_TRANSPORT", "stdio").strip() or "stdio"
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise ValueError(
            "MCP_TRANSPORT must be one of: stdio, sse, streamable-http "
            "(default stdio for IDE-spawned MCP; set env only for standalone HTTP/SSE)"
        )
    if transport == "streamable-http":
        import anyio

        anyio.run(_run_streamable_http_combined_async)
    else:
        mcp.run(transport=transport)


if __name__ == "__main__":
    main()

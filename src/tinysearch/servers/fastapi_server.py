from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from tinysearch import core
from tinysearch.services import browser_tool_service
from tinysearch.services.tinysearch_config_service import (
    load_tinysearch_config,
    save_tinysearch_config,
)
from tinysearch.services.site_crawl_service import BrowserCrawlerSession
from tinysearch.telemetry import configure_from_environment, shutdown as shutdown_telemetry


# Settings that let a caller reach outside the HTTP surface: an external
# browser endpoint, a cookie/storage file, and the switch that launches or
# attaches to Playwright. All are operator decisions made at startup, never
# over the API.
_OPERATOR_MANAGED_CONFIG_FIELDS = frozenset({
    "browser_cdp_url",
    "browser_backend",
    "browser_storage_state_path",
})


_crawler_session: BrowserCrawlerSession | None = None


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _crawler_session
    configure_from_environment()
    _crawler_session = BrowserCrawlerSession()
    try:
        yield
    finally:
        crawler_session, _crawler_session = _crawler_session, None
        if crawler_session is not None:
            await crawler_session.close()
        await core.close_browser_sessions()
        shutdown_telemetry()


def _tinysearch_version() -> str:
    return os.environ.get("TINYSEARCH_VERSION", "dev").strip() or "dev"


app = FastAPI(
    title="TinySearch API",
    description=(
        "HTTP API mirroring the TinySearch MCP tools. POST /search provides "
        "fast backend-ordered discovery; /scrape provides deep grounded "
        "retrieval for known URLs; /browser/* provides the same narrow "
        "browser interaction surface."
    ),
    version=_tinysearch_version(),
    lifespan=_lifespan,
)


class ScrapeBatchItem(BaseModel):
    url: HttpUrl
    query: str | None = None


class ScrapeBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ScrapeBatchItem] = Field(..., min_length=1, max_length=5)
    max_tokens: int = Field(4000, ge=1)


class SearchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    domains: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[SearchItem] | None = Field(default=None, min_length=1, max_length=5)
    # Deprecated single-query compatibility fields; prefer items. Kept so callers
    # written against the pre-batch /search contract do not break on upgrade.
    query: str | None = Field(default=None, min_length=1)
    domains: list[str] | None = None

    @model_validator(mode="after")
    def _coerce_single_query(self) -> "SearchRequest":
        if self.items is None:
            if self.query is None:
                raise ValueError("provide items (1 to 5 search items) or a query")
            self.items = [SearchItem(query=self.query, domains=self.domains or [])]
        return self

    def normalized_items(self) -> list[dict[str, Any]]:
        return [item.model_dump() for item in self.items or []]


class BrowserNavigateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl
    find: str = ""
    depth: int = Field(0, ge=0)

    def arguments(self) -> dict[str, Any]:
        arguments: dict[str, Any] = {"url": str(self.url)}
        if self.find:
            arguments["find"] = self.find
        if self.depth:
            arguments["depth"] = self.depth
        return arguments


class BrowserActRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(..., min_length=1)
    target: str = ""
    text: str = ""
    submit: bool = False
    find: str = ""
    depth: int = Field(0, ge=0)
    time: float = Field(0.0, ge=0)
    text_gone: str = ""
    tab_action: str = "list"
    index: int | None = None

    def arguments(self) -> dict[str, Any]:
        return browser_tool_service.resolve_act_arguments(
            self.action,
            {
                "target": self.target,
                "text": self.text,
                "submit": self.submit,
                "find": self.find,
                "depth": self.depth,
                "time_seconds": self.time,
                "text_gone": self.text_gone,
                "action": self.tab_action,
                "index": self.index,
            },
        )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/current_datetime")
async def current_datetime_endpoint() -> dict[str, str]:
    return core.get_current_datetime()


@app.get("/config")
async def get_config_endpoint() -> dict[str, Any]:
    config = load_tinysearch_config()
    return {
        key: (
            "***"
            if (key in ("browser_cdp_url", "browser_storage_state_path") and value)
            or any(marker in key.lower() for marker in ("secret", "token", "api_key"))
            else value
        )
        for key, value in config.items()
    }


@app.put("/config")
async def put_config_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("TINYSEARCH_CONFIG_WRITABLE", "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "config_read_only",
                "message": "Set TINYSEARCH_CONFIG_WRITABLE=1 to enable config updates.",
            },
        )
    if not os.environ.get("TINYSEARCH_CONFIG_PATH", "").strip():
        raise HTTPException(
            status_code=403,
            detail={
                "code": "config_path_required",
                "message": (
                    "Set TINYSEARCH_CONFIG_PATH to an explicit writable override "
                    "file before enabling config updates."
                ),
            },
        )
    operator_managed_fields = sorted(
        _OPERATOR_MANAGED_CONFIG_FIELDS.intersection(payload)
    )
    if operator_managed_fields:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "operator_managed_config",
                "message": (
                    f"{', '.join(operator_managed_fields)} cannot be changed over "
                    "HTTP. Configure these at startup with "
                    f"{', '.join('TINYSEARCH_' + f.upper() for f in operator_managed_fields)} "
                    "or in the file selected by TINYSEARCH_CONFIG_PATH, then "
                    "restart TinySearch. Omit them when updating other settings."
                ),
                "fields": operator_managed_fields,
            },
        )
    try:
        return save_tinysearch_config(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_config", "message": str(exc)},
        ) from exc


@app.post("/scrape")
async def scrape_endpoint(request: ScrapeBatchRequest) -> dict[str, Any]:
    """Scrape one to five independent URL/query pairs with per-item outcomes."""
    return await core.scrape_urls(
        [item.model_dump(mode="json") for item in request.items],
        max_tokens=request.max_tokens,
        config=load_tinysearch_config(),
        crawler_session=_crawler_session,
    )


async def _browser_endpoint(name: str, arguments: dict[str, Any]) -> dict[str, str]:
    try:
        result = await browser_tool_service.call_tool(
            name,
            load_tinysearch_config(),
            **arguments,
        )
    except browser_tool_service.BrowserDisabledError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "browser_disabled", "message": str(exc)},
        ) from exc
    except browser_tool_service.BrowserToolError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "browser_error", "message": str(exc)},
        ) from exc
    return {"result": result}


@app.post("/browser/navigate")
async def browser_navigate_endpoint(
    request: BrowserNavigateRequest,
) -> dict[str, str]:
    """Open an exact URL and return the same accessibility view as MCP."""
    return await _browser_endpoint("navigate", request.arguments())


@app.post("/browser/act")
async def browser_act_endpoint(request: BrowserActRequest) -> dict[str, str]:
    """Perform one folded browser action against the current page."""
    try:
        arguments = request.arguments()
    except browser_tool_service.BrowserToolError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "browser_error", "message": str(exc)},
        ) from exc
    return await _browser_endpoint(request.action, arguments)


@app.post("/search")
async def search_endpoint(request: SearchRequest) -> dict[str, Any]:
    if request.query is not None:
        print(
            "[tinysearch] /search called with deprecated single-query shape; prefer items",
            file=sys.stderr,
            flush=True,
        )
    return await core.search(
        request.normalized_items(),
        config=load_tinysearch_config(),
    )

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from tinysearch import core
from tinysearch.services.tinysearch_config_service import (
    load_tinysearch_config,
    save_tinysearch_config,
)
from tinysearch.telemetry import configure_from_environment, shutdown as shutdown_telemetry


_OPERATOR_MANAGED_CONFIG_FIELDS = frozenset({"browser_cdp_url"})


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    configure_from_environment()
    try:
        yield
    finally:
        shutdown_telemetry()


def _tinysearch_version() -> str:
    return os.environ.get("TINYSEARCH_VERSION", "dev").strip() or "dev"


app = FastAPI(
    title="TinySearch API",
    description=(
        "HTTP API mirroring the TinySearch MCP tools. POST /search provides "
        "fast backend-ordered discovery; /scrape provides deep grounded "
        "retrieval for known URLs."
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
            if (key == "browser_cdp_url" and value)
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
                    "browser_cdp_url cannot be changed over HTTP. Configure it "
                    "at startup with TINYSEARCH_BROWSER_CDP_URL or in the file "
                    "selected by TINYSEARCH_CONFIG_PATH, then restart TinySearch. "
                    "Omit browser_cdp_url when updating other settings."
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
    )


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

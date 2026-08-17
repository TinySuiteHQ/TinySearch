from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from tinysearch import core
from tinysearch.services.tinysearch_config_service import (
    load_tinysearch_config,
    save_tinysearch_config,
)


_OPERATOR_MANAGED_CONFIG_FIELDS = frozenset({"browser_cdp_url"})


def _tinysearch_version() -> str:
    return os.environ.get("TINYSEARCH_VERSION", "dev").strip() or "dev"


app = FastAPI(
    title="TinySearch API",
    description=(
        "HTTP API mirroring the TinySearch MCP tools. POST /search provides "
        "fast backend-ordered discovery; /research and /scrape provide deep "
        "grounded retrieval."
    ),
    version=_tinysearch_version(),
)


class ScrapeBatchItem(BaseModel):
    url: HttpUrl
    query: str | None = None


class ScrapeBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ScrapeBatchItem] = Field(..., min_length=1, max_length=5)
    max_tokens: int = Field(4000, ge=1)


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    output_format: Literal["prompt", "json"] = "prompt"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(10, ge=1, le=50)
    output_format: Literal["prompt", "json"] = "prompt"


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


@app.post("/research")
async def research_endpoint(request: ResearchRequest) -> dict[str, Any]:
    result = await core.research(request.query, config=load_tinysearch_config())
    if request.output_format == "json":
        return result
    from tinysearch.prompts import to_prompt

    return {"answer": to_prompt(result)}


@app.post("/search")
async def search_endpoint(request: SearchRequest) -> dict[str, Any]:
    try:
        result = await core.search(
            request.query,
            limit=request.limit,
            config=load_tinysearch_config(),
        )
    except Exception as exc:
        from tinysearch.services.web_search_service import SearchBackendError

        if isinstance(exc, SearchBackendError):
            raise HTTPException(
                status_code=502,
                detail={"code": "search_backend_error", "message": str(exc)},
            ) from exc
        raise
    if request.output_format == "json":
        return result
    from tinysearch.prompts import to_prompt

    return {"answer": to_prompt(result)}

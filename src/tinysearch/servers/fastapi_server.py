from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from tinysearch import core
from tinysearch.services.tinysearch_config_service import (
    load_tinysearch_config,
    save_tinysearch_config,
    tokenizer_name_for_config,
)
from tinysearch.services.scrape_service import (
    SCRAPE_ERROR_MAP,
    ScrapeError,
)
from tinysearch.services.url_safety_service import BlockedUrlError, InvalidUrlError


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


class ScrapeRequest(BaseModel):
    url: HttpUrl
    query: str = Field(..., min_length=1)
    output_format: Literal["prompt", "json"] = "prompt"


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
        key: ("***" if any(marker in key.lower() for marker in ("secret", "token", "api_key")) else value)
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
    try:
        return save_tinysearch_config(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_config", "message": str(exc)},
        ) from exc


def _raise_scrape_http_error(exc: Exception) -> None:
    mapping = SCRAPE_ERROR_MAP.get(type(exc))
    if mapping is None:
        raise HTTPException(
            status_code=500,
            detail={"code": "internal_error", "message": "internal error"},
        ) from exc
    code, status_code = mapping
    raise HTTPException(
        status_code=status_code,
        detail={"code": code, "message": str(exc)},
    ) from exc


@app.post("/scrape")
async def scrape_endpoint(request: ScrapeRequest) -> dict[str, Any]:
    try:
        result = await core.scrape_url(
            str(request.url),
            request.query,
            config=load_tinysearch_config(),
        )
    except (InvalidUrlError, BlockedUrlError, ScrapeError) as exc:
        _raise_scrape_http_error(exc)
    if request.output_format == "json":
        return result
    from tinysearch.prompts import to_prompt
    from tinysearch.services.token_counter_service import token_count

    prompt = to_prompt(result)
    source = result["sources"][0]
    config = load_tinysearch_config()
    return {
        "answer": prompt,
        "url": source["url"],
        "title": source["title"],
        "content_tokens": result["stats"]["content_tokens"],
        "answer_tokens": token_count(prompt, tokenizer_name_for_config(config)),
        "truncated": result["stats"]["truncated"],
        "retrieved_at": result["retrieved_at"],
    }


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

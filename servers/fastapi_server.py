from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from pipelines.agentic_research import agentic_run
from services.current_datetime_service import current_datetime_payload
from services.embedding_service import normalize_embedding_backend
from services.research_config_service import (
    config_trace_path,
    load_research_config,
    normalize_research_query,
    research_run_kwargs,
    research_tokenizer_name,
    save_research_config,
)
from services.scrape_service import (
    DEFAULT_SCRAPE_MAX_TOKENS,
    SCRAPE_ERROR_MAP,
    ScrapeError,
    ScrapeResult,
    scrape_url,
)
from services.url_safety_service import BlockedUrlError, InvalidUrlError


async def _ensure_local_bundle_for_config(config: dict[str, Any]) -> None:
    if normalize_embedding_backend(str(config["embedding_backend"])) != "onnx":
        return
    from services.onnx_bundle_service import ensure_onnx_bundle_sync

    await asyncio.to_thread(ensure_onnx_bundle_sync, str(config["embedding_model"]))


def _tinysearch_version() -> str:
    return os.environ.get("TINYSEARCH_VERSION", "dev").strip() or "dev"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    cfg = load_research_config()
    await _ensure_local_bundle_for_config(cfg)
    yield


app = FastAPI(
    title="TinySearch API",
    description="HTTP API mirroring the TinySearch MCP tools.",
    version=_tinysearch_version(),
    lifespan=_lifespan,
)


class ScrapeRequest(BaseModel):
    url: HttpUrl
    query: str = Field(..., min_length=1)


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/current_datetime")
async def current_datetime_endpoint() -> dict[str, str]:
    return current_datetime_payload()


@app.get("/config")
async def get_config_endpoint() -> dict[str, Any]:
    return load_research_config()


@app.put("/config")
async def put_config_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return save_research_config(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_config", "message": str(exc)},
        ) from exc


def _scrape_response(result: ScrapeResult) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "url": result.url,
        "title": result.title,
        "content_tokens": result.content_tokens,
        "answer_tokens": result.answer_tokens,
        "truncated": result.truncated,
        "retrieved_at": result.retrieved_at,
    }


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
    config = load_research_config()
    await _ensure_local_bundle_for_config(config)
    tokenizer = research_tokenizer_name(config)
    try:
        result = await scrape_url(
            str(request.url),
            request.query,
            max_tokens=DEFAULT_SCRAPE_MAX_TOKENS,
            include_metadata=True,
            config=config,
            tokenizer_name=tokenizer,
        )
    except (InvalidUrlError, BlockedUrlError, ScrapeError) as exc:
        _raise_scrape_http_error(exc)
    return _scrape_response(result)


@app.post("/research")
async def research_endpoint(request: ResearchRequest) -> dict[str, Any]:
    config = load_research_config()
    await _ensure_local_bundle_for_config(config)
    query = normalize_research_query(request.query)
    result = await agentic_run(
        query,
        trace_path=config_trace_path(config),
        **research_run_kwargs(config),
    )
    return {"answer": result.answer}

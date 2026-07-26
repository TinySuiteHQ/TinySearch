"""Compatibility helpers around the public :mod:`tinysearch.config` API.

Core library calls use ``TinySearchConfig`` directly. Environment and native
path discovery live here for CLI/server compatibility only.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tinysearch.config import (
    DEFAULT_CONFIG,
    TinySearchConfig,
    normalize_config,
    save_config,
)
from tinysearch.services.embedding_service import (
    normalize_embedding_backend,
    resolve_embedding_tokenizer_name,
    resolve_local_embedding_model_spec,
)

DEFAULT_RESEARCH_CONFIG = DEFAULT_CONFIG


def _coerce_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    return normalize_config(raw)


def resolve_research_config_path(path: str | Path | None = None) -> Path:
    """Resolve an explicit/server config path without consulting the checkout."""
    if path is not None:
        return Path(path).expanduser()
    env_path = os.environ.get("TINYSEARCH_CONFIG_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    from tinysearch.paths import native_config_path

    return native_config_path()


def load_research_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load explicit/server configuration and apply server environment overrides."""
    config_path = resolve_research_config_path(path)
    config = (
        TinySearchConfig.from_json(config_path)
        if config_path.exists()
        else TinySearchConfig()
    )
    overrides: dict[str, Any] = {}
    search_backend = os.environ.get("TINYSEARCH_SEARCH_BACKEND", "").strip()
    if search_backend:
        overrides["search_backend"] = search_backend
    searxng_url = os.environ.get("SEARXNG_URL", "").strip()
    if searxng_url:
        overrides["search_backend_url"] = searxng_url
    return config.with_overrides(overrides).to_dict() if overrides else config.to_dict()


def save_research_config(
    raw: Mapping[str, Any],
    path: str | Path | None = None,
) -> dict[str, Any]:
    config_path = resolve_research_config_path(path)
    existing = (
        TinySearchConfig.from_json(config_path).to_dict()
        if config_path.exists()
        else {}
    )
    merged = dict(existing)
    merged.update(raw)
    return save_config(merged, config_path).to_dict()


def research_run_kwargs(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved = normalize_config(config)
    keys = (
        "search_top_k",
        "search_rrf_cutoff",
        "search_dense_weight",
        "search_max_results_to_keep",
        "chunk_rrf_cutoff",
        "chunk_dense_weight",
        "chunk_max_results_to_keep",
        "chunk_rank_oversample",
        "chunk_dedupe_jaccard_threshold",
        "chunk_max_per_source_url",
        "max_concurrent_crawls",
        "max_concurrent_embedding_calls",
        "pipeline_timeout_seconds",
        "embedding_timeout_seconds",
        "embedding_timeout_retries",
        "crawl_fit_markdown_mode",
        "crawl_fit_min_chars",
        "crawl_bm25_threshold",
        "crawl_bm25_language",
        "crawl_pruning_threshold",
        "crawl_max_chunk_tokens",
        "crawl_overlap_tokens",
        "crawl_max_page_tokens",
        "encoding_name",
        "embedding_backend",
        "embedding_model",
        "embedding_openai_env_file",
        "dense_query_prefix",
        "dense_document_prefix",
        "dense_document_embed_batch_size",
        "blocked_domains",
    )
    return {key: resolved[key] for key in keys}


def normalize_research_query(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    return query


def research_embedding_model_info(
    config: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    resolved = normalize_config(config)
    backend = normalize_embedding_backend(str(resolved["embedding_backend"]))
    if backend == "openai_compatible":
        return {"requested_model": "", "repo_id": "", "local_dir": ""}
    spec = resolve_local_embedding_model_spec(str(resolved["embedding_model"]))
    return {
        "requested_model": spec.requested_model,
        "repo_id": spec.repo_id,
        "local_dir": str(spec.local_dir),
    }


def research_tokenizer_name(config: Mapping[str, Any] | None = None) -> str:
    resolved = normalize_config(config)
    encoding_name = str(resolved.get("encoding_name") or "").strip()
    if encoding_name and encoding_name.lower() != "embedding":
        return encoding_name
    backend = normalize_embedding_backend(str(resolved["embedding_backend"]))
    return resolve_embedding_tokenizer_name(
        backend=backend,
        embedding_model=str(resolved["embedding_model"]),
        openai_env_file=(
            str(resolved["embedding_openai_env_file"])
            if backend == "openai_compatible"
            else None
        ),
    )


def config_trace_path(config: Mapping[str, Any] | None = None) -> Path | None:
    resolved = normalize_config(config)
    value = str(resolved.get("trace_path") or "").strip()
    if not value:
        return None
    return Path(value).expanduser()

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from services.embedding_service import (
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_OPENAI_ENV_FILE,
    normalize_embedding_backend,
    resolve_local_embedding_model_spec,
    resolve_embedding_tokenizer_name,
)
from services.web_search_service import ALLOWED_SEARCH_BACKENDS, DEFAULT_SEARXNG_URL, normalize_domain


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESEARCH_CONFIG_PATH = PROJECT_ROOT / "configs" / "research_config.json"

DEFAULT_RESEARCH_CONFIG: dict[str, Any] = {
    "search_top_k": 10,
    "search_rrf_cutoff": 0.0,
    "search_dense_weight": 0.5,
    "search_max_results_to_keep": 5,
    "chunk_rrf_cutoff": 0.0,
    "chunk_dense_weight": 0.5,
    "chunk_max_results_to_keep": 2,
    "chunk_rank_oversample": 3,
    "chunk_dedupe_jaccard_threshold": 0.92,
    "chunk_max_per_source_url": 4,
    "max_concurrent_crawls": 5,
    "max_concurrent_embedding_calls": 3,
    "pipeline_timeout_seconds": 120.0,
    "embedding_timeout_seconds": 60.0,
    "embedding_timeout_retries": 2,
    "crawl_fit_markdown_mode": "bm25",
    "crawl_fit_min_chars": 200,
    "crawl_bm25_threshold": 1.5,
    "crawl_bm25_language": "english",
    "crawl_pruning_threshold": 0.48,
    "crawl_max_chunk_tokens": 300,
    "crawl_overlap_tokens": 80,
    "crawl_max_page_tokens": 0,
    "encoding_name": "o200k_base",
    "embedding_backend": DEFAULT_EMBEDDING_BACKEND,
    "embedding_model": DEFAULT_EMBEDDING_MODEL,
    "embedding_openai_env_file": DEFAULT_EMBEDDING_OPENAI_ENV_FILE,
    "dense_query_prefix": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
    "dense_document_prefix": "",
    "dense_document_embed_batch_size": 32,
    "blocked_domains": [],
    "search_backend": "searxng",
    "search_backend_url": DEFAULT_SEARXNG_URL,
    "search_engines": [],
    "search_region": "",
    "search_backend_fallback": True,
    "trace_path": "",
}

_INT_FIELDS = {
    "search_top_k",
    "search_max_results_to_keep",
    "chunk_max_results_to_keep",
    "chunk_rank_oversample",
    "chunk_max_per_source_url",
    "max_concurrent_crawls",
    "max_concurrent_embedding_calls",
    "embedding_timeout_retries",
    "crawl_fit_min_chars",
    "crawl_max_chunk_tokens",
    "crawl_overlap_tokens",
    "crawl_max_page_tokens",
    "dense_document_embed_batch_size",
}
_FLOAT_FIELDS = {
    "search_rrf_cutoff",
    "search_dense_weight",
    "chunk_rrf_cutoff",
    "chunk_dense_weight",
    "chunk_dedupe_jaccard_threshold",
    "crawl_bm25_threshold",
    "crawl_pruning_threshold",
    "embedding_timeout_seconds",
}


def _coerce_config(raw: dict[str, Any]) -> dict[str, Any]:
    config = dict(DEFAULT_RESEARCH_CONFIG)
    config.update(raw)
    for legacy in ("embedding_gguf_file", "mcp_transport"):
        config.pop(legacy, None)
    for key in _INT_FIELDS:
        config[key] = int(config[key])
    for key in _FLOAT_FIELDS:
        config[key] = float(config[key])
    raw_timeout = config.get("pipeline_timeout_seconds")
    config["pipeline_timeout_seconds"] = float(raw_timeout) if raw_timeout is not None else None
    for key in (
        "encoding_name",
        "embedding_backend",
        "embedding_model",
        "embedding_openai_env_file",
        "dense_query_prefix",
        "dense_document_prefix",
        "crawl_fit_markdown_mode",
        "crawl_bm25_language",
        "trace_path",
    ):
        if config.get(key) is not None:
            config[key] = str(config[key])
    blocked_domains = config.get("blocked_domains", [])
    if not isinstance(blocked_domains, list):
        raise ValueError("research config blocked_domains must be a JSON list")
    config["blocked_domains"] = list(
        dict.fromkeys(
            normalized
            for item in blocked_domains
            if isinstance(item, str)
            for normalized in [normalize_domain(item)]
            if normalized
        )
    )

    backend = str(config.get("search_backend") or "searxng").strip().lower()
    if backend not in ALLOWED_SEARCH_BACKENDS:
        raise ValueError(
            "research config search_backend must be one of "
            f"{sorted(ALLOWED_SEARCH_BACKENDS)}"
        )
    config["search_backend"] = backend

    env_searxng_url = os.environ.get("SEARXNG_URL", "").strip()
    if env_searxng_url:
        config["search_backend_url"] = env_searxng_url
    else:
        config["search_backend_url"] = str(
            config.get("search_backend_url") or DEFAULT_SEARXNG_URL
        ).strip() or DEFAULT_SEARXNG_URL

    engines_raw = config.get("search_engines")
    if engines_raw is None or engines_raw == "":
        engines_list: list[str] = []
    elif isinstance(engines_raw, str):
        engines_list = [part.strip() for part in engines_raw.split(",") if part.strip()]
    elif isinstance(engines_raw, list):
        engines_list = [
            str(item).strip() for item in engines_raw if str(item).strip()
        ]
    else:
        raise ValueError(
            "research config search_engines must be a list or comma-separated string"
        )
    config["search_engines"] = engines_list

    region_raw = config.get("search_region")
    if not region_raw:
        region_raw = config.get("search_country")
    config["search_region"] = str(region_raw or "").strip()
    config.pop("search_country", None)

    config["search_backend_fallback"] = bool(
        config.get("search_backend_fallback", True)
    )

    return config


def resolve_research_config_path(path: str | Path | None = None) -> Path:
    raw_path = path if path is not None else os.environ.get("TINYSEARCH_CONFIG_PATH")
    config_path = Path(raw_path) if raw_path else DEFAULT_RESEARCH_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return config_path


def load_research_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_research_config_path(path)
    if not config_path.exists():
        return dict(DEFAULT_RESEARCH_CONFIG)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"research config must be a JSON object: {config_path}")
    return _coerce_config(raw)


def save_research_config(
    raw: dict[str, Any], path: str | Path | None = None
) -> dict[str, Any]:
    config_path = resolve_research_config_path(path)
    merged = load_research_config(config_path) if config_path.exists() else dict(DEFAULT_RESEARCH_CONFIG)
    merged.update(raw)
    config = _coerce_config(merged)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(config_path)
    return config


def research_run_kwargs(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_research_config() if config is None else config
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
    return {key: config[key] for key in keys}


def normalize_research_query(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    return query


def research_embedding_model_info(config: dict[str, Any] | None = None) -> dict[str, str]:
    config = load_research_config() if config is None else config
    backend = normalize_embedding_backend(str(config["embedding_backend"]))
    if backend == "openai_compatible":
        return {
            "requested_model": "",
            "repo_id": "",
            "local_dir": "",
        }
    spec = resolve_local_embedding_model_spec(str(config["embedding_model"]))
    return {
        "requested_model": spec.requested_model,
        "repo_id": spec.repo_id,
        "local_dir": str(spec.local_dir),
    }


def research_tokenizer_name(config: dict[str, Any] | None = None) -> str:
    config = load_research_config() if config is None else config
    encoding_name = str(config.get("encoding_name") or "").strip()
    if encoding_name and encoding_name.lower() != "embedding":
        return encoding_name
    backend = normalize_embedding_backend(str(config["embedding_backend"]))
    return resolve_embedding_tokenizer_name(
        backend=backend,
        embedding_model=str(config["embedding_model"]),
        openai_env_file=(
            str(config["embedding_openai_env_file"])
            if backend == "openai_compatible"
            else None
        ),
    )


def config_trace_path(config: dict[str, Any] | None = None) -> Path | None:
    config = load_research_config() if config is None else config
    value = str(config.get("trace_path") or "").strip()
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

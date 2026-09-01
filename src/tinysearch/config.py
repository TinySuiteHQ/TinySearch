"""Public, dependency-free TinySearch configuration."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from tinysearch.services.embedding_service import (
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_OPENAI_ENV_FILE,
    SUPPORTED_EMBEDDING_BACKENDS,
    normalize_embedding_backend,
)
from tinysearch.services.web_search_service import (
    ALLOWED_SEARCH_BACKENDS,
    DEFAULT_SEARXNG_URL,
    normalize_domain,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "search_max_results": 10,
    "search_max_concurrent_items": 3,
    "scrape_max_tokens": 2000,
    "scrape_max_links": 8,
    "scrape_max_link_tokens": 500,
    "search_top_k": 10,
    "chunk_rrf_cutoff": 0.0,
    "chunk_dense_weight": 0.5,
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
    "dense_query_prefix": (
        "Instruct: Given a web search query, retrieve relevant passages that "
        "answer the query\nQuery:"
    ),
    "dense_document_prefix": "",
    "dense_document_embed_batch_size": 32,
    "blocked_domains": [],
    "search_backend": "ddgs",
    "search_backend_url": DEFAULT_SEARXNG_URL,
    "search_engines": [],
    "search_region": "",
    "search_backend_fallback": True,
    "ddgs_timeout_seconds": 20.0,
    "ddgs_backend": "auto",
    "browser_cdp_url": "",
}

_INT_FIELDS = {
    "search_max_results",
    "search_max_concurrent_items",
    "scrape_max_tokens",
    "scrape_max_links",
    "scrape_max_link_tokens",
    "search_top_k",
    "max_concurrent_embedding_calls",
    "embedding_timeout_retries",
    "crawl_fit_min_chars",
    "crawl_max_chunk_tokens",
    "crawl_overlap_tokens",
    "crawl_max_page_tokens",
    "dense_document_embed_batch_size",
}
_FLOAT_FIELDS = {
    "chunk_rrf_cutoff",
    "chunk_dense_weight",
    "crawl_bm25_threshold",
    "crawl_pruning_threshold",
    "embedding_timeout_seconds",
    "ddgs_timeout_seconds",
}


def normalize_config(raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Merge a partial mapping onto canonical defaults and validate it."""
    config = dict(DEFAULT_CONFIG)
    config.update(
        {
            key: value
            for key, value in dict(raw or {}).items()
            if not str(key).startswith("_comment")
        }
    )
    for legacy in ("embedding_gguf_file", "mcp_transport"):
        config.pop(legacy, None)
    for key in _INT_FIELDS:
        config[key] = int(config[key])
    for key in ("search_max_concurrent_items", "scrape_max_link_tokens"):
        if config[key] <= 0:
            raise ValueError(f"tinysearch config {key} must be positive")
    for key in _FLOAT_FIELDS:
        config[key] = float(config[key])
    raw_timeout = config.get("pipeline_timeout_seconds")
    config["pipeline_timeout_seconds"] = (
        float(raw_timeout) if raw_timeout is not None else None
    )
    for key in (
        "encoding_name",
        "embedding_model",
        "embedding_openai_env_file",
        "dense_query_prefix",
        "dense_document_prefix",
        "crawl_fit_markdown_mode",
        "crawl_bm25_language",
        "ddgs_backend",
        "browser_cdp_url",
    ):
        if config.get(key) is not None:
            config[key] = str(config[key])

    browser_cdp_url = config["browser_cdp_url"].strip()
    if browser_cdp_url:
        parsed_cdp_url = urlsplit(browser_cdp_url)
        if parsed_cdp_url.scheme.lower() not in {"http", "https", "ws", "wss"}:
            raise ValueError(
                "tinysearch config browser_cdp_url must use http, https, ws, or wss"
            )
        if not parsed_cdp_url.hostname:
            raise ValueError(
                "tinysearch config browser_cdp_url must include a hostname"
            )
    config["browser_cdp_url"] = browser_cdp_url

    embedding_backend = normalize_embedding_backend(
        str(config.get("embedding_backend") or DEFAULT_EMBEDDING_BACKEND)
    )
    if embedding_backend not in SUPPORTED_EMBEDDING_BACKENDS:
        raise ValueError(
            "tinysearch config embedding_backend must be one of "
            f"{list(SUPPORTED_EMBEDDING_BACKENDS)}"
        )
    config["embedding_backend"] = embedding_backend

    blocked_domains = config.get("blocked_domains", [])
    if not isinstance(blocked_domains, list):
        raise ValueError("tinysearch config blocked_domains must be a JSON list")
    config["blocked_domains"] = list(
        dict.fromkeys(
            normalized
            for item in blocked_domains
            if isinstance(item, str)
            for normalized in [normalize_domain(item)]
            if normalized
        )
    )

    backend = str(config.get("search_backend") or "ddgs").strip().lower()
    if backend not in ALLOWED_SEARCH_BACKENDS:
        raise ValueError(
            "tinysearch config search_backend must be one of "
            f"{sorted(ALLOWED_SEARCH_BACKENDS)}"
        )
    config["search_backend"] = backend
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
            "tinysearch config search_engines must be a list or comma-separated string"
        )
    config["search_engines"] = engines_list
    config["search_region"] = str(
        config.get("search_region") or config.get("search_country") or ""
    ).strip()
    config.pop("search_country", None)
    config["search_backend_fallback"] = bool(
        config.get("search_backend_fallback", True)
    )
    return config


@dataclass(frozen=True)
class TinySearchConfig(Mapping[str, Any]):
    """Validated immutable wrapper around TinySearch's flat configuration."""

    _values: dict[str, Any]

    def __init__(self, **values: Any) -> None:
        object.__setattr__(self, "_values", normalize_config(values))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None = None) -> TinySearchConfig:
        instance = cls.__new__(cls)
        object.__setattr__(instance, "_values", normalize_config(values))
        return instance

    @classmethod
    def from_json(cls, path: str | Path) -> TinySearchConfig:
        config_path = Path(path)
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"tinysearch config must be a JSON object: {config_path}")
        return cls.from_mapping(raw)

    def to_dict(self) -> dict[str, Any]:
        values = dict(self._values)
        values["blocked_domains"] = list(values["blocked_domains"])
        values["search_engines"] = list(values["search_engines"])
        return values

    def with_overrides(self, values: Mapping[str, Any]) -> TinySearchConfig:
        merged = self.to_dict()
        merged.update(values)
        return TinySearchConfig.from_mapping(merged)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, key: str) -> Any:
        try:
            return object.__getattribute__(self, "_values")[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


ConfigInput = TinySearchConfig | Mapping[str, Any]


def resolve_config(
    config: ConfigInput | None = None,
    *,
    path: str | Path | None = None,
) -> TinySearchConfig:
    """Resolve explicit file values and per-call overrides onto defaults."""
    base = TinySearchConfig.from_json(path) if path is not None else TinySearchConfig()
    if config is None:
        return base
    if isinstance(config, TinySearchConfig):
        return config if path is None else base.with_overrides(config.to_dict())
    return base.with_overrides(config)


def save_config(config: ConfigInput, path: str | Path) -> TinySearchConfig:
    resolved = resolve_config(config)
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(resolved.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(config_path)
    return resolved

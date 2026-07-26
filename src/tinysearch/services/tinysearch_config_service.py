"""Filesystem and environment configuration for TinySearch server processes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tinysearch.config import TinySearchConfig, normalize_config, save_config
from tinysearch.services.embedding_service import (
    normalize_embedding_backend,
    resolve_embedding_tokenizer_name,
)


def resolve_tinysearch_config_path(path: str | Path | None = None) -> Path:
    """Resolve an explicit/server config path without consulting the checkout."""
    if path is not None:
        return Path(path).expanduser()
    env_path = os.environ.get("TINYSEARCH_CONFIG_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    from tinysearch.paths import native_config_path

    return native_config_path()


def load_tinysearch_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load explicit/server configuration and apply server environment overrides."""
    config_path = resolve_tinysearch_config_path(path)
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
    embedding_backend = os.environ.get(
        "TINYSEARCH_EMBEDDING_BACKEND", ""
    ).strip()
    if embedding_backend:
        overrides["embedding_backend"] = embedding_backend
    embedding_model = os.environ.get("TINYSEARCH_EMBEDDING_MODEL", "").strip()
    if embedding_model:
        overrides["embedding_model"] = embedding_model
    return (
        config.with_overrides(overrides).to_dict()
        if overrides
        else config.to_dict()
    )


def save_tinysearch_config(
    raw: Mapping[str, Any],
    path: str | Path | None = None,
) -> dict[str, Any]:
    config_path = resolve_tinysearch_config_path(path)
    existing = (
        TinySearchConfig.from_json(config_path).to_dict()
        if config_path.exists()
        else {}
    )
    merged = dict(existing)
    merged.update(raw)
    return save_config(merged, config_path).to_dict()


def normalize_query(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    return query


def tokenizer_name_for_config(config: Mapping[str, Any] | None = None) -> str:
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

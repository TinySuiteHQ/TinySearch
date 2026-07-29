"""Inspect configuration, browser, model, and writable directories without downloading anything.

All output goes to stderr, this can run while an MCP stdio server is
elsewhere on the same machine, so stdout must stay clean.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from tinysearch.services.embedding_service import (
    normalize_embedding_backend,
    resolve_local_embedding_model_spec,
)
from tinysearch.services.tinysearch_config_service import (
    load_tinysearch_config,
    resolve_tinysearch_config_path,
)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _check_chromium() -> tuple[bool, str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False, "playwright is not installed"
    try:
        with sync_playwright() as p:
            executable = Path(p.chromium.executable_path)
    except Exception as exc:
        return False, f"could not resolve the Chromium executable path: {exc}"
    if executable.exists():
        return True, str(executable)
    return False, f"Chromium executable not found at {executable} (run `tinysearch setup`)"


def _check_model(config: dict[str, Any]) -> tuple[bool, str]:
    backend = normalize_embedding_backend(str(config["embedding_backend"]))
    if backend != "onnx":
        return True, f"embedding_backend={backend!r}; no local ONNX bundle required"
    spec = resolve_local_embedding_model_spec(str(config["embedding_model"]))
    if any((spec.local_dir / rel).exists() for rel in spec.onnx_paths):
        return True, f"model bundle present at {spec.local_dir}"
    return False, f"model bundle missing at {spec.local_dir} (run `tinysearch setup`)"


def _check_writable(path: Path) -> tuple[bool, str]:
    target = path if path.is_dir() else path.parent
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"{target} could not be created: {exc}"
    if os.access(target, os.W_OK):
        return True, f"{target} is writable"
    return False, f"{target} is not writable"


def run() -> int:
    """Print a readiness report to stderr. Returns 0 if every check passes, else 1."""
    config_path = resolve_tinysearch_config_path()
    config = load_tinysearch_config()
    _log(
        f"config: {config_path} "
        f"({'found' if config_path.exists() else 'not found, using built-in defaults'})"
    )

    checks = [
        ("chromium", *_check_chromium()),
        ("model", *_check_model(config)),
        ("config dir writable", *_check_writable(config_path.parent)),
    ]

    all_ok = True
    for name, ok, message in checks:
        _log(f"{name}: {'ok' if ok else 'MISSING'} - {message}")
        all_ok = all_ok and ok

    _log("all checks passed" if all_ok else "some checks failed; run `tinysearch setup`")
    return 0 if all_ok else 1

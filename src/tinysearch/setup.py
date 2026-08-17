"""Install required local browser and embedding bundles.

All output goes to stderr, matching `doctor`, this can run adjacent to an
MCP stdio server, so stdout must stay clean.
"""

from __future__ import annotations

import sys

from tinysearch.services.browser_bundle_service import install_chromium
from tinysearch.services.embedding_service import normalize_embedding_backend
from tinysearch.services.tinysearch_config_service import load_tinysearch_config


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run(with_system_deps: bool = False) -> int:
    config = load_tinysearch_config()
    if str(config.get("browser_cdp_url") or "").strip():
        _log("browser_cdp_url is configured; skipping bundled Chromium install")
    else:
        install_chromium(with_system_deps)

    if normalize_embedding_backend(str(config["embedding_backend"])) == "onnx":
        from tinysearch.services.onnx_bundle_service import ensure_onnx_bundle_sync

        _log(f"downloading ONNX embedding bundle model={config['embedding_model']!r}")
        ensure_onnx_bundle_sync(str(config["embedding_model"]))
    else:
        _log(
            f"embedding_backend={config['embedding_backend']!r}; "
            "skipping ONNX model download"
        )

    _log("setup complete")
    return 0

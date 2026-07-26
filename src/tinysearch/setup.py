"""Install Chromium and download the configured ONNX embedding model.

All output goes to stderr, matching `doctor` — this can run adjacent to an
MCP stdio server, so stdout must stay clean.
"""

from __future__ import annotations

import platform
import subprocess
import sys

from tinysearch.services.embedding_service import normalize_embedding_backend
from tinysearch.services.tinysearch_config_service import load_tinysearch_config


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _install_chromium(with_system_deps: bool) -> None:
    args = [sys.executable, "-m", "playwright", "install"]
    if with_system_deps:
        if platform.system() == "Linux":
            args.append("--with-deps")
        else:
            _log(
                "--with-system-deps has no effect on "
                f"{platform.system()}; installing Chromium only"
            )
    args.append("chromium")
    _log(f"running: {' '.join(args)}")
    subprocess.run(args, check=True, stdout=sys.stderr, stderr=sys.stderr)


def run(with_system_deps: bool = False) -> int:
    _install_chromium(with_system_deps)

    config = load_tinysearch_config()
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

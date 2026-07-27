"""Thin wrapper preserving `uvicorn servers.fastapi_server:app` for Docker/compose.

The real implementation lives at `tinysearch.servers.fastapi_server`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from tinysearch.servers.fastapi_server import app

__all__ = ["app"]

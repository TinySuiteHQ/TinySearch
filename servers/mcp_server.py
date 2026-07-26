"""Thin wrapper preserving `python servers/mcp_server.py` for Docker/compose/mcp_templates.

The real implementation lives at `tinysearch.servers.mcp_server`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from tinysearch.servers.mcp_server import main

if __name__ == "__main__":
    main()

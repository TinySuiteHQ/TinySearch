"""Initialize the installed TinySearch MCP server over a real stdio transport."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


EXPECTED_TOOLS = {"get_current_datetime", "research", "scrape_url"}


async def smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="tinysearch-mcp-smoke-") as temp_dir:
        temp_path = Path(temp_dir)
        config_path = temp_path / "config.json"
        config_path.write_text(
            json.dumps({"embedding_backend": "openai_compatible"}),
            encoding="utf-8",
        )
        child_env = os.environ.copy()
        child_env.update(
            {
                "MCP_TRANSPORT": "stdio",
                "TINYSEARCH_CONFIG_PATH": str(config_path),
                "TINYSEARCH_MODELS_DIR": str(temp_path / "models"),
            }
        )
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "tinysearch.cli", "mcp"],
            env=child_env,
        )
        async with asyncio.timeout(45):
            async with stdio_client(parameters) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    response = await session.list_tools()
                    names = {tool.name for tool in response.tools}
                    if names != EXPECTED_TOOLS:
                        raise RuntimeError(
                            f"unexpected MCP tools: {sorted(names)}; "
                            f"expected {sorted(EXPECTED_TOOLS)}"
                        )


def main() -> int:
    asyncio.run(smoke())
    print("MCP stdio smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

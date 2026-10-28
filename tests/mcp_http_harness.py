"""Run the real streamable-HTTP MCP app on a loopback port for end-to-end tests."""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import httpx
import uvicorn

from tinysearch.servers import mcp_server


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class RunningMcpServer:
    """uvicorn on a background thread, serving a freshly built MCP app."""

    def __init__(self) -> None:
        # The SDK's session manager runs once per instance; each test gets its own.
        mcp_server.mcp._session_manager = None
        self.port = _free_port()
        config = uvicorn.Config(
            mcp_server._build_streamable_http_app(),
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    def __enter__(self) -> RunningMcpServer:
        # Declared like a real HTTP deployment: code that tells stdio from
        # HTTP by the configured transport must see HTTP here.
        self._transport = patch.dict(os.environ, {"MCP_TRANSPORT": "streamable-http"})
        self._transport.start()
        self._thread.start()
        deadline = time.monotonic() + 15
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("MCP test server did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=15)
        self._transport.stop()
        mcp_server.mcp._session_manager = None


@asynccontextmanager
async def mcp_session(
    url: str,
    headers: Callable[[], dict[str, str]] | dict[str, str] | None = None,
) -> AsyncIterator[Any]:
    """One MCP client session. ``headers`` may be a callable, read per HTTP request."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def apply(request: httpx.Request) -> None:
        values = headers() if callable(headers) else (headers or {})
        for name, value in values.items():
            request.headers[name] = value

    def factory(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:  # noqa: ANN001
        return httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            auth=auth,
            follow_redirects=True,
            event_hooks={"request": [apply]},
        )

    async with streamablehttp_client(url, httpx_client_factory=factory) as (read, write, _):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=15)
            yield session

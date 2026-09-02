"""Managed child `@playwright/mcp` server, exposed through a narrow allowlist.

TinySearch does not implement browser automation. It supervises Microsoft's
`@playwright/mcp` as a stdio child process and re-exports a small subset of
its tools, because a real Playwright driver -- trusted CDP input events,
accessibility-tree refs, auto-waiting -- is strictly better than anything we
would hand-roll in injected JavaScript.

Three things are owned here and nowhere else:

* **Lifecycle.** One child process, started lazily on first use and stopped
  on idle or at server shutdown.
* **The allowlist.** Only `_EXPOSED_TOOLS` is registered. `browser_evaluate`
  and `browser_run_code_unsafe` are absent from the schema rather than
  discouraged by prompt, so a page that injects instructions into its own
  rendered text has no arbitrary-code tool to reach for. The same applies to
  the side-effecting `fill_form` / `file_upload` / `drag` / `drop`.
* **Response capping.** Accessibility snapshots are the dominant token cost
  of browser use. Oversized upstream output is spilled to a file and replaced
  by a head plus a path, in code, so it cannot be forgotten.

Upstream is version-pinned: it is a 0.0.x package that ships breaking changes
between patch releases. Argument schemas are fetched from the running child
and re-exported verbatim (see `servers/mcp_server.py`), so argument-level
drift on a version bump is inherited rather than hand-maintained.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tinysearch.telemetry import span_scope

PINNED_PLAYWRIGHT_MCP_VERSION = "0.0.80"

# The nine tools TinySearch re-exports, keyed by upstream name. Everything
# else the child offers stays unregistered. Descriptions are TinySearch's
# own: they place each tool inside our research workflow (Playwright as the
# exception path after `scrape_urls` returns thin content), which upstream's
# generic wording does not do.
_EXPOSED_TOOLS: dict[str, str] = {
    "browser_navigate": (
        "Open an exact URL in the live browser. Use only after scrape_urls returned "
        "thin, empty, or clearly incomplete content for this URL, or when the page "
        "visibly requires a small read-only interaction to reveal already-requested "
        "information. Not a discovery tool; prefer search and scrape_urls."
    ),
    "browser_find": (
        "Locate specific text or an interface element on the current page. This is the "
        "normal way to find a target: it returns matching accessibility-tree nodes and "
        "nearby context far more cheaply than a full snapshot. Reuse the returned refs "
        "for the next interaction instead of looking at the page again."
    ),
    "browser_snapshot": (
        "Capture the page's accessibility tree. Use only when a targeted find cannot "
        "explain the page or produce a usable target. Scope it to a known element and "
        "the smallest useful depth. Never snapshot the whole page merely to search for "
        "text you could find with browser_find."
    ),
    "browser_click": (
        "Click an element located by a ref from browser_find or browser_snapshot. "
        "Read-only interactions that reveal already-public content (pagination, "
        "expanding a section, accepting a cookie banner) need no confirmation; any "
        "real-world side effect does."
    ),
    "browser_type": (
        "Type text into an element located by a ref. Do not enter credentials or other "
        "sensitive data, and confirm before submitting anything with a side effect."
    ),
    "browser_wait_for": (
        "Wait for text to appear or disappear, or for a fixed delay, when a page needs "
        "time to render after an interaction."
    ),
    "browser_take_screenshot": (
        "Capture a screenshot to a file, for when a visual check genuinely matters. The "
        "image is written to disk and its path returned; it is never inlined into the "
        "conversation. Do not use it to choose an interaction target."
    ),
    "browser_tabs": (
        "List, select, open, or close browser tabs."
    ),
    "browser_close": (
        "Close the browser session. Call this when the current research task is "
        "complete; do not leave a session open for speculative exploration."
    ),
}

# Registered names are prefixed so a client sees the browser surface as one
# coherent group next to `search` / `scrape_urls` / `get_current_datetime`.
TOOL_NAME_PREFIX = "playwright_"


def exposed_tool_names() -> tuple[str, ...]:
    return tuple(_EXPOSED_TOOLS)


def public_tool_name(upstream_name: str) -> str:
    return f"{TOOL_NAME_PREFIX}{upstream_name}"


def tool_description(upstream_name: str) -> str:
    return _EXPOSED_TOOLS[upstream_name]


class PlaywrightMcpError(Exception):
    """The child Playwright MCP server could not be started or called."""


class PlaywrightMcpDisabledError(PlaywrightMcpError):
    """Browser tools were requested while `browser_backend` is "off"."""


def _npx_command() -> str:
    """Resolve the npx executable, preferring the Windows shim when present."""
    for candidate in ("npx.cmd", "npx") if os.name == "nt" else ("npx",):
        found = shutil.which(candidate)
        if found:
            return found
    raise PlaywrightMcpError(
        "Node.js is required for browser_backend='playwright_mcp' but npx was not "
        "found on PATH. Install Node.js 18+ or set browser_backend to 'off'."
    )


def build_launch_args(config: Mapping[str, Any]) -> list[str]:
    """Build the child's argv from TinySearch config.

    `--isolated` plus `--storage-state` is deliberate. A persistent
    `--user-data-dir` can only be held by one browser instance at a time,
    which would make concurrent TinySearch clients conflict. An isolated
    in-memory context seeded from a shared storage-state file keeps cookie
    persistence (so consent interstitials are paid for once, not on every
    navigation) without taking a profile lock -- and keeps that state a
    server-side path the model never sees or transmits.

    `--caps` is never passed: the vision, pdf, devtools, network, storage,
    and testing tool groups stay off at the source.
    """
    args = [
        "-y",
        f"@playwright/mcp@{PINNED_PLAYWRIGHT_MCP_VERSION}",
        "--headless",
        "--isolated",
        "--image-responses",
        "omit",
        "--block-service-workers",
    ]

    storage_state = str(config.get("browser_storage_state_path") or "").strip()
    if storage_state:
        Path(storage_state).parent.mkdir(parents=True, exist_ok=True)
        args += ["--storage-state", storage_state]

    output_dir = str(config.get("browser_output_dir") or "").strip()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        args += ["--output-dir", output_dir]

    cdp_url = str(config.get("browser_cdp_url") or "").strip()
    if cdp_url:
        args += ["--cdp-endpoint", cdp_url]

    timeout_seconds = float(config.get("browser_action_timeout_seconds") or 10.0)
    args += ["--timeout-action", str(int(timeout_seconds * 1000))]

    return args


def _resolve_output_dir(config: Mapping[str, Any]) -> Path:
    configured = str(config.get("browser_output_dir") or "").strip()
    directory = Path(configured) if configured else Path.cwd() / ".tinysearch" / "browser"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def cap_response(
    text: str,
    *,
    char_budget: int,
    output_dir: Path,
    tool_name: str,
) -> str:
    """Spill an oversized upstream response to disk, returning a head and a path.

    Accessibility snapshots of a large page dwarf every other response
    TinySearch produces. Truncating in the proxy -- rather than asking a
    model to remember to be frugal -- is what makes the browser affordable
    to keep enabled.
    """
    if char_budget <= 0 or len(text) <= char_budget:
        return text

    path = output_dir / f"{tool_name}-{int(time.time() * 1000)}.txt"
    try:
        path.write_text(text, encoding="utf-8")
        pointer = str(path)
    except OSError as exc:  # disk full, read-only mount, bad path
        pointer = f"<unavailable: {exc}>"

    head = text[:char_budget]
    return (
        f"{head}\n\n"
        f"[tinysearch: truncated {tool_name} response at {char_budget} of "
        f"{len(text)} characters. Full output saved to {pointer}. "
        f"Prefer browser_find over a broader snapshot rather than re-reading this.]"
    )


class PlaywrightMcpClient:
    """Owns one child `@playwright/mcp` process and forwards allowlisted calls.

    The child is started on first use and stopped after `idle_seconds`
    without a call, so an enabled-but-unused browser backend costs nothing
    but a config key. A single shared child (rather than one per caller)
    keeps memory bounded; `--isolated` means it holds no profile lock.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)
        self._session: Any = None
        self._runner: asyncio.Task[None] | None = None
        self._ready: asyncio.Event = asyncio.Event()
        self._stop: asyncio.Event = asyncio.Event()
        self._startup_error: BaseException | None = None
        self._lock = asyncio.Lock()
        self._last_used = time.monotonic()
        self._idle_task: asyncio.Task[None] | None = None
        self._upstream_tools: dict[str, Any] = {}

    @property
    def running(self) -> bool:
        return self._runner is not None and not self._runner.done()

    async def _run_child(self) -> None:
        """Hold the stdio client open for the child's whole lifetime.

        `stdio_client` and `ClientSession` are async context managers, so a
        long-lived session needs a task that stays parked inside them. This
        task owns both and only unwinds when `_stop` is set, which keeps the
        process teardown on the same task that created it.
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=_npx_command(),
            args=build_launch_args(self._config),
            env=dict(os.environ),
        )
        try:
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    listing = await session.list_tools()
                    self._upstream_tools = {tool.name: tool for tool in listing.tools}
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001 - re-raised to the caller below
            self._startup_error = exc
            self._ready.set()
            raise
        finally:
            self._session = None

    async def start(self) -> None:
        async with self._lock:
            if self.running:
                return
            self._ready = asyncio.Event()
            self._stop = asyncio.Event()
            self._startup_error = None
            self._runner = asyncio.create_task(self._run_child())
            await self._ready.wait()
            if self._startup_error is not None or self._session is None:
                error = self._startup_error
                self._runner = None
                raise PlaywrightMcpError(
                    f"failed to start {PINNED_PLAYWRIGHT_MCP_VERSION} playwright mcp child: {error}"
                ) from error
            missing = [name for name in _EXPOSED_TOOLS if name not in self._upstream_tools]
            if missing:
                raise PlaywrightMcpError(
                    "playwright mcp child does not expose expected tools: "
                    f"{sorted(missing)}. Pinned version is "
                    f"{PINNED_PLAYWRIGHT_MCP_VERSION}."
                )
            if self._idle_task is None or self._idle_task.done():
                self._idle_task = asyncio.create_task(self._idle_watchdog())

    async def _idle_watchdog(self) -> None:
        idle_seconds = float(self._config.get("browser_idle_shutdown_seconds") or 300.0)
        try:
            while self.running:
                await asyncio.sleep(min(30.0, idle_seconds))
                if not self.running:
                    return
                if time.monotonic() - self._last_used > idle_seconds:
                    await self.stop()
                    return
        except asyncio.CancelledError:
            raise

    async def upstream_schema(self, upstream_name: str) -> dict[str, Any]:
        await self.start()
        tool = self._upstream_tools.get(upstream_name)
        if tool is None:
            raise PlaywrightMcpError(f"playwright mcp child has no tool {upstream_name!r}")
        return dict(tool.inputSchema or {"type": "object", "properties": {}})

    async def call(self, upstream_name: str, arguments: Mapping[str, Any]) -> str:
        if upstream_name not in _EXPOSED_TOOLS:
            raise PlaywrightMcpError(
                f"{upstream_name!r} is not an allowlisted TinySearch browser tool"
            )
        await self.start()
        self._last_used = time.monotonic()

        with span_scope(
            "tinysearch.playwright",
            attributes={"tinysearch.playwright.tool": upstream_name},
            operation="playwright_tool",
        ) as telemetry:
            session = self._session
            if session is None:
                raise PlaywrightMcpError("playwright mcp child is not running")
            result = await session.call_tool(upstream_name, dict(arguments))
            telemetry.complete()

        self._last_used = time.monotonic()
        text = "\n".join(
            block.text
            for block in (result.content or [])
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        ).strip()

        if getattr(result, "isError", False):
            raise PlaywrightMcpError(text or f"{upstream_name} failed")

        capped = cap_response(
            text,
            char_budget=int(self._config.get("browser_response_char_budget") or 0),
            output_dir=_resolve_output_dir(self._config),
            tool_name=upstream_name,
        )
        if upstream_name == "browser_close":
            await self.stop()
        return capped

    async def stop(self) -> None:
        runner, self._runner = self._runner, None
        idle_task, self._idle_task = self._idle_task, None
        self._stop.set()
        if idle_task is not None and idle_task is not asyncio.current_task():
            idle_task.cancel()
        if runner is not None:
            try:
                await asyncio.wait_for(asyncio.shield(runner), timeout=15.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                runner.cancel()
            except Exception:
                pass
        self._session = None
        self._upstream_tools = {}


async def fetch_tool_schemas(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Start a throwaway child purely to read the allowlisted tools' schemas.

    Tool registration happens before the serving event loop exists, so this
    deliberately does not touch the process-wide client: an `asyncio.Lock`
    or `Event` awaited on a registration loop would be bound to a loop that
    is closed by the time requests arrive. The long-lived client is
    constructed lazily on the serving loop instead.
    """
    client = PlaywrightMcpClient(config)
    try:
        return {
            name: await client.upstream_schema(name) for name in exposed_tool_names()
        }
    finally:
        await client.stop()


_client: PlaywrightMcpClient | None = None


def browser_backend_enabled(config: Mapping[str, Any]) -> bool:
    return str(config.get("browser_backend") or "off").strip().lower() == "playwright_mcp"


def get_client(config: Mapping[str, Any]) -> PlaywrightMcpClient:
    """Return the process-wide child client, creating it on first use."""
    global _client
    if not browser_backend_enabled(config):
        raise PlaywrightMcpDisabledError(
            "browser tools are disabled; set browser_backend to 'playwright_mcp' in "
            "the TinySearch config to enable them"
        )
    if _client is None:
        _client = PlaywrightMcpClient(config)
    return _client


async def shutdown_client() -> None:
    """Stop the child process during a server's graceful shutdown."""
    global _client
    client = _client
    _client = None
    if client is not None:
        await client.stop()

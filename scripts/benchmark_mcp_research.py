"""
Spawn TinySearch MCP over stdio, initialize, optionally call `research`, and print timings.

Uses the same transport as Cursor (stdio). Inherits your current environment so
embedding/API settings match a normal shell.

Edit the variables under `if __name__ == "__main__"` (no CLI args). By default
this runs one `research` call; set `_LIST_TOOLS_ONLY = True` for a quick
connect + tools/list only.

Logging toggles: `_PHASE_LOG`, `_SHOW_PROGRESS` (MCP tool progress), `_SHOW_MCP_LOG`
(server `ctx.info` / etc. via MCP logging), `_SERVER_UNBUFFERED` (child stderr, e.g.
`[research]` / `[tinysearch]` lines, appears sooner). `_LOG_EMBED_TIMING` turns on
pipeline embedding timing (`TINYSEARCH_LOG_EMBED_TIMING` in the child process).

By default the benchmark **requires** the ONNX bundle (onnxruntime path) so timings
match the intended fast path. With `embedding_backend` `onnx` in
`configs/research_config.json`, this script **prefetches** the bundle via the same
`ensure_onnx_bundle_sync()` used by `servers/mcp_server.py` before spawning the child,
so the first run does not fail when weights are gitignored. Set
`_REQUIRE_ONNX_BUNDLE = False` to benchmark with `openai_compatible` embeddings.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import anyio
import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from tinysearch.services.embedding_service import (
    normalize_embedding_backend,
    onnx_backend_will_use_onnx_bundle,
)
from tinysearch.services.onnx_bundle_service import ensure_onnx_bundle_sync
from tinysearch.services.research_config_service import load_research_config


def _phase(label: str, t0: float, enabled: bool) -> None:
    if not enabled:
        return
    print(f"[benchmark] {label} (+{time.perf_counter() - t0:.3f}s)", flush=True)


def _tool_result_summary(result: types.CallToolResult) -> dict[str, object]:
    out: dict[str, object] = {"isError": result.isError}
    if result.structuredContent is not None:
        out["structured_keys"] = list(result.structuredContent.keys())
        ans = result.structuredContent.get("answer")
        if isinstance(ans, str):
            out["answer_chars"] = len(ans)
    text_parts: list[str] = []
    for block in result.content:
        if isinstance(block, types.TextContent):
            text_parts.append(block.text)
    if text_parts:
        joined = "\n".join(text_parts)
        out["text_chars"] = len(joined)
        try:
            data = json.loads(joined)
            if isinstance(data, dict) and "answer" in data:
                out["parsed_answer_chars"] = len(str(data["answer"]))
        except json.JSONDecodeError:
            out["text_preview"] = joined[:200].replace("\n", " ")
    return out


async def _run(
    python_exe: str,
    server_script: Path,
    query: str | None,
    list_tools: bool,
    show_progress: bool,
    show_mcp_log: bool,
    phase_log: bool,
    server_unbuffered: bool,
    embed_timing_log: bool,
    tool_timeout: timedelta,
    cwd: Path,
) -> None:
    child_env = os.environ.copy()
    if server_unbuffered:
        child_env["PYTHONUNBUFFERED"] = "1"
    if embed_timing_log:
        child_env["TINYSEARCH_LOG_EMBED_TIMING"] = "1"

    params = StdioServerParameters(
        command=python_exe,
        args=[str(server_script)],
        cwd=str(cwd),
        env=child_env,
    )

    t0 = time.perf_counter()
    _phase("starting stdio client (spawning mcp_server.py)", t0, phase_log)

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        if not show_progress:
            return
        t = total if total is not None else "?"
        print(f"  [progress] {progress}/{t} {message or ''}", flush=True)

    async def on_mcp_log(params: types.LoggingMessageNotificationParams) -> None:
        if not show_mcp_log:
            return
        logger = params.logger or "server"
        print(f"  [mcp {params.level}] [{logger}] {params.data}", flush=True)

    async with stdio_client(params) as (read_stream, write_stream):
        t_after_spawn = time.perf_counter()
        _phase("stdio streams ready (subprocess alive)", t0, phase_log)
        async with ClientSession(
            read_stream,
            write_stream,
            logging_callback=on_mcp_log,
        ) as session:
            await session.initialize()
            t_after_init = time.perf_counter()
            _phase("MCP initialize complete", t0, phase_log)

            if show_mcp_log:
                for level in ("debug", "info"):
                    try:
                        await session.set_logging_level(level)
                        break
                    except Exception as exc:
                        logging.getLogger("benchmark_mcp").warning(
                            "set_logging_level(%r) failed (%s); continuing", level, exc
                        )

            if list_tools:
                tools = await session.list_tools()
                names = [t.name for t in tools.tools]
                print(f"tools/list: {names}")
                t_end = time.perf_counter()
                print(
                    f"timings: spawn_to_session_ready_s={t_after_spawn - t0:.3f} "
                    f"initialize_s={t_after_init - t_after_spawn:.3f} "
                    f"total_s={t_end - t0:.3f}"
                )
                return

            if not query:
                raise RuntimeError("set a non-empty query when list_tools is False")

            _phase(f"tools/call research query={query!r} …", t0, phase_log)
            t_before_tool = time.perf_counter()
            result = await session.call_tool(
                "research",
                {"query": query},
                read_timeout_seconds=tool_timeout,
                progress_callback=on_progress if show_progress else None,
            )
            t_after_tool = time.perf_counter()
            _phase("tools/call research returned", t0, phase_log)

            summary = _tool_result_summary(result)
            print(f"tools/call research: {summary}")
            if result.isError:
                for block in result.content:
                    if isinstance(block, types.TextContent):
                        print(block.text[:2000])
                raise RuntimeError("research tool returned isError=true")

            print(
                "timings: "
                f"spawn_to_session_ready_s={t_after_spawn - t0:.3f} "
                f"initialize_s={t_after_init - t_after_spawn:.3f} "
                f"research_s={t_after_tool - t_before_tool:.3f} "
                f"total_wall_s={t_after_tool - t0:.3f}"
            )


if __name__ == "__main__":
    _PYTHON_EXE = sys.executable
    _SERVER_SCRIPT = _PROJECT_ROOT / "servers" / "mcp_server.py"
    _CWD = _PROJECT_ROOT
    _TOOL_TIMEOUT_SECONDS = 900
    _PHASE_LOG = True
    _SHOW_PROGRESS = False
    _SHOW_MCP_LOG = False
    _SERVER_UNBUFFERED = True
    _LOG_EMBED_TIMING = True
    _LIST_TOOLS_ONLY = False
    _QUERY = "what is the walrus operator in Python"
    _REQUIRE_ONNX_BUNDLE = True

    if not _SERVER_SCRIPT.is_file():
        raise SystemExit(f"server script not found: {_SERVER_SCRIPT}")

    if not _LIST_TOOLS_ONLY and not (_QUERY or "").strip():
        raise SystemExit("set _QUERY when _LIST_TOOLS_ONLY is False")

    _cfg = load_research_config()
    _backend = normalize_embedding_backend(str(_cfg["embedding_backend"]))
    if _REQUIRE_ONNX_BUNDLE and _backend != "onnx":
        raise SystemExit(
            "_REQUIRE_ONNX_BUNDLE is True but configs/research_config.json has "
            f"embedding_backend={_cfg['embedding_backend']!r} (resolved {_backend!r}). "
            "Use onnx local embeddings or set _REQUIRE_ONNX_BUNDLE = False."
        )
    if _REQUIRE_ONNX_BUNDLE:
        ensure_onnx_bundle_sync(str(_cfg["embedding_model"]))

    _onnx_ok = onnx_backend_will_use_onnx_bundle(str(_cfg["embedding_model"]))
    _kind = "onnx bundle (onnxruntime)" if _onnx_ok else "missing ONNX bundle"
    print(f"[benchmark] repo onnx local embeddings -> {_kind}", flush=True)
    if _REQUIRE_ONNX_BUNDLE and not _onnx_ok:
        raise SystemExit(
            "Benchmark requires a complete ONNX bundle under "
            f"{_PROJECT_ROOT / 'models'} for embedding_model={_cfg['embedding_model']!r}. "
            "Prefetch failed; check network/Hugging Face access, run "
            "scripts/export_embedding_onnx.py for fast, or set _REQUIRE_ONNX_BUNDLE = False."
        )

    _tool_timeout = timedelta(seconds=max(1, _TOOL_TIMEOUT_SECONDS))
    _query_for_run = None if _LIST_TOOLS_ONLY else str(_QUERY).strip()

    try:
        anyio.run(
            _run,
            _PYTHON_EXE,
            _SERVER_SCRIPT,
            _query_for_run,
            _LIST_TOOLS_ONLY,
            _SHOW_PROGRESS,
            _SHOW_MCP_LOG,
            _PHASE_LOG,
            _SERVER_UNBUFFERED,
            _LOG_EMBED_TIMING,
            _tool_timeout,
            _CWD,
            backend="asyncio",
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

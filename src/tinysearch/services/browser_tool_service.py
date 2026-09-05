"""Browser automation implemented directly on Playwright's Python API.

TinySearch already depends on Playwright (through Crawl4AI) and already
installs its Chromium, so browser automation needs no second runtime, no
second browser, and no child process: the tools below are thin adapters over
the same driver the scrape pipeline uses.

What makes this workable is Playwright's own accessibility snapshot. It
labels every node with a stable `[ref=eNN]`, and the `aria-ref=` selector
engine turns one of those labels back into a real locator. So a model reads
a compact tree, names a node, and TinySearch clicks it with genuine CDP
input events -- trusted, auto-waiting, and framework-agnostic. Nothing here
synthesizes DOM events or invents CSS selectors.

Token cost is controlled structurally rather than by truncation: the
snapshot's `depth` produces a shallower *valid* tree instead of a string cut
in half. On a large page that is the difference between ~600 and ~33,000
characters.

Sessions are isolated and seeded from an optional storage-state file, so a
cookie banner accepted once is not paid for on every later navigation while
no browser profile lock is taken.

`aria_snapshot(mode="ai")` is a recent and historically unstable API -- the
parameter was `ref: bool` in Playwright 1.52, absent in 1.54-1.58, and
became `mode`/`depth` in 1.59. `ensure_supported()` therefore checks for it
explicitly rather than failing later with an opaque TypeError.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tinysearch.telemetry import span_scope

MINIMUM_PLAYWRIGHT_VERSION = "1.59"

# The tool surface TinySearch exposes. Deliberately excludes anything that
# evaluates arbitrary code in the page, and anything with a real-world side
# effect (form fill, file upload, drag/drop): a page's own rendered text is
# untrusted input, and it must not be able to reach a tool that runs code.
TOOL_NAMES = (
    "navigate",
    "look",
    "click",
    "type",
    "wait_for",
    "take_screenshot",
    "tabs",
    "close",
)

# Arguments that shape what a call *returns* rather than what it does. Every
# operation handing back a view of the page accepts them, because finding is
# not a sibling of clicking -- it is a filter on the result. Without that, an
# agent pays for a whole tree after each action and then pays again to narrow
# it; with it, a click that reveals a table can return only the table.
VIEW_ARGUMENTS = ("find", "depth")

# MCP has no way to group or nest tools: `tools/list` is flat, and every
# schema is re-sent to the model on every request. So the eight operations are
# published as two tools. `navigate` stays first-class -- it is the entry
# point, the only operation that does not need a page already open, and the
# one a model reaches for first -- while the remaining seven are one page
# session's lifecycle and fold behind `browser_act(action=...)`.
ACT_ACTIONS = (
    "look",
    "click",
    "type",
    "wait_for",
    "take_screenshot",
    "tabs",
    "close",
)

# Which arguments each folded action actually consumes. A flat dispatcher
# cannot express "click requires target" in JSON Schema, so the requirement is
# enforced here and reported as a precise error instead of a TypeError.
_ACT_PARAMETERS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # action: (accepted, required)
    "look": (VIEW_ARGUMENTS, ()),
    "click": (("target", *VIEW_ARGUMENTS), ("target",)),
    "type": (("target", "text", "submit", *VIEW_ARGUMENTS), ("target", "text")),
    "wait_for": (("time_seconds", "text", "text_gone", *VIEW_ARGUMENTS), ()),
    "take_screenshot": (("full_page",), ()),
    "tabs": (("action", "index", *VIEW_ARGUMENTS), ()),
    "close": ((), ()),
}

_REF_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


class BrowserToolError(Exception):
    """A browser tool could not complete."""


class BrowserDisabledError(BrowserToolError):
    """Browser tools were called while `browser_backend` is "off"."""


def resolve_act_arguments(action: str, supplied: Mapping[str, Any]) -> dict[str, Any]:
    """Select the arguments one folded action uses, and check the required ones.

    Arguments meant for a different action are dropped rather than rejected:
    a model that sends a stray `depth` alongside a click should get the click,
    not an error about an argument that simply does not apply here.
    """
    if action not in _ACT_PARAMETERS:
        raise BrowserToolError(
            f"action must be one of {list(ACT_ACTIONS)}, not {action!r}"
        )
    accepted, required = _ACT_PARAMETERS[action]
    # The MCP dispatcher uses "" / 0 rather than null for absent optional
    # arguments: a nullable JSON-Schema type costs an anyOf block per
    # parameter, and this tool has nine of them. Booleans are exempt --
    # False is a real value -- as is any argument whose own zero is
    # meaningful, such as a tab index.
    resolved = {}
    for key, value in supplied.items():
        if key not in accepted or value is None:
            continue
        if not isinstance(value, bool) and key != "index" and value in ("", 0, 0.0):
            continue
        resolved[key] = value
    missing = [key for key in required if resolved.get(key) in (None, "")]
    if missing:
        raise BrowserToolError(
            f"browser action {action!r} requires {', '.join(missing)}"
        )
    return resolved


def browser_backend_enabled(config: Mapping[str, Any]) -> bool:
    return str(config.get("browser_backend") or "playwright").strip().lower() != "off"


def ensure_supported() -> None:
    """Fail early, and legibly, on a Playwright too old for AI-mode snapshots."""
    from playwright.async_api import Locator

    signature = inspect.signature(Locator.aria_snapshot)
    if "mode" not in signature.parameters:
        import importlib.metadata as metadata

        try:
            found = metadata.version("playwright")
        except Exception:  # pragma: no cover - metadata should always exist
            found = "unknown"
        raise BrowserToolError(
            f"browser tools need playwright >= {MINIMUM_PLAYWRIGHT_VERSION} for "
            f"accessibility snapshots with element refs, but {found} is installed"
        )


def _validate_ref(target: str) -> str:
    """Accept only a snapshot ref, never a raw selector.

    Constraining the target to a ref the model actually observed keeps a page
    from talking a model into addressing arbitrary nodes, and keeps this from
    becoming a selector-injection surface.
    """
    value = (target or "").strip()
    if not value or not _REF_PATTERN.match(value):
        raise BrowserToolError(
            f"target must be an element ref from a snapshot, such as 'e42', not {target!r}"
        )
    return value


def cap(text: str, char_budget: int) -> str:
    """Trim a response that outgrew its budget, and say how to ask for less.

    `find` and `depth` are the better levers, so the message points at them
    rather than silently swallowing the rest of the page.
    """
    if char_budget <= 0 or len(text) <= char_budget:
        return text
    return (
        text[:char_budget]
        + f"\n\n[tinysearch: truncated at {char_budget} of {len(text)} characters. "
        "Pass find= to return only the matching nodes, or a smaller depth.]"
    )


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _record_bounds(lines: list[str], index: int, max_lines: int = 30) -> tuple[int, int]:
    """Grow a match outward to its enclosing record instead of a fixed window.

    An aria-snapshot nests each record (one video, one result row) several
    levels deep -- a heading and the views/date a few fields below it share a
    common ancestor well above the heading's immediate parent, so a fixed
    +/-2 line window around the heading cuts the date off entirely. Climbing
    to successive ancestors and taking each one's whole subtree, as long as
    it still fits `max_lines`, finds that common ancestor without knowing
    anything about the page: the shared list container (every sibling
    record at once) blows the budget and stops the climb one level below
    it, which is the record boundary.
    """
    best = max(0, index - 2), min(len(lines), index + 3)
    root = index
    indent = _indent(lines[root])
    while True:
        # Search from the current root, not the padded window -- starting
        # from the fallback's edge would skip straight past the true
        # immediate parent whenever it falls inside that +/-2 padding.
        parent = root - 1
        while parent >= 0 and _indent(lines[parent]) >= indent:
            parent -= 1
        if parent < 0:
            return best
        parent_indent = _indent(lines[parent])
        subtree_end = parent + 1
        while subtree_end < len(lines) and _indent(lines[subtree_end]) > parent_indent:
            subtree_end += 1
        if subtree_end - parent > max_lines:
            return best
        best = (parent, subtree_end)
        root, indent = parent, parent_indent


def filter_snapshot(snapshot: str, find: str | None) -> str | None:
    """Reduce a snapshot to the nodes matching `find`, with surrounding context.

    `find` is tried as a regular expression first, so alternation and other
    patterns work without a second parameter to remember or a "no matches"
    dead end when a model reaches for `a|b` syntax on the wrong argument.
    Text that fails to compile (an unescaped `[` or a bare `*`) falls back to
    a literal, case-insensitive substring search instead of erroring, so
    plain text never needs escaping.

    Returns None when no filter was asked for, which is how a caller tells
    "show me the whole tree" apart from "nothing matched" -- the second is a
    real answer about the page and must not be mistaken for the first.
    """
    if not find:
        return None

    lines = snapshot.splitlines()
    try:
        search = re.compile(find, re.IGNORECASE).search
    except re.error:
        lowered = find.lower()
        search = lambda line: lowered in line.lower()  # noqa: E731

    blocks: list[str] = []
    covered: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        if not search(line):
            continue
        if any(start <= index < end for start, end in covered):
            continue
        start, end = _record_bounds(lines, index)
        covered.append((start, end))
        blocks.append("\n".join(lines[start:end]))
    if not blocks:
        return "No matches."
    return f"Found {len(blocks)} matches:\n\n" + "\n\n----\n\n".join(blocks)


class BrowserSession:
    """One Playwright browser, context, and page, shared across tool calls."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._owns_browser = True
        self._lock = asyncio.Lock()
        self._last_used = time.monotonic()
        self._idle_task: asyncio.Task[None] | None = None

    @property
    def started(self) -> bool:
        return self._context is not None

    def _storage_state_path(self) -> Path | None:
        raw = str(self._config.get("browser_storage_state_path") or "").strip()
        return Path(raw) if raw else None

    def _output_dir(self) -> Path:
        raw = str(self._config.get("browser_output_dir") or "").strip()
        directory = Path(raw) if raw else Path.cwd() / ".tinysearch" / "browser"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    async def start(self) -> None:
        self._cancel_idle_shutdown()
        async with self._lock:
            if self.started:
                return
            ensure_supported()
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            cdp_url = str(self._config.get("browser_cdp_url") or "").strip()
            if cdp_url:
                # An operator-supplied browser owns its own profile and
                # fingerprint; connect to it instead of launching one.
                self._browser = await self._playwright.chromium.connect_over_cdp(cdp_url)
                self._owns_browser = False
            else:
                self._browser = await self._playwright.chromium.launch(headless=True)
                self._owns_browser = True

            state = self._storage_state_path()
            context_options: dict[str, Any] = {"locale": "en-US"}
            if state is not None and state.is_file():
                context_options["storage_state"] = str(state)
            self._context = await self._browser.new_context(**context_options)
            self._context.set_default_timeout(
                float(self._config.get("browser_action_timeout_seconds") or 10.0) * 1000
            )
            self._page = await self._context.new_page()
            self._last_used = time.monotonic()

    async def _save_storage_state(self) -> None:
        """Persist cookies so a consent banner is a one-time cost."""
        state = self._storage_state_path()
        if state is None or self._context is None:
            return
        try:
            state.parent.mkdir(parents=True, exist_ok=True)
            await self._context.storage_state(path=str(state))
        except Exception:
            pass

    async def close(self) -> None:
        idle_task = self._idle_task
        self._idle_task = None
        if idle_task is not None and idle_task is not asyncio.current_task():
            idle_task.cancel()
        async with self._lock:
            await self._save_storage_state()
            for closer in (self._context, self._browser if self._owns_browser else None):
                if closer is not None:
                    try:
                        await closer.close()
                    except Exception:
                        pass
            if self._playwright is not None:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
            self._playwright = self._browser = self._context = self._page = None

    def _cancel_idle_shutdown(self) -> None:
        task, self._idle_task = self._idle_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _shutdown_after_idle(self) -> None:
        try:
            idle_seconds = float(
                self._config.get("browser_idle_shutdown_seconds") or 300.0
            )
            await asyncio.sleep(idle_seconds)
            if self.started and self.idle_seconds() >= idle_seconds:
                await self.close()
        except asyncio.CancelledError:
            return

    def schedule_idle_shutdown(self) -> None:
        """Restart the inactivity timer after a completed browser tool call."""
        self._cancel_idle_shutdown()
        if self.started:
            self._last_used = time.monotonic()
            self._idle_task = asyncio.create_task(self._shutdown_after_idle())

    async def page(self) -> Any:
        self._cancel_idle_shutdown()
        await self.start()
        self._last_used = time.monotonic()
        if self._page is None or self._page.is_closed():
            self._page = await self._context.new_page()
        return self._page

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_used

    # --- snapshot helpers -------------------------------------------------

    async def _snapshot(self, depth: int | None = None) -> str:
        """Snapshot the page, tolerating a navigation the last action started.

        Snapshotting at page level rather than through a `body` locator
        matters: a click that triggers navigation swaps the document out, and
        a body-scoped locator can resolve against the torn-down page and fail.
        """
        page = await self.page()
        resolved = depth if depth is not None else self._config.get("browser_snapshot_depth")
        kwargs: dict[str, Any] = {"mode": "ai"}
        if resolved:
            kwargs["depth"] = int(resolved)
        try:
            await page.wait_for_load_state("domcontentloaded")
        except Exception:
            pass
        return await page.aria_snapshot(**kwargs)

    async def _page_header(self) -> str:
        page = await self.page()
        try:
            title = await page.title()
        except Exception:
            title = ""
        return f"- Page URL: {page.url}\n- Page Title: {title}"

    async def _observation(
        self,
        depth: int | None = None,
        find: str | None = None,
    ) -> str:
        """Return the page view every action hands back, narrowed if asked.

        Filtering here rather than in a separate tool is what lets one call
        both act and report: the snapshot is taken once either way, so the
        narrow form costs strictly less than the tree it replaces.
        """
        header = await self._page_header()
        snapshot = await self._snapshot(depth)
        matches = filter_snapshot(snapshot, find)
        if matches is not None:
            return f"{header}\n\n{matches}"
        return f"{header}\n\n### Snapshot\n{snapshot}"

    # --- the eight operations ---------------------------------------------

    async def navigate(
        self,
        url: str,
        depth: int | None = None,
        find: str | None = None,
    ) -> str:
        page = await self.page()
        await page.goto(url, wait_until="domcontentloaded")
        return await self._observation(depth, find)

    async def look(
        self,
        depth: int | None = None,
        find: str | None = None,
    ) -> str:
        """Read the current page without touching it.

        With `find` this is the cheap way to get a ref -- on a large page a
        few lines where the whole tree would be tens of thousands. Without
        one it is the full snapshot, for when no filter can name the target.
        """
        return await self._observation(depth, find)

    async def click(
        self,
        target: str,
        depth: int | None = None,
        find: str | None = None,
    ) -> str:
        page = await self.page()
        await page.locator(f"aria-ref={_validate_ref(target)}").click()
        return await self._observation(depth, find)

    async def type(
        self,
        target: str,
        text: str,
        submit: bool = False,
        depth: int | None = None,
        find: str | None = None,
    ) -> str:
        page = await self.page()
        locator = page.locator(f"aria-ref={_validate_ref(target)}")
        await locator.fill(text)
        if submit:
            await locator.press("Enter")
        return await self._observation(depth, find)

    async def wait_for(
        self,
        time_seconds: float | None = None,
        text: str | None = None,
        text_gone: str | None = None,
        depth: int | None = None,
        find: str | None = None,
    ) -> str:
        page = await self.page()
        provided = [value is not None for value in (time_seconds, text, text_gone)]
        if sum(provided) != 1:
            raise BrowserToolError("provide exactly one of time, text, text_gone")
        if time_seconds is not None:
            await page.wait_for_timeout(float(time_seconds) * 1000)
        elif text is not None:
            await page.get_by_text(text).first.wait_for(state="visible")
        else:
            await page.get_by_text(text_gone).first.wait_for(state="hidden")
        return await self._observation(depth, find)

    async def take_screenshot(self, full_page: bool = False) -> str:
        """Write a screenshot to disk and return its path.

        The image is never inlined: a screenshot costs far more than the
        snapshot text that should be driving decisions anyway.
        """
        page = await self.page()
        path = self._output_dir() / f"screenshot-{int(time.time() * 1000)}.png"
        await page.screenshot(path=str(path), full_page=bool(full_page))
        return f"Screenshot saved to {path}"

    async def tabs(
        self,
        action: str = "list",
        index: int | None = None,
        depth: int | None = None,
        find: str | None = None,
    ) -> str:
        await self.start()
        pages = self._context.pages
        if action == "list":
            rows = [f"{i}: {p.url}" for i, p in enumerate(pages)]
            return "Open tabs:\n" + ("\n".join(rows) if rows else "(none)")
        if action == "new":
            self._page = await self._context.new_page()
            return f"Opened tab {len(self._context.pages) - 1}"
        if index is None or not 0 <= index < len(pages):
            raise BrowserToolError(f"tab index {index!r} is out of range")
        if action == "select":
            self._page = pages[index]
            await self._page.bring_to_front()
            return await self._observation(depth, find)
        if action == "close":
            await pages[index].close()
            remaining = self._context.pages
            self._page = remaining[0] if remaining else None
            return f"Closed tab {index}"
        raise BrowserToolError(f"unknown tabs action {action!r}")

    async def close_browser(self) -> str:
        if not self.started:
            return "Browser is not open."
        await self.close()
        return "Browser closed."


_session: BrowserSession | None = None


def get_session(config: Mapping[str, Any]) -> BrowserSession:
    global _session
    if not browser_backend_enabled(config):
        raise BrowserDisabledError(
            "browser tools are disabled; set browser_backend to 'playwright' to enable them"
        )
    if _session is None:
        _session = BrowserSession(config)
    return _session


async def shutdown_session() -> None:
    """Close the browser during a server's graceful shutdown."""
    global _session
    session, _session = _session, None
    if session is not None:
        await session.close()


async def call_tool(name: str, config: Mapping[str, Any], **arguments: Any) -> str:
    """Run one browser tool and return its capped text response."""
    if name not in TOOL_NAMES:
        raise BrowserToolError(f"unknown browser tool {name!r}")
    session = get_session(config)

    with span_scope(
        "tinysearch.browser_tool",
        attributes={"tinysearch.browser_tool.name": name},
        operation="browser_tool",
    ) as telemetry:
        method = {
            "navigate": session.navigate,
            "look": session.look,
            "click": session.click,
            "type": session.type,
            "wait_for": session.wait_for,
            "take_screenshot": session.take_screenshot,
            "tabs": session.tabs,
            "close": session.close_browser,
        }[name]
        try:
            result = await method(**arguments)
        except BrowserToolError:
            raise
        except Exception as exc:
            raise BrowserToolError(f"{name} failed: {exc}") from exc
        finally:
            session.schedule_idle_shutdown()
        telemetry.complete()

    return cap(result, int(config.get("browser_response_char_budget") or 0))

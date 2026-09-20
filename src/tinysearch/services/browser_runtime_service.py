"""Shared Playwright runtime and browser-context configuration."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

# Connection.cleanup() (playwright._impl._connection) cancels its internal
# `_init_task` without awaiting or clearing the reference, so the Task (and
# the TargetClosedError future it captured) survive until a later garbage
# collection pass. Python's C-accelerated Task/Future __del__ writes that
# warning straight to sys.stderr rather than through asyncio's exception
# handler (confirmed: neither a loop exception handler nor a patched
# `call_exception_handler` ever sees it), so it can't be filtered any other
# way. It's cosmetic -- results are correct whenever it fires -- and
# internal to the `playwright` package; match on its exact, stable markers
# so nothing else written to stderr is ever affected.
_BENIGN_STDERR_MARKERS = (
    ("Task was destroyed but it is pending!", "Connection.run"),
    ("Future exception was never retrieved", "TargetClosedError"),
)
_shutdown_noise_filter_installed = False


class _StderrNoiseFilter:
    def __init__(self, target: Any) -> None:
        self._target = target

    def write(self, data: str) -> int:
        for markers in _BENIGN_STDERR_MARKERS:
            if all(marker in data for marker in markers):
                return len(data)
        return self._target.write(data)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def _ensure_shutdown_noise_filtered() -> None:
    global _shutdown_noise_filter_installed
    if _shutdown_noise_filter_installed:
        return
    sys.stderr = _StderrNoiseFilter(sys.stderr)
    _shutdown_noise_filter_installed = True


class PlaywrightRuntime:
    """Own one Playwright driver/browser and create isolated contexts on demand."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._config = dict(config or {})
        self._playwright: Any = None
        self._browser: Any = None
        self._owns_browser = True

    @property
    def started(self) -> bool:
        return self._browser is not None

    def storage_state_path(self) -> Path | None:
        raw = str(self._config.get("browser_storage_state_path") or "").strip()
        return Path(raw) if raw else None

    async def start(self) -> None:
        if self.started:
            return

        _ensure_shutdown_noise_filtered()
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        cdp_url = str(self._config.get("browser_cdp_url") or "").strip()
        if cdp_url:
            self._browser = await self._playwright.chromium.connect_over_cdp(cdp_url)
            self._owns_browser = False
        else:
            self._browser = await self._playwright.chromium.launch(headless=True)
            self._owns_browser = True

    async def new_context(self, *, apply_action_timeout: bool = False) -> Any:
        await self.start()
        if self._browser is None:
            raise RuntimeError("Playwright browser failed to start")

        options: dict[str, Any] = {"locale": "en-US"}
        state = self.storage_state_path()
        if state is not None and state.is_file():
            options["storage_state"] = str(state)

        context = await self._browser.new_context(**options)
        if apply_action_timeout:
            context.set_default_timeout(
                float(self._config.get("browser_action_timeout_seconds") or 10.0) * 1000
            )
        return context

    async def persist_storage_state(self, context: Any | None) -> None:
        """Persist cookies/local storage for an interactive context when configured."""
        state = self.storage_state_path()
        if state is None or context is None:
            return
        try:
            state.parent.mkdir(parents=True, exist_ok=True)
            await context.storage_state(path=str(state))
        except Exception:
            # Storage persistence is an optimization, never a reason to fail a
            # browser operation or graceful shutdown.
            pass

    async def close(self) -> None:
        browser, self._browser = self._browser, None
        playwright, self._playwright = self._playwright, None
        owns_browser, self._owns_browser = self._owns_browser, True

        if browser is not None and owns_browser:
            with suppress(Exception):
                await browser.close()
        if playwright is not None:
            with suppress(Exception):
                await playwright.stop()

"""Shared Playwright runtime and browser-context configuration."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any


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

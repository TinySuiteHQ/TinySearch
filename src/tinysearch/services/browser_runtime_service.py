"""Shared Playwright runtime and browser-context configuration."""

from __future__ import annotations

import hashlib
import os
import secrets
import sys
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from tinysearch.paths import private_file

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


_SHARED = object()


def storage_state_for(config: Mapping[str, Any], principal: str | None) -> Path | None:
    """The saved cookies and local storage a browser for ``principal`` may use.

    A server that does not tell callers apart keeps the one configured file,
    as before. One that does (a trusted identity is configured) never shares
    it: each caller gets their own file next to it, named by a salted hash,
    and a caller with no identity -- or a browser shared by every caller,
    such as the page fetcher's -- gets none. Otherwise one person's logins
    would be loaded into everyone else's browser.
    """
    raw = str(config.get("browser_storage_state_path") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    from tinysearch.services.management_service import LOCAL_PRINCIPAL, multi_tenant

    if not multi_tenant() or principal == LOCAL_PRINCIPAL:
        return path
    if principal is None:
        return None
    digest = hashlib.sha256(f"{_storage_salt(path)}:{principal}".encode()).hexdigest()[:32]
    return path.with_name(f"{path.stem}.{digest}{path.suffix or '.json'}")


def _storage_salt(path: Path) -> str:
    """A random salt kept beside the storage files, so their names reveal no one."""
    salt_path = path.with_name(f".{path.name}.salt")
    try:
        salt = salt_path.read_text(encoding="utf-8").strip()
        if salt:
            return salt
    except FileNotFoundError:
        pass
    salt_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    salt = secrets.token_hex(16)
    try:
        fd = os.open(salt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:  # another call created it first
        return salt_path.read_text(encoding="utf-8").strip()
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(salt)
    return salt


class PlaywrightRuntime:
    """Own one Playwright driver/browser and create isolated contexts on demand."""

    def __init__(self, config: Mapping[str, Any] | None = None, *, storage_state: Path | None | object = _SHARED) -> None:
        self._config = dict(config or {})
        # Unless told whose browser this is, it is shared: storage_state_for(None).
        self._storage_state = storage_state_for(self._config, None) if storage_state is _SHARED else storage_state
        self._playwright: Any = None
        self._browser: Any = None
        self._owns_browser = True
        self._next_proxy_index = 0

    @property
    def started(self) -> bool:
        return self._browser is not None

    def storage_state_path(self) -> Path | None:
        return self._storage_state  # type: ignore[return-value]

    def _next_proxy(self) -> dict[str, str] | None:
        """Pick the next configured proxy, round-robin, for rotation across contexts."""
        servers = [
            entry.strip()
            for entry in str(self._config.get("browser_proxy_server") or "").split(",")
            if entry.strip()
        ]
        if not servers:
            return None

        server = servers[self._next_proxy_index % len(servers)]
        self._next_proxy_index += 1

        proxy: dict[str, str] = {"server": server}
        username = str(self._config.get("browser_proxy_username") or "").strip()
        password = str(self._config.get("browser_proxy_password") or "").strip()
        bypass = str(self._config.get("browser_proxy_bypass") or "").strip()
        if username:
            proxy["username"] = username
        if password:
            proxy["password"] = password
        if bypass:
            proxy["bypass"] = bypass
        return proxy

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
        proxy = self._next_proxy()
        if proxy is not None:
            options["proxy"] = proxy

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
            # Cookies can be logins: owner-only before and after Playwright writes them.
            private_file(state)
            await context.storage_state(path=str(state))
            private_file(state)
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

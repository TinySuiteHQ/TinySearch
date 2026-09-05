"""Ensure Chromium is installed for crawl4ai/Playwright, on demand or at startup."""

from __future__ import annotations

import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


def _chromium_executable() -> Path | None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    with sync_playwright() as p:
        return Path(p.chromium.executable_path)


@lru_cache(maxsize=1)
def chromium_ready() -> bool:
    """Return whether the configured Playwright Chromium exists.

    Resolving ``chromium.executable_path`` starts the Playwright driver even
    though no browser is launched.  A server process only needs to pay that
    setup cost once; an installation below clears the cached negative result.
    """
    executable = _chromium_executable()
    return executable is not None and executable.exists()


def install_chromium(with_system_deps: bool = False) -> None:
    args = [sys.executable, "-m", "playwright", "install"]
    if with_system_deps:
        if platform.system() == "Linux":
            args.append("--with-deps")
        else:
            print(
                "[tinysearch] --with-system-deps has no effect on "
                f"{platform.system()}; installing Chromium only",
                file=sys.stderr,
                flush=True,
            )
    args.append("chromium")
    print(f"[tinysearch] running: {' '.join(args)}", file=sys.stderr, flush=True)
    subprocess.run(args, check=True, stdout=sys.stderr, stderr=sys.stderr)


def ensure_chromium_sync(with_system_deps: bool = False) -> None:
    """If Chromium is missing, install it once. Cheap no-op when already present."""
    if chromium_ready():
        return
    install_chromium(with_system_deps=with_system_deps)
    chromium_ready.cache_clear()

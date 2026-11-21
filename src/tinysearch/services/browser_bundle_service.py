"""Ensure Playwright Chromium is installed, on demand or at startup.

TinySearch only launches Chromium headless, and Playwright serves a headless
launch from its separate Chrome Headless Shell build, never from full Chromium.
So only the shell is installed (``--only-shell``) and only the shell is checked
for: full Chromium would be several hundred MB that nothing runs.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

# Playwright writes this into a browser's folder once its download is complete.
_INSTALL_MARKER = "INSTALLATION_COMPLETE"


def _chromium_executable() -> Path | None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    with sync_playwright() as p:
        return Path(p.chromium.executable_path)


def headless_shell_dir(chromium_executable: Path) -> Path | None:
    """Return the headless shell's install folder for this Playwright.

    Playwright reports only full Chromium's executable path, which exists only
    when full Chromium is installed. The two builds share a revision and sit
    side by side in the browsers folder (``chromium-1255`` beside
    ``chromium_headless_shell-1255``), so the shell's folder is derived from it.
    """
    for folder in chromium_executable.parents:
        if folder.name.startswith("chromium-"):
            revision = folder.name.removeprefix("chromium-")
            return folder.with_name(f"chromium_headless_shell-{revision}")
    return None


def headless_shell_installed(chromium_executable: Path) -> bool:
    shell = headless_shell_dir(chromium_executable)
    return shell is not None and (shell / _INSTALL_MARKER).is_file()


@lru_cache(maxsize=1)
def chromium_ready() -> bool:
    """Return whether the Chromium that headless launches use is installed.

    Resolving ``chromium.executable_path`` starts the Playwright driver even
    though no browser is launched.  A server process only needs to pay that
    setup cost once; an installation below clears the cached negative result.
    """
    executable = _chromium_executable()
    return executable is not None and headless_shell_installed(executable)


def install_chromium(with_system_deps: bool = False) -> None:
    args = [sys.executable, "-m", "playwright", "install", "--only-shell"]
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

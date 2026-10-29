"""Native (non-Docker, non-source-checkout) default storage locations.

These are only consulted when neither an explicit env var override
(`TINYSEARCH_CONFIG_PATH`, `TINYSEARCH_MODELS_DIR`) nor a repo-relative
`configs/`/`models/` directory is present, i.e. the genuinely-new case of a
`pip`/`uvx`-installed wheel with no surrounding checkout. Docker and
from-source dev keep their existing behavior unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs

APP_NAME = "tinysearch"


def native_config_path() -> Path:
    return Path(platformdirs.user_config_dir(APP_NAME)) / "tinysearch_config.json"


def native_models_dir() -> Path:
    return Path(platformdirs.user_data_dir(APP_NAME)) / "models"

def private_file(path: Path) -> None:
    """Make a data file owner-only before anything is written to it.

    The file is created 0600, and its folder 0700 when this creates it. A
    folder that already exists is left as it is: it may be one the operator
    chose and shares on purpose. The file, and SQLite's -wal and -shm files
    beside it (which SQLite gives the database's own mode), are tightened if
    an older version left them readable by others. Windows is left alone:
    the per-user AppData folder already belongs to the user alone.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    except FileExistsError:
        pass
    if os.name == "nt":
        return
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            if candidate.stat().st_mode & 0o077:
                os.chmod(candidate, 0o600)
        except (FileNotFoundError, PermissionError):
            pass

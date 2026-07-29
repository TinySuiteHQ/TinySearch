"""Native (non-Docker, non-source-checkout) default storage locations.

These are only consulted when neither an explicit env var override
(`TINYSEARCH_CONFIG_PATH`, `TINYSEARCH_MODELS_DIR`) nor a repo-relative
`configs/`/`models/` directory is present, i.e. the genuinely-new case of a
`pip`/`uvx`-installed wheel with no surrounding checkout. Docker and
from-source dev keep their existing behavior unchanged.
"""

from __future__ import annotations

from pathlib import Path

import platformdirs

APP_NAME = "tinysearch"


def native_config_path() -> Path:
    return Path(platformdirs.user_config_dir(APP_NAME)) / "tinysearch_config.json"


def native_models_dir() -> Path:
    return Path(platformdirs.user_data_dir(APP_NAME)) / "models"

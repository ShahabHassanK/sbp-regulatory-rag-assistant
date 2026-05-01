"""Centralised config loader — reads config/config.yaml once and returns a dict."""

from pathlib import Path
from functools import lru_cache
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).parents[2] / "config" / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load and cache config/config.yaml.

    Returns:
        Parsed YAML as a nested dict.

    Raises:
        FileNotFoundError: If config.yaml does not exist at the expected path.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

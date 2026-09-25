"""Load config/config.yaml and resolve project-relative paths."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for an entry under `paths:` in config.yaml."""
    return PROJECT_ROOT / load_config()["paths"][key]


def sql_path(key: str) -> Path:
    """Absolute path for an entry under `sql:` in config.yaml."""
    return PROJECT_ROOT / load_config()["sql"][key]

"""Project configuration loader.

All paths in config.yaml are relative to the project root, so notebooks,
scripts and the Streamlit app resolve the same files regardless of the
current working directory.
"""
from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Read config.yaml once and cache it."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for an entry of the `paths` section; creates dirs for folders."""
    p = PROJECT_ROOT / load_config()["paths"][key]
    if p.suffix == "":
        p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int | None = None) -> int:
    """Fix every random number generator we use, for reproducibility."""
    seed = load_config()["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed

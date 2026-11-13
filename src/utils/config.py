"""Configuration loading and project path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root (the directory that contains ``src``)."""
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping, got {type(config).__name__}")
    return config


def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (root or project_root()) / path

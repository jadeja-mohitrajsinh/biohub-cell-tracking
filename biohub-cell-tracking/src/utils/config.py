"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(config_name: str = "default", config_dir: str | Path | None = None) -> DictConfig:
    """Load a YAML config with optional defaults inheritance."""
    base_dir = Path(config_dir) if config_dir is not None else Path(__file__).resolve().parents[2] / "configs"
    config_path = base_dir / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = OmegaConf.load(config_path)
    defaults = config.pop("defaults", None)
    if defaults:
        merged: dict[str, Any] = {}
        for entry in defaults:
            if isinstance(entry, str):
                merged = OmegaConf.to_container(
                    OmegaConf.merge(OmegaConf.create(merged), load_config(entry, base_dir)),
                    resolve=True,
                )
        config = OmegaConf.merge(OmegaConf.create(merged), config)
    return OmegaConf.create(config)


def resolve_path(config: DictConfig, key: str, base_dir: str | Path | None = None) -> Path:
    """Resolve a config path relative to the project root."""
    project_root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[2]
    value = OmegaConf.select(config, key)
    if value is None:
        raise KeyError(f"Missing config key: {key}")
    path = Path(str(value))
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path

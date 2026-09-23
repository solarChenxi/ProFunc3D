"""Configuration loading and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    required = ("dataset_root", "work_dir", "proposal_bank", "fun3du_root")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"missing configuration keys: {', '.join(missing)}")
    # Public configs use repository-relative paths and remain valid even when
    # the CLI is launched outside the repository root.
    repository_root = path.parent.parent
    path_keys = (
        "fun3du_root", "dataset_root", "work_dir", "proposal_bank",
        "mask3d_processed", "grounding_observations", "contracts",
        "track_dir", "track_report", "pga_output", "prediction_dir",
        "frozen_track_dir", "parent_overrides", "mask3d_checkpoint",
    )
    for key in path_keys:
        value = config.get(key)
        if value:
            candidate = Path(value).expanduser()
            config[key] = str(candidate if candidate.is_absolute()
                              else (repository_root / candidate).resolve())
    config["config_path"] = str(path)
    config["repository_root"] = str(repository_root)
    return config


def path_value(config: dict[str, Any], key: str) -> Path:
    return Path(config[key]).expanduser().resolve()

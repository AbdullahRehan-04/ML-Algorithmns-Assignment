"""
config.py
---------
Centralized project configuration for dataset location and schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("config/dataset_config.json")

DEFAULTS: dict[str, Any] = {
    "dataset_path": "data/telecom_churn.csv",
    "target_column": "churn",
    "id_columns": ["customer_id"],
    "numeric_features": [],
    "categorical_features": [],
    "random_seed": 42,
}


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def load_project_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load config JSON and merge it with defaults.

    If the config file is missing, defaults are used so the project still runs.
    """
    merged = DEFAULTS.copy()
    path = Path(config_path)

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("dataset config must be a JSON object")
        merged.update(loaded)

    merged["dataset_path"] = str(merged["dataset_path"])
    merged["target_column"] = str(merged["target_column"])
    merged["id_columns"] = _ensure_list(merged.get("id_columns", []))
    merged["numeric_features"] = _ensure_list(merged.get("numeric_features", []))
    merged["categorical_features"] = _ensure_list(merged.get("categorical_features", []))
    merged["random_seed"] = int(merged.get("random_seed", 42))
    return merged
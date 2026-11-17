from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GEO_DIR = DATA_DIR / "geo"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MAPS_DIR = REPORTS_DIR / "maps"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


CONFIG = load_config()
RANDOM_STATE = CONFIG.get("random_state", 42)
WEIGHTS = CONFIG.get("weights", {})

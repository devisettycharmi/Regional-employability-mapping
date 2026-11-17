from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, PROCESSED_DIR


def processed_csv_path(filename: str) -> Path:
    configured_dir = os.environ.get("EMPLOYABILITY_PROCESSED_DATA_DIR")
    if configured_dir:
        path = Path(configured_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Processed data file not found: {path}")
        return path

    local_path = PROCESSED_DIR / filename
    if local_path.is_file():
        return local_path

    demo_path = DATA_DIR / "demo" / "processed" / filename
    if demo_path.is_file():
        return demo_path

    raise FileNotFoundError(
        f"Processed data file not found at {local_path} or {demo_path}"
    )


def read_processed_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(processed_csv_path(filename))

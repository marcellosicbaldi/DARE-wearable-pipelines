from __future__ import annotations

from pathlib import Path

import pandas as pd


def find_empatica_heart_rate_files(
    input_root: str | Path,
    participant: str,
    visit: str = "T0",
    sensor: str = "Empatica",
) -> dict[str, Path | None]:
    """Locate expected Empatica heart-rate files for one participant/visit."""
    base = Path(input_root) / str(participant) / str(visit) / str(sensor)
    candidates = {
        "ppg": base / "ppg.parquet",
        "acc": base / "acc.parquet",
        "bvp": base / "bvp.parquet",
        "ibi": base / "ibi.parquet",
        "hr": base / "hr.parquet",
    }
    return {key: path if path.exists() else None for key, path in candidates.items()}


def load_ppg(path: str | Path) -> pd.DataFrame:
    """Load Empatica PPG parquet data using the validated HRV notebook convention."""
    return pd.read_parquet(path).sort_index()


def load_acc(path: str | Path) -> pd.DataFrame:
    """Load Empatica accelerometer parquet data using the validated HRV notebook convention."""
    return pd.read_parquet(path).sort_index()


def load_bvp(path: str | Path) -> pd.DataFrame:
    """Load BVP data when present in a study export."""
    return pd.read_parquet(path).sort_index()


def load_ibi(path: str | Path) -> pd.DataFrame:
    """Load IBI data when present in a study export."""
    return pd.read_parquet(path).sort_index()


def load_heart_rate(path: str | Path) -> pd.DataFrame:
    """Load heart-rate data when present in a study export."""
    return pd.read_parquet(path).sort_index()

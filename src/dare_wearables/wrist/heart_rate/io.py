from __future__ import annotations

from pathlib import Path

import pandas as pd


def find_empatica_ppg_acc_files(
    input_root: str | Path,
    *,
    participant: str,
    visit: str,
    sensor: str = "Empatica",
) -> dict[str, Path | None]:
    base = Path(input_root) / str(participant) / str(visit) / str(sensor)
    candidates = {
        "ppg": base / "ppg.parquet",
        "acc": base / "acc.parquet",
    }
    return {key: path if path.exists() else None for key, path in candidates.items()}


def load_ppg(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path).sort_index()


def load_acc(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path).sort_index()

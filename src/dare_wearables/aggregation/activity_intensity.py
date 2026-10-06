from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_VISIT = "T0"
DEFAULT_ACTIVITY_FILENAME = "activity_intensity_summary.csv"

SELECTED_COLUMNS = {
    "n_days": "activity_n_valid_days",
    "dur_day_window_min_mean": "activity_day_window_min_mean",
    "dur_day_window_observed_min_mean": "activity_day_observed_min_mean",
    "dur_day_window_imputed_min_mean": "activity_day_imputed_min_mean",
    "dur_day_total_IN_min_mean": "activity_day_inactive_min_mean",
    "dur_day_total_LIG_min_mean": "activity_day_light_min_mean",
    "dur_day_total_MOD_min_mean": "activity_day_moderate_min_mean",
    "dur_day_total_VIG_min_mean": "activity_day_vigorous_min_mean",
    "dur_day_total_MVPA_min_mean": "activity_day_mvpa_min_mean",
}


def _discover_activity_summary_files(
    silver_root: Path,
    *,
    visit: str,
    activity_subdir: Path,
    activity_filename: str,
) -> list[tuple[str, Path]]:
    pattern = str(Path("*") / visit / activity_subdir / activity_filename)
    files = sorted(silver_root.glob(pattern))
    return [(path.parents[3].name, path) for path in files]


def aggregate_activity_intensity(
    silver_root: str | Path,
    *,
    visit: str = DEFAULT_VISIT,
    activity_subdir: str | Path,
    activity_filename: str = DEFAULT_ACTIVITY_FILENAME,
) -> pd.DataFrame:
    silver_root = Path(silver_root).expanduser()
    activity_subdir = Path(activity_subdir)

    rows: list[dict[str, object]] = []
    for subject, path in _discover_activity_summary_files(
        silver_root,
        visit=visit,
        activity_subdir=activity_subdir,
        activity_filename=activity_filename,
    ):
        summary_df = pd.read_csv(path)
        if summary_df.empty:
            continue

        first_row = summary_df.iloc[0]
        row: dict[str, object] = {"subject": str(subject), "visit": str(visit)}
        for source_column, output_column in SELECTED_COLUMNS.items():
            row[output_column] = first_row[source_column] if source_column in summary_df.columns else np.nan
        rows.append(row)

    columns = ["subject", "visit", *SELECTED_COLUMNS.values()]
    return pd.DataFrame(rows).reindex(columns=columns).sort_values("subject").reset_index(drop=True)


def write_activity_intensity_exports(
    silver_root: str | Path,
    *,
    visit: str = DEFAULT_VISIT,
    output_dir: str | Path | None = None,
    activity_subdir: str | Path,
    activity_filename: str = DEFAULT_ACTIVITY_FILENAME,
) -> tuple[Path, pd.DataFrame]:
    silver_root = Path(silver_root).expanduser()
    output_dir = Path(output_dir).expanduser() if output_dir is not None else silver_root / "aggregation"
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_df = aggregate_activity_intensity(
        silver_root,
        visit=visit,
        activity_subdir=activity_subdir,
        activity_filename=activity_filename,
    )
    output_path = output_dir / f"activity_intensity_{visit}.csv"
    subject_df.to_csv(output_path, index=False)
    return output_path, subject_df



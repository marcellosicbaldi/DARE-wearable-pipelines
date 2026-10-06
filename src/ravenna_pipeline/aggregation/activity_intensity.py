from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


from dare_wearables.aggregation import activity_intensity as _shared
from dare_wearables.aggregation.activity_intensity import (
    SELECTED_COLUMNS,
    _discover_activity_summary_files,
)

DEFAULT_SILVER_ROOT = Path("~/dare-data/ravenna/silver").expanduser()
DEFAULT_VISIT = "T0"
DEFAULT_ACTIVITY_SUBDIR = Path("GENEActiv") / "sleep_circadian"
DEFAULT_ACTIVITY_FILENAME = "activity_intensity_summary.csv"


def aggregate_activity_intensity(
    silver_root: str | Path = DEFAULT_SILVER_ROOT,
    *,
    visit: str = DEFAULT_VISIT,
    activity_subdir: str | Path = DEFAULT_ACTIVITY_SUBDIR,
    activity_filename: str = DEFAULT_ACTIVITY_FILENAME,
) -> pd.DataFrame:
    return _shared.aggregate_activity_intensity(
        silver_root=silver_root,
        visit=visit,
        activity_subdir=activity_subdir,
        activity_filename=activity_filename,
    )


def write_activity_intensity_exports(
    silver_root: str | Path = DEFAULT_SILVER_ROOT,
    *,
    visit: str = DEFAULT_VISIT,
    output_dir: str | Path | None = None,
    activity_subdir: str | Path = DEFAULT_ACTIVITY_SUBDIR,
    activity_filename: str = DEFAULT_ACTIVITY_FILENAME,
) -> tuple[Path, pd.DataFrame]:
    return _shared.write_activity_intensity_exports(
        silver_root=silver_root,
        visit=visit,
        output_dir=output_dir,
        activity_subdir=activity_subdir,
        activity_filename=activity_filename,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate activity-intensity silver-layer outputs.")
    parser.add_argument(
        "--silver-root",
        default=str(DEFAULT_SILVER_ROOT),
        help="Root folder containing SILVER/<subject>/<visit>/GENEActiv/sleep_circadian outputs.",
    )
    parser.add_argument(
        "--visit",
        default=DEFAULT_VISIT,
        help="Visit to aggregate. Defaults to T0.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder for aggregate CSV exports. Defaults to <silver-root>/aggregation.",
    )
    parser.add_argument(
        "--activity-subdir",
        default=str(DEFAULT_ACTIVITY_SUBDIR),
        help="Path under each subject/visit folder containing activity_intensity_summary.csv.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path, subject_df = write_activity_intensity_exports(
        args.silver_root,
        visit=args.visit,
        output_dir=args.output_dir,
        activity_subdir=args.activity_subdir,
    )
    print(f"Wrote activity-intensity aggregation: {output_path} ({len(subject_df)} subjects)")


if __name__ == "__main__":
    main()

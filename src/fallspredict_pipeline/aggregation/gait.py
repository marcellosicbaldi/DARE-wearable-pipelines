from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


from dare_wearables.aggregation import gait as _shared
from dare_wearables.aggregation.gait import (
    ADMIN_COLUMNS,
    EXPLICIT_AMOUNT_FEATURES,
    _discover_day_aggregation_files,
    _is_amount_feature,
    _numeric_feature_columns,
    _prefix_gait_columns,
)

DEFAULT_SILVER_ROOT = Path("~/dare-data/ravenna/silver").expanduser()
DEFAULT_VISIT = "T0"
DEFAULT_SENSOR = "McRoberts"
DEFAULT_GAIT_SUBDIR = "gait"
DEFAULT_MIN_VALID_HOURS = 16.0


def aggregate_gait(
    silver_root: str | Path = DEFAULT_SILVER_ROOT,
    *,
    visit: str = DEFAULT_VISIT,
    sensor: str = DEFAULT_SENSOR,
    gait_subdir: str = DEFAULT_GAIT_SUBDIR,
    min_valid_hours: float = DEFAULT_MIN_VALID_HOURS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _shared.aggregate_gait(
        silver_root=silver_root,
        visit=visit,
        sensor=sensor,
        gait_subdir=gait_subdir,
        min_valid_hours=min_valid_hours,
        allow_legacy_hours=False,
    )


def write_gait_exports(
    silver_root: str | Path = DEFAULT_SILVER_ROOT,
    *,
    visit: str = DEFAULT_VISIT,
    output_dir: str | Path | None = None,
    sensor: str = DEFAULT_SENSOR,
    gait_subdir: str = DEFAULT_GAIT_SUBDIR,
    min_valid_hours: float = DEFAULT_MIN_VALID_HOURS,
) -> tuple[Path, pd.DataFrame]:
    return _shared.write_gait_exports(
        silver_root=silver_root,
        visit=visit,
        output_dir=output_dir,
        sensor=sensor,
        gait_subdir=gait_subdir,
        min_valid_hours=min_valid_hours,
        allow_legacy_hours=False,
    )


def _load_gait_day_aggregation(
    silver_root: str | Path = DEFAULT_SILVER_ROOT,
    *,
    visit: str = DEFAULT_VISIT,
    sensor: str = DEFAULT_SENSOR,
    gait_subdir: str = DEFAULT_GAIT_SUBDIR,
) -> pd.DataFrame:
    return _shared._load_gait_day_aggregation(
        silver_root=silver_root,
        visit=visit,
        sensor=sensor,
        gait_subdir=gait_subdir,
        allow_legacy_hours=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate McRoberts gait silver-layer outputs.")
    parser.add_argument(
        "--silver-root",
        default=str(DEFAULT_SILVER_ROOT),
        help="Root folder containing silver/<subject>/<visit>/McRoberts/gait outputs.",
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
        "--sensor",
        default=DEFAULT_SENSOR,
        help="Sensor folder under each subject/visit. Defaults to McRoberts.",
    )
    parser.add_argument(
        "--min-valid-hours",
        type=float,
        default=DEFAULT_MIN_VALID_HOURS,
        help="Minimum valid hours required for amount-feature daily averages. Defaults to 16.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path, subject_df = write_gait_exports(
        args.silver_root,
        visit=args.visit,
        output_dir=args.output_dir,
        sensor=args.sensor,
        min_valid_hours=args.min_valid_hours,
    )
    print(f"Wrote gait aggregation: {output_path} ({len(subject_df)} subjects)")


if __name__ == "__main__":
    main()

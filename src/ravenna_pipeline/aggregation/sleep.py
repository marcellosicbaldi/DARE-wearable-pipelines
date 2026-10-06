from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


from dare_wearables.aggregation import sleep as _shared
from dare_wearables.aggregation.sleep import (
    AGGREGATION_METHODS,
    AggregationMethod,
    GUIDER_ORDER,
    GUIDER_COLUMN_PREFIX,
    GUIDER_ALIASES,
    TIME_OF_DAY_COLUMNS,
    DURATION_MIN_COLUMNS,
    NUMERIC_SLEEP_COLUMNS,
    CIRCADIAN_COLUMNS,
    _CLOCK_RE,
    _normalize_guider_source,
    _is_true,
    _aggregate_numeric,
    _aggregate_timedelta_minutes,
    _extract_clock_seconds,
    _aggregate_clock_hours,
    _parse_datetime_series,
    _aggregate_sol_minutes,
    _sleep_columns_for_guider,
    _output_columns,
    _read_csv_if_exists,
    _valid_sleep_rows,
    _add_empty_sleep_values,
    _add_sleep_values,
    _add_circadian_values,
    _discover_sleep_dirs,
    aggregate_subject_sleep,
)

DEFAULT_SILVER_ROOT = Path("~/dare-data/ravenna/silver").expanduser()
DEFAULT_VISIT = "T0"
DEFAULT_SLEEP_SUBDIR = Path("GENEActiv") / "sleep_circadian"


def aggregate_sleep(
    silver_root: str | Path = DEFAULT_SILVER_ROOT,
    *,
    visit: str = DEFAULT_VISIT,
    sleep_subdir: str | Path = DEFAULT_SLEEP_SUBDIR,
    method: AggregationMethod = "mean",
) -> pd.DataFrame:
    return _shared.aggregate_sleep(
        silver_root=silver_root,
        visit=visit,
        sleep_subdir=sleep_subdir,
        method=method,
    )


def write_sleep_exports(
    silver_root: str | Path = DEFAULT_SILVER_ROOT,
    *,
    visit: str = DEFAULT_VISIT,
    output_dir: str | Path | None = None,
    sleep_subdir: str | Path = DEFAULT_SLEEP_SUBDIR,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    return _shared.write_sleep_exports(
        silver_root=silver_root,
        visit=visit,
        output_dir=output_dir,
        sleep_subdir=sleep_subdir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Ravenna sleep and circadian silver-layer outputs.")
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
        "--sleep-subdir",
        default=str(DEFAULT_SLEEP_SUBDIR),
        help="Path under each subject/visit folder containing sleep/circadian outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mean_path, median_path, mean_df, median_df = write_sleep_exports(
        args.silver_root,
        visit=args.visit,
        output_dir=args.output_dir,
        sleep_subdir=args.sleep_subdir,
    )
    print(f"Wrote mean sleep aggregation: {mean_path} ({len(mean_df)} subjects)")
    print(f"Wrote median sleep aggregation: {median_path} ({len(median_df)} subjects)")


if __name__ == "__main__":
    main()

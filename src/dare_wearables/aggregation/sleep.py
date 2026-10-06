from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


DEFAULT_VISIT = "T0"

AGGREGATION_METHODS = ("mean", "median")
AggregationMethod = Literal["mean", "median"]

GUIDER_ORDER = ("sleep_diary", "HDCZA", "lower_back_tib")
GUIDER_COLUMN_PREFIX = {
    "sleep_diary": "sleep_diary",
    "HDCZA": "hdcza",
    "lower_back_tib": "lower_back",
}
GUIDER_ALIASES = {
    "sleep_diary": "sleep_diary",
    "sleep diary": "sleep_diary",
    "diary": "sleep_diary",
    "hdcza": "HDCZA",
    "lower_back_tib": "lower_back_tib",
    "lower back tib": "lower_back_tib",
    "lower_back": "lower_back_tib",
    "lower back": "lower_back_tib",
}

TIME_OF_DAY_COLUMNS = {
    "guider_start": "guider_start_clock_h",
    "guider_end": "guider_end_clock_h",
    "spt_start": "spt_start_clock_h",
    "spt_end": "spt_end_clock_h",
}
DURATION_MIN_COLUMNS = {
    "TST": "tst_min",
    "WASO": "waso_min",
    "longest_bout": "longest_sleep_bout_min",
    "max_wake_gap": "longest_wake_bout_min",
}
NUMERIC_SLEEP_COLUMNS = {
    "sleep_efficiency": "sleep_efficiency",
    "n_awakenings": "n_awakenings",
    "fragmentation_awakenings_per_hour_tst": "sleep_fragmentation_awakenings_per_hour_tst",
}

CIRCADIAN_COLUMNS = (
    "IS",
    "IV",
    "L5VALUE",
    "L5TIME_num",
    "L5TIME_clock",
    "M5VALUE",
    "M5TIME_num",
    "M5TIME_clock",
    "cosinor_timeOffsetHours",
    "cosinor_mes",
    "cosinor_amp",
    "cosinor_acrophase",
    "cosinor_acrotime",
    "cosinor_days",
    "MESOR_log1p_mg",
    "Amplitude_log1p_mg",
    "Phase_rad_series_start",
    "Acrotime_hour",
    "Cosinor_n_days_used",
    "Cosinor_n_valid_minutes",
)

_CLOCK_RE = re.compile(
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2})(?:\.(?P<fraction>\d+))?)?"
)


def _normalize_guider_source(value: object) -> str | None:
    key = str(value).strip().lower()
    return GUIDER_ALIASES.get(key)


def _is_true(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _aggregate_numeric(series: pd.Series, method: AggregationMethod) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    if method == "mean":
        return float(values.mean())
    if method == "median":
        return float(values.median())
    raise ValueError(f"Unsupported aggregation method: {method!r}")


def _aggregate_timedelta_minutes(series: pd.Series, method: AggregationMethod) -> float:
    values = pd.to_timedelta(series, errors="coerce").dt.total_seconds().dropna() / 60.0
    if values.empty:
        return np.nan
    if method == "mean":
        return float(values.mean())
    if method == "median":
        return float(values.median())
    raise ValueError(f"Unsupported aggregation method: {method!r}")


def _extract_clock_seconds(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    parts = text.str.extract(_CLOCK_RE)
    hour = pd.to_numeric(parts["hour"], errors="coerce")
    minute = pd.to_numeric(parts["minute"], errors="coerce")
    second = pd.to_numeric(parts["second"], errors="coerce").fillna(0)
    fraction = parts["fraction"].fillna("")
    fractional_seconds = pd.to_numeric("0." + fraction, errors="coerce").fillna(0)

    valid = hour.between(0, 23) & minute.between(0, 59) & second.between(0, 59)
    seconds = hour * 60 * 60 + minute * 60 + second + fractional_seconds
    return seconds.where(valid)


def _aggregate_clock_hours(
    series: pd.Series,
    method: AggregationMethod,
    *,
    anchor_hour: float = 12.0,
) -> float:
    seconds_in_day = 24 * 60 * 60
    anchor_seconds = anchor_hour * 60 * 60
    seconds = _extract_clock_seconds(series).dropna()
    if seconds.empty:
        return np.nan

    shifted = (seconds - anchor_seconds) % seconds_in_day
    if method == "mean":
        aggregated = float(shifted.mean())
    elif method == "median":
        aggregated = float(shifted.median())
    else:
        raise ValueError(f"Unsupported aggregation method: {method!r}")

    return ((aggregated + anchor_seconds) % seconds_in_day) / 3600.0


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        missing = parsed.isna() & text.notna()
        if not missing.any():
            break
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], format=fmt, errors="coerce")
    return parsed


def _aggregate_sol_minutes(sleep_rows: pd.DataFrame, method: AggregationMethod) -> float:
    spt_start = _parse_datetime_series(sleep_rows["spt_start"])
    guider_start = _parse_datetime_series(sleep_rows["guider_start"])
    sol_min = (spt_start - guider_start).dt.total_seconds() / 60.0
    return _aggregate_numeric(sol_min, method)


def _sleep_columns_for_guider(guider_source: str) -> list[str]:
    prefix = GUIDER_COLUMN_PREFIX[guider_source]
    columns = [f"{prefix}_n_valid_nights"]
    columns.extend(f"{prefix}_{name}" for name in TIME_OF_DAY_COLUMNS.values())
    columns.extend(f"{prefix}_{name}" for name in DURATION_MIN_COLUMNS.values())
    columns.extend(f"{prefix}_{name}" for name in NUMERIC_SLEEP_COLUMNS.values())
    if guider_source == "lower_back_tib":
        columns.append(f"{prefix}_sol_min")
    return columns


def _output_columns() -> list[str]:
    columns = ["subject", "visit"]
    for guider_source in GUIDER_ORDER:
        columns.extend(_sleep_columns_for_guider(guider_source))
    columns.extend(f"circadian_{column}" for column in CIRCADIAN_COLUMNS)
    return columns


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _valid_sleep_rows(sleep_df: pd.DataFrame) -> pd.DataFrame:
    if sleep_df.empty or "guider_source" not in sleep_df.columns:
        return pd.DataFrame()

    out = sleep_df.copy()
    out["_guider_source"] = out["guider_source"].map(_normalize_guider_source)
    out = out.loc[out["_guider_source"].notna()].copy()
    if "spt_found" in out.columns:
        out = out.loc[_is_true(out["spt_found"])].copy()
    return out


def _add_empty_sleep_values(row: dict[str, object], guider_source: str) -> None:
    prefix = GUIDER_COLUMN_PREFIX[guider_source]
    row[f"{prefix}_n_valid_nights"] = 0
    for output_name in TIME_OF_DAY_COLUMNS.values():
        row[f"{prefix}_{output_name}"] = np.nan
    for output_name in DURATION_MIN_COLUMNS.values():
        row[f"{prefix}_{output_name}"] = np.nan
    for output_name in NUMERIC_SLEEP_COLUMNS.values():
        row[f"{prefix}_{output_name}"] = np.nan
    if guider_source == "lower_back_tib":
        row[f"{prefix}_sol_min"] = np.nan


def _add_sleep_values(
    row: dict[str, object],
    sleep_df: pd.DataFrame,
    *,
    guider_source: str,
    method: AggregationMethod,
) -> None:
    prefix = GUIDER_COLUMN_PREFIX[guider_source]
    source_rows = sleep_df.loc[sleep_df["_guider_source"] == guider_source].copy()
    row[f"{prefix}_n_valid_nights"] = int(len(source_rows))

    if source_rows.empty:
        _add_empty_sleep_values(row, guider_source)
        return

    for source_column, output_name in TIME_OF_DAY_COLUMNS.items():
        row[f"{prefix}_{output_name}"] = (
            _aggregate_clock_hours(source_rows[source_column], method)
            if source_column in source_rows.columns
            else np.nan
        )
    for source_column, output_name in DURATION_MIN_COLUMNS.items():
        row[f"{prefix}_{output_name}"] = (
            _aggregate_timedelta_minutes(source_rows[source_column], method)
            if source_column in source_rows.columns
            else np.nan
        )
    for source_column, output_name in NUMERIC_SLEEP_COLUMNS.items():
        row[f"{prefix}_{output_name}"] = (
            _aggregate_numeric(source_rows[source_column], method)
            if source_column in source_rows.columns
            else np.nan
        )
    if guider_source == "lower_back_tib":
        row[f"{prefix}_sol_min"] = _aggregate_sol_minutes(source_rows, method)


def _add_circadian_values(row: dict[str, object], circadian_df: pd.DataFrame) -> None:
    if circadian_df.empty:
        for column in CIRCADIAN_COLUMNS:
            row[f"circadian_{column}"] = np.nan
        return

    first_row = circadian_df.iloc[0]
    for column in CIRCADIAN_COLUMNS:
        row[f"circadian_{column}"] = first_row[column] if column in circadian_df.columns else np.nan


def _discover_sleep_dirs(
    silver_root: Path,
    *,
    visit: str,
    sleep_subdir: Path,
) -> list[tuple[str, Path]]:
    pattern = str(Path("*") / visit / sleep_subdir / "sleep_output_all_guiders.csv")
    sleep_files = sorted(silver_root.glob(pattern))
    return [(path.parents[3].name, path.parent) for path in sleep_files]


def aggregate_subject_sleep(
    subject: str,
    visit: str,
    sleep_dir: Path,
    *,
    method: AggregationMethod,
) -> dict[str, object]:
    row: dict[str, object] = {"subject": str(subject), "visit": str(visit)}
    sleep_df = _valid_sleep_rows(_read_csv_if_exists(sleep_dir / "sleep_output_all_guiders.csv"))
    circadian_df = _read_csv_if_exists(sleep_dir / "circadian_metrics.csv")

    for guider_source in GUIDER_ORDER:
        if sleep_df.empty:
            _add_empty_sleep_values(row, guider_source)
        else:
            _add_sleep_values(row, sleep_df, guider_source=guider_source, method=method)
    _add_circadian_values(row, circadian_df)
    return row


def aggregate_sleep(
    silver_root: str | Path,
    *,
    visit: str = DEFAULT_VISIT,
    sleep_subdir: str | Path,
    method: AggregationMethod = "mean",
) -> pd.DataFrame:
    silver_root = Path(silver_root).expanduser()
    sleep_subdir = Path(sleep_subdir)
    if method not in AGGREGATION_METHODS:
        raise ValueError(f"method must be one of {AGGREGATION_METHODS}; got {method!r}")

    rows = [
        aggregate_subject_sleep(subject, visit, sleep_dir, method=method)
        for subject, sleep_dir in _discover_sleep_dirs(
            silver_root,
            visit=visit,
            sleep_subdir=sleep_subdir,
        )
    ]
    return pd.DataFrame(rows).reindex(columns=_output_columns()).sort_values("subject").reset_index(drop=True)


def write_sleep_exports(
    silver_root: str | Path,
    *,
    visit: str = DEFAULT_VISIT,
    output_dir: str | Path | None = None,
    sleep_subdir: str | Path,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    silver_root = Path(silver_root).expanduser()
    output_dir = Path(output_dir).expanduser() if output_dir is not None else silver_root / "aggregation"
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_df = aggregate_sleep(silver_root, visit=visit, sleep_subdir=sleep_subdir, method="mean")
    median_df = aggregate_sleep(silver_root, visit=visit, sleep_subdir=sleep_subdir, method="median")

    mean_path = output_dir / f"sleep_{visit}_mean.csv"
    median_path = output_dir / f"sleep_{visit}_median.csv"
    mean_df.to_csv(mean_path, index=False)
    median_df.to_csv(median_path, index=False)
    return mean_path, median_path, mean_df, median_df



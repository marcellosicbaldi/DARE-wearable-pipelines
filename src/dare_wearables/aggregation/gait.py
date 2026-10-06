from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_VISIT = "T0"
DEFAULT_SENSOR = "McRoberts"
DEFAULT_GAIT_SUBDIR = "gait"
DEFAULT_MIN_VALID_HOURS = 16.0

ADMIN_COLUMNS = {
    "unprocessed_wear_seconds", "processing_status", "date",
    "ID",
    "day",
    "hours",
    "nonwear_time_minutes",
    "nonwear_time_percent",
    "subject",
    "visit",
    "day_count",
    "valid_hours",
    "valid_hours_source",
    "valid_day",
}
EXPLICIT_AMOUNT_FEATURES = {
    "total_walking_duration_min",
    "step_count",
}


def _discover_day_aggregation_files(
    silver_root: Path,
    *,
    visit: str,
    sensor: str,
    gait_subdir: str,
) -> list[tuple[str, Path]]:
    pattern = str(Path("*") / visit / sensor / gait_subdir / "*_day_aggregation.csv")
    files = sorted(silver_root.glob(pattern))
    return [(path.parents[3].name, path) for path in files]


def _is_amount_feature(column: str) -> bool:
    if column in EXPLICIT_AMOUNT_FEATURES:
        return True
    if not column.startswith("wb_"):
        return False
    return column.endswith("__count") or column.endswith("__sum")


def _prepare_daily_qc(df: pd.DataFrame, *, path: Path, allow_legacy_hours: bool = False) -> pd.DataFrame:
    """Validate daily coverage, with explicit opt-in for wear-filtered legacy hours."""
    out = df.copy()
    if "nonwear_time_minutes" not in out and not allow_legacy_hours:
        raise ValueError(f"{path}: missing gait QC column 'nonwear_time_minutes'.")
    if "hours" not in out:
        raise ValueError(f"{path}: missing gait QC column 'hours'.")
    out["hours"] = pd.to_numeric(out["hours"], errors="coerce")
    if not np.isfinite(out["hours"]).all():
        raise ValueError(f"{path}: missing or nonnumeric gait 'hours'; cannot determine valid days.")
    if "nonwear_time_minutes" in out:
        out["nonwear_time_minutes"] = pd.to_numeric(out["nonwear_time_minutes"], errors="coerce")
        if not np.isfinite(out["nonwear_time_minutes"]).all():
            raise ValueError(f"{path}: missing or nonnumeric 'nonwear_time_minutes'; cannot determine valid days.")
        out["valid_hours"] = out["hours"] - out["nonwear_time_minutes"] / 60.0
        out["valid_hours_source"] = "recorded_hours_minus_nonwear"
    else:
        if "nonwear_time_percent" in out:
            raise ValueError(f"{path}: nonwear_time_percent is present but nonwear_time_minutes is missing.")
        out["valid_hours"] = out["hours"]
        out["valid_hours_source"] = "legacy_wear_filtered_hours"
        # Preserve unknown nonwear duration rather than inventing a zero.
        out["nonwear_time_minutes"] = np.nan
    return out


def _load_gait_day_aggregation(
    silver_root: str | Path,
    *,
    visit: str = DEFAULT_VISIT,
    sensor: str = DEFAULT_SENSOR,
    gait_subdir: str = DEFAULT_GAIT_SUBDIR,
    allow_legacy_hours: bool = False,
) -> pd.DataFrame:
    silver_root = Path(silver_root).expanduser()
    frames: list[pd.DataFrame] = []
    for subject, path in _discover_day_aggregation_files(
        silver_root,
        visit=visit,
        sensor=sensor,
        gait_subdir=gait_subdir,
    ):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = _prepare_daily_qc(df, path=path, allow_legacy_hours=allow_legacy_hours)
        if not allow_legacy_hours:
            df = df.drop(columns=["valid_hours_source", "valid_hours"])
        df["subject"] = str(subject)
        df["visit"] = str(visit)
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _numeric_feature_columns(gait: pd.DataFrame) -> list[str]:
    numeric_columns = gait.select_dtypes(include="number").columns.tolist()
    return [column for column in numeric_columns if column not in ADMIN_COLUMNS]


def _prefix_gait_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        column: f"gait_{column}"
        for column in df.columns
        if column not in {"subject", "visit"} and not column.startswith("gait_")
    }
    return df.rename(columns=rename_map)


def aggregate_gait(
    silver_root: str | Path,
    *,
    visit: str = DEFAULT_VISIT,
    sensor: str = DEFAULT_SENSOR,
    gait_subdir: str = DEFAULT_GAIT_SUBDIR,
    min_valid_hours: float = DEFAULT_MIN_VALID_HOURS,
    allow_legacy_hours: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gait = _load_gait_day_aggregation(
        silver_root,
        visit=visit,
        sensor=sensor,
        gait_subdir=gait_subdir,
        allow_legacy_hours=allow_legacy_hours,
    )
    if gait.empty:
        return pd.DataFrame(columns=["subject", "visit"]), gait

    if not allow_legacy_hours:
        gait["valid_hours"] = gait["hours"] - gait["nonwear_time_minutes"] / 60.0

    # valid_hours is prepared per input file so mixed legacy/current schemas
    # cannot turn absent nonwear fields into NaN > threshold -> False.
    gait["valid_day"] = gait["valid_hours"] > float(min_valid_hours)
    if "processing_status" in gait:
        gait["valid_day"] &= gait["processing_status"].eq("completed")
    gait["day_count"] = gait.groupby(["subject", "visit"])["day"].transform("nunique")

    feature_columns = _numeric_feature_columns(gait)
    amount_features = [column for column in feature_columns if _is_amount_feature(column)]
    quality_features = [column for column in feature_columns if column not in amount_features]

    valid_gait = gait.loc[gait["valid_day"]].copy()
    group_cols = ["subject", "visit"]

    if amount_features:
        daily_amount = valid_gait.groupby(group_cols, sort=True)[amount_features].mean().reset_index()
    else:
        daily_amount = gait[group_cols].drop_duplicates().sort_values(group_cols).reset_index(drop=True)

    if quality_features:
        quality_avg = gait.groupby(group_cols, sort=True)[quality_features].mean().reset_index()
    else:
        quality_avg = gait[group_cols].drop_duplicates().sort_values(group_cols).reset_index(drop=True)

    subject_level = daily_amount.merge(quality_avg, on=group_cols, how="outer")

    day_qc = (
        gait.groupby(group_cols, sort=True)
        .agg(
            gait_n_days=("day", "nunique"),
            gait_n_valid_days=("valid_day", "sum"),
            gait_mean_valid_hours=("valid_hours", "mean"),
        )
        .reset_index()
    )
    subject_level = day_qc.merge(subject_level, on=group_cols, how="left")
    subject_level = subject_level.sort_values(group_cols).reset_index(drop=True)
    return _prefix_gait_columns(subject_level), gait


def write_gait_exports(
    silver_root: str | Path,
    *,
    visit: str = DEFAULT_VISIT,
    output_dir: str | Path | None = None,
    sensor: str = DEFAULT_SENSOR,
    gait_subdir: str = DEFAULT_GAIT_SUBDIR,
    min_valid_hours: float = DEFAULT_MIN_VALID_HOURS,
    allow_legacy_hours: bool = False,
) -> tuple[Path, pd.DataFrame]:
    silver_root = Path(silver_root).expanduser()
    output_dir = Path(output_dir).expanduser() if output_dir is not None else silver_root / "aggregation"
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_df, _ = aggregate_gait(
        silver_root,
        visit=visit,
        sensor=sensor,
        gait_subdir=gait_subdir,
        min_valid_hours=min_valid_hours,
        allow_legacy_hours=allow_legacy_hours,
    )
    output_path = output_dir / f"gait_{visit}_mean.csv"
    subject_df.to_csv(output_path, index=False)
    return output_path, subject_df



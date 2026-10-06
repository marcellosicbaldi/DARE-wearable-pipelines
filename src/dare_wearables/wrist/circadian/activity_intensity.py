from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from dare_wearables.wrist.circadian.metrics import (
    detect_nonwear_intervals,
    load_and_calibrate_accelerometer,
    load_ggir_part5_intensity,
)


INTENSITY_LEVELS = ("IN", "LIG", "MOD", "VIG")
LEGACY_SPT_START_COL = "spt_start_sleeplog"
LEGACY_SPT_END_COL = "spt_end_sleeplog"
DIARY_BED_COL = "Datetime_bed"
DIARY_WAKE_COL = "Datetime_wake"


def _normalize_subject_token(value: object) -> str:
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not extract a numeric subject identifier from {value!r}.")
    return str(int(digits))


def _coerce_to_timezone(series: pd.Series, timezone: str) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        tz = ZoneInfo(timezone)
        try:
            return ts.dt.tz_localize(
                tz,
                ambiguous="infer",
                nonexistent=pd.Timedelta(hours=1),
            )
        except Exception as exc:
            if exc.__class__.__name__ != "AmbiguousTimeError":
                raise
            return ts.dt.tz_localize(
                tz,
                ambiguous=False,
                nonexistent=pd.Timedelta(hours=1),
            )
    return ts.dt.tz_convert(ZoneInfo(timezone))


def _resolve_subject_directory(raw_root: str | Path, subject: str) -> Path:
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw accelerometer root not found: {raw_root}")

    exact = raw_root / str(subject)
    if exact.is_dir():
        return exact

    subject_norm = _normalize_subject_token(subject)
    matches = [
        path for path in raw_root.iterdir()
        if path.is_dir() and path.name.isdigit() and _normalize_subject_token(path.name) == subject_norm
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not find a raw-data directory for subject {subject!r} under {raw_root}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple raw-data directories for subject {subject!r}: "
            f"{[path.name for path in matches]}"
        )
    return matches[0]


def _resolve_visit_directory(subject_dir: str | Path, visit: str = "T0") -> Path:
    subject_dir = Path(subject_dir)
    candidates = sorted(
        path for path in subject_dir.iterdir()
        if path.is_dir() and path.name.startswith(visit)
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a visit folder starting with {visit!r} in {subject_dir}."
        )
    if len(candidates) > 1:
        exact = [path for path in candidates if path.name == visit]
        if len(exact) == 1:
            return exact[0]
    return candidates[0]


def find_wrist_accelerometer_file(raw_root: str | Path,
                                  subject: str,
                                  visit: str = "T0",
                                  sensor_folder: str = "GeneActivPolso") -> Path:
    """
    Locate the non-dominant wrist GENEActiv `.bin` file for one subject/visit.
    """
    subject_dir = _resolve_subject_directory(raw_root, subject)
    visit_dir = _resolve_visit_directory(subject_dir, visit=visit)
    wrist_dir = visit_dir / sensor_folder
    if not wrist_dir.exists():
        raise FileNotFoundError(f"Wrist sensor folder not found: {wrist_dir}")

    bin_files = sorted(
        path for path in wrist_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".bin"
    )
    if not bin_files:
        raise FileNotFoundError(f"No `.bin` files found in {wrist_dir}")
    if len(bin_files) > 1:
        raise ValueError(
            f"Found multiple wrist accelerometer files in {wrist_dir}: "
            f"{[path.name for path in bin_files]}"
        )
    return bin_files[0]


def load_sleep_windows(sleep_csv_path: str | Path,
                       subject: str,
                       visit: str = "T0",
                       timezone: str = "Europe/Rome",
                       sleep_subject: Optional[str] = None) -> pd.DataFrame:
    """
    Load sleep-period time windows derived from the external sleep pipeline.

    Accepted input formats
    ----------------------
    1. legacy sleep-pipeline export with:
       `subject`, `visit`, `night_id`, `spt_start_sleeplog`, `spt_end_sleeplog`
    2. diary export with:
       `id`, `Datetime_bed`, `Datetime_wake`, and optionally `night`

    Matching logic
    --------------
    - exact match on the detected subject column if present
    - otherwise numeric-normalized match, so `782` and `00782` are treated as
      the same subject ID

    Returned timestamps are localized to `timezone`, because the accelerometer
    stream is converted from UTC to local clock time before window matching.
    """
    sleep_csv_path = Path(sleep_csv_path)
    if not sleep_csv_path.exists():
        raise FileNotFoundError(f"Sleep aggregation file not found: {sleep_csv_path}")

    sleep_df = pd.read_csv(sleep_csv_path)
    columns = set(sleep_df.columns)

    if {LEGACY_SPT_START_COL, LEGACY_SPT_END_COL, "subject"}.issubset(columns):
        subject_col = "subject"
        visit_col = "visit" if "visit" in columns else None
        night_col = "night_id" if "night_id" in columns else None
        start_col = LEGACY_SPT_START_COL
        end_col = LEGACY_SPT_END_COL
    elif {DIARY_BED_COL, DIARY_WAKE_COL, "id"}.issubset(columns):
        subject_col = "id"
        visit_col = "visit" if "visit" in columns else None
        night_col = "night" if "night" in columns else None
        start_col = DIARY_BED_COL
        end_col = DIARY_WAKE_COL
    else:
        raise ValueError(
            f"Unrecognized sleep CSV format for {sleep_csv_path}. "
            "Expected either legacy SPT columns or diary columns "
            f"({DIARY_BED_COL}, {DIARY_WAKE_COL})."
        )

    sleep_df[subject_col] = sleep_df[subject_col].astype(str).str.strip()
    if visit_col is not None:
        sleep_df[visit_col] = sleep_df[visit_col].astype(str).str.strip()

    target_subject = str(sleep_subject if sleep_subject is not None else subject).strip()
    exact = sleep_df.loc[sleep_df[subject_col] == target_subject].copy()
    if visit_col is not None:
        exact = exact.loc[exact[visit_col] == visit].copy()

    if exact.empty:
        target_norm = _normalize_subject_token(target_subject)
        candidates = sleep_df.copy()
        if visit_col is not None:
            candidates = candidates.loc[candidates[visit_col] == visit].copy()
        candidates["subject_norm"] = candidates[subject_col].map(_normalize_subject_token)
        exact = candidates.loc[candidates["subject_norm"] == target_norm].copy()

    if exact.empty:
        raise FileNotFoundError(
            f"No sleep windows found in {sleep_csv_path} for subject {target_subject!r} and visit {visit!r}."
        )

    exact["spt_start"] = _coerce_to_timezone(exact[start_col], timezone)
    exact["spt_end"] = _coerce_to_timezone(exact[end_col], timezone)
    exact = exact.loc[exact["spt_start"].notna() & exact["spt_end"].notna()].copy()
    exact = exact.loc[exact["spt_end"] > exact["spt_start"]].copy()
    exact = exact.sort_values("spt_start").reset_index(drop=True)

    if exact.empty:
        raise ValueError(
            f"Sleep windows were found for subject {target_subject!r}, but none had valid SPT bounds."
        )

    out = pd.DataFrame({
        "subject": exact[subject_col].astype(str),
        "visit": (exact[visit_col].astype(str) if visit_col is not None else visit),
        "night_id": (exact[night_col] if night_col is not None else np.arange(len(exact))),
        "spt_start": exact["spt_start"],
        "spt_end": exact["spt_end"],
    })
    return out.reset_index(drop=True)


def build_epoch_table(calibrated_df,
                      nonwear_df: pd.DataFrame,
                      epoch_seconds: int = 5,
                      dayborder_hours: float = 0.0,
                      timezone: str = "Europe/Rome") -> pd.DataFrame:
    """
    Convert calibrated tri-axial acceleration to a regular ENMO epoch table.

    Output columns
    --------------
    enmo_mg       : mean ENMO in mg on the epoch grid
    invalid       : True for epochs with missing samples or overlapping non-wear
    analysis_date : day label after applying dayborder
    epoch_of_day  : epoch position within the analysis day
    clock_hour    : local clock hour
    """
    if epoch_seconds <= 0 or 86400 % int(epoch_seconds) != 0:
        raise ValueError("epoch_seconds must be a positive divisor of 86400.")

    freq = f"{int(epoch_seconds)}s"
    epoch_delta = pd.Timedelta(seconds=int(epoch_seconds))

    df = calibrated_df.select(["time", "x_cal", "y_cal", "z_cal"]).to_pandas()
    df["time"] = _coerce_to_timezone(df["time"], timezone)
    df = df.sort_values("time")

    mag_g = np.sqrt(df["x_cal"] ** 2 + df["y_cal"] ** 2 + df["z_cal"] ** 2)
    df["enmo_mg"] = np.maximum((mag_g - 1.0) * 1000.0, 0.0)

    epoch_series = (
        df.set_index("time")["enmo_mg"]
        .resample(freq)
        .mean()
    )
    full_index = pd.date_range(
        start=epoch_series.index.min().floor(freq),
        end=epoch_series.index.max().floor(freq),
        freq=freq,
        tz=epoch_series.index.tz,
    )

    epoch_df = pd.DataFrame(index=full_index)
    epoch_df.index.name = "time"
    epoch_df["enmo_mg"] = epoch_series.reindex(full_index)
    epoch_df["invalid"] = epoch_df["enmo_mg"].isna()

    if nonwear_df is not None and not nonwear_df.empty:
        nonwear = nonwear_df.copy()
        nonwear["start"] = _coerce_to_timezone(nonwear["start"], timezone)
        nonwear["end"] = _coerce_to_timezone(nonwear["end"], timezone)
        epoch_start = epoch_df.index
        epoch_end = epoch_df.index + epoch_delta
        for row in nonwear.itertuples(index=False):
            mask = (epoch_start < row.end) & (epoch_end > row.start)
            epoch_df.loc[mask, "invalid"] = True

    shifted = epoch_df.index - pd.to_timedelta(dayborder_hours, unit="h")
    seconds_of_day = shifted.hour * 3600 + shifted.minute * 60 + shifted.second
    epoch_df["analysis_date"] = shifted.date
    epoch_df["epoch_of_day"] = (seconds_of_day // int(epoch_seconds)).astype(int)
    epoch_df["clock_hour"] = shifted.hour + shifted.minute / 60.0 + shifted.second / 3600.0

    return epoch_df


def summarize_epoch_days(epoch_df: pd.DataFrame,
                         epoch_seconds: int = 5,
                         min_wear_hours: float = 18.0,
                         require_full_24h: bool = False,
                         max_nonwear_hours: Optional[float] = None) -> pd.DataFrame:
    """
    Summarize each analysis day on the epoch grid and flag valid days.
    """
    if max_nonwear_hours is not None:
        min_wear_hours = 24.0 - float(max_nonwear_hours)

    epochs_per_day = 86400 // int(epoch_seconds)
    day_summary = (
        epoch_df.groupby("analysis_date")
        .agg(
            total_epochs=("enmo_mg", "size"),
            invalid_epochs=("invalid", "sum"),
        )
        .reset_index()
    )

    day_summary["valid_epochs"] = day_summary["total_epochs"] - day_summary["invalid_epochs"]
    day_summary["invalid_hours"] = day_summary["invalid_epochs"] * float(epoch_seconds) / 3600.0
    day_summary["valid_hours"] = day_summary["valid_epochs"] * float(epoch_seconds) / 3600.0
    day_summary["is_full_24h"] = day_summary["total_epochs"] == epochs_per_day
    day_summary["is_valid_day"] = day_summary["valid_hours"] >= float(min_wear_hours)

    if require_full_24h:
        day_summary["is_valid_day"] &= day_summary["is_full_24h"]

    return day_summary


def select_valid_epoch_days(epoch_df: pd.DataFrame,
                            day_summary: pd.DataFrame,
                            max_days: Optional[int] = 7,
                            min_valid_days: Optional[int] = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep valid days only, optionally limiting to the earliest `max_days`.
    """
    valid_days = day_summary.loc[day_summary["is_valid_day"]].copy()
    valid_days = valid_days.sort_values("analysis_date").reset_index(drop=True)

    if max_days is not None:
        valid_days = valid_days.iloc[:max_days].copy()

    if min_valid_days is not None and len(valid_days) < min_valid_days:
        raise ValueError(
            f"Only {len(valid_days)} valid days found, but min_valid_days={min_valid_days}."
        )

    keep_dates = set(valid_days["analysis_date"])
    epoch_selected = epoch_df.loc[epoch_df["analysis_date"].isin(keep_dates)].copy()
    epoch_selected = epoch_selected.sort_index()

    return epoch_selected, valid_days


def impute_invalid_epochs_by_clocktime(epoch_df: pd.DataFrame,
                                       value_col: str = "enmo_mg",
                                       fallback_value: float = 0.0) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Impute invalid epochs using the mean ENMO at the same clock-time position
    across the other retained valid days.

    This mirrors the custom non-wear rule you described:
    - keep only days with at least 18 hours of wear
    - keep only recordings with at least 3 valid days
    - replace invalid epochs on those retained days with the average ENMO from
      the same epoch-of-day across the other retained days
    """
    out = epoch_df.copy()
    out["was_imputed"] = out["invalid"].astype(bool)
    out[f"{value_col}_imputed"] = out[value_col]

    valid_means = (
        out.loc[~out["invalid"]]
        .groupby("epoch_of_day")[value_col]
        .mean()
    )

    mask = out["invalid"]
    out.loc[mask, f"{value_col}_imputed"] = out.loc[mask, "epoch_of_day"].map(valid_means)
    out[f"{value_col}_imputed"] = out[f"{value_col}_imputed"].fillna(float(fallback_value))

    imputed_epochs_per_day = (
        out.groupby("analysis_date")["was_imputed"]
        .sum()
        .astype(int)
    )
    return out, imputed_epochs_per_day


def annotate_sleep_period_window(epoch_df: pd.DataFrame,
                                 sleep_windows: pd.DataFrame,
                                 epoch_seconds: int = 5) -> pd.DataFrame:
    """
    Mark whether each epoch falls inside the externally-derived SPT window.

    Because the separate sleep pipeline provides only `spt_start_sleeplog` and
    `spt_end_sleeplog`, this function segments epochs into:
    - `day`: outside the sleep period time window
    - `spt`: inside the sleep period time window

    It does not attempt to split SPT into sleep versus wake epochs, because
    that would require finer sleep-state labels that are not available in the
    uploaded CSV.
    """
    out = epoch_df.copy()
    midpoint = out.index + pd.Timedelta(seconds=int(epoch_seconds) / 2.0)
    in_spt = np.zeros(len(out), dtype=bool)
    night_id = pd.Series(pd.NA, index=out.index, dtype="Int64")

    for row in sleep_windows.itertuples(index=False):
        mask = (midpoint >= row.spt_start) & (midpoint < row.spt_end)
        in_spt |= mask
        if "night_id" in sleep_windows.columns:
            night_id.loc[mask] = int(row.night_id)

    out["in_spt"] = in_spt
    out["spt_night_id"] = night_id
    out["segment"] = pd.Categorical(
        np.where(out["in_spt"], "spt", "day"),
        categories=["day", "spt"],
        ordered=True,
    )
    return out


def classify_epoch_intensity(epoch_df: pd.DataFrame,
                             value_col: str = "enmo_mg_imputed",
                             thresholds_mg: Tuple[float, float, float] = (40.0, 100.0, 400.0),
                             output_col: str = "intensity_level") -> pd.DataFrame:
    """
    Label each epoch using GGIR's ENMO cut-points:
    - < 40 mg  -> IN
    - 40-100   -> LIG
    - 100-400  -> MOD
    - >= 400   -> VIG
    """
    thr_lig, thr_mod, thr_vig = thresholds_mg
    if not (thr_lig < thr_mod < thr_vig):
        raise ValueError("thresholds_mg must be strictly increasing.")

    out = epoch_df.copy()
    values = out[value_col].to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError(
            f"Column {value_col!r} contains NaN/inf; classify intensity after imputation."
        )

    labels = np.where(
        values < thr_lig,
        "IN",
        np.where(values < thr_mod, "LIG", np.where(values < thr_vig, "MOD", "VIG")),
    )
    out[output_col] = pd.Categorical(labels, categories=list(INTENSITY_LEVELS), ordered=True)
    out["threshold_lig_mg"] = float(thr_lig)
    out["threshold_mod_mg"] = float(thr_mod)
    out["threshold_vig_mg"] = float(thr_vig)
    return out


def summarize_activity_intensity(epoch_df: pd.DataFrame,
                                 epoch_seconds: int = 5,
                                 level_col: str = "intensity_level",
                                 segment_col: str = "segment",
                                 imputed_col: str = "was_imputed") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize intensity time per retained analysis day and average across days.

    Non-wear handling
    -----------------
    - non-wear epochs are first flagged as invalid
    - only days with at least 18 hours of observed wear are kept
    - only recordings with at least 3 valid days are summarized
    - invalid epochs on retained days are imputed from the same clock-time
      position across the other retained valid days
    - intensity classification is then applied to the imputed ENMO series

    To make the impact of non-wear transparent, the daily summary reports
    observed and imputed minutes separately for each segment and intensity level.
    """
    required_cols = {"analysis_date", level_col, segment_col, imputed_col}
    missing = required_cols.difference(epoch_df.columns)
    if missing:
        raise ValueError(f"epoch_df is missing required columns: {sorted(missing)}")

    minutes_per_epoch = float(epoch_seconds) / 60.0
    per_day_rows: List[Dict[str, object]] = []

    for analysis_date, day_df in epoch_df.groupby("analysis_date", sort=True):
        row: Dict[str, object] = {
            "analysis_date": analysis_date,
            "n_epochs_total": int(len(day_df)),
            "n_epochs_imputed": int(day_df[imputed_col].sum()),
            "n_epochs_observed": int((~day_df[imputed_col].astype(bool)).sum()),
            "dur_full_total_min": float(len(day_df) * minutes_per_epoch),
            "dur_full_imputed_min": float(day_df[imputed_col].sum() * minutes_per_epoch),
            "dur_full_observed_min": float((~day_df[imputed_col].astype(bool)).sum() * minutes_per_epoch),
        }

        for segment in ("day", "spt", "full"):
            if segment == "full":
                segment_df = day_df
            else:
                segment_df = day_df.loc[day_df[segment_col] == segment]

            row[f"n_epochs_{segment}"] = int(len(segment_df))
            row[f"n_epochs_{segment}_imputed"] = int(segment_df[imputed_col].sum())
            row[f"dur_{segment}_window_min"] = float(len(segment_df) * minutes_per_epoch)
            row[f"dur_{segment}_window_imputed_min"] = float(segment_df[imputed_col].sum() * minutes_per_epoch)
            row[f"dur_{segment}_window_observed_min"] = float(
                (~segment_df[imputed_col].astype(bool)).sum() * minutes_per_epoch
            )

            for level in INTENSITY_LEVELS:
                level_mask = segment_df[level_col] == level
                row[f"dur_{segment}_total_{level}_min"] = float(level_mask.sum() * minutes_per_epoch)
                row[f"dur_{segment}_total_{level}_imputed_min"] = float(
                    (level_mask & segment_df[imputed_col].astype(bool)).sum() * minutes_per_epoch
                )
                row[f"dur_{segment}_total_{level}_observed_min"] = float(
                    (level_mask & ~segment_df[imputed_col].astype(bool)).sum() * minutes_per_epoch
                )

            row[f"dur_{segment}_total_MVPA_min"] = (
                float(row[f"dur_{segment}_total_MOD_min"] + row[f"dur_{segment}_total_VIG_min"])
            )
            row[f"dur_{segment}_total_MVPA_imputed_min"] = (
                float(
                    row[f"dur_{segment}_total_MOD_imputed_min"] +
                    row[f"dur_{segment}_total_VIG_imputed_min"]
                )
            )
            row[f"dur_{segment}_total_MVPA_observed_min"] = (
                float(
                    row[f"dur_{segment}_total_MOD_observed_min"] +
                    row[f"dur_{segment}_total_VIG_observed_min"]
                )
            )

        per_day_rows.append(row)

    daily_df = pd.DataFrame(per_day_rows)
    if daily_df.empty:
        return daily_df, pd.DataFrame()

    summary: Dict[str, object] = {"n_days": int(len(daily_df))}
    numeric_cols = daily_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        summary[f"{col}_mean"] = float(daily_df[col].mean())

    summary_df = pd.DataFrame([summary])
    return daily_df, summary_df


def compare_activity_intensity_against_ggir(intensity_daily_df: pd.DataFrame,
                                            ggir_ms5_rdata_path: str | Path,
                                            window_type: str = "MM") -> Dict[str, pd.DataFrame]:
    """
    Compare Python full-day threshold totals against GGIR part-5 full-day totals.

    Comparison note
    ---------------
    This Python module segments the recording into day versus SPT windows using
    the uploaded sleep CSV, but it does not have per-epoch sleep-versus-wake
    labels inside SPT. GGIR's part-5 full-day inactivity total includes
    `spt_sleep`, so small residual differences are expected even when the epoch
    cut-points and non-wear handling are close.
    """
    ggir_daily = load_ggir_part5_intensity(
        ms5_rdata_path=str(ggir_ms5_rdata_path),
        window_type=window_type,
        drop_incomplete_days=True,
    )

    py_daily = intensity_daily_df.copy()
    py_daily["analysis_date"] = pd.to_datetime(py_daily["analysis_date"]).dt.date
    merged = py_daily.merge(ggir_daily, on="analysis_date", how="inner", suffixes=("", "_ggir"))

    metric_cols = [
        "dur_full_total_IN_min",
        "dur_full_total_LIG_min",
        "dur_full_total_MOD_min",
        "dur_full_total_VIG_min",
        "dur_full_total_MVPA_min",
    ]

    per_day_rows: List[Dict[str, object]] = []
    for _, row in merged.iterrows():
        for metric in metric_cols:
            ggir_metric = f"{metric}_ggir"
            per_day_rows.append(
                {
                    "analysis_date": row["analysis_date"],
                    "metric": metric,
                    "python": float(row[metric]),
                    "ggir": float(row[ggir_metric]),
                    "difference": float(row[metric] - row[ggir_metric]),
                    "abs_difference": float(abs(row[metric] - row[ggir_metric])),
                    "ggir_nonwear_perc_day": float(row["nonwear_perc_day"]),
                }
            )

    per_day_df = pd.DataFrame(per_day_rows)
    if per_day_df.empty:
        summary_df = pd.DataFrame([{"n_overlap_days": 0}])
    else:
        summary_df = (
            per_day_df.groupby("metric")
            .agg(
                n_overlap_days=("analysis_date", "nunique"),
                mean_difference=("difference", "mean"),
                mean_abs_difference=("abs_difference", "mean"),
                max_abs_difference=("abs_difference", "max"),
            )
            .reset_index()
        )

    return {
        "ggir_daily_df": ggir_daily,
        "comparison_daily_df": per_day_df,
        "comparison_summary_df": summary_df,
    }


def run_activity_intensity_pipeline(raw_root: str | Path,
                                    subject: str,
                                    sleep_csv_path: str | Path,
                                    visit: str = "T0",
                                    raw_file_path: Optional[str | Path] = None,
                                    sleep_subject: Optional[str] = None,
                                    nonwear_method: str = "vanhees2013",
                                    epoch_seconds: int = 5,
                                    thresholds_mg: Tuple[float, float, float] = (40.0, 100.0, 400.0),
                                    dayborder_hours: float = 0.0,
                                    timezone: str = "Europe/Rome",
                                    min_wear_hours: float = 18.0,
                                    min_valid_days: int = 3,
                                    max_days: Optional[int] = 7,
                                    require_spt_overlap: bool = True,
                                    ggir_ms5_rdata_path: Optional[str | Path] = None) -> Dict[str, object]:
    """
    Run a GGIR-oriented activity-intensity pipeline on one raw wrist file.

    Processing steps
    ----------------
    1. load and autocalibrate the raw GENEActiv recording
    2. detect non-wear on the calibrated signal
    3. compute ENMO on a short epoch grid (default 5 seconds)
    4. retain only days with at least 18 hours of wear
    5. require at least 3 valid days
    6. impute invalid epochs using the same clock-time across the other valid days
    7. classify imputed ENMO epochs with the 40/100/400 mg cut-points
    8. segment epochs into day versus SPT using `spt_start_sleeplog` and `spt_end_sleeplog`
    9. average daily intensity totals across the retained days
    """
    if raw_file_path is None:
        raw_file_path = find_wrist_accelerometer_file(raw_root=raw_root, subject=subject, visit=visit)
    raw_file_path = Path(raw_file_path)

    calibrated_df, info = load_and_calibrate_accelerometer(str(raw_file_path))
    sample_rate = getattr(info, "fs", np.nan)
    if not np.isfinite(sample_rate):
        sample_rate = 100.0

    nonwear_df = detect_nonwear_intervals(
        calibrated_df=calibrated_df,
        fs=int(round(float(sample_rate))),
        method=nonwear_method,
    )
    epoch_df = build_epoch_table(
        calibrated_df=calibrated_df,
        nonwear_df=nonwear_df,
        epoch_seconds=epoch_seconds,
        dayborder_hours=dayborder_hours,
        timezone=timezone,
    )
    day_summary = summarize_epoch_days(
        epoch_df=epoch_df,
        epoch_seconds=epoch_seconds,
        min_wear_hours=min_wear_hours,
    )
    epoch_selected, valid_day_summary = select_valid_epoch_days(
        epoch_df=epoch_df,
        day_summary=day_summary,
        max_days=max_days,
        min_valid_days=min_valid_days,
    )
    epoch_imputed, imputed_epochs_per_day = impute_invalid_epochs_by_clocktime(epoch_selected)

    sleep_windows = load_sleep_windows(
        sleep_csv_path=sleep_csv_path,
        subject=subject,
        visit=visit,
        timezone=timezone,
        sleep_subject=sleep_subject,
    )
    epoch_segmented = annotate_sleep_period_window(
        epoch_df=epoch_imputed,
        sleep_windows=sleep_windows,
        epoch_seconds=epoch_seconds,
    )
    if require_spt_overlap and not bool(epoch_segmented["in_spt"].any()):
        raw_start = epoch_segmented.index.min()
        raw_end = epoch_segmented.index.max()
        sleep_start = sleep_windows["spt_start"].min()
        sleep_end = sleep_windows["spt_end"].max()
        raise ValueError(
            "The retained accelerometer epochs do not overlap any supplied SPT window. "
            f"Accelerometer range: {raw_start} to {raw_end}. "
            f"Sleep-window range: {sleep_start} to {sleep_end}."
        )
    epoch_classified = classify_epoch_intensity(
        epoch_df=epoch_segmented,
        thresholds_mg=thresholds_mg,
    )
    intensity_daily_df, intensity_summary_df = summarize_activity_intensity(
        epoch_df=epoch_classified,
        epoch_seconds=epoch_seconds,
    )

    out: Dict[str, object] = {
        "raw_file_path": raw_file_path,
        "sleep_windows": sleep_windows,
        "calibrated_df": calibrated_df,
        "nonwear_df": nonwear_df,
        "epoch_df": epoch_df,
        "day_summary": day_summary,
        "valid_day_summary": valid_day_summary,
        "epoch_imputed": epoch_imputed,
        "epoch_classified": epoch_classified,
        "imputed_epochs_per_day": imputed_epochs_per_day,
        "intensity_daily_df": intensity_daily_df,
        "intensity_summary_df": intensity_summary_df,
    }

    if ggir_ms5_rdata_path is not None:
        out["ggir_comparison"] = compare_activity_intensity_against_ggir(
            intensity_daily_df=intensity_daily_df,
            ggir_ms5_rdata_path=ggir_ms5_rdata_path,
        )

    return out


def write_activity_intensity_outputs(results: Dict[str, object],
                                     output_dir: str | Path,
                                     include_epoch_table: bool = False) -> Dict[str, Path]:
    """
    Write the main activity-intensity outputs to CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    file_map = {
        "sleep_windows": "sleep_windows_used.csv",
        "day_summary": "activity_intensity_day_summary.csv",
        "valid_day_summary": "activity_intensity_valid_days.csv",
        "intensity_daily_df": "activity_intensity_daily.csv",
        "intensity_summary_df": "activity_intensity_summary.csv",
    }

    if include_epoch_table:
        file_map["epoch_classified"] = "activity_intensity_epochs.csv"

    for key, filename in file_map.items():
        value = results.get(key)
        if isinstance(value, pd.DataFrame):
            path = output_dir / filename
            value.to_csv(path, index=(key == "epoch_classified"))
            paths[key] = path

    ggir_comparison = results.get("ggir_comparison")
    if isinstance(ggir_comparison, dict):
        for key, filename in {
            "ggir_daily_df": "activity_intensity_ggir_daily.csv",
            "comparison_daily_df": "activity_intensity_vs_ggir_daily.csv",
            "comparison_summary_df": "activity_intensity_vs_ggir_summary.csv",
        }.items():
            value = ggir_comparison.get(key)
            if isinstance(value, pd.DataFrame):
                path = output_dir / filename
                value.to_csv(path, index=False)
                paths[key] = path

    return paths

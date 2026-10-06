from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from dare_wearables.common.autocalibrate import autocalibrate
from dare_wearables.wrist.circadian.metrics import (
    build_mean_24h_profile,
    build_minute_table,
    classify_intensity_minutes,
    compute_ggir_is_iv,
    compute_l5_m5,
    fit_cosinor,
    impute_invalid_by_clocktime,
    plot_cosinor,
    plot_nonparametric_profile,
    prepare_cosinor_series,
    select_valid_days,
    summarize_days,
    summarize_intensity_levels,
)
from dare_wearables.wrist.nonwear.nimbaldetach import nimbaldetach


def find_empatica_parquet_files(silver_root: str | Path,
                                participant: str,
                                visit: str = "T0",
                                sensor: str = "Empatica") -> Dict[str, Path]:
    """
    Locate the Empatica EmbracePlus silver parquet files for one participant/visit.
    """
    base = Path(silver_root) / str(participant) / str(visit) / str(sensor)
    acc_path = base / "acc.parquet"
    temp_path = base / "temp.parquet"

    if not acc_path.exists():
        raise FileNotFoundError(f"Accelerometer parquet not found: {acc_path}")
    if not temp_path.exists():
        raise FileNotFoundError(f"Temperature parquet not found: {temp_path}")

    return {
        "acc_parquet_path": acc_path,
        "temp_parquet_path": temp_path,
    }


def _infer_sampling_frequency(index: pd.DatetimeIndex) -> float:
    diffs = index.to_series().diff().dt.total_seconds().dropna()
    diffs = diffs.loc[diffs > 0]
    if diffs.empty:
        raise ValueError("Could not infer sampling frequency from the timestamp index.")
    return float(1.0 / diffs.median())


def load_empatica_recording(acc_parquet_path: str | Path,
                            temp_parquet_path: str | Path,
                            recruitment_tracker_path: Optional[str | Path] = None,
                            parquet_engine: str = "fastparquet") -> Tuple[pl.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Load the preprocessed Empatica EmbracePlus silver files and apply the same
    Van Hees / GGIR-style autocalibration used in the GENEActiv pipeline.

    The GP silver accelerometer data are already stored in gravitational units,
    but we still calibrate them to local gravity the same way as in
    `circadian_pipeline.py`.

    Temperature is recorded at a lower frequency than acceleration, so before
    calibration we align the temperature stream to the accelerometer timestamps
    with a nearest-neighbour merge and a small tolerance. This preserves the
    original charging gaps instead of forward-filling temperature values across
    long missing segments.
    """
    visit = acc_parquet_path.parent.parent.name
    subject = acc_parquet_path.parent.parent.parent.name
    acc_df = pd.read_parquet(acc_parquet_path, engine=parquet_engine).copy()
    temp_df = pd.read_parquet(temp_parquet_path, engine=parquet_engine).copy()
    if recruitment_tracker_path is not None:
        recruitment_tracker = pd.read_excel(
            recruitment_tracker_path,
            sheet_name="Recruitment",
            skiprows=1,
        )

        if visit == "T0":
            start_date = recruitment_tracker[
                recruitment_tracker["codice pseudo partecipante"] == int(subject)
            ]["Data T0 reale"].iloc[0]
        elif visit == "T1":
            start_date = recruitment_tracker[
                recruitment_tracker["codice pseudo partecipante"] == int(subject)
            ]["Data T1 reale"].iloc[0]
        else:
            start_date = None

        if start_date is not None:
            acc_df = acc_df[acc_df.index >= start_date]

    if not isinstance(acc_df.index, pd.DatetimeIndex):
        raise TypeError("acc.parquet must have a DatetimeIndex.")
    if not isinstance(temp_df.index, pd.DatetimeIndex):
        raise TypeError("temp.parquet must have a DatetimeIndex.")

    acc_df = acc_df.sort_index()
    temp_df = temp_df.sort_index()

    required_acc_cols = {"x", "y", "z"}
    missing_acc = required_acc_cols.difference(acc_df.columns)
    if missing_acc:
        raise ValueError(f"acc.parquet is missing required columns: {sorted(missing_acc)}")
    if "temp" not in temp_df.columns:
        raise ValueError("temp.parquet must contain a 'temp' column.")

    accel_freq_hz = _infer_sampling_frequency(acc_df.index)
    temperature_freq_hz = _infer_sampling_frequency(temp_df.index)

    acc_reset = acc_df.reset_index()
    acc_time_col = acc_reset.columns[0]
    acc_reset = acc_reset.rename(
        columns={
            acc_time_col: "time",
        }
    )
    acc_reset["time"] = pd.to_datetime(acc_reset["time"])

    temp_reset = temp_df.reset_index()
    temp_time_col = temp_reset.columns[0]
    temp_reset = temp_reset.rename(
        columns={
            temp_time_col: "time",
            "temp": "temperature",
        }
    )
    temp_reset["time"] = pd.to_datetime(temp_reset["time"])

    raw_df = pd.merge_asof(
        acc_reset.sort_values("time"),
        temp_reset.sort_values("time")[["time", "temperature"]],
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=2),
    )

    raw_pl_df = pl.from_pandas(raw_df[["time", "x", "y", "z", "temperature"]])
    calibrated = autocalibrate(
        raw_pl_df,
        ts_col="time",
        x_col="x",
        y_col="y",
        z_col="z",
        temp_col="temperature",
    )
    calibrated_df = calibrated["df"]
    cal_params = calibrated["params"]
    if not {"x_cal", "y_cal", "z_cal"}.issubset(set(calibrated_df.columns)):
        calibrated_df = calibrated_df.with_columns(
            [
                pl.col("x").alias("x_cal"),
                pl.col("y").alias("y_cal"),
                pl.col("z").alias("z_cal"),
            ]
        )

    info = {
        "device": "Empatica EmbracePlus",
        "accel_freq_hz": float(accel_freq_hz),
        "temperature_freq_hz": float(temperature_freq_hz),
        "acc_parquet_path": str(acc_parquet_path),
        "temp_parquet_path": str(temp_parquet_path),
        "start_time": acc_df.index.min(),
        "end_time": acc_df.index.max(),
        "n_acc_samples": int(len(acc_df)),
        "n_temp_samples": int(len(temp_df)),
        "calibration_valid": bool(cal_params.valid) if cal_params is not None else False,
        "calibration_n_epochs_used": int(cal_params.n_epochs_used) if cal_params is not None else 0,
        "calibration_error_start": float(cal_params.cal_error_start) if cal_params is not None and cal_params.cal_error_start is not None else np.nan,
        "calibration_error_end": float(cal_params.cal_error_end) if cal_params is not None and cal_params.cal_error_end is not None else np.nan,
    }

    return calibrated_df, acc_df, temp_df, info


def find_charging_periods(acc_df: pd.DataFrame,
                          gap_seconds: float = 60.0) -> pd.DataFrame:
    """
    Identify charging periods as timestamp gaps larger than `gap_seconds`.
    """
    gap_mask = acc_df.index.to_series().diff().dt.total_seconds() > float(gap_seconds)
    charge_end = acc_df.index[gap_mask]

    if len(charge_end) == 0:
        return pd.DataFrame(columns=["start", "end"])

    charge_start = acc_df.index[np.flatnonzero(gap_mask.to_numpy()) - 1]
    charge_df = pd.DataFrame({"start": charge_start, "end": charge_end})
    charge_df = charge_df.sort_values("start").reset_index(drop=True)
    return charge_df


def _build_noncharging_chunks(acc_df: pd.DataFrame,
                              temp_df: pd.DataFrame,
                              charge_df: pd.DataFrame,
                              min_intermediate_duration: str | pd.Timedelta = "10 min",
                              min_last_duration: str | pd.Timedelta = "10 min") -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Segment the recording into non-charging chunks.

    This mirrors the structure used in the GP notebook:
    - the first chunk before the first charging period is always considered
    - intermediate chunks shorter than 10 minutes are skipped
    - the last chunk after the final charging period is kept only if it lasts
      at least 10 minutes
    """
    if charge_df.empty:
        return [(acc_df, temp_df)]

    min_intermediate_duration = pd.Timedelta(min_intermediate_duration)
    min_last_duration = pd.Timedelta(min_last_duration)

    chunks: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    first_charge_start = charge_df.loc[0, "start"]
    chunks.append((acc_df.loc[:first_charge_start].copy(), temp_df.loc[:first_charge_start].copy()))

    if len(charge_df) > 1:
        good_portions = pd.DataFrame({
            "start": charge_df["end"].iloc[:-1].reset_index(drop=True),
            "end": charge_df["start"].iloc[1:].reset_index(drop=True),
        })
        for row in good_portions.itertuples(index=False):
            if row.end - row.start < min_intermediate_duration:
                continue
            chunks.append((acc_df.loc[row.start:row.end].copy(), temp_df.loc[row.start:row.end].copy()))

    last_charge_end = charge_df.loc[len(charge_df) - 1, "end"]
    if acc_df.index.max() - last_charge_end >= min_last_duration:
        chunks.append((acc_df.loc[last_charge_end:].copy(), temp_df.loc[last_charge_end:].copy()))

    return chunks


def build_nonwear_periods(t_charge: pd.DataFrame,
                          start_stop_nw_datetime: pd.DataFrame,
                          *,
                          charge_start_col: str = "start",
                          charge_end_col: str = "end",
                          nw_start_col: str = "start_time",
                          nw_end_col: str = "end_time",
                          merge_gap: str | pd.Timedelta = "10 min") -> pd.DataFrame:
    """
    Combine charging and DETACH non-wear intervals, merge overlaps, then merge
    intervals separated by <= `merge_gap`.
    """
    gap = pd.Timedelta(merge_gap) if not isinstance(merge_gap, pd.Timedelta) else merge_gap

    t_charge_ = t_charge.copy()
    nw_ = start_stop_nw_datetime.copy()

    frames = []
    if not t_charge_.empty:
        t_charge_["type"] = "charge"
        frames.append(
            t_charge_[[charge_start_col, charge_end_col, "type"]].rename(
                columns={charge_start_col: "start", charge_end_col: "end"}
            )
        )
    if not nw_.empty:
        nw_["type"] = "nonwear"
        frames.append(
            nw_[[nw_start_col, nw_end_col, "type"]].rename(
                columns={nw_start_col: "start", nw_end_col: "end"}
            )
        )

    if not frames:
        return pd.DataFrame(columns=["start", "end"])

    nonwear_periods = pd.concat(frames, ignore_index=True)
    nonwear_periods = nonwear_periods.dropna(subset=["start", "end"])
    nonwear_periods["start"] = pd.to_datetime(nonwear_periods["start"])
    nonwear_periods["end"] = pd.to_datetime(nonwear_periods["end"])
    nonwear_periods = nonwear_periods.loc[nonwear_periods["end"] >= nonwear_periods["start"]].copy()
    nonwear_periods = nonwear_periods.sort_values("start").reset_index(drop=True)

    if nonwear_periods.empty:
        return pd.DataFrame(columns=["start", "end"])

    merged = []
    current_start = nonwear_periods.loc[0, "start"]
    current_end = nonwear_periods.loc[0, "end"]

    for i in range(1, len(nonwear_periods)):
        start = nonwear_periods.loc[i, "start"]
        end = nonwear_periods.loc[i, "end"]
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append({"start": current_start, "end": current_end})
            current_start, current_end = start, end
    merged.append({"start": current_start, "end": current_end})

    merged_df = pd.DataFrame(merged).sort_values("start").reset_index(drop=True)
    final = []
    current_start = merged_df.loc[0, "start"]
    current_end = merged_df.loc[0, "end"]

    for i in range(1, len(merged_df)):
        start = merged_df.loc[i, "start"]
        end = merged_df.loc[i, "end"]
        if (start - current_end) <= gap:
            current_end = max(current_end, end)
        else:
            final.append({"start": current_start, "end": current_end})
            current_start, current_end = start, end
    final.append({"start": current_start, "end": current_end})

    return pd.DataFrame(final).sort_values("start").reset_index(drop=True)


def detect_empatica_nonwear_intervals(acc_df: pd.DataFrame,
                                      temp_df: pd.DataFrame,
                                      accel_freq: Optional[float] = None,
                                      temperature_freq: Optional[float] = None,
                                      nonwear_method: str = "empatica_detach",
                                      charge_gap_seconds: float = 60.0,
                                      merge_gap: str | pd.Timedelta = "10 min",
                                      min_intermediate_chunk: str | pd.Timedelta = "10 min",
                                      min_detach_chunk_minutes: float = 5.0) -> pd.DataFrame:
    """
    GP / Empatica-specific non-wear detection.

    Default behaviour reproduces the logic used in the GP notebook:
    1. identify charging periods from large timestamp gaps
    2. split the recording into non-charging chunks
    3. run DETACH on those chunks
    4. merge charging and DETACH intervals into final non-wear windows

    Supported methods
    -----------------
    - `empatica_detach` / `detach` / `nimbaldetach`: notebook-style charging + DETACH
    - `charge_only`: use charging periods only
    """
    method = nonwear_method.lower()
    if accel_freq is None:
        accel_freq = _infer_sampling_frequency(acc_df.index)
    if temperature_freq is None:
        temperature_freq = _infer_sampling_frequency(temp_df.index)

    charge_df = find_charging_periods(acc_df=acc_df, gap_seconds=charge_gap_seconds)
    if method == "charge_only":
        return charge_df
    if method not in {"empatica_detach", "detach", "nimbaldetach"}:
        raise ValueError(
            f"Unsupported nonwear_method {nonwear_method!r}. "
            "Use 'empatica_detach'/'detach'/'nimbaldetach' or 'charge_only'."
        )

    chunks = _build_noncharging_chunks(
        acc_df=acc_df,
        temp_df=temp_df,
        charge_df=charge_df,
        min_intermediate_duration=min_intermediate_chunk,
        min_last_duration=min_intermediate_chunk,
    )

    start_stop_nw_datetime = []
    min_detach_samples = int(round(float(accel_freq) * 60.0 * float(min_detach_chunk_minutes)))

    for acc_loop, temp_loop in chunks:
        if len(acc_loop) < min_detach_samples or temp_loop.empty:
            continue

        start_stop_nw, _ = nimbaldetach(
            acc_loop["x"].to_numpy(),
            acc_loop["y"].to_numpy(),
            acc_loop["z"].to_numpy(),
            temp_loop["temp"].to_numpy(),
            accel_freq=float(accel_freq),
            temperature_freq=float(temperature_freq),
            quiet=True,
        )

        if start_stop_nw.empty:
            continue

        for row in start_stop_nw.itertuples(index=False):
            start_idx = int(row[0])
            end_idx = min(int(row[1]), len(acc_loop) - 1)
            start_stop_nw_datetime.append(
                {
                    "start_time": acc_loop.index[start_idx],
                    "end_time": acc_loop.index[end_idx],
                }
            )

    detected_df = pd.DataFrame(start_stop_nw_datetime)
    if detected_df.empty and charge_df.empty:
        return pd.DataFrame(columns=["start", "end"])
    if detected_df.empty:
        return charge_df.copy()

    return build_nonwear_periods(
        t_charge=charge_df,
        start_stop_nw_datetime=detected_df,
        merge_gap=merge_gap,
    )


def preprocess_empatica_recording(acc_parquet_path: str | Path,
                                  temp_parquet_path: str | Path,
                                  recruitment_tracker_path: Optional[str | Path] = None,
                                  parquet_engine: str = "fastparquet",
                                  fs: Optional[float] = None,
                                  nonwear_method: str = "empatica_detach",
                                  charge_gap_seconds: float = 60.0,
                                  merge_gap: str | pd.Timedelta = "10 min",
                                  min_intermediate_chunk: str | pd.Timedelta = "10 min",
                                  min_detach_chunk_minutes: float = 5.0) -> Dict[str, object]:
    """
    Shared GP / EmbracePlus preprocessing:
    - load silver parquet accelerometer + temperature
    - autocalibrate the accelerometer stream
    - detect non-wear using the GP-specific charging + DETACH workflow

    This helper centralizes the preprocessing so the circadian, intensity, and
    sleep pipelines all operate on the same calibrated and non-wear-masked
    recording.
    """
    calibrated_df, acc_df, temp_df, info = load_empatica_recording(
        acc_parquet_path=acc_parquet_path,
        temp_parquet_path=temp_parquet_path,
        recruitment_tracker_path=recruitment_tracker_path,
        parquet_engine=parquet_engine,
    )
    if fs is None:
        fs = float(info["accel_freq_hz"])

    nonwear_df = detect_empatica_nonwear_intervals(
        acc_df=acc_df,
        temp_df=temp_df,
        accel_freq=float(fs),
        temperature_freq=float(info["temperature_freq_hz"]),
        nonwear_method=nonwear_method,
        charge_gap_seconds=charge_gap_seconds,
        merge_gap=merge_gap,
        min_intermediate_chunk=min_intermediate_chunk,
        min_detach_chunk_minutes=min_detach_chunk_minutes,
    )

    return {
        "calibrated_df": calibrated_df,
        "acc_df": acc_df,
        "temp_df": temp_df,
        "info": info,
        "fs": float(fs),
        "nonwear_df": nonwear_df,
    }


def run_circadian_pipeline_gp_from_preprocessed(calibrated_df: pl.DataFrame,
                                                acc_df: pd.DataFrame,
                                                temp_df: pd.DataFrame,
                                                info: dict,
                                                nonwear_df: pd.DataFrame,
                                                *,
                                                save_folder: Optional[str | Path] = None,
                                                nonwear_method: str = "empatica_detach",
                                                dayborder_hours: float = 0.0,
                                                min_wear_hours: float = 18.0,
                                                max_nonwear_hours: Optional[float] = None,
                                                require_full_24h: bool = False,
                                                max_days: Optional[int] = 7,
                                                min_valid_days: Optional[int] = 3,
                                                activity_threshold_mg: float = 40.0,
                                                intensity_thresholds_mg: Tuple[float, float, float] = (40.0, 100.0, 400.0),
                                                l5_m5_step_minutes: int = 10,
                                                plot: bool = True) -> Dict[str, object]:
    """
    Run the GP circadian analysis starting from a shared preprocessed recording.

    This helper avoids repeating Empatica loading, calibration, and non-wear
    detection when another pipeline stage has already produced them.
    """
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    effective_min_wear_hours = (
        24.0 - float(max_nonwear_hours)
        if max_nonwear_hours is not None
        else float(min_wear_hours)
    )

    minute_df = build_minute_table(
        calibrated_df=calibrated_df,
        nonwear_df=nonwear_df,
        dayborder_hours=dayborder_hours,
    )
    day_summary = summarize_days(
        minute_df=minute_df,
        min_wear_hours=effective_min_wear_hours,
        require_full_24h=require_full_24h,
        max_nonwear_hours=max_nonwear_hours,
    )
    minute_selected, valid_day_summary = select_valid_days(
        minute_df=minute_df,
        day_summary=day_summary,
        max_days=max_days,
        min_valid_days=min_valid_days,
    )

    minute_imputed, imputed_minutes_per_day = impute_invalid_by_clocktime(
        minute_df=minute_selected,
        value_col="enmo_mg",
        fallback_value=0.0,
    )
    minute_imputed = classify_intensity_minutes(
        minute_df=minute_imputed,
        value_col="enmo_mg_imputed",
        thresholds_mg=intensity_thresholds_mg,
        output_col="intensity_level",
    )
    intensity_daily_df, intensity_summary_df = summarize_intensity_levels(
        minute_df=minute_imputed,
        level_col="intensity_level",
        imputed_col="was_imputed",
    )

    profile_df = build_mean_24h_profile(
        minute_df=minute_imputed,
        value_col="enmo_mg_imputed",
        dayborder_hours=dayborder_hours,
    )
    ivis = compute_ggir_is_iv(
        minute_df=minute_selected,
        value_col="enmo_mg",
        threshold_mg=activity_threshold_mg,
    )
    l5m5 = compute_l5_m5(
        minute_df=minute_imputed,
        value_col="enmo_mg_imputed",
        step_minutes=l5_m5_step_minutes,
    )

    cosinor_df = prepare_cosinor_series(
        minute_df=minute_selected,
        value_col="enmo_mg",
    )
    cosinor_out = fit_cosinor(cosinor_df)

    if plot:
        plot_nonparametric_profile(profile_df, l5m5)
        plot_cosinor(cosinor_df, cosinor_out)

    metrics = {
        "n_valid_days_selected": int(len(valid_day_summary)),
        "min_wear_hours_per_day": float(effective_min_wear_hours),
        "max_nonwear_hours_per_day": (np.nan if max_nonwear_hours is None else float(max_nonwear_hours)),
        "dayborder_hours": float(dayborder_hours),
        "nonwear_method": nonwear_method,
        "activity_threshold_mg": float(activity_threshold_mg),
        "intensity_threshold_lig_mg": float(intensity_thresholds_mg[0]),
        "intensity_threshold_mod_mg": float(intensity_thresholds_mg[1]),
        "intensity_threshold_vig_mg": float(intensity_thresholds_mg[2]),
        **ivis,
        **l5m5,
        "cosinor_timeOffsetHours": cosinor_out["time_offset_hours"],
        "cosinor_mes": cosinor_out["MESOR_log1p_mg"],
        "cosinor_amp": cosinor_out["Amplitude_log1p_mg"],
        "cosinor_acrophase": cosinor_out["Acrophase_rad_clock"],
        "cosinor_acrotime": cosinor_out["Acrotime_hour"],
        "cosinor_days": cosinor_out["n_days_used"],
        "MESOR_log1p_mg": cosinor_out["MESOR_log1p_mg"],
        "Amplitude_log1p_mg": cosinor_out["Amplitude_log1p_mg"],
        "Phase_rad_series_start": cosinor_out["Phase_rad_series_start"],
        "Acrotime_hour": cosinor_out["Acrotime_hour"],
        "Cosinor_n_days_used": cosinor_out["n_days_used"],
        "Cosinor_n_valid_minutes": cosinor_out["n_valid_minutes"],
        "device": info["device"],
        "accel_freq_hz": info["accel_freq_hz"],
        "temperature_freq_hz": info["temperature_freq_hz"],
        "calibration_valid": info["calibration_valid"],
        "calibration_n_epochs_used": info["calibration_n_epochs_used"],
        "calibration_error_start": info["calibration_error_start"],
        "calibration_error_end": info["calibration_error_end"],
    }
    if not intensity_summary_df.empty:
        for col, value in intensity_summary_df.iloc[0].items():
            metrics[col] = value
    for i, row in valid_day_summary.reset_index(drop=True).iterrows():
        day = row["analysis_date"]
        metrics[f"day_{i + 1}_date"] = str(day)
        metrics[f"day_{i + 1}_invalid_min"] = int(row["invalid_minutes"])
        metrics[f"day_{i + 1}_imputed_min"] = int(imputed_minutes_per_day.get(day, 0))

    metrics_df = pd.DataFrame([metrics])

    if save_folder is not None:
        day_summary.to_csv(os.path.join(save_folder, "circadian_day_summary_all.csv"), index=False)
        valid_day_summary.to_csv(os.path.join(save_folder, "circadian_day_summary_selected.csv"), index=False)
        profile_df.to_csv(os.path.join(save_folder, "circadian_profile.csv"), index=False)
        metrics_df.to_csv(os.path.join(save_folder, "circadian_metrics.csv"), index=False)
        intensity_daily_df.to_csv(os.path.join(save_folder, "circadian_intensity_daily.csv"), index=False)
        intensity_summary_df.to_csv(os.path.join(save_folder, "circadian_intensity_summary.csv"), index=False)
        minute_imputed.to_csv(os.path.join(save_folder, "circadian_1min_selected.csv"))
        nonwear_df.to_csv(os.path.join(save_folder, "circadian_nonwear_windows.csv"), index=False)

    return {
        "info": info,
        "acc_df": acc_df,
        "temp_df": temp_df,
        "nonwear_df": nonwear_df,
        "minute_df": minute_df,
        "day_summary": day_summary,
        "valid_day_summary": valid_day_summary,
        "minute_imputed": minute_imputed,
        "intensity_daily_df": intensity_daily_df,
        "intensity_summary_df": intensity_summary_df,
        "profile_df": profile_df,
        "cosinor_df": cosinor_df,
        "metrics_df": metrics_df,
    }


def circadian_pipeline_gp(acc_parquet_path: str | Path,
                          temp_parquet_path: str | Path,
                          recruitment_tracker_path: Optional[str | Path] = None,
                          save_folder: Optional[str] = None,
                          parquet_engine: str = "fastparquet",
                          fs: Optional[float] = None,
                          nonwear_method: str = "empatica_detach",
                          dayborder_hours: float = 0.0,
                          min_wear_hours: float = 18.0,
                          max_nonwear_hours: Optional[float] = None,
                          require_full_24h: bool = False,
                          max_days: Optional[int] = 7,
                          min_valid_days: Optional[int] = 3,
                          activity_threshold_mg: float = 40.0,
                          intensity_thresholds_mg: Tuple[float, float, float] = (40.0, 100.0, 400.0),
                          l5_m5_step_minutes: int = 10,
                          plot: bool = True) -> Dict[str, object]:
    """
    Circadian pipeline for the GP EmbracePlus dataset.

    This keeps the circadian analysis logic aligned with `circadian_pipeline.py`
    while swapping in the GP notebook's loading and non-wear preprocessing:
    - load Empatica silver `acc.parquet` and `temp.parquet`
    - treat charging gaps as non-wear
    - run DETACH on non-charging chunks
    - merge charging + DETACH intervals before the 1-minute circadian pipeline
    """
    preprocessed = preprocess_empatica_recording(
        acc_parquet_path=acc_parquet_path,
        temp_parquet_path=temp_parquet_path,
        recruitment_tracker_path=recruitment_tracker_path,
        parquet_engine=parquet_engine,
        fs=fs,
        nonwear_method=nonwear_method,
    )
    if save_folder is None:
        save_folder = str(Path(acc_parquet_path).parent)

    return run_circadian_pipeline_gp_from_preprocessed(
        calibrated_df=preprocessed["calibrated_df"],
        acc_df=preprocessed["acc_df"],
        temp_df=preprocessed["temp_df"],
        info=preprocessed["info"],
        nonwear_df=preprocessed["nonwear_df"],
        save_folder=save_folder,
        nonwear_method=nonwear_method,
        dayborder_hours=dayborder_hours,
        min_wear_hours=min_wear_hours,
        max_nonwear_hours=max_nonwear_hours,
        require_full_24h=require_full_24h,
        max_days=max_days,
        min_valid_days=min_valid_days,
        activity_threshold_mg=activity_threshold_mg,
        intensity_thresholds_mg=intensity_thresholds_mg,
        l5_m5_step_minutes=l5_m5_step_minutes,
        plot=plot,
    )


def circadian_pipeline_gp_from_silver(silver_root: str | Path,
                                      participant: str,
                                      visit: str = "T0",
                                      sensor: str = "Empatica",
                                      recruitment_tracker_path: Optional[str | Path] = None,
                                      **kwargs) -> Dict[str, object]:
    """
    Convenience wrapper to run the GP pipeline directly from the silver folder.
    """
    paths = find_empatica_parquet_files(
        silver_root=silver_root,
        participant=participant,
        visit=visit,
        sensor=sensor,
    )
    return circadian_pipeline_gp(
        acc_parquet_path=paths["acc_parquet_path"],
        temp_parquet_path=paths["temp_parquet_path"],
        **kwargs,
    )

run_circadian_from_preprocessed = run_circadian_pipeline_gp_from_preprocessed
run_circadian_from_empatica = circadian_pipeline_gp
run_circadian_from_empatica_silver = circadian_pipeline_gp_from_silver

import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt


def load_and_calibrate_accelerometer(file_path: str,
                                     device: str = "GENEActiv") -> Tuple[pl.DataFrame, dict]:
    """
    Load raw accelerometer data and apply autocalibration.

    Returns
    -------
    calibrated_df : polars.DataFrame
        Must contain time, x_cal, y_cal, z_cal, and optionally temperature.
    info : dict
        Device metadata returned by the reader, if available.
    """
    if device != "GENEActiv":
        raise NotImplementedError("At the moment this loader only supports device='GENEActiv'.")

    from dare_wearables.wrist.data_io.geneactiv import as_polars_dataframe, read_geneactiv_bin
    from dare_wearables.common.autocalibrate import autocalibrate

    info, data = read_geneactiv_bin(file_path)
    raw_df = as_polars_dataframe(info, data)

    calibrated = autocalibrate(
        raw_df,
        ts_col="time",
        x_col="x",
        y_col="y",
        z_col="z",
        temp_col="temperature",
    )

    return calibrated["df"], info


def detect_nonwear_intervals(calibrated_df: pl.DataFrame,
                             fs: int = 100,
                             method: str = "vanhees2013") -> pd.DataFrame:
    """
    Run non-wear detection and return a pandas DataFrame with columns start, end.

    Supported methods
    -----------------
    - vanhees2013 / ggir : closer to the GGIR non-wear logic
    - detach / nimbaldetach : temperature-assisted DETACH implementation
    """
    method = method.lower()

    if method in {"vanhees2013", "ggir"}:
        from dare_wearables.wrist.nonwear.vanhees2013 import vanhees2013

        out = vanhees2013(
            calibrated_df,
            ts_col="time",
            x_col="x_cal",
            y_col="y_cal",
            z_col="z_cal",
            freq=fs,
            return_debug=True,
        )
        nonwear_df = out.nonwear_df
    elif method in {"detach", "nimbaldetach"}:
        from dare_wearables.wrist.nonwear.detach import detach

        out = detach(
            calibrated_df,
            ts_col="time",
            x_col="x_cal",
            y_col="y_cal",
            z_col="z_cal",
            temp_col_in_acc="temperature",
            temp_df=None,
            accel_freq=fs,
            temperature_freq=fs / 300,
            num_axes=3,
            return_debug=True,
        )
        nonwear_df = out.nonwear_df
    else:
        raise ValueError(
            f"Unsupported nonwear method '{method}'. "
            "Use 'vanhees2013'/'ggir' or 'detach'/'nimbaldetach'."
        )

    if nonwear_df.height == 0:
        return pd.DataFrame(columns=["start", "end"])

    nw = nonwear_df.select(["start", "end"]).to_pandas()
    nw["start"] = pd.to_datetime(nw["start"])
    nw["end"] = pd.to_datetime(nw["end"])
    nw = nw.sort_values("start").reset_index(drop=True)
    return nw


def build_minute_table(calibrated_df: pl.DataFrame,
                       nonwear_df: pd.DataFrame,
                       dayborder_hours: float = 0.0) -> pd.DataFrame:
    """
    Create the core 1-minute table used by the rest of the pipeline.

    Output columns
    --------------
    enmo_mg       : 1-min mean ENMO in mg
    invalid       : True for minutes overlapping non-wear or lacking data
    analysis_date : day label after applying dayborder
    minute_of_day : 0..1439 relative to analysis_date start
    clock_hour    : local clock hour (0..24)
    """
    df = calibrated_df.select(["time", "x_cal", "y_cal", "z_cal"]).to_pandas()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    mag_g = np.sqrt(df["x_cal"] ** 2 + df["y_cal"] ** 2 + df["z_cal"] ** 2)
    df["enmo_mg"] = np.maximum((mag_g - 1.0) * 1000.0, 0.0)

    minute_series = (
        df.set_index("time")["enmo_mg"]
        .resample("1min")
        .mean()
    )

    full_index = pd.date_range(
        start=minute_series.index.min().floor("min"),
        end=minute_series.index.max().floor("min"),
        freq="1min",
    )

    minute_df = pd.DataFrame(index=full_index)
    minute_df.index.name = "time"
    minute_df["enmo_mg"] = minute_series.reindex(full_index)
    minute_df["invalid"] = minute_df["enmo_mg"].isna()

    if nonwear_df is not None and not nonwear_df.empty:
        for row in nonwear_df.itertuples(index=False):
            start = pd.Timestamp(row.start).floor("min")
            end = pd.Timestamp(row.end).ceil("min")
            mask = (minute_df.index >= start) & (minute_df.index < end)
            minute_df.loc[mask, "invalid"] = True

    shifted = minute_df.index - pd.to_timedelta(dayborder_hours, unit="h")
    minute_df["analysis_date"] = shifted.date
    minute_df["minute_of_day"] = shifted.hour * 60 + shifted.minute
    minute_df["clock_hour"] = minute_df.index.hour + minute_df.index.minute / 60.0

    return minute_df


def summarize_days(minute_df: pd.DataFrame,
                   min_wear_hours: float = 18.0,
                   require_full_24h: bool = False,
                   max_nonwear_hours: Optional[float] = None) -> pd.DataFrame:
    """
    Summarize each analysis day and flag valid days.
    """
    if max_nonwear_hours is not None:
        min_wear_hours = 24.0 - float(max_nonwear_hours)

    day_summary = (
        minute_df.groupby("analysis_date")
        .agg(
            total_minutes=("enmo_mg", "size"),
            invalid_minutes=("invalid", "sum"),
        )
        .reset_index()
    )

    day_summary["valid_minutes"] = day_summary["total_minutes"] - day_summary["invalid_minutes"]
    day_summary["invalid_hours"] = day_summary["invalid_minutes"] / 60.0
    day_summary["valid_hours"] = day_summary["valid_minutes"] / 60.0
    day_summary["is_full_24h"] = day_summary["total_minutes"] == 1440
    day_summary["is_valid_day"] = day_summary["valid_hours"] >= min_wear_hours

    if require_full_24h:
        day_summary["is_valid_day"] &= day_summary["is_full_24h"]

    return day_summary


def select_valid_days(minute_df: pd.DataFrame,
                      day_summary: pd.DataFrame,
                      max_days: Optional[int] = 7,
                      min_valid_days: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep valid days only, optionally limiting to the first `max_days`.
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
    minute_selected = minute_df.loc[minute_df["analysis_date"].isin(keep_dates)].copy()
    minute_selected = minute_selected.sort_index()

    return minute_selected, valid_days


def impute_invalid_by_clocktime(minute_df: pd.DataFrame,
                                value_col: str = "enmo_mg",
                                fallback_value: float = 0.0) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Impute invalid minutes using the mean value at the same minute-of-day on other days.
    If no valid value exists at that minute-of-day across the recording, use fallback_value.
    """
    out = minute_df.copy()
    out["was_imputed"] = out["invalid"].astype(bool)
    out[f"{value_col}_imputed"] = out[value_col]

    valid_means = (
        out.loc[~out["invalid"]]
        .groupby("minute_of_day")[value_col]
        .mean()
    )

    mask = out["invalid"]
    out.loc[mask, f"{value_col}_imputed"] = (
        out.loc[mask, "minute_of_day"].map(valid_means)
    )
    out[f"{value_col}_imputed"] = out[f"{value_col}_imputed"].fillna(fallback_value)

    imputed_minutes_per_day = (
        out.groupby("analysis_date")["was_imputed"]
        .sum()
        .astype(int)
    )

    return out, imputed_minutes_per_day


def build_mean_24h_profile(minute_df: pd.DataFrame,
                           value_col: str = "enmo_mg_imputed",
                           dayborder_hours: float = 0.0) -> pd.DataFrame:
    """
    Average the selected 1-minute series over minute-of-day.
    """
    profile = (
        minute_df.groupby("minute_of_day")[value_col]
        .mean()
        .reindex(range(1440))
    )

    profile_df = pd.DataFrame({
        "minute_of_day": np.arange(1440),
        "mean_24h_enmo_mg": profile.to_numpy(),
    })
    profile_df["clock_hour"] = ((profile_df["minute_of_day"] / 60.0) + dayborder_hours) % 24

    return profile_df


def prepare_ggir_series(minute_df: pd.DataFrame,
                        value_col: str = "enmo_mg") -> pd.DataFrame:
    """
    Prepare a GGIR-like 1-minute series:
    - start at the first valid minute
    - keep invalid minutes as NaN
    - trim to an integer number of days
    """
    series = minute_df[[value_col, "invalid"]].copy().sort_index()

    valid_mask = (~series["invalid"]) & series[value_col].notna()
    if not valid_mask.any():
        raise ValueError("No valid minutes available to build a GGIR-like series.")

    first_valid = series.index[np.flatnonzero(valid_mask.to_numpy())[0]]
    series = series.loc[first_valid:].copy()

    total_minutes = int((series.index.max() - first_valid) / pd.Timedelta(minutes=1)) + 1
    n_full_days = total_minutes // 1440
    if n_full_days < 1:
        raise ValueError("Not enough recording span for at least one full day.")

    end_time = first_valid + pd.Timedelta(minutes=n_full_days * 1440 - 1)
    full_index = pd.date_range(start=first_valid, end=end_time, freq="1min")

    series = series.reindex(full_index)
    series["invalid"] = series["invalid"].fillna(True) | series[value_col].isna()
    series["ggir_input"] = np.where(
        series["invalid"],
        np.nan,
        series[value_col].astype(float),
    )

    return series


def compute_ggir_is_iv(minute_df: pd.DataFrame,
                       value_col: str = "enmo_mg",
                       threshold_mg: float = 40.0) -> Dict[str, float]:
    """
    Approximate the current GGIR IV/IS implementation on a 1-minute ENMO series.

    GGIR computes IV/IS on an hourly series and keeps invalid data as missing.
    When a threshold is provided, the signal is converted to a binary
    activity/inactivity indicator before hourly aggregation.
    """
    series = prepare_ggir_series(minute_df=minute_df, value_col=value_col)
    xi = series["ggir_input"].to_numpy(dtype=float)

    if threshold_mg is not None:
        xi = np.where(
            np.isnan(xi),
            np.nan,
            np.where(xi > float(threshold_mg), 0.0, 1.0),
        )

    hourly = (
        pd.Series(xi)
        .groupby(np.arange(len(xi)) // 60)
        .mean()
    )
    xi_hour = hourly.to_numpy(dtype=float)

    if len(xi_hour) <= 1:
        return {"IS": np.nan, "IV": np.nan}

    hour_of_day = np.arange(len(xi_hour)) % 24
    ave_day = (
        pd.DataFrame({"hour_of_day": hour_of_day, "Xi": xi_hour})
        .groupby("hour_of_day", sort=True)["Xi"]
        .mean()
        .to_numpy(dtype=float)
    )

    x_mean = np.nanmean(ave_day)
    denom = np.nansum((xi_hour - x_mean) ** 2)
    n_obs = int(np.sum(~np.isnan(xi_hour)))

    if denom == 0 or n_obs <= 1:
        return {"IS": np.nan, "IV": np.nan}

    is_ = (np.nansum((ave_day - x_mean) ** 2) * n_obs) / (24.0 * denom)
    iv = (np.nansum(np.diff(xi_hour) ** 2) * n_obs) / ((n_obs - 1) * denom)

    return {"IS": float(is_), "IV": float(iv)}


def compute_classic_iv_is(minute_df: pd.DataFrame,
                          value_col: str = "enmo_mg_imputed") -> Dict[str, float]:
    """
    Classical nonparametric IV and IS formulas on a 1-minute series.
    """
    x = minute_df[value_col].to_numpy(dtype=float)

    if np.any(~np.isfinite(x)):
        raise ValueError("The imputed minute-level series still contains NaN/inf values.")

    n = len(x)
    x_mean = x.mean()

    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return {"IV_classic": np.nan, "IS_classic": np.nan}

    iv = (np.sum((x[1:] - x[:-1]) ** 2) / (n - 1)) / (denom / n)

    mean_24h = (
        minute_df.groupby("minute_of_day")[value_col]
        .mean()
        .reindex(range(1440))
        .to_numpy()
    )

    if np.any(~np.isfinite(mean_24h)):
        raise ValueError("The 24h profile is incomplete; cannot compute IS_classic.")

    is_ = (n * np.sum((mean_24h - x_mean) ** 2)) / (1440 * denom)

    return {"IV_classic": iv, "IS_classic": is_}


def _circular_window_mean(profile: np.ndarray, window_minutes: int) -> np.ndarray:
    """
    Rolling mean on a circular 24h profile.
    """
    if len(profile) != 1440:
        raise ValueError("Profile must have length 1440.")
    if np.any(~np.isfinite(profile)):
        raise ValueError("Profile contains NaN/inf; cannot compute circular windows.")

    extended = np.concatenate([profile, profile[:window_minutes - 1]])
    return np.array([
        extended[i:i + window_minutes].mean()
        for i in range(1440)
    ])


def compute_l5_m10_ra(profile_df: pd.DataFrame,
                      dayborder_hours: float = 0.0) -> Dict[str, float]:
    """
    Compute L5, M10, and RA from a circular mean 24h profile.
    """
    profile = profile_df["mean_24h_enmo_mg"].to_numpy(dtype=float)

    l5_means = _circular_window_mean(profile, 5 * 60)
    m10_means = _circular_window_mean(profile, 10 * 60)

    l5_idx = int(np.argmin(l5_means))
    m10_idx = int(np.argmax(m10_means))

    l5 = float(l5_means[l5_idx])
    m10 = float(m10_means[m10_idx])

    ra = np.nan if (m10 + l5) == 0 else (m10 - l5) / (m10 + l5)

    return {
        "L5": l5,
        "M10": m10,
        "RA": ra,
        "L5_start_hour": ((l5_idx / 60.0) + dayborder_hours) % 24,
        "M10_start_hour": ((m10_idx / 60.0) + dayborder_hours) % 24,
    }


def _time_num_to_clock_string(time_num: float) -> Optional[str]:
    if not np.isfinite(time_num):
        return None

    hr = int(np.floor(time_num))
    minute = int(np.floor((time_num - hr) * 60.0))
    second = int(np.floor((time_num - hr - (minute / 60.0)) * 3600.0))

    if second >= 60:
        second = 59
    if hr >= 24:
        hr -= 24

    return f"{hr}:{minute}:{second}"


def compute_l5_m5(minute_df: pd.DataFrame,
                  value_col: str = "enmo_mg_imputed",
                  window_hours: float = 5.0,
                  step_minutes: int = 10) -> Dict[str, float]:
    """
    Compute GGIR-style L5/M5 summaries from selected valid days.

    The implementation follows the part-6 idea more closely than a 24h mean
    profile: derive L5/M5 inside each valid day/window first, then average the
    daily values and times across the recording.
    """
    per_day_rows: List[Dict[str, float]] = []
    window_minutes = int(round(window_hours * 60.0))

    for _, day_df in minute_df.groupby("analysis_date", sort=True):
        day_df = day_df.sort_index()
        values = day_df[value_col].to_numpy(dtype=float)

        if len(values) < window_minutes or np.any(~np.isfinite(values)):
            continue

        window_length_hours = len(values) / 60.0
        endd = np.floor(window_length_hours * 10.0) / 10.0
        n_windows = int((endd - window_hours) * (60.0 / step_minutes))
        if n_windows < 1:
            continue

        run_means = []
        run_starts = []
        for hri in range(n_windows):
            start_idx = hri * step_minutes
            end_idx = min(start_idx + window_minutes, len(values))
            if end_idx <= start_idx:
                continue
            run_means.append(float(np.mean(values[start_idx:end_idx])))
            run_starts.append(day_df.index[start_idx])

        if not run_means:
            continue

        run_means_arr = np.asarray(run_means, dtype=float)
        l5_pos = int(np.argmin(run_means_arr))
        m5_pos = int(np.argmax(run_means_arr))

        l5_start = pd.Timestamp(run_starts[l5_pos])
        m5_start = pd.Timestamp(run_starts[m5_pos])

        l5_time_num = (
            l5_start.hour +
            (l5_start.minute / 60.0) +
            (l5_start.second / 3600.0)
        )
        if l5_time_num <= 12:
            l5_time_num += 24.0

        m5_time_num = (
            m5_start.hour +
            (m5_start.minute / 60.0) +
            (m5_start.second / 3600.0)
        )

        per_day_rows.append({
            "L5VALUE": float(run_means_arr[l5_pos]),
            "M5VALUE": float(run_means_arr[m5_pos]),
            "L5TIME_num": float(l5_time_num),
            "M5TIME_num": float(m5_time_num),
        })

    if not per_day_rows:
        return {
            "L5VALUE": np.nan,
            "M5VALUE": np.nan,
            "L5TIME_num": np.nan,
            "M5TIME_num": np.nan,
            "L5TIME_clock": None,
            "M5TIME_clock": None,
        }

    per_day = pd.DataFrame(per_day_rows)
    l5_time_num = float(per_day["L5TIME_num"].mean())
    m5_time_num = float(per_day["M5TIME_num"].mean())

    return {
        "L5VALUE": float(per_day["L5VALUE"].mean()),
        "M5VALUE": float(per_day["M5VALUE"].mean()),
        "L5TIME_num": l5_time_num,
        "M5TIME_num": m5_time_num,
        "L5TIME_clock": _time_num_to_clock_string(l5_time_num),
        "M5TIME_clock": _time_num_to_clock_string(m5_time_num),
    }


def classify_intensity_minutes(minute_df: pd.DataFrame,
                               value_col: str = "enmo_mg_imputed",
                               thresholds_mg: Tuple[float, float, float] = (40.0, 100.0, 400.0),
                               output_col: str = "intensity_level") -> pd.DataFrame:
    """
    Label each minute into GGIR-like intensity classes using the supplied ENMO
    thresholds.

    Classification rules mirror GGIR's threshold logic:
    - IN:  value <  threshold.lig
    - LIG: threshold.lig <= value < threshold.mod
    - MOD: threshold.mod <= value < threshold.vig
    - VIG: value >= threshold.vig
    """
    thr_lig, thr_mod, thr_vig = thresholds_mg
    if not (thr_lig < thr_mod < thr_vig):
        raise ValueError("thresholds_mg must be strictly increasing.")

    out = minute_df.copy()
    values = out[value_col].to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError(
            f"Column '{value_col}' contains NaN/inf; classify intensity after imputation."
        )

    labels = np.where(
        values < thr_lig,
        "IN",
        np.where(
            values < thr_mod,
            "LIG",
            np.where(values < thr_vig, "MOD", "VIG"),
        ),
    )
    out[output_col] = pd.Categorical(labels, categories=["IN", "LIG", "MOD", "VIG"], ordered=True)
    out["intensity_threshold_lig_mg"] = float(thr_lig)
    out["intensity_threshold_mod_mg"] = float(thr_mod)
    out["intensity_threshold_vig_mg"] = float(thr_vig)

    return out


def summarize_intensity_levels(minute_df: pd.DataFrame,
                               level_col: str = "intensity_level",
                               imputed_col: str = "was_imputed") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize time spent in each intensity level per analysis day and on average.

    Non-wear handling in this summary:
    - invalid/non-wear minutes have already been imputed before classification
    - total minutes therefore reflect a complete day for each selected day
    - imputed minutes are tracked separately per intensity class to document
      how much of each daily total is driven by imputation
    """
    required_cols = {"analysis_date", level_col, imputed_col}
    missing = required_cols.difference(minute_df.columns)
    if missing:
        raise ValueError(f"minute_df is missing required columns: {sorted(missing)}")

    per_day_rows: List[Dict[str, object]] = []
    level_order = ["IN", "LIG", "MOD", "VIG"]

    for analysis_date, day_df in minute_df.groupby("analysis_date", sort=True):
        row: Dict[str, object] = {
            "analysis_date": analysis_date,
            "n_minutes_total": int(len(day_df)),
            "n_minutes_imputed": int(day_df[imputed_col].sum()),
            "n_minutes_observed": int((~day_df[imputed_col].astype(bool)).sum()),
        }

        for level in level_order:
            level_mask = day_df[level_col] == level
            row[f"dur_{level}_min"] = float(level_mask.sum())
            row[f"dur_{level}_imputed_min"] = float((level_mask & day_df[imputed_col].astype(bool)).sum())
            row[f"dur_{level}_observed_min"] = float((level_mask & ~day_df[imputed_col].astype(bool)).sum())

        row["dur_MVPA_min"] = float(row["dur_MOD_min"] + row["dur_VIG_min"])
        row["dur_MVPA_imputed_min"] = float(row["dur_MOD_imputed_min"] + row["dur_VIG_imputed_min"])
        row["dur_MVPA_observed_min"] = float(row["dur_MOD_observed_min"] + row["dur_VIG_observed_min"])
        per_day_rows.append(row)

    daily_df = pd.DataFrame(per_day_rows)
    if daily_df.empty:
        return daily_df, pd.DataFrame()

    summary = {
        "n_days": int(len(daily_df)),
        "n_minutes_total_mean": float(daily_df["n_minutes_total"].mean()),
        "n_minutes_imputed_mean": float(daily_df["n_minutes_imputed"].mean()),
        "n_minutes_observed_mean": float(daily_df["n_minutes_observed"].mean()),
    }

    for col in daily_df.columns:
        if col == "analysis_date":
            continue
        if col in summary:
            continue
        summary[f"{col}_mean"] = float(daily_df[col].mean())

    summary_df = pd.DataFrame([summary])
    return daily_df, summary_df


def prepare_cosinor_series(minute_df: pd.DataFrame,
                           value_col: str = "enmo_mg") -> pd.DataFrame:
    """
    Prepare a valid-only, non-imputed 1-minute series for cosinor.
    Uses the first valid minute and keeps the maximum integer number of days after it.
    Invalid minutes are kept as NaN.
    """
    series = minute_df[[value_col, "invalid"]].copy().sort_index()

    valid_mask = (~series["invalid"]) & series[value_col].notna()
    if not valid_mask.any():
        raise ValueError("No valid minutes available for cosinor.")

    first_valid = series.index[valid_mask.argmax()]
    series = series.loc[first_valid:].copy()

    total_minutes = int((series.index.max() - first_valid) / pd.Timedelta(minutes=1)) + 1
    n_full_days = total_minutes // 1440
    if n_full_days < 1:
        raise ValueError("Not enough valid recording span for at least one full day of cosinor.")

    end_time = first_valid + pd.Timedelta(minutes=n_full_days * 1440 - 1)
    full_index = pd.date_range(start=first_valid, end=end_time, freq="1min")

    series = series.reindex(full_index)
    series["invalid"] = series["invalid"].fillna(True) | series[value_col].isna()
    series["cosinor_input"] = np.where(
        series["invalid"],
        np.nan,
        np.log1p(series[value_col].astype(float))
    )

    return series


def fit_cosinor(cosinor_df: pd.DataFrame,
                period_minutes: int = 1440) -> Dict[str, object]:
    """
    Standard 24h linear cosinor:
        y = MESOR + a*cos(wt) + b*sin(wt)
    fitted on the valid rows only of cosinor_input = log1p(enmo_mg).
    """
    y = cosinor_df["cosinor_input"].to_numpy(dtype=float)
    t = np.arange(len(y), dtype=float)
    valid = np.isfinite(y)

    if valid.sum() < 10:
        raise ValueError("Too few valid points for cosinor fit.")

    omega = 2 * np.pi / period_minutes
    X = np.column_stack([
        np.ones(valid.sum()),
        np.cos(omega * t[valid]),
        np.sin(omega * t[valid]),
    ])
    beta, *_ = np.linalg.lstsq(X, y[valid], rcond=None)

    mesor = float(beta[0])
    a = float(beta[1])
    b = float(beta[2])

    amplitude = float(np.sqrt(a ** 2 + b ** 2))
    phase_rad = float(np.arctan2(-b, a))
    acrophase_min_from_start = float(((-phase_rad) / omega) % period_minutes)

    start_ts = pd.Timestamp(cosinor_df.index[0])
    start_clock_min = cosinor_df.index[0].hour * 60 + cosinor_df.index[0].minute
    acrotime_hour = ((start_clock_min + acrophase_min_from_start) / 60.0) % 24
    acrophase_clock_rad = float((acrotime_hour / 24.0) * (2.0 * np.pi))

    next_midnight = start_ts.normalize()
    if start_ts != next_midnight:
        next_midnight = next_midnight + pd.Timedelta(days=1)
    time_offset_hours = float((next_midnight - start_ts).total_seconds() / 3600.0)

    fitted_log = mesor + a * np.cos(omega * t) + b * np.sin(omega * t)
    fitted_mg = np.expm1(fitted_log)

    return {
        "time_offset_hours": time_offset_hours,
        "MESOR_log1p_mg": mesor,
        "Amplitude_log1p_mg": amplitude,
        "Phase_rad_series_start": phase_rad,
        "Acrophase_rad_clock": acrophase_clock_rad,
        "Acrophase_min_from_series_start": acrophase_min_from_start,
        "Acrotime_hour": acrotime_hour,
        "fitted_log": fitted_log,
        "fitted_mg": fitted_mg,
        "n_minutes": len(cosinor_df),
        "n_valid_minutes": int(valid.sum()),
        "n_days_used": len(cosinor_df) / 1440.0,
    }


def plot_nonparametric_profile(profile_df: pd.DataFrame,
                               l5_m5: Dict[str, float]) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(profile_df["clock_hour"], profile_df["mean_24h_enmo_mg"])
    if np.isfinite(l5_m5.get("L5TIME_num", np.nan)):
        plt.axvline(l5_m5["L5TIME_num"] % 24.0, linestyle="--")
    if np.isfinite(l5_m5.get("M5TIME_num", np.nan)):
        plt.axvline(l5_m5["M5TIME_num"] % 24.0, linestyle="--")
    plt.xlim(0, 24)
    plt.xlabel("Clock time (hours)")
    plt.ylabel("Mean ENMO (mg)")
    plt.title("Mean 24h activity profile")
    plt.tight_layout()
    plt.show()


def plot_cosinor(cosinor_df: pd.DataFrame,
                 cosinor_out: Dict[str, object],
                 max_days_to_plot: int = 7) -> None:
    n_plot = min(len(cosinor_df), max_days_to_plot * 1440)
    plot_idx = np.arange(n_plot)

    plt.figure(figsize=(12, 4))
    plt.plot(plot_idx / 60.0, cosinor_df["cosinor_input"].to_numpy()[:n_plot], alpha=0.5, label="log1p(ENMO mg), valid only")
    plt.plot(plot_idx / 60.0, cosinor_out["fitted_log"][:n_plot], linewidth=2, label="Cosinor fit")
    plt.xlabel("Hours since cosinor series start")
    plt.ylabel("log1p(ENMO mg)")
    plt.title("Cosinor fit")
    plt.legend()
    plt.tight_layout()
    plt.show()


def circadian_pipeline(file_path: str,
                       fs: int = 100,
                       device: str = "GENEActiv",
                       save_folder: Optional[str] = None,
                       nonwear_method: str = "vanhees2013", # "vanhees2013"/"ggir" or "detach"/"nimbaldetach"
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
    GGIR-inspired circadian pipeline.

    Main design choices
    -------------------
    - One minute is the core analysis grid.
    - Invalid minutes are imputed only for nonparametric/profile summaries.
    - Cosinor uses the non-imputed valid-only series.
    - L5/M5 are computed per valid day/window and then averaged.
    """
    if save_folder is None:
        save_folder = str(Path(file_path).parent)
    os.makedirs(save_folder, exist_ok=True)

    calibrated_df, info = load_and_calibrate_accelerometer(file_path, device=device)
    nonwear_df = detect_nonwear_intervals(calibrated_df, fs=fs, method=nonwear_method)
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
    }

    if not intensity_summary_df.empty:
        for col, value in intensity_summary_df.iloc[0].items():
            metrics[col] = value

    for i, row in valid_day_summary.reset_index(drop=True).iterrows():
        day = row["analysis_date"]
        metrics[f"day_{i+1}_date"] = str(day)
        metrics[f"day_{i+1}_invalid_min"] = int(row["invalid_minutes"])
        metrics[f"day_{i+1}_imputed_min"] = int(imputed_minutes_per_day.get(day, 0))

    metrics_df = pd.DataFrame([metrics])

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


GGIR_PART6_METRICS = [
    "cosinor_mes",
    "cosinor_amp",
    "cosinor_acrophase",
    "cosinor_acrotime",
    "cosinor_ndays",
    "IS",
    "IV",
    "L5VALUE",
    "M5VALUE",
    "L5TIME_num",
    "M5TIME_num",
    "L5TIME_clock",
    "M5TIME_clock",
]


def load_ggir_part6_metrics(rdata_path: str) -> pd.Series:
    """
    Load the GGIR part-6 summary metrics stored in ms6.out/*.RData.
    """
    import pyreadr

    out = pyreadr.read_r(rdata_path)["output_part6"]
    metrics = out.iloc[0].copy()
    metrics["cosinor_days"] = metrics.get("cosinor_ndays", np.nan)
    return metrics


def load_ggir_part5_intensity(ms5_rdata_path: str,
                              window_type: str = "MM",
                              drop_incomplete_days: bool = True) -> pd.DataFrame:
    """
    Load GGIR part-5 intensity output and derive full-window intensity totals.

    For MM windows, the derived totals correspond to a full 24 h day by combining
    daytime and SPT components:
    - IN  = day inactivity + SPT wake inactivity + SPT sleep
    - LIG = day light + SPT wake light
    - MOD = day moderate + SPT wake moderate
    - VIG = day vigorous + SPT wake vigorous
    """
    import pyreadr

    out = pyreadr.read_r(ms5_rdata_path)["output"].copy()
    out = out.loc[out["window"] == window_type].copy()

    numeric_cols = [
        "nonwear_perc_day",
        "dur_spt_sleep_min",
        "dur_spt_wake_IN_min",
        "dur_spt_wake_LIG_min",
        "dur_spt_wake_MOD_min",
        "dur_spt_wake_VIG_min",
        "dur_day_total_IN_min",
        "dur_day_total_LIG_min",
        "dur_day_total_MOD_min",
        "dur_day_total_VIG_min",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["analysis_date"] = pd.to_datetime(out["calendar_date"]).dt.date
    out["dur_full_total_IN_min"] = (
        out["dur_day_total_IN_min"] +
        out["dur_spt_wake_IN_min"] +
        out["dur_spt_sleep_min"]
    )
    out["dur_full_total_LIG_min"] = out["dur_day_total_LIG_min"] + out["dur_spt_wake_LIG_min"]
    out["dur_full_total_MOD_min"] = out["dur_day_total_MOD_min"] + out["dur_spt_wake_MOD_min"]
    out["dur_full_total_VIG_min"] = out["dur_day_total_VIG_min"] + out["dur_spt_wake_VIG_min"]
    out["dur_full_total_MVPA_min"] = out["dur_full_total_MOD_min"] + out["dur_full_total_VIG_min"]
    out["dur_full_total_min"] = (
        out["dur_full_total_IN_min"] +
        out["dur_full_total_LIG_min"] +
        out["dur_full_total_MOD_min"] +
        out["dur_full_total_VIG_min"]
    )

    if drop_incomplete_days:
        out = out.loc[np.isclose(out["dur_full_total_min"], 1440.0)].copy()

    keep_cols = [
        "analysis_date",
        "calendar_date",
        "window_number",
        "nonwear_perc_day",
        "dur_full_total_IN_min",
        "dur_full_total_LIG_min",
        "dur_full_total_MOD_min",
        "dur_full_total_VIG_min",
        "dur_full_total_MVPA_min",
        "dur_full_total_min",
    ]
    return out[keep_cols].reset_index(drop=True)


def compare_intensity_against_ggir(intensity_daily_df: pd.DataFrame,
                                   ggir_ms5_rdata_path: str,
                                   window_type: str = "MM") -> Dict[str, pd.DataFrame]:
    """
    Compare Python full-day intensity totals against GGIR part-5 output on
    overlapping dates.
    """
    ggir_daily = load_ggir_part5_intensity(
        ms5_rdata_path=ggir_ms5_rdata_path,
        window_type=window_type,
        drop_incomplete_days=True,
    )

    py_daily = intensity_daily_df.copy()
    py_daily["analysis_date"] = pd.to_datetime(py_daily["analysis_date"]).dt.date
    py_daily = py_daily.rename(
        columns={
            "dur_IN_min": "python_dur_full_total_IN_min",
            "dur_LIG_min": "python_dur_full_total_LIG_min",
            "dur_MOD_min": "python_dur_full_total_MOD_min",
            "dur_VIG_min": "python_dur_full_total_VIG_min",
            "dur_MVPA_min": "python_dur_full_total_MVPA_min",
        }
    )

    merged = py_daily.merge(
        ggir_daily,
        on="analysis_date",
        how="inner",
        suffixes=("", "_ggir"),
    )

    comparison_cols = [
        ("dur_full_total_IN_min", "python_dur_full_total_IN_min", "dur_full_total_IN_min"),
        ("dur_full_total_LIG_min", "python_dur_full_total_LIG_min", "dur_full_total_LIG_min"),
        ("dur_full_total_MOD_min", "python_dur_full_total_MOD_min", "dur_full_total_MOD_min"),
        ("dur_full_total_VIG_min", "python_dur_full_total_VIG_min", "dur_full_total_VIG_min"),
        ("dur_full_total_MVPA_min", "python_dur_full_total_MVPA_min", "dur_full_total_MVPA_min"),
    ]

    per_day_rows: List[Dict[str, object]] = []
    for _, row in merged.iterrows():
        for metric, py_col, ggir_col in comparison_cols:
            per_day_rows.append({
                "analysis_date": row["analysis_date"],
                "metric": metric,
                "python": float(row[py_col]),
                "ggir": float(row[ggir_col]),
                "difference": float(row[py_col] - row[ggir_col]),
                "abs_difference": float(abs(row[py_col] - row[ggir_col])),
                "ggir_nonwear_perc_day": float(row["nonwear_perc_day"]),
            })

    per_day_df = pd.DataFrame(per_day_rows)

    summary_rows = []
    for metric, py_col, ggir_col in comparison_cols:
        py_mean = merged[py_col].mean()
        ggir_mean = merged[ggir_col].mean()
        summary_rows.append({
            "metric": metric,
            "n_overlap_days": int(len(merged)),
            "python_mean": float(py_mean),
            "ggir_mean": float(ggir_mean),
            "difference": float(py_mean - ggir_mean),
            "abs_difference": float(abs(py_mean - ggir_mean)),
        })

    summary_df = pd.DataFrame(summary_rows)
    return {
        "ggir_daily_df": ggir_daily,
        "comparison_per_day_df": per_day_df,
        "comparison_summary_df": summary_df,
    }


def compare_against_ggir(metrics_df: pd.DataFrame,
                         ggir_rdata_path: str) -> pd.DataFrame:
    """
    Compare one-row Python circadian metrics against GGIR part-6 output.
    """
    if len(metrics_df) != 1:
        raise ValueError("metrics_df must contain exactly one row.")

    python_metrics = metrics_df.iloc[0]
    ggir_metrics = load_ggir_part6_metrics(ggir_rdata_path)

    rows = []
    metric_names = [
        "cosinor_mes",
        "cosinor_amp",
        "cosinor_acrophase",
        "cosinor_acrotime",
        "cosinor_days",
        "IS",
        "IV",
        "L5VALUE",
        "M5VALUE",
        "L5TIME_num",
        "M5TIME_num",
        "L5TIME_clock",
        "M5TIME_clock",
    ]

    for metric in metric_names:
        ggir_key = "cosinor_ndays" if metric == "cosinor_days" else metric
        python_value = python_metrics.get(metric, np.nan)
        ggir_value = ggir_metrics.get(ggir_key, np.nan)

        numeric = (
            pd.api.types.is_number(python_value) and
            pd.api.types.is_number(ggir_value)
        )

        rows.append({
            "metric": metric,
            "python": python_value,
            "ggir": ggir_value,
            "difference": (python_value - ggir_value) if numeric else np.nan,
            "abs_difference": abs(python_value - ggir_value) if numeric else np.nan,
        })

    return pd.DataFrame(rows)

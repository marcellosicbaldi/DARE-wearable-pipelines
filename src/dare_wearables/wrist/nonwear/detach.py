from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import polars as pl
from scipy.signal import butter, filtfilt


@dataclass
class DetachResult:
    nonwear_df: pl.DataFrame      # columns: start, end (Datetime)
    mask: np.ndarray              # accel-length boolean mask (debug)
    temp_level_df: pl.DataFrame   # temp-rate table used for decisions (debug)


def _lowpass(temp: np.ndarray, fs: float, cutoff_hz: float = 0.005, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    wn = cutoff_hz / nyq
    b, a = butter(N=order, Wn=wn, btype="lowpass")
    return filtfilt(b, a, x=temp)


def _prepare_temp_timeline(
    acc_df: pl.DataFrame,
    *,
    ts_col: str,
    temp_col_in_acc: Optional[str],
    temp_df: Optional[pl.DataFrame],
    temp_ts_col: str,
    temp_value_col: str,
    temperature_freq: float,
) -> pl.DataFrame:
    """
    Build a temperature timeline with ~1/temperature_freq spacing.
    - If temp_df is provided: resample it to regular bins.
    - Else: downsample temperature from acc_df (temperature stored at accel rate).
    """
    every_sec = float(1.0 / temperature_freq)
    # Use millisecond resolution string to be safe for non-integer seconds (if ever)
    every = f"{int(round(every_sec * 1000))}ms"

    if temp_df is not None:
        tdf = temp_df.select([
            pl.col(temp_ts_col).alias("ts_temp"),
            pl.col(temp_value_col).cast(pl.Float64).alias("temperature"),
        ]).sort("ts_temp")

        # Resample to regular bins (mean temperature per bin)
        tdf = (
            tdf.group_by_dynamic(
                index_column="ts_temp",
                every=every,
                period=every,
                closed="left",
                label="left",
            )
            .agg(pl.col("temperature").mean().alias("temperature"))
            .drop_nulls(["temperature"])
            .sort("ts_temp")
        )
        return tdf

    if temp_col_in_acc is None:
        raise ValueError("Provide either temp_df or temp_col_in_acc (temperature column in acc_df).")

    tdf = (
        acc_df.select([pl.col(ts_col).alias("ts_temp"), pl.col(temp_col_in_acc).alias("temperature")])
        .sort("ts_temp")
        .group_by_dynamic(
            index_column="ts_temp",
            every=every,
            period=every,
            closed="left",
            label="left",
        )
        .agg(pl.col("temperature").mean().cast(pl.Float64).alias("temperature"))
        .drop_nulls(["temperature"])
        .sort("ts_temp")
    )
    return tdf


def detach(
    acc_df: pl.DataFrame,
    *,
    # accel columns
    ts_col: str = "ts",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    accel_freq: float = 100.0,
    # temperature: EITHER provide temp_col_in_acc OR temp_df if temperature is a separate stream
    temp_col_in_acc: Optional[str] = "temperature",
    temp_df: Optional[pl.DataFrame] = None,
    temp_ts_col: str = "ts",
    temp_value_col: str = "temperature",
    temperature_freq: float = 0.25,
    # DETACH params
    std_thresh_mg: float = 8.0,
    low_temperature_cutoff: float = 26.0,
    high_temperature_cutoff: float = 30.0,
    temp_dec_roc: float = -0.2,
    temp_inc_roc: float = 0.1,
    num_axes: int = 2,
    quiet: bool = False,
    return_debug: bool = True,
) -> DetachResult:
    """
    DETACH nonwear detection returning a Polars nonwear_df (start/end timestamps)

    Works with:
    1) temperature at accel rate (temp_col_in_acc in acc_df), OR
    2) separate low-rate temperature stream (temp_df with ts + temperature).
    """
    for c in (ts_col, x_col, y_col, z_col):
        if c not in acc_df.columns:
            raise ValueError(f"Missing required column in acc_df: '{c}'")

    acc_df = acc_df.sort(ts_col)
    n = acc_df.height
    if n == 0:
        empty = pl.DataFrame(schema={"start": pl.Datetime("ns"), "end": pl.Datetime("ns")})
        return DetachResult(nonwear_df=empty, mask=np.zeros(0, dtype=bool), temp_level_df=pl.DataFrame())

    if not quiet:
        print("Starting DETACH (polars, timestamp-aligned)...")

    std_thresh_g = std_thresh_mg / 1000.0

    # --- 1) Accel-derived features at accel rate (same logic as official: 1min std, 5min forward/back percent) ---
    win_1m = int(round(accel_freq * 60))
    win_5m = int(round(accel_freq * 60 * 5))

    # backward-looking 1-min std
    feat = acc_df.with_columns([
        pl.col(x_col).rolling_std(win_1m, min_periods=win_1m).alias("x_std_back"),
        pl.col(y_col).rolling_std(win_1m, min_periods=win_1m).alias("y_std_back"),
        pl.col(z_col).rolling_std(win_1m, min_periods=win_1m).alias("z_std_back"),
    ])

    # forward-looking 1-min std (reverse trick)
    fwd = (
        acc_df.reverse()
        .with_columns([
            pl.col(x_col).rolling_std(win_1m, min_periods=win_1m).alias("x_std_fwd"),
            pl.col(y_col).rolling_std(win_1m, min_periods=win_1m).alias("y_std_fwd"),
            pl.col(z_col).rolling_std(win_1m, min_periods=win_1m).alias("z_std_fwd"),
        ])
        .reverse()
        .select(["x_std_fwd", "y_std_fwd", "z_std_fwd"])
    )
    feat = feat.hstack(fwd)

    feat = feat.with_columns([
        (
            (pl.col("x_std_fwd") < std_thresh_g).cast(pl.Int8) +
            (pl.col("y_std_fwd") < std_thresh_g).cast(pl.Int8) +
            (pl.col("z_std_fwd") < std_thresh_g).cast(pl.Int8)
        ).alias("num_axes_fwd"),
        (
            (pl.col("x_std_back") < std_thresh_g).cast(pl.Int8) +
            (pl.col("y_std_back") < std_thresh_g).cast(pl.Int8) +
            (pl.col("z_std_back") < std_thresh_g).cast(pl.Int8)
        ).alias("num_axes_back"),
    ])

    feat = feat.with_columns([
        (
            (pl.col("num_axes_fwd") >= num_axes).cast(pl.Float32)
            .reverse()
            .rolling_mean(win_5m, min_periods=win_5m)
            .reverse()
        ).alias("perc_next5m_fwd"),
        (
            (pl.col("num_axes_back") >= num_axes).cast(pl.Float32)
            .reverse()
            .rolling_mean(win_5m, min_periods=win_5m)
            .reverse()
        ).alias("perc_next5m_back"),
    ]).select([pl.col(ts_col), "num_axes_fwd", "num_axes_back", "perc_next5m_fwd", "perc_next5m_back"]).sort(ts_col)

    # --- 2) Build temperature timeline (either downsample from accel or use separate stream) ---
    temp_tl = _prepare_temp_timeline(
        acc_df,
        ts_col=ts_col,
        temp_col_in_acc=temp_col_in_acc if temp_df is None else None,
        temp_df=temp_df,
        temp_ts_col=temp_ts_col,
        temp_value_col=temp_value_col,
        temperature_freq=temperature_freq,
    )
    if temp_tl.height < 3:
        empty = pl.DataFrame(schema={"start": pl.Datetime("ns"), "end": pl.Datetime("ns")})
        return DetachResult(nonwear_df=empty, mask=np.zeros(n, dtype=bool), temp_level_df=temp_tl)

    # --- 3) Smooth temperature + ROC (numpy; we assume approx regular after resampling) ---
    temp_vals = temp_tl.select("temperature").to_numpy().ravel().astype(float)
    sm = _lowpass(temp_vals, fs=temperature_freq, cutoff_hz=0.005, order=2)

    # deg/min; avoid prepend=1 bug
    roc = np.diff(sm, prepend=sm[0]) * 60.0 * temperature_freq

    temp_tl = temp_tl.with_columns([
        pl.Series("smoothed_temp", sm),
        pl.Series("temp_deg_per_min", roc),
    ])

    win_5m_temp = int(round(5 * 60 * temperature_freq))

    temp_tl = temp_tl.with_columns([
        pl.col("smoothed_temp").reverse().rolling_max(win_5m_temp, min_periods=win_5m_temp).reverse().alias("max_temp_next5m"),
        pl.col("smoothed_temp").reverse().rolling_min(win_5m_temp, min_periods=win_5m_temp).reverse().alias("min_temp_next5m"),
        pl.col("temp_deg_per_min").reverse().rolling_mean(win_5m_temp, min_periods=win_5m_temp).reverse().alias("mean_5m_temp_change"),
    ])

    # --- 4) Align accel features to temperature timeline (robust: join on epoch ms integers) ---
    tol_ms = 10_000  # 10 seconds tolerance

    feat_ms = (
        feat.with_columns(pl.col(ts_col).dt.epoch("ms").alias("ts_ms"))
            .select(["ts_ms", "num_axes_fwd", "num_axes_back", "perc_next5m_fwd", "perc_next5m_back"])
            .sort("ts_ms")
    )

    temp_ms = (
        temp_tl.with_columns(pl.col("ts_temp").dt.epoch("ms").alias("ts_ms"))
            .sort("ts_ms")
    )

    temp_level = (
        temp_ms.join_asof(
            feat_ms,
            left_on="ts_ms",
            right_on="ts_ms",
            strategy="backward",
            tolerance=tol_ms,   # integer tolerance (ms)
        )
        .drop("ts_ms")
        .sort("ts_temp")
    )

    # --- 5) DETACH start/end logic on temperature timeline ---
    cand = temp_level.select(
        ((pl.col("num_axes_fwd") >= num_axes) & (pl.col("perc_next5m_fwd") >= 0.9)).alias("cand")
    ).to_numpy().ravel().astype(bool)
    candidate_idx = np.where(cand)[0]

    e1 = temp_level.select(
        (
            (pl.col("num_axes_back") == 0) &
            (pl.col("perc_next5m_back") <= 0.50) &
            (pl.col("mean_5m_temp_change") > temp_inc_roc)
        ).alias("e1")
    ).to_numpy().ravel().astype(bool)

    e2 = temp_level.select(
        (
            (pl.col("num_axes_back") == 0) &
            (pl.col("perc_next5m_back") <= 0.50) &
            (pl.col("min_temp_next5m") > low_temperature_cutoff)
        ).alias("e2")
    ).to_numpy().ravel().astype(bool)

    end_crit_combined = np.sort(np.unique(np.concatenate([np.where(e1)[0], np.where(e2)[0]])))

    max_temp_next5m = temp_level.select("max_temp_next5m").to_numpy().ravel()
    mean_5m_temp_change = temp_level.select("mean_5m_temp_change").to_numpy().ravel()
    temp_ts = temp_level.select("ts_temp").to_numpy().ravel()

    acc_ts = acc_df.select(ts_col).to_numpy().ravel()

    nonwear_mask = np.zeros(n, dtype=bool)
    start_idx_list: List[int] = []
    end_idx_list: List[int] = []

    previous_end_temp = -1
    last_temp_index = temp_level.height - 1

    for ind in candidate_idx:
        if ind < previous_end_temp:
            continue

        start_ind = int(ind)
        end_ind_min = int(ind + win_5m_temp)  # at least 5 min ahead

        # start criteria
        valid_start = False
        if (max_temp_next5m[start_ind] < high_temperature_cutoff) and (mean_5m_temp_change[start_ind] < temp_dec_roc):
            valid_start = True
        elif max_temp_next5m[start_ind] < low_temperature_cutoff:
            valid_start = True
        if not valid_start:
            continue

        # end criterion: first end point after start+5min
        end_candidates = end_crit_combined[end_crit_combined > end_ind_min]
        bout_end_temp = int(end_candidates[0]) if end_candidates.size > 0 else last_temp_index

        start_time = temp_ts[start_ind]
        end_time = temp_ts[bout_end_temp]

        # map times to accel indices (inclusive end)
        a0 = int(np.searchsorted(acc_ts, start_time, side="left"))
        a1_excl = int(np.searchsorted(acc_ts, end_time, side="right"))
        if a1_excl <= a0:
            previous_end_temp = bout_end_temp
            continue

        nonwear_mask[a0:a1_excl] = True
        start_idx_list.append(a0)
        end_idx_list.append(a1_excl - 1)  # inclusive end index

        previous_end_temp = bout_end_temp

    # --- 6) Build the nonwear_df that VH2018 expects ---
    ts_dtype = acc_df.schema[ts_col]

    if len(start_idx_list) == 0:
        nonwear_df = pl.DataFrame(schema={"start": ts_dtype, "end": ts_dtype})
    else:
        idx_df = pl.DataFrame(
            {"start_idx": start_idx_list, "end_idx": end_idx_list},
            schema={"start_idx": pl.Int64, "end_idx": pl.Int64},
        )

        lookup_start = acc_df.select(pl.col(ts_col).alias("start")).with_row_count("start_idx")
        lookup_end   = acc_df.select(pl.col(ts_col).alias("end")).with_row_count("end_idx")

        nonwear_df = (
            idx_df
            .join(lookup_start, on="start_idx", how="left")
            .join(lookup_end, on="end_idx", how="left")
            .select([
                pl.col("start").cast(ts_dtype),
                pl.col("end").cast(ts_dtype),
            ])
            .drop_nulls()
            .sort("start")
        )

    if not quiet:
        print("Finished DETACH (polars).")

    if return_debug:
        return DetachResult(nonwear_df=nonwear_df, mask=nonwear_mask, temp_level_df=temp_level)
    else:
        return DetachResult(nonwear_df=nonwear_df, mask=nonwear_mask, temp_level_df=pl.DataFrame())



""" Example usage:
# Case A — temperature stored at accel rate (your current dataset)
detach = detach(
    acc_df,
    ts_col="ts", x_col="x", y_col="y", z_col="z",
    temp_col_in_acc="temperature",
    temp_df=None,
    accel_freq=75, temperature_freq=0.25,
    return_debug=False
)

nonwear_df = detach.nonwear_df

# Case B — temperature as a separate low-rate stream
detach = detach(
    acc_df,
    ts_col="ts", x_col="x", y_col="y", z_col="z",
    temp_df=temp_df,           # must have columns ts + temperature
    temp_ts_col="ts",
    temp_value_col="temperature",
    accel_freq=75, temperature_freq=0.25,
    return_debug=False
)

nonwear_df = detach.nonwear_df
"""
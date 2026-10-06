from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import polars as pl


@dataclass
class VanHeesNonWearResult:
    nonwear_df: pl.DataFrame        # columns: start, end (Datetime)
    window_df: pl.DataFrame         # debug: per-window stats + flags


def vanhees2013(
    acc_df: pl.DataFrame,
    *,
    ts_col: str = "ts",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    non_wear_window: float = 60.0,         # minutes
    window_step_size: float = 15.0,        # minutes
    std_thresh_mg: float = 13.0,
    value_range_thresh_mg: float = 50.0,
    num_axes_required: int = 2,
    freq: float = 75.0,
    quiet: bool = False,
    return_debug: bool = True,
) -> VanHeesNonWearResult:
    """
    Van Hees / GGIR non-wear:
      - sliding windows of length non_wear_window (min), stepped by window_step_size (min)
      - mark a window as non-wear if >= num_axes_required axes have:
          std < std_thresh OR range < value_range_thresh
      - then apply GGIR-style border criteria to convert short "wear" gaps based on adjacent durations
    Returns:
      - nonwear_df: Polars DF with start/end timestamps (inclusive end) for masking in VH2018
      - window_df: debug DF (optional)
    """
    for c in (ts_col, x_col, y_col, z_col):
        if c not in acc_df.columns:
            raise ValueError(f"Missing required column '{c}'")

    if not quiet:
        print("Starting Van Hees (GGIR) non-wear...")

    acc_df = acc_df.sort(ts_col)
    n = acc_df.height
    if n == 0:
        empty = pl.DataFrame(schema={"start": pl.Datetime("ns"), "end": pl.Datetime("ns")})
        return VanHeesNonWearResult(nonwear_df=empty, window_df=pl.DataFrame())

    # thresholds mg -> g
    std_thresh_g = std_thresh_mg / 1000.0
    range_thresh_g = value_range_thresh_mg / 1000.0

    # window setup
    win_m = float(non_wear_window)
    step_m = float(window_step_size)
    win_str = f"{int(round(win_m))}m"
    step_str = f"{int(round(step_m))}m"

    expected_samples = int(round(win_m * 60.0 * freq))  # for "full window" check

    # Sliding windows via group_by_dynamic (overlapping when period > every)
    wdf = (
        acc_df.group_by_dynamic(
            index_column=ts_col,
            every=step_str,
            period=win_str,
            closed="left",
            label="left",
        )
        .agg([
            pl.len().alias("n"),
            pl.col(x_col).std().alias("x_std"),
            pl.col(y_col).std().alias("y_std"),
            pl.col(z_col).std().alias("z_std"),
            (pl.col(x_col).max() - pl.col(x_col).min()).alias("x_rng"),
            (pl.col(y_col).max() - pl.col(y_col).min()).alias("y_rng"),
            (pl.col(z_col).max() - pl.col(z_col).min()).alias("z_rng"),
        ])
        .rename({ts_col: "win_start"})
        .with_columns([
            (pl.col("win_start") + pl.duration(minutes=win_m)).alias("win_end_excl"),
        ])
        # mimic "drop final partial windows"
        .filter(pl.col("n") >= expected_samples)
        .with_columns([
            (
                (pl.col("x_std") < std_thresh_g).cast(pl.Int8) +
                (pl.col("y_std") < std_thresh_g).cast(pl.Int8) +
                (pl.col("z_std") < std_thresh_g).cast(pl.Int8)
            ).alias("std_axes_count"),
            (
                (pl.col("x_rng") < range_thresh_g).cast(pl.Int8) +
                (pl.col("y_rng") < range_thresh_g).cast(pl.Int8) +
                (pl.col("z_rng") < range_thresh_g).cast(pl.Int8)
            ).alias("rng_axes_count"),
        ])
        .with_columns([
            (
                (pl.col("std_axes_count") >= num_axes_required) |
                (pl.col("rng_axes_count") >= num_axes_required)
            ).alias("is_nonwear_window")
        ])
        .sort("win_start")
    )

    # No nonwear windows → empty
    nw_windows = wdf.filter(pl.col("is_nonwear_window"))
    if nw_windows.height == 0:
        empty = pl.DataFrame(schema={"start": acc_df.schema[ts_col], "end": acc_df.schema[ts_col]})
        return VanHeesNonWearResult(nonwear_df=empty, window_df=(wdf if return_debug else pl.DataFrame()))

    # Map window start/end to sample indices using epoch ns (avoid object datetimes)
    ts_ns = acc_df.select(pl.col(ts_col).dt.epoch("ns")).to_numpy().ravel()

    win_start_ns = nw_windows.select(pl.col("win_start").dt.epoch("ns")).to_numpy().ravel()
    win_end_ns = nw_windows.select(pl.col("win_end_excl").dt.epoch("ns")).to_numpy().ravel()

    # Convert windows to index-intervals [a0, a1_excl)
    intervals: List[tuple[int, int]] = []
    for s_ns, e_ns in zip(win_start_ns, win_end_ns):
        a0 = int(np.searchsorted(ts_ns, s_ns, side="left"))
        a1 = int(np.searchsorted(ts_ns, e_ns, side="left"))  # exclusive end
        if a1 > a0:
            intervals.append((a0, a1))

    if not intervals:
        empty = pl.DataFrame(schema={"start": acc_df.schema[ts_col], "end": acc_df.schema[ts_col]})
        return VanHeesNonWearResult(nonwear_df=empty, window_df=(wdf if return_debug else pl.DataFrame()))

    # Merge overlapping index-intervals
    intervals.sort()
    merged: List[tuple[int, int]] = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:   # overlap/adjacent
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))

    # Build bout list across full recording: (is_nonwear, start_idx, end_idx_excl)
    bouts: List[tuple[bool, int, int]] = []
    cur = 0
    for s, e in merged:
        if cur < s:
            bouts.append((False, cur, s))  # wear
        bouts.append((True, s, e))          # nonwear
        cur = e
    if cur < n:
        bouts.append((False, cur, n))

    # Border criteria (same rules as your pandas code), applied on bout durations
    # duration in minutes
    durs = np.array([(b[2] - b[1]) / (freq * 60.0) for b in bouts], dtype=float)

    converted = [b[0] for b in bouts]  # mutable is_nonwear flags
    for i, (is_nw, s, e) in enumerate(bouts):
        if is_nw:
            continue  # only consider wear bouts
        dur = durs[i]
        prev_d = durs[i - 1] if i > 0 else 0.0
        next_d = durs[i + 1] if i < (len(bouts) - 1) else 0.0
        adj = prev_d + next_d
        if adj <= 0:
            continue
        ratio = dur / adj

        if dur <= 180:
            if ratio < 0.8:
                converted[i] = True
        elif dur <= 360:
            if ratio < 0.3:
                converted[i] = True

    # Merge consecutive nonwear bouts after conversion → final nonwear intervals
    final_intervals: List[tuple[int, int]] = []
    i = 0
    while i < len(bouts):
        is_nw = converted[i]
        s, e = bouts[i][1], bouts[i][2]
        if not is_nw:
            i += 1
            continue
        # extend while next bouts also nonwear
        j = i + 1
        end = e
        while j < len(bouts) and converted[j]:
            end = max(end, bouts[j][2])
            j += 1
        final_intervals.append((s, end))
        i = j

    # Build nonwear_df (start/end timestamps, inclusive end) via index lookup (no datetime objects)
    ts_dtype = acc_df.schema[ts_col]
    if len(final_intervals) == 0:
        nonwear_df = pl.DataFrame(schema={"start": ts_dtype, "end": ts_dtype})
    else:
        start_idx_list = [s for s, _ in final_intervals]
        end_idx_list = [max(s, e - 1) for s, e in final_intervals]  # inclusive end index

        idx_df = pl.DataFrame(
            {"start_idx": start_idx_list, "end_idx": end_idx_list},
            schema={"start_idx": pl.Int64, "end_idx": pl.Int64},
        )
        lookup_start = acc_df.select(pl.col(ts_col).alias("start")).with_row_count("start_idx")
        lookup_end = acc_df.select(pl.col(ts_col).alias("end")).with_row_count("end_idx")

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
        print("Finished Van Hees (GGIR) non-wear.")

    return VanHeesNonWearResult(
        nonwear_df=nonwear_df,
        window_df=(wdf if return_debug else pl.DataFrame()),
    )

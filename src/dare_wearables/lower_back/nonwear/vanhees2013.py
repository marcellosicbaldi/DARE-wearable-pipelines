from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    non_wear_window: float = 60.0,
    window_step_size: float = 15.0,
    std_thresh_mg: float = 13.0,
    value_range_thresh_mg: float = 50.0,
    num_axes_required: int = 2,
    freq: Optional[float] = None,
    edge_correction: bool = True,
    quiet: bool = False,
    return_debug: bool = True,
) -> VanHeesNonWearResult:
    """
    Non-wear detection aligned with GGIR's current implementation of the
    2013 Van Hees approach:
      - 15-minute medium epochs
      - 60-minute long windows centred on each medium epoch
      - an axis only counts as non-wear when BOTH range and SD are below threshold
      - single-axis windows surrounded by >=2-axis windows are promoted
      - short wear gaps are relabeled with the iterative `g.weardec` rules
    """
    for c in (ts_col, x_col, y_col, z_col):
        if c not in acc_df.columns:
            raise ValueError(f"Missing required column '{c}'")

    if not quiet:
        print("Starting Van Hees (GGIR) non-wear...")

    acc_df = acc_df.sort(ts_col)

    # Keep timestamps in Polars/Python datetime format.
    # This avoids ns/us unit confusion when building output DataFrames.
    acc_df = acc_df.with_columns(
        pl.col(ts_col).cast(pl.Datetime("us")).alias(ts_col)
    )

    n = acc_df.height
    ts_dtype = pl.Datetime("us")

    if n == 0:
        empty = pl.DataFrame(schema={"start": ts_dtype, "end": ts_dtype})
        return VanHeesNonWearResult(nonwear_df=empty, window_df=pl.DataFrame())

    ts_series = acc_df.get_column(ts_col)

    x = acc_df[x_col].to_numpy()
    y = acc_df[y_col].to_numpy()
    z = acc_df[z_col].to_numpy()

    if freq is None:
        ts_ns = ts_series.dt.epoch("ns").to_numpy()
        if len(ts_ns) < 2:
            raise ValueError("Cannot infer sampling frequency from fewer than 2 timestamps.")
        dt_s = np.median(np.diff(ts_ns)) / 1e9
        if dt_s <= 0:
            raise ValueError("Could not infer a positive sampling frequency from timestamps.")
        freq = 1.0 / dt_s

    std_thresh_g = std_thresh_mg / 1000.0
    range_thresh_g = value_range_thresh_mg / 1000.0

    medium_samples = int(round(window_step_size * 60.0 * freq))
    long_samples = int(round(non_wear_window * 60.0 * freq))
    if medium_samples <= 0 or long_samples <= 0:
        raise ValueError("Window sizes and frequency must produce positive sample counts.")

    n_medium = n // medium_samples
    if n_medium == 0:
        empty = pl.DataFrame(schema={"start": ts_dtype, "end": ts_dtype})
        return VanHeesNonWearResult(nonwear_df=empty, window_df=pl.DataFrame())

    minimum_epoch_count = int((long_samples / medium_samples) / 2 + 1)
    medium_starts = np.arange(n_medium, dtype=int) * medium_samples
    medium_ends_excl = medium_starts + medium_samples

    x_std = np.full(n_medium, np.nan)
    y_std = np.full(n_medium, np.nan)
    z_std = np.full(n_medium, np.nan)
    x_rng = np.full(n_medium, np.nan)
    y_rng = np.full(n_medium, np.nan)
    z_rng = np.full(n_medium, np.nan)
    x_axis = np.zeros(n_medium, dtype=np.int8)
    y_axis = np.zeros(n_medium, dtype=np.int8)
    z_axis = np.zeros(n_medium, dtype=np.int8)
    long_starts = np.zeros(n_medium, dtype=int)
    long_ends_excl = np.zeros(n_medium, dtype=int)

    axis_arrays = (x, y, z)
    std_arrays = (x_std, y_std, z_std)
    rng_arrays = (x_rng, y_rng, z_rng)
    flag_arrays = (x_axis, y_axis, z_axis)

    for h in range(n_medium):
        if h + 1 <= minimum_epoch_count:
            long_start = 0
            long_end_excl = min(long_samples, n)
        elif h + 1 >= (n_medium - minimum_epoch_count):
            long_start = max((n_medium - minimum_epoch_count) * medium_samples, 0)
            long_end_excl = n_medium * medium_samples
        else:
            medium_center = medium_starts[h] + medium_samples * 0.5
            long_start = int(round(medium_center - long_samples * 0.5))
            long_end_excl = int(round(medium_center + long_samples * 0.5))

        long_start = max(long_start, 0)
        long_end_excl = min(long_end_excl, n)
        long_starts[h] = long_start
        long_ends_excl[h] = long_end_excl

        for axis_values, std_store, rng_store, flag_store in zip(
            axis_arrays, std_arrays, rng_arrays, flag_arrays
        ):
            segment = axis_values[long_start:long_end_excl]
            if segment.size == 0:
                continue

            axis_range = float(np.nanmax(segment) - np.nanmin(segment))
            axis_std = float(np.nanstd(segment, ddof=1)) if segment.size > 1 else 0.0

            rng_store[h] = axis_range
            std_store[h] = axis_std
            if axis_range < range_thresh_g and axis_std < std_thresh_g:
                flag_store[h] = 1

    score = x_axis + y_axis + z_axis
    isolated_one = np.where(score == 1)[0]
    for idx in isolated_one:
        if idx == 0 or idx == (len(score) - 1):
            continue
        if score[idx - 1] > 1 and score[idx + 1] > 1:
            score[idx] = 2

    base_nonwear = score >= num_axes_required

    def _additional_nonwear_detection(s1, s3, *, apply_edge_correction):
        ch1 = np.where(np.diff(s1) == 1)[0] + 1
        ch2 = np.where(np.diff(s1) == -1)[0] + 1
        ws2 = window_step_size * 60.0
        three_h = 3 * 3600.0 / ws2
        six_h = 6 * 3600.0 / ws2
        one_h = 3600.0 / ws2
        twentyfour_h = 24 * 3600.0 / ws2

        if len(ch1) > 1:
            for wear_i in range(len(ch1) - 1):
                wear_len = abs(ch1[wear_i + 1] - ch2[wear_i])
                nw_after = abs(ch2[wear_i + 1] - ch1[wear_i + 1])
                nw_before = abs(ch2[wear_i] - ch1[wear_i])
                nearby = nw_after + nw_before
                if nearby > 0 and wear_len < six_h and (wear_len / nearby) < 0.3:
                    s3[ch2[wear_i] : ch1[wear_i + 1]] = 1
                if nearby > 0 and wear_len < three_h and (wear_len / nearby) < 0.8:
                    s3[ch2[wear_i] : ch1[wear_i + 1]] = 1
                if ch1[wear_i] > (n_medium - twentyfour_h):
                    if wear_len < three_h and nw_before > one_h:
                        s3[ch2[wear_i] : ch1[wear_i + 1]] = 1

        if apply_edge_correction and len(ch1) > 0:
            if ch1[0] < three_h and ch1[0] > 1:
                s3[1:ch1[0]] = 1
            if ch2[-1] > (len(s3) - three_h) and ch2[-1] != len(s3):
                s3[ch2[-1] : len(s3)] = 1

        return s1, s3

    r1 = base_nonwear.astype(np.int8)
    r3 = np.zeros_like(r1)
    r1_pad = np.concatenate(([0], r1, [0]))
    r3_pad = np.concatenate(([0], r3, [0]))
    _, r3_pad = _additional_nonwear_detection(
        r1_pad,
        r3_pad,
        apply_edge_correction=edge_correction,
    )
    r3 = r3_pad[1:-1]

    for _ in range(2):
        r1b = np.clip(r1 + r3, 0, 1)
        r1b_pad = np.concatenate(([0], r1b, [0]))
        r3b_pad = np.concatenate(([0], r3, [0]))
        _, r3b_pad = _additional_nonwear_detection(
            r1b_pad,
            r3b_pad,
            apply_edge_correction=False,
        )
        r3 = r3b_pad[1:-1]

    final_nonwear = (r1 + r3) > 0

    intervals = []
    i = 0
    while i < n_medium:
        if not final_nonwear[i]:
            i += 1
            continue
        start_idx = medium_starts[i]
        j = i + 1
        end_excl = medium_ends_excl[i]
        while j < n_medium and final_nonwear[j]:
            end_excl = medium_ends_excl[j]
            j += 1
        intervals.append((start_idx, end_excl))
        i = j

    if intervals:
        nonwear_df = pl.DataFrame(
            {
                "start": [ts_series.item(int(s)) for s, _ in intervals],
                "end": [ts_series.item(int(e - 1)) for _, e in intervals],
            },
            schema={
                "start": ts_dtype,
                "end": ts_dtype,
            },
        )
    else:
        nonwear_df = pl.DataFrame(schema={"start": ts_dtype, "end": ts_dtype})

    if return_debug:
        window_df = pl.DataFrame(
        {
            "win_start": [ts_series.item(int(i)) for i in medium_starts],
            "win_end": [ts_series.item(int(i - 1)) for i in medium_ends_excl],
            "long_start": [ts_series.item(int(i)) for i in long_starts],
            "long_end": [ts_series.item(int(i - 1)) for i in long_ends_excl],
            "x_std": x_std,
            "y_std": y_std,
            "z_std": z_std,
            "x_rng": x_rng,
            "y_rng": y_rng,
            "z_rng": z_rng,
            "x_axis_nonwear": x_axis.astype(bool),
            "y_axis_nonwear": y_axis.astype(bool),
            "z_axis_nonwear": z_axis.astype(bool),
            "nonwear_score": score,
            "is_nonwear_window": base_nonwear,
            "is_added_nonwear": r3.astype(bool),
            "is_final_nonwear": final_nonwear,
        }
    ).with_columns([
        pl.col("win_start").cast(ts_dtype),
        pl.col("win_end").cast(ts_dtype),
        pl.col("long_start").cast(ts_dtype),
        pl.col("long_end").cast(ts_dtype),
    ])
    else:
        window_df = pl.DataFrame()

    if not quiet:
        print("Finished Van Hees (GGIR) non-wear.")

    return VanHeesNonWearResult(nonwear_df=nonwear_df, window_df=window_df)

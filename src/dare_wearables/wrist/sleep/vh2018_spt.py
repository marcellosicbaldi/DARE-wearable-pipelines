from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from dare_wearables.wrist.sleep.angle_utils import build_anglez_5s_pl


def label_epochs_from_windows(
    epochs: pl.DataFrame,
    windows: pl.DataFrame,
    *,
    ts_col: str = "ts",
    start_col: str = "spt_start",
    end_col: str = "spt_end",
    label_col: str = "in_spt",
) -> pl.DataFrame:
    """
    Label epochs whose timestamp falls inside any SPT window.
    """
    if windows.is_empty():
        return epochs.with_columns(pl.lit(False).alias(label_col))

    ts_dtype = epochs.schema[ts_col]
    epochs_sorted = epochs.sort(ts_col)
    windows_sorted = windows.with_columns(
        [
            pl.col(start_col).cast(ts_dtype),
            pl.col(end_col).cast(ts_dtype),
        ]
    ).sort(start_col)

    joined = epochs_sorted.join_asof(
        windows_sorted,
        left_on=ts_col,
        right_on=start_col,
        strategy="backward",
    )

    return joined.with_columns(
        (
            pl.col(end_col).is_not_null()
            & (pl.col(ts_col) <= pl.col(end_col))
        ).alias(label_col)
    )


def _normalize_haspt_ignore_invalid(value: Any) -> float | bool:
    """
    Match GGIR's three-state logic for HASPT.ignore.invalid.

    - False: use imputed angle/activity during invalid epochs
    - True: invalid epochs cannot contribute to nomovement
    - NA: invalid epochs are treated as nomovement
    """
    if value is None:
        return np.nan
    if isinstance(value, str) and value.strip().lower() in {"na", "nan", "none"}:
        return np.nan
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return np.nan
    return bool(value)


def _rolling_median_absdiff(angle: np.ndarray, *, window_size: int) -> np.ndarray:
    angle = np.asarray(angle, dtype=float)
    if angle.size == 0:
        return angle.copy()

    def medabsdi(window: pl.Series) -> float:
        x = window.to_numpy()
        if len(x) <= 1:
            return 0.0
        return float(np.median(np.abs(np.diff(x))))

    feature = (
        pl.Series("angle", angle)
        .rolling_map(medabsdi, window_size=window_size, min_samples=window_size, center=True)
        .fill_null(0.0)
        .to_numpy()
    )
    return feature.astype(float, copy=False)


def _resolve_hdcza_threshold(
    feature: np.ndarray,
    *,
    hdcza_threshold: float | tuple[float, float] | list[float] | None = None,
) -> float:
    if hdcza_threshold is None:
        hdcza_threshold = (10.0, 15.0)

    if isinstance(hdcza_threshold, (tuple, list, np.ndarray)) and len(hdcza_threshold) == 2:
        percentile = float(hdcza_threshold[0]) / 100.0
        multiplier = float(hdcza_threshold[1])
        threshold = float(np.nanquantile(feature, percentile) * multiplier)
        return max(0.13, min(0.50, threshold))

    return float(hdcza_threshold)


def _rebuild_rle(values: np.ndarray, lengths: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    rebuilt = np.repeat(values, lengths)[:n]
    rle_values: list[int] = []
    rle_lengths: list[int] = []
    prev = None
    count = 0
    for item in rebuilt:
        item = int(item)
        if prev is None or item != prev:
            if prev is not None:
                rle_values.append(prev)
                rle_lengths.append(count)
            prev = item
            count = 1
        else:
            count += 1
    if prev is not None:
        rle_values.append(prev)
        rle_lengths.append(count)
    return np.asarray(rle_values, dtype=int), np.asarray(rle_lengths, dtype=int)


def _encode_rle(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _rebuild_rle(np.asarray(x, dtype=int), np.ones(len(x), dtype=int), len(x))


def _haspt_hdcza_window(
    angle_imputed: np.ndarray,
    invalid: np.ndarray,
    *,
    epoch_seconds: int = 5,
    haspt_ignore_invalid: float | bool = False,
    hdcza_threshold: float | tuple[float, float] | list[float] | None = None,
    spt_min_block_dur_minutes: int = 30,
    spt_max_gap_dur_minutes: int = 60,
    spt_max_gap_ratio: float = 1.0,
) -> dict[str, Any]:
    """
    Apply GGIR-like HDCZA to one 24-hour detection window.
    """
    angle_imputed = np.asarray(angle_imputed, dtype=float)
    invalid = np.asarray(invalid, dtype=bool)
    n = len(angle_imputed)
    if n == 0:
        return {
            "spt_start_idx": None,
            "spt_end_idx": None,
            "threshold": np.nan,
            "part3_guider": "HDCZA",
            "feature_5min": np.asarray([], dtype=float),
            "nomov": np.asarray([], dtype=int),
            "spt_crude_estimate": np.asarray([], dtype=float),
        }

    k1 = int(round(5 * (60 / int(epoch_seconds))))
    feature = _rolling_median_absdiff(angle_imputed, window_size=k1)
    threshold = _resolve_hdcza_threshold(feature, hdcza_threshold=hdcza_threshold)
    ignore_setting = _normalize_haspt_ignore_invalid(haspt_ignore_invalid)

    nomov = np.zeros(n, dtype=int)
    if isinstance(ignore_setting, (float, np.floating)) and np.isnan(ignore_setting):
        nomov[(feature < threshold) | invalid] = 1
    elif ignore_setting is False:
        nomov[feature < threshold] = 1
    else:
        nomov[(feature < threshold) & (~invalid)] = 1

    part3_guider = "HDCZA"
    fraction_invalid = float(invalid.mean()) if n else 1.0
    if fraction_invalid >= 1.0:
        return {
            "spt_start_idx": None,
            "spt_end_idx": None,
            "threshold": threshold,
            "part3_guider": part3_guider,
            "feature_5min": feature,
            "nomov": nomov,
            "spt_crude_estimate": np.full(n, np.nan),
        }

    padded = np.concatenate([[0], nomov, [0]])
    rle_values, rle_lengths = _encode_rle(padded)

    min_block_len = int((60 / int(epoch_seconds)) * int(spt_min_block_dur_minutes))
    remove_idx = np.flatnonzero((rle_values == 1) & (rle_lengths <= min_block_len))
    remove_idx = remove_idx[(remove_idx != 0) & (remove_idx != len(rle_values) - 1)]
    if remove_idx.size:
        rle_values = rle_values.copy()
        rle_values[remove_idx] = 0
        rle_values, rle_lengths = _rebuild_rle(rle_values, rle_lengths, n)

    max_gap_len = int((60 / int(epoch_seconds)) * int(spt_max_gap_dur_minutes))
    n_segments = len(rle_lengths)
    if spt_max_gap_ratio is not None and float(spt_max_gap_ratio) < 1.0 and n_segments > 3:
        length_after = np.concatenate([rle_lengths[1:], [0]])
        length_before = np.concatenate([[0], rle_lengths[:-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_after = np.where(length_after > 0, rle_lengths / length_after, np.inf)
            ratio_before = np.where(length_before > 0, rle_lengths / length_before, np.inf)
        gap_mask = (
            (rle_values == 0)
            & (rle_lengths < max_gap_len)
            & (ratio_after < float(spt_max_gap_ratio))
            & (ratio_before < float(spt_max_gap_ratio))
        )
        gaps_to_fill = np.flatnonzero(gap_mask)
    else:
        gaps_to_fill = np.flatnonzero((rle_values == 0) & (rle_lengths < max_gap_len))

    gaps_to_fill = gaps_to_fill[(gaps_to_fill != 0) & (gaps_to_fill != len(rle_values) - 1)]
    if gaps_to_fill.size:
        rle_values = rle_values.copy()
        rle_values[gaps_to_fill] = 1
        rle_values, rle_lengths = _rebuild_rle(rle_values, rle_lengths, n)

    spt_crude_estimate = np.repeat(rle_values, rle_lengths)[:n].astype(float)

    spt_start_idx = None
    spt_end_idx = None
    if 1 in rle_values:
        max_length = int(rle_lengths[rle_values == 1].max())
        longest = np.flatnonzero((rle_values == 1) & (rle_lengths == max_length))[0]
        keep_values = np.zeros_like(rle_values)
        keep_values[longest] = 1
        spt_estimate = np.repeat(keep_values, rle_lengths)[:n]

        changes = np.diff(np.concatenate([[0], spt_estimate, [0]]))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        if starts.size and ends.size:
            spt_start_idx = int(starts[0])
            spt_end_idx = int(ends[0])
            spt_crude_estimate[spt_start_idx : spt_end_idx + 1] = 2

            if isinstance(ignore_setting, (float, np.floating)) and np.isnan(ignore_setting):
                if invalid[spt_start_idx : spt_end_idx + 1].any():
                    part3_guider = "HDCZA+invalid"

    return {
        "spt_start_idx": spt_start_idx,
        "spt_end_idx": spt_end_idx,
        "threshold": threshold,
        "part3_guider": part3_guider,
        "feature_5min": feature,
        "nomov": nomov,
        "spt_crude_estimate": spt_crude_estimate,
    }


def _prepare_angle_epoch_table_pl(
    angle_5s: pl.DataFrame,
    *,
    ts_col: str = "ts",
    value_col: str = "z_angle_5s",
    nonwear_df: pl.DataFrame | None = None,
    epoch_seconds: int = 5,
    start_col: str = "start",
    end_col: str = "end",
) -> pl.DataFrame:
    if value_col not in angle_5s.columns:
        raise ValueError(f"angle_5s must contain {value_col!r}.")
    if angle_5s.is_empty():
        raise ValueError("angle_5s is empty.")

    ts_dtype = angle_5s.schema[ts_col]
    start_ts = angle_5s.get_column(ts_col)[0]
    end_ts = angle_5s.get_column(ts_col)[-1]

    full_grid = pl.DataFrame(
        {
            ts_col: pl.datetime_range(
                start=start_ts,
                end=end_ts,
                interval=f"{int(epoch_seconds)}s",
                eager=True,
            )
        }
    ).with_columns(pl.col(ts_col).cast(ts_dtype))

    out = (
        full_grid.join(angle_5s.select([ts_col, value_col]), on=ts_col, how="left")
        .with_columns(
            [
                pl.col(value_col).is_null().alias("invalid_missing"),
                pl.lit(False).alias("invalid_nonwear"),
            ]
        )
    )

    if nonwear_df is not None and not nonwear_df.is_empty():
        if not {start_col, end_col}.issubset(set(nonwear_df.columns)):
            raise ValueError(f"nonwear_df must contain {start_col!r} and {end_col!r}.")
        nw = (
            nonwear_df.select(
                [
                    pl.col(start_col).cast(ts_dtype, strict=False).alias("start"),
                    pl.col(end_col).cast(ts_dtype, strict=False).alias("end"),
                ]
            )
            .drop_nulls()
            .filter(pl.col("end") >= pl.col("start"))
            .sort("start")
        )
        if nw.height > 0:
            conds = [
                (pl.col(ts_col) < pl.lit(row["end"]))
                & ((pl.col(ts_col) + pl.duration(seconds=int(epoch_seconds))) > pl.lit(row["start"]))
                for row in nw.to_dicts()
            ]
            if conds:
                out = out.with_columns(
                    pl.fold(pl.lit(False), lambda acc, x: acc | x, conds).alias("invalid_nonwear")
                )

    return (
        out.with_columns(
            [
                (pl.col("invalid_missing") | pl.col("invalid_nonwear")).alias("invalid"),
                (
                    (
                        pl.col(ts_col).dt.hour() * 3600
                        + pl.col(ts_col).dt.minute() * 60
                        + pl.col(ts_col).dt.second()
                    )
                    // int(epoch_seconds)
                ).cast(pl.Int32).alias("epoch_of_day"),
                (pl.col(ts_col) - pl.duration(hours=12)).dt.date().alias("analysis_date_noon"),
            ]
        )
        .sort(ts_col)
    )


def impute_angle_by_clocktime_pl(
    epoch_df: pl.DataFrame,
    *,
    value_col: str = "z_angle_5s",
    epoch_seconds: int = 5,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Impute invalid 5-second angle epochs using the same epoch-of-day across days.
    """
    if value_col not in epoch_df.columns:
        raise ValueError(f"epoch_df must contain {value_col!r}.")
    if not {"invalid", "epoch_of_day", "analysis_date_noon"}.issubset(set(epoch_df.columns)):
        raise ValueError("epoch_df must contain 'invalid', 'epoch_of_day', and 'analysis_date_noon'.")

    out = epoch_df.with_columns(
        [
            pl.col("invalid").cast(pl.Boolean).alias("was_imputed"),
            pl.col(value_col).cast(pl.Float64).alias(f"{value_col}_imputed"),
        ]
    )

    valid_means = (
        out.filter(~pl.col("invalid") & pl.col(value_col).is_not_null())
        .group_by("epoch_of_day")
        .agg(pl.col(value_col).mean().alias("clock_mean"))
        .sort("epoch_of_day")
    )
    if valid_means.is_empty():
        raise ValueError("No valid angle epochs are available for imputation.")

    epochs_per_day = int(86400 // int(epoch_seconds))
    clock_grid = (
        pl.DataFrame({"epoch_of_day": np.arange(epochs_per_day, dtype=np.int32)})
        .join(valid_means, on="epoch_of_day", how="left")
        .sort("epoch_of_day")
    )

    means = clock_grid.get_column("clock_mean").to_numpy()
    valid_idx = np.flatnonzero(np.isfinite(means))
    if valid_idx.size == 0:
        raise ValueError("Clock-time means could not be estimated for angle imputation.")
    if np.any(~np.isfinite(means)):
        fill_idx = np.flatnonzero(~np.isfinite(means))
        means = means.astype(float, copy=True)
        means[fill_idx] = np.interp(fill_idx, valid_idx, means[valid_idx])

    clock_grid = clock_grid.with_columns(pl.Series("clock_mean_filled", means))
    out = out.join(clock_grid.select(["epoch_of_day", "clock_mean_filled"]), on="epoch_of_day", how="left")
    out = out.with_columns(
        pl.when(pl.col("invalid"))
        .then(pl.col("clock_mean_filled"))
        .otherwise(pl.col(value_col).cast(pl.Float64))
        .alias(f"{value_col}_imputed")
    ).drop("clock_mean_filled")

    imputed_epochs_per_day = (
        out.group_by("analysis_date_noon")
        .agg(pl.col("was_imputed").sum().cast(pl.Int64).alias("n_imputed_epochs"))
        .sort("analysis_date_noon")
    )
    return out, imputed_epochs_per_day


def _extract_window_pl(
    epoch_df: pl.DataFrame,
    *,
    ts_col: str = "ts",
    start: Any,
    end: Any,
) -> pl.DataFrame:
    return epoch_df.filter((pl.col(ts_col) >= pl.lit(start)) & (pl.col(ts_col) <= pl.lit(end))).sort(ts_col)


def _empty_spt_windows(ts_dtype: pl.DataType) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "night_id": pl.Int64,
            "window_date": pl.Date,
            "window_start": ts_dtype,
            "window_end": ts_dtype,
            "window_hours": pl.Float64,
            "is_full_window": pl.Boolean,
            "used_window_anchor_hours": pl.Int64,
            "spt_start": ts_dtype,
            "spt_end": ts_dtype,
            "spt_duration": pl.Duration("ns"),
            "spt_found": pl.Boolean,
            "fraction_invalid_window": pl.Float64,
            "n_epochs_window": pl.Int64,
            "n_epochs_imputed_window": pl.Int64,
            "n_epochs_imputed_day_mean": pl.Float64,
            "hdcza_threshold": pl.Float64,
            "part3_guider": pl.String,
        }
    )


def _empty_night_epochs(ts_col: str, ts_dtype: pl.DataType) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            ts_col: ts_dtype,
            "z_angle_5s": pl.Float64,
            "invalid_missing": pl.Boolean,
            "invalid_nonwear": pl.Boolean,
            "invalid": pl.Boolean,
            "epoch_of_day": pl.Int32,
            "analysis_date_noon": pl.Date,
            "was_imputed": pl.Boolean,
            "z_angle_5s_imputed": pl.Float64,
            "night_id": pl.Int64,
            "window_date": pl.Date,
            "used_window_anchor_hours": pl.Int64,
            "hdcza_feature_5min": pl.Float64,
            "hdcza_threshold": pl.Float64,
            "nomov": pl.Int8,
            "spt_crude_estimate": pl.Float64,
            "in_spt": pl.Boolean,
        }
    )


def _empty_imputed_epochs_per_day() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "analysis_date_noon": pl.Date,
            "n_imputed_epochs": pl.Int64,
        }
    )


def _is_empty_angle_imputation_error(exc: ValueError) -> bool:
    message = str(exc)
    return (
        "No valid angle epochs are available for imputation." in message
        or "Clock-time means could not be estimated for angle imputation." in message
    )


def vh2018_spt(
    acc_df: pl.DataFrame,
    nonwear_df: pl.DataFrame | None = None,
    *,
    ts_col: str = "ts",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    epoch_seconds: int = 5,
    haspt_ignore_invalid: float | bool = False,
    hdcza_threshold: float | tuple[float, float] | list[float] | None = None,
    spt_min_block_dur_minutes: int = 30,
    spt_max_gap_dur_minutes: int = 60,
    spt_max_gap_ratio: float = 1.0,
    rerun_with_6pm_window: bool = True,
    return_epochs: bool = True,
    return_intermediates: bool = False,
) -> dict[str, Any]:
    """
    Estimate the Sleep Period Time (SPT) window with the HDCZA algorithm.

    This is the native polars implementation. It follows the same workflow as
    GGIR's HDCZA guider as closely as practical:
    - derive a 5-second z-angle series from raw wrist acceleration
    - build a complete 5-second grid and flag missing / non-wear epochs as invalid
    - impute invalid 5-second angles from the same clock-time across days
    - compute the 5-minute rolling median of absolute angle changes
    - use the 10th percentile * 15 threshold (clamped to 0.13-0.50) by default
    - keep blocks longer than 30 minutes, bridge gaps shorter than 60 minutes,
      and retain the longest resulting block
    - if a detected block reaches the end of a noon-noon window, optionally
      rerun the search on a 6pm-6pm window as GGIR does for day-sleep cases

    Non-wear / invalid handling
    ---------------------------
    By default, invalid 5-second epochs are imputed before HDCZA is applied.
    That means the z-angle threshold is computed on an imputed angle series,
    which mirrors GGIR's default `HASPT.ignore.invalid = FALSE`.

    The `haspt_ignore_invalid` parameter controls whether invalid epochs can
    contribute to the final no-movement blocks:
    - `False`: invalid epochs contribute through their imputed angle values
    - `True`: invalid epochs are excluded from the final no-movement blocks
    - `np.nan` / `"NA"`: invalid epochs are treated as explicit no-movement
    """
    if ts_col not in acc_df.columns:
        raise ValueError(f"acc_df must contain a datetime column {ts_col!r}.")

    angle_5s = build_anglez_5s_pl(
        acc_df,
        ts_col=ts_col,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        rolling_5s="5s",
        resample_5s=f"{int(epoch_seconds)}s",
        angle_col_out="z_angle_5s",
    )
    epoch_df = _prepare_angle_epoch_table_pl(
        angle_5s,
        ts_col=ts_col,
        value_col="z_angle_5s",
        nonwear_df=nonwear_df,
        epoch_seconds=epoch_seconds,
    )
    ts_dtype = epoch_df.schema[ts_col]
    try:
        epoch_imputed, imputed_epochs_per_day = impute_angle_by_clocktime_pl(
            epoch_df,
            value_col="z_angle_5s",
            epoch_seconds=epoch_seconds,
        )
    except ValueError as exc:
        if not _is_empty_angle_imputation_error(exc):
            raise
        out: dict[str, Any] = {
            "spt_windows": _empty_spt_windows(ts_dtype),
            "imputed_epochs_per_day": _empty_imputed_epochs_per_day(),
            "status": "skipped",
            "reason": str(exc),
        }
        if return_epochs:
            out["spt_night_epochs"] = _empty_night_epochs(ts_col, ts_dtype)
        if return_intermediates:
            out["angle_5s"] = angle_5s
            out["epoch_df"] = epoch_df
        return out

    epochs_per_hour = int(3600 / int(epoch_seconds))
    imputed_day_mean = (
        float(imputed_epochs_per_day.get_column("n_imputed_epochs").mean())
        if not imputed_epochs_per_day.is_empty()
        else 0.0
    )

    nightly_rows: list[dict[str, Any]] = []
    nightly_epoch_frames: list[pl.DataFrame] = []

    analysis_dates = (
        epoch_imputed.select(pl.col("analysis_date_noon").unique().sort())
        .to_series()
        .to_list()
    )

    for night_id, analysis_date in enumerate(analysis_dates):
        window_df = (
            epoch_imputed.filter(pl.col("analysis_date_noon") == pl.lit(analysis_date))
            .sort(ts_col)
        )
        if window_df.height < 60:
            continue

        angle_imputed = window_df.get_column("z_angle_5s_imputed").to_numpy()
        invalid = window_df.get_column("invalid").to_numpy()
        result = _haspt_hdcza_window(
            angle_imputed,
            invalid,
            epoch_seconds=epoch_seconds,
            haspt_ignore_invalid=haspt_ignore_invalid,
            hdcza_threshold=hdcza_threshold,
            spt_min_block_dur_minutes=spt_min_block_dur_minutes,
            spt_max_gap_dur_minutes=spt_max_gap_dur_minutes,
            spt_max_gap_ratio=spt_max_gap_ratio,
        )

        used_anchor_hours = 12
        candidate_df = window_df
        chosen = result

        if (
            rerun_with_6pm_window
            and result["spt_end_idx"] is not None
            and result["spt_end_idx"] >= window_df.height - epochs_per_hour - 1
        ):
            shifted_start = window_df.get_column(ts_col)[0] + timedelta(hours=6)
            shifted_end = window_df.get_column(ts_col)[-1] + timedelta(hours=6)
            shifted_df = _extract_window_pl(
                epoch_imputed,
                ts_col=ts_col,
                start=shifted_start,
                end=shifted_end,
            )
            if shifted_df.height > 23 * epochs_per_hour:
                shifted_result = _haspt_hdcza_window(
                    shifted_df.get_column("z_angle_5s_imputed").to_numpy(),
                    shifted_df.get_column("invalid").to_numpy(),
                    epoch_seconds=epoch_seconds,
                    haspt_ignore_invalid=haspt_ignore_invalid,
                    hdcza_threshold=hdcza_threshold,
                    spt_min_block_dur_minutes=spt_min_block_dur_minutes,
                    spt_max_gap_dur_minutes=spt_max_gap_dur_minutes,
                    spt_max_gap_ratio=spt_max_gap_ratio,
                )
                if shifted_result["spt_start_idx"] is not None:
                    shifted_end_ts = shifted_df.get_column(ts_col)[shifted_result["spt_end_idx"]]
                    original_window_end_ts = window_df.get_column(ts_col)[-1]
                    if shifted_end_ts >= original_window_end_ts:
                        candidate_df = shifted_df
                        chosen = shifted_result
                        used_anchor_hours = 18

        spt_found = chosen["spt_start_idx"] is not None and chosen["spt_end_idx"] is not None
        spt_start = None
        spt_end = None
        spt_duration = None
        if spt_found:
            spt_start = candidate_df.get_column(ts_col)[int(chosen["spt_start_idx"])]
            spt_end = candidate_df.get_column(ts_col)[int(chosen["spt_end_idx"])]
            spt_duration = spt_end - spt_start

        window_start = candidate_df.get_column(ts_col)[0]
        window_end = candidate_df.get_column(ts_col)[-1]
        nightly_rows.append(
            {
                "night_id": int(night_id),
                "window_date": analysis_date,
                "window_start": window_start,
                "window_end": window_end,
                "window_hours": float((window_end - window_start).total_seconds() / 3600.0),
                "is_full_window": bool(candidate_df.height >= (23 * epochs_per_hour)),
                "used_window_anchor_hours": int(used_anchor_hours),
                "spt_start": spt_start,
                "spt_end": spt_end,
                "spt_duration": spt_duration,
                "spt_found": bool(spt_found),
                "fraction_invalid_window": float(candidate_df.get_column("invalid").cast(pl.Float64).mean()),
                "n_epochs_window": int(candidate_df.height),
                "n_epochs_imputed_window": int(candidate_df.get_column("was_imputed").sum()),
                "n_epochs_imputed_day_mean": imputed_day_mean,
                "hdcza_threshold": float(chosen["threshold"]) if np.isfinite(chosen["threshold"]) else np.nan,
                "part3_guider": chosen["part3_guider"],
            }
        )

        if return_epochs:
            in_spt = np.zeros(candidate_df.height, dtype=bool)
            if spt_found:
                start_i = int(chosen["spt_start_idx"])
                end_i = int(chosen["spt_end_idx"])
                in_spt[start_i : end_i + 1] = True

            night_epochs = candidate_df.with_columns(
                [
                    pl.lit(int(night_id)).alias("night_id"),
                    pl.lit(analysis_date).cast(pl.Date).alias("window_date"),
                    pl.lit(int(used_anchor_hours)).alias("used_window_anchor_hours"),
                    pl.Series("hdcza_feature_5min", chosen["feature_5min"]),
                    pl.lit(float(chosen["threshold"]) if np.isfinite(chosen["threshold"]) else None).alias("hdcza_threshold"),
                    pl.Series("nomov", chosen["nomov"]).cast(pl.Int8),
                    pl.Series("spt_crude_estimate", chosen["spt_crude_estimate"]),
                    pl.Series("in_spt", in_spt),
                ]
            )
            nightly_epoch_frames.append(night_epochs)

    spt_windows = pl.DataFrame(nightly_rows) if nightly_rows else _empty_spt_windows(ts_dtype)
    out: dict[str, Any] = {
        "spt_windows": spt_windows,
        "imputed_epochs_per_day": imputed_epochs_per_day,
    }

    if return_epochs:
        out["spt_night_epochs"] = (
            pl.concat(nightly_epoch_frames, how="vertical") if nightly_epoch_frames else _empty_night_epochs(ts_col, ts_dtype)
        )
    if return_intermediates:
        out["angle_5s"] = angle_5s
        out["epoch_df"] = epoch_df
        out["epoch_imputed"] = epoch_imputed
    return out


def vh2018_spt_pd(
    acc: pd.DataFrame,
    nonwear: pd.DataFrame | None = None,
    *,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    epoch_seconds: int = 5,
    haspt_ignore_invalid: float | bool = False,
    hdcza_threshold: float | tuple[float, float] | list[float] | None = None,
    spt_min_block_dur_minutes: int = 30,
    spt_max_gap_dur_minutes: int = 60,
    spt_max_gap_ratio: float = 1.0,
    rerun_with_6pm_window: bool = True,
    return_epochs: bool = True,
    return_intermediates: bool = False,
) -> dict[str, Any]:
    """
    Pandas compatibility wrapper around the native polars implementation.
    """
    if not isinstance(acc.index, pd.DatetimeIndex):
        raise ValueError("acc must have a DatetimeIndex.")

    acc_pd = acc[[x_col, y_col, z_col]].copy().sort_index().reset_index()
    ts_col = acc_pd.columns[0]
    acc_pl = pl.from_pandas(acc_pd)
    nonwear_pl = pl.from_pandas(nonwear) if nonwear is not None and len(nonwear) > 0 else None

    out_pl = vh2018_spt(
        acc_pl,
        nonwear_df=nonwear_pl,
        ts_col=ts_col,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        epoch_seconds=epoch_seconds,
        haspt_ignore_invalid=haspt_ignore_invalid,
        hdcza_threshold=hdcza_threshold,
        spt_min_block_dur_minutes=spt_min_block_dur_minutes,
        spt_max_gap_dur_minutes=spt_max_gap_dur_minutes,
        spt_max_gap_ratio=spt_max_gap_ratio,
        rerun_with_6pm_window=rerun_with_6pm_window,
        return_epochs=return_epochs,
        return_intermediates=return_intermediates,
    )

    out: dict[str, Any] = {
        "spt_windows": out_pl["spt_windows"].to_pandas(),
        "imputed_epochs_per_day": (
            out_pl["imputed_epochs_per_day"].to_pandas().set_index("analysis_date_noon")["n_imputed_epochs"]
            if not out_pl["imputed_epochs_per_day"].is_empty()
            else pd.Series(dtype="int64", name="n_imputed_epochs")
        ),
    }

    if return_epochs:
        epochs_pd = out_pl["spt_night_epochs"].to_pandas()
        if not epochs_pd.empty and ts_col in epochs_pd.columns:
            epochs_pd[ts_col] = pd.to_datetime(epochs_pd[ts_col], errors="coerce")
            epochs_pd = epochs_pd.set_index(ts_col)
        out["spt_night_epochs"] = epochs_pd

    if return_intermediates:
        angle_pd = out_pl["angle_5s"].to_pandas()
        if not angle_pd.empty and ts_col in angle_pd.columns:
            angle_pd[ts_col] = pd.to_datetime(angle_pd[ts_col], errors="coerce")
            angle_pd = angle_pd.set_index(ts_col)
        out["angle_5s"] = angle_pd
        for key in ("epoch_df", "epoch_imputed"):
            df_pd = out_pl[key].to_pandas()
            if not df_pd.empty and ts_col in df_pd.columns:
                df_pd[ts_col] = pd.to_datetime(df_pd[ts_col], errors="coerce")
                df_pd = df_pd.set_index(ts_col)
            out[key] = df_pd

    return out

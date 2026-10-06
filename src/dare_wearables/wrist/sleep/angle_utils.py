from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


def build_anglez_5s_pl(
    acc_df: pl.DataFrame,
    *,
    ts_col: str = "ts",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    rolling_5s: str = "5s",
    resample_5s: str = "5s",
    angle_col_out: str = "z_angle_5s",
) -> pl.DataFrame:
    """
    Shared 5-second z-angle construction for the sleep algorithms.

    Steps:
    1. apply a 5-second rolling median to x/y/z
    2. compute z-angle in degrees
    3. resample the angle to a 5-second grid by mean
    """
    if ts_col not in acc_df.columns:
        raise ValueError(f"acc_df must contain a datetime column {ts_col!r}.")
    for col in (x_col, y_col, z_col):
        if col not in acc_df.columns:
            raise ValueError(f"acc_df is missing column {col!r}.")

    df = acc_df.sort(ts_col)
    if df.is_empty():
        raise ValueError("acc_df is empty.")

    return (
        df.with_columns(
            [
                pl.col(x_col).rolling_median_by(ts_col, window_size=rolling_5s).alias("x_med_5s"),
                pl.col(y_col).rolling_median_by(ts_col, window_size=rolling_5s).alias("y_med_5s"),
                pl.col(z_col).rolling_median_by(ts_col, window_size=rolling_5s).alias("z_med_5s"),
            ]
        )
        .with_columns(
            (
                (pl.col("z_med_5s") / (pl.col("x_med_5s") ** 2 + pl.col("y_med_5s") ** 2).sqrt())
                .arctan()
                * (180.0 / pl.lit(np.pi))
            ).alias("z_angle")
        )
        .group_by_dynamic(
            index_column=ts_col,
            every=resample_5s,
            closed="left",
            label="left",
        )
        .agg(pl.col("z_angle").mean().alias(angle_col_out))
        .sort(ts_col)
    )


def build_anglez_5s_pd(
    acc: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    rolling_5s: str = "5s",
    resample_5s: str = "5s",
    angle_col_out: str = "z_angle_5s",
) -> pd.DataFrame:
    """
    Pandas convenience wrapper around the shared polars implementation.
    """
    if not isinstance(acc.index, pd.DatetimeIndex):
        raise ValueError("acc must have a DatetimeIndex.")

    acc_pd = acc[[x_col, y_col, z_col]].copy().sort_index().reset_index()
    ts_col = acc_pd.columns[0]
    acc_pl = pl.from_pandas(acc_pd)

    angle_pl = build_anglez_5s_pl(
        acc_pl,
        ts_col=ts_col,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        rolling_5s=rolling_5s,
        resample_5s=resample_5s,
        angle_col_out=angle_col_out,
    )
    angle_pd = angle_pl.to_pandas()
    angle_pd[ts_col] = pd.to_datetime(angle_pd[ts_col], errors="coerce")
    angle_pd = angle_pd.dropna(subset=[ts_col]).set_index(ts_col)
    angle_pd.index.name = acc.index.name or "time"
    return angle_pd[[angle_col_out]]

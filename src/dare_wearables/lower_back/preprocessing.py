from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dare_wearables.common.autocalibrate import autocalibrate
from dare_wearables.lower_back.io.mcroberts_loader import read_dp7_as_dataframe_fast


GRAV_MS2 = 9.80665
MAX_SAMPLES = 100 * 60 * 60 * 24 * 7


def _build_wear_mask_from_nonwear(
    timestamps: pd.Series | pd.DatetimeIndex,
    nonwear_bouts: pd.DataFrame,
) -> np.ndarray:
    wear_mask = np.ones(len(timestamps), dtype=bool)
    if nonwear_bouts.empty:
        return wear_mask

    ts_ns = pd.to_datetime(pd.Index(timestamps)).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    for row in nonwear_bouts.itertuples(index=False):
        start_ns = pd.Timestamp(row.start).value
        end_ns = pd.Timestamp(row.end).value
        start_idx = int(np.searchsorted(ts_ns, start_ns, side="left"))
        end_idx = int(np.searchsorted(ts_ns, end_ns, side="right"))
        wear_mask[start_idx:end_idx] = False
    return wear_mask


def _pl_to_pandas(df_pl) -> pd.DataFrame:
    if df_pl.is_empty():
        return pd.DataFrame(columns=df_pl.columns)
    return pd.DataFrame(df_pl.to_dict(as_series=False))


def preprocess_lowerback_file(
    filepath: str,
    *,
    sample_rate_hz: float = 100.0,
    max_samples: int = MAX_SAMPLES,
    verbose: bool = True,
    autocalibration_kwargs: dict[str, Any] | None = None,
    nonwear_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        import polars as pl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Custom lower-back preprocessing requires `polars` to run."
        ) from exc

    from dare_wearables.lower_back.nonwear.vanhees2013 import vanhees2013

    autocalibration_kwargs = autocalibration_kwargs or {}
    nonwear_kwargs = nonwear_kwargs or {}

    df = read_dp7_as_dataframe_fast(filepath, verbose=int(bool(verbose)))
    if not np.isclose(df.attrs.get("sample_rate_hz", sample_rate_hz), sample_rate_hz):
        raise ValueError("Recording sample rate does not match preprocessing sample_rate_hz.")
    df = df.rename(
        columns={
            "ax": "acc_x",
            "ay": "acc_y",
            "az": "acc_z",
            "gx": "gyr_x",
            "gy": "gyr_y",
            "gz": "gyr_z",
        }
    )

    timestamp_series = pd.Series(df["datetime"].to_numpy(copy=False), index=df.index)
    df["is_valid"] = ~timestamp_series.duplicated() & (timestamp_series >= timestamp_series.cummax())
    len_original = len(df)
    df = df[df["is_valid"]].drop(columns="is_valid").copy()

    if verbose:
        dropped = len_original - len(df)
        print(f"Found {dropped} bad timestamps ({np.round(dropped / sample_rate_hz / 60, 3)} minutes)")

    if max_samples is not None:
        df = df.iloc[:max_samples].copy()

    df["timestamp"] = (df["datetime"] - df["datetime"].min()).dt.total_seconds()

    if df.empty:
        df["wear"] = pd.Series(dtype=bool)
        return {
            "df": df,
            "calibration": None,
            "nonwear_bouts": pd.DataFrame(columns=["start", "end"]),
            "nonwear_windows": pd.DataFrame(),
        }

    accel_pl = pl.DataFrame(
        {
            "ts": df["datetime"].to_numpy(copy=False),
            "x": df["acc_x"].to_numpy(copy=False),
            "y": df["acc_y"].to_numpy(copy=False),
            "z": df["acc_z"].to_numpy(copy=False),
        }
    )

    calibration = autocalibrate(
        accel_pl,
        ts_col="ts",
        x_col="x",
        y_col="y",
        z_col="z",
        apply=True,
        **autocalibration_kwargs,
    )

    calibrated_df = calibration["df"]
    if {"x_cal", "y_cal", "z_cal"}.issubset(set(calibrated_df.columns)):
        df["acc_x"] = calibrated_df["x_cal"].to_numpy()
        df["acc_y"] = calibrated_df["y_cal"].to_numpy()
        df["acc_z"] = calibrated_df["z_cal"].to_numpy()
        accel_pl = pl.DataFrame(
            {
                "ts": df["datetime"].to_numpy(copy=False),
                "x": df["acc_x"].to_numpy(copy=False),
                "y": df["acc_y"].to_numpy(copy=False),
                "z": df["acc_z"].to_numpy(copy=False),
            }
        )

    nonwear_result = vanhees2013(
        accel_pl,
        ts_col="ts",
        x_col="x",
        y_col="y",
        z_col="z",
        freq=sample_rate_hz,
        quiet=not verbose,
        return_debug=True,
        **nonwear_kwargs,
    )

    nonwear_bouts = _pl_to_pandas(nonwear_result.nonwear_df)
    if not nonwear_bouts.empty:
        nonwear_bouts["start"] = pd.to_datetime(nonwear_bouts["start"])
        nonwear_bouts["end"] = pd.to_datetime(nonwear_bouts["end"])

    df["wear"] = _build_wear_mask_from_nonwear(df["datetime"], nonwear_bouts)
    df[["acc_x", "acc_y", "acc_z"]] *= GRAV_MS2

    if verbose:
        nonwear_minutes = np.round((len(df) - int(df["wear"].sum())) / sample_rate_hz / 60, 3)
        print(f"Non wear time: {nonwear_minutes} minutes")

    return {
        "df": df,
        "calibration": calibration,
        "nonwear_bouts": nonwear_bouts,
        "nonwear_windows": _pl_to_pandas(nonwear_result.window_df),
    }

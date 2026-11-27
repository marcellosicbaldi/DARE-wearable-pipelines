from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dare_wearables.common.autocalibrate import autocalibrate
from dare_wearables.lower_back.io.mcroberts_loader import read_dp7_as_dataframe_fast
from dare_wearables.lower_back.nonwear.respiration import RespirationOptions, override_frame
from dare_wearables.lower_back.nonwear.ggir2023 import NonwearOptions, detect_nonwear
from dare_wearables.lower_back.io.timestamp_quality import TimestampOptions, processing_segments, invalid_intervals


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
    protocol_hours: float | None = None,
    verbose: bool = True,
    autocalibration_kwargs: dict[str, Any] | None = None,
    nonwear_kwargs: dict[str, Any] | None = None,
    respiration: RespirationOptions | None = None,
    nonwear: NonwearOptions | None = None,
    timestamps: TimestampOptions | None = None,
) -> dict[str, Any]:
    nonwear = nonwear or NonwearOptions()
    nonwear.validate()
    timestamp_options = timestamps or TimestampOptions()
    timestamp_options.validate()
    if protocol_hours is not None and (isinstance(protocol_hours, bool) or not np.isfinite(protocol_hours) or not 0 < protocol_hours <= 168):
        raise ValueError("protocol_hours must be in (0, 168], or None for the historical sample cap.")
    if timestamp_options.policy != "strict" and nonwear.method != "ggir_2023_matched":
        raise ValueError("Timestamp recovery requires the ggir_2023_matched baseline.")
    if nonwear.method == "ggir_2023_matched" and nonwear_kwargs:
        raise ValueError("The GGIR matched profile fixes SD=13 mg, range=50 mg and two axes. "
                         "nonwear_kwargs applies only to nonwear.method='current'.")
    try:
        import polars as pl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Custom lower-back preprocessing requires `polars` to run."
        ) from exc

    from dare_wearables.lower_back.nonwear.vanhees2013 import vanhees2013

    autocalibration_kwargs = autocalibration_kwargs or {}
    nonwear_kwargs = nonwear_kwargs or {}

    reader_kwargs = {"timestamps": timestamp_options} if timestamps is not None else {}
    df = read_dp7_as_dataframe_fast(filepath, verbose=int(bool(verbose)), **reader_kwargs)
    quality = df.attrs.get("timestamp_quality", {"policy": timestamp_options.policy, "events": [],
                                               "recording_discontinuities": [], "streams": []})
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

    if protocol_hours is not None and not df.empty:
        # Match the reviewed benchmark's elapsed seven-day window. Missing time
        # does not extend the protocol to accumulate seven days of samples.
        end = df.datetime.iloc[0] + pd.Timedelta(hours=protocol_hours)
        df = df.loc[df.datetime < end].copy()
    elif max_samples is not None:
        df = df.iloc[:max_samples].copy()

    df["timestamp"] = (df["datetime"] - df["datetime"].min()).dt.total_seconds()
    segments = processing_segments(df, sample_rate_hz, quality)
    if nonwear.method == "ggir_2023_matched" and nonwear.step_seconds == 60 and not df.empty:
        # The reviewed one-minute benchmark always restarted at every raw gap,
        # including gaps in files that did not need anchor quarantine.
        from dare_wearables.lower_back.nonwear.respiration import _segment_bounds
        observed = _segment_bounds(pd.DatetimeIndex(df.datetime), sample_rate_hz)
        edges = np.unique(np.r_[observed[1:, 0], np.asarray(segments, dtype=int)[1:, 0]
                                if segments is not None else np.empty(0, dtype=int)])
        segments = np.column_stack((np.r_[0, edges], np.r_[edges, len(df)])).tolist()
    if segments is not None:
        df.attrs["processing_segments"] = segments
        bounds = np.asarray(segments)
        df["time_segment"] = np.repeat(np.arange(len(bounds), dtype=np.int32), bounds[:, 1]-bounds[:, 0])
    quality["analysis"] = {"n_samples": len(df), "segment_count": len(segments) if segments else 1,
        "first_sample": str(df.datetime.iloc[0]) if len(df) else None,
        "last_sample": str(df.datetime.iloc[-1]) if len(df) else None}
    df.attrs["timestamp_quality"] = quality
    time_invalid = invalid_intervals(df, sample_rate_hz, quality)
    timing = {"protocol_hours": protocol_hours,
              "timestamp_options": timestamp_options, "timestamp_quality": quality,
              "time_invalid_intervals": time_invalid}

    if df.empty:
        df["wear"] = pd.Series(dtype=bool)
        diagnostics = {}
        if respiration is not None and respiration.enabled:
            diagnostics["respiration"] = override_frame(df, sample_rate_hz, respiration)
        return {
            "df": df,
            "calibration": None,
            "nonwear_bouts": pd.DataFrame(columns=["start", "end"]),
            "nonwear_windows": pd.DataFrame(),
            "nonwear_options": nonwear,
            "nonwear_provenance": {"method": nonwear.method, "decision": "empty recording"},
            **timing,
            **diagnostics,
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

    if nonwear.method == "ggir_2023_matched":
        segment_kwargs = {"segment_bounds": segments} if segments is not None else {}
        nonwear_result = detect_nonwear(accel_pl, freq=sample_rate_hz, options=nonwear, **segment_kwargs)
        # Polars to_dict uses Python datetimes and loses sub-microsecond ends.
        nonwear_bouts = nonwear_result.nonwear_df.to_pandas()
        nonwear_windows = nonwear_result.window_df.to_pandas()
        nonwear_provenance = nonwear_result.provenance
    else:
        nonwear_result = vanhees2013(
            accel_pl, ts_col="ts", x_col="x", y_col="y", z_col="z",
            freq=sample_rate_hz, quiet=not verbose, return_debug=True, **nonwear_kwargs,
        )
        nonwear_bouts = _pl_to_pandas(nonwear_result.nonwear_df)
        nonwear_windows = _pl_to_pandas(nonwear_result.window_df)
        nonwear_provenance = {"method": "current", "overrides": nonwear_kwargs}
    if not nonwear_bouts.empty:
        nonwear_bouts["start"] = pd.to_datetime(nonwear_bouts["start"])
        nonwear_bouts["end"] = pd.to_datetime(nonwear_bouts["end"])

    df["wear"] = _build_wear_mask_from_nonwear(df["datetime"], nonwear_bouts)
    diagnostics = {}
    if respiration is not None and respiration.enabled:
        diagnostics["accelerometer_nonwear_bouts"] = nonwear_bouts
        diagnostics["respiration"] = override_frame(df, sample_rate_hz, respiration)
        final = diagnostics["respiration"]["nonwear_bouts"]
        nonwear_bouts = final[["start", "end"]].copy()
        # Existing preprocessing consumers expect inclusive timestamp ends.
        nonwear_bouts["end"] = pd.DatetimeIndex(df.datetime)[final.stop_sample.to_numpy(dtype=int) - 1]
    df[["acc_x", "acc_y", "acc_z"]] *= GRAV_MS2

    if verbose:
        nonwear_minutes = np.round((len(df) - int(df["wear"].sum())) / sample_rate_hz / 60, 3)
        print(f"Non wear time: {nonwear_minutes} minutes")

    return {
        "df": df,
        "calibration": calibration,
        "nonwear_bouts": nonwear_bouts,
        "nonwear_windows": nonwear_windows,
        "nonwear_options": nonwear,
        "nonwear_provenance": nonwear_provenance,
        **timing,
        **diagnostics,
    }

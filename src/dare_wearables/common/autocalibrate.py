from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Optional, Tuple

import numpy as np
import polars as pl


@dataclass
class CalibrationParams:
    offset: np.ndarray        # shape (3,)
    scale: np.ndarray         # shape (3,)
    temp_scale: np.ndarray    # shape (3,)
    temp_mean: float
    cal_error_start: float
    cal_error_end: float
    n_epochs_used: int
    valid: bool
    used_temperature: bool


def _default_params(
    *,
    n_epochs_used: int = 0,
    cal_error_start: float = np.nan,
    cal_error_end: float = np.nan,
    used_temperature: bool = False,
) -> CalibrationParams:
    return CalibrationParams(
        offset=np.zeros(3, dtype=float),
        scale=np.ones(3, dtype=float),
        temp_scale=np.zeros(3, dtype=float),
        temp_mean=0.0,
        cal_error_start=cal_error_start,
        cal_error_end=cal_error_end,
        n_epochs_used=int(n_epochs_used),
        valid=False,
        used_temperature=used_temperature,
    )


def _temperature_is_usable(df: pl.DataFrame, temp_col: Optional[str]) -> bool:
    """
    GGIR-like temperature handling.

    GGIR uses temperature only if:
      - the column exists
      - the values are not unrealistically high
      - there is enough temperature variation

    In GGIR:
      - temperature is ignored if mean of first 10 values > 120
      - temperature is ignored if SD < 0.01
    """
    if temp_col is None:
        return False

    if temp_col not in df.columns:
        raise ValueError(f"temp_col='{temp_col}' not found in df")

    temp_values = (
        df.select(pl.col(temp_col).cast(pl.Float64, strict=False))
        .to_numpy()
        .ravel()
    )

    temp_values = temp_values[np.isfinite(temp_values)]

    if temp_values.size == 0:
        return False

    first_values = temp_values[:10]

    if first_values.size > 0 and float(np.nanmean(first_values)) > 120:
        return False

    if float(np.nanstd(temp_values)) < 0.01:
        return False

    return True


def _weighted_linreg_2d(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> Tuple[float, float, float]:
    """
    GGIR-like weighted regression:

        y ~ b0 + b1*x1 + b2*x2

    Important:
    GGIR uses lm.wfit(), which tolerates rank-deficient design matrices.
    This matters when temperature is not used, because GGIR passes a dummy
    zero temperature column.

    Therefore, this function uses weighted least squares via np.linalg.lstsq()
    rather than solving X'WX directly with np.linalg.solve().
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    X = np.column_stack([np.ones_like(x1), x1, x2])

    mask = (
        np.isfinite(X).all(axis=1)
        & np.isfinite(y)
        & np.isfinite(w)
        & (w > 0)
    )

    if mask.sum() < 2:
        return 0.0, 1.0, 0.0

    X = X[mask]
    y = y[mask]
    w = w[mask]

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w

    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    if beta.shape[0] < 3:
        beta = np.pad(beta, (0, 3 - beta.shape[0]))

    b0, b1, b2 = beta[:3]

    if not np.isfinite(b0):
        b0 = 0.0
    if not np.isfinite(b1):
        b1 = 1.0
    if not np.isfinite(b2):
        b2 = 0.0

    return float(b0), float(b1), float(b2)


def _icp_fit_unit_sphere(
    acc_rm: np.ndarray,
    tmp_rm: Optional[np.ndarray],
    *,
    sphere_crit: float = 0.3,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> Tuple[bool, CalibrationParams]:
    """
    Iterative closest-point fitting to the unit sphere.

    This follows the GGIR logic:
      - still epochs are fitted to the closest point on the unit sphere
      - each axis is updated with weighted regression
      - temperature is represented as a centered covariate if available
      - if temperature is unavailable, GGIR still uses a dummy zero column
    """
    acc_rm = np.asarray(acc_rm, dtype=float)

    if acc_rm.ndim != 2 or acc_rm.shape[1] != 3:
        raise ValueError("acc_rm must have shape (N, 3)")

    if tmp_rm is not None:
        tmp_rm = np.asarray(tmp_rm, dtype=float).reshape(-1)
        if tmp_rm.shape[0] != acc_rm.shape[0]:
            raise ValueError("tmp_rm must have the same number of rows as acc_rm")

    finite_mask = np.isfinite(acc_rm).all(axis=1)

    if tmp_rm is not None:
        finite_mask &= np.isfinite(tmp_rm)

    acc_rm = acc_rm[finite_mask]

    if tmp_rm is not None:
        tmp_rm = tmp_rm[finite_mask]

    N = acc_rm.shape[0]
    use_temp = tmp_rm is not None

    if N < 10:
        params = _default_params(
            n_epochs_used=N,
            used_temperature=use_temp,
        )
        return False, params

    tel = (
        (acc_rm.min(axis=0) < -sphere_crit)
        & (acc_rm.max(axis=0) > sphere_crit)
    ).sum()

    if tel != 3:
        params = _default_params(
            n_epochs_used=N,
            used_temperature=use_temp,
        )
        return False, params

    cal_error_start = float(
        np.mean(np.abs(np.linalg.norm(acc_rm, axis=1) - 1.0))
    )

    offset = np.zeros(3, dtype=float)
    scale = np.ones(3, dtype=float)
    temp_scale = np.zeros(3, dtype=float)

    if use_temp:
        tmp_mean = float(np.mean(tmp_rm))
        tmp_centered = (tmp_rm - tmp_mean).reshape(-1, 1)
    else:
        tmp_mean = 0.0
        tmp_centered = np.zeros((N, 1), dtype=float)

    w = np.ones(N, dtype=float)
    res_prev = np.inf

    for _ in range(max_iter):
        curr = (acc_rm + offset) * scale + tmp_centered * temp_scale

        denom = np.linalg.norm(curr, axis=1, keepdims=True)
        denom[denom == 0] = 1e-12

        closest = curr / denom

        offsetch = np.zeros(3, dtype=float)
        scalech = np.ones(3, dtype=float)
        toffch = np.zeros(3, dtype=float)

        for k in range(3):
            temp_predictor = tmp_centered[:, 0] if use_temp else np.zeros(N)

            b0, b1, b2 = _weighted_linreg_2d(
                x1=curr[:, k],
                x2=temp_predictor,
                y=closest[:, k],
                w=w,
            )

            offsetch[k] = b0
            scalech[k] = b1

            if use_temp:
                toffch[k] = b2
            else:
                toffch[k] = 0.0

            curr[:, k] = b0 + b1 * curr[:, k] + b2 * temp_predictor

        update_denom = scale * scalech

        if (
            not np.isfinite(update_denom).all()
            or np.any(np.abs(update_denom) < 1e-12)
        ):
            params = _default_params(
                n_epochs_used=N,
                cal_error_start=cal_error_start,
                used_temperature=use_temp,
            )
            return False, params

        offset = offset + offsetch / update_denom

        if use_temp:
            temp_scale = temp_scale * scalech + toffch

        scale = scale * scalech

        if (
            not np.isfinite(offset).all()
            or not np.isfinite(scale).all()
            or not np.isfinite(temp_scale).all()
        ):
            params = _default_params(
                n_epochs_used=N,
                cal_error_start=cal_error_start,
                used_temperature=use_temp,
            )
            return False, params

        diff = curr - closest

        res = 3.0 * np.mean((w[:, None] * (diff ** 2)) / w.sum())

        errnorm = np.linalg.norm(diff, axis=1)
        errnorm[errnorm == 0] = 1e-12

        w = np.minimum(1.0 / errnorm, 100.0)

        if abs(res - res_prev) < tol:
            break

        res_prev = res

    acc_cal_epochs = (acc_rm + offset) * scale + tmp_centered * temp_scale

    cal_error_end = float(
        np.mean(np.abs(np.linalg.norm(acc_cal_epochs, axis=1) - 1.0))
    )

    valid = (
        np.isfinite(cal_error_start)
        and np.isfinite(cal_error_end)
        and cal_error_end < cal_error_start
        and cal_error_end < 0.01
    )

    params = CalibrationParams(
        offset=offset,
        scale=scale,
        temp_scale=temp_scale,
        temp_mean=tmp_mean,
        cal_error_start=cal_error_start,
        cal_error_end=cal_error_end,
        n_epochs_used=N,
        valid=valid,
        used_temperature=use_temp,
    )

    return valid, params


def autocalibrate(
    df: pl.DataFrame,
    *,
    ts_col: str = "ts",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    temp_col: Optional[str] = None,
    sd_crit: float = 0.013,
    sphere_crit: float = 0.3,
    min_hours: int = 72,
    extend_hours: int = 12,
    min_required_hours: int = 24,
    max_iter: int = 1000,
    tol: float = 1e-10,
    apply: bool = True,
    out_cols: Tuple[str, str, str] = ("x_cal", "y_cal", "z_cal"),
) -> dict[str, Any]:
    """
    Autocalibrate raw accelerometer data using GGIR-like logic.

    Assumptions:
      - acceleration is in g
      - timestamps are datetime-like
      - calibration uses still 10-second windows
      - optional temperature correction is used only if temperature is valid

    Returns:
      - df: calibrated dataframe if calibration is valid and apply=True
      - params: CalibrationParams
      - epochs: 10-second epoch dataframe
    """
    if df.height == 0:
        params = _default_params()
        return {
            "df": df,
            "params": params,
            "epochs": pl.DataFrame(),
        }

    if ts_col not in df.columns:
        raise ValueError(f"df must contain datetime column '{ts_col}'")

    for col in (x_col, y_col, z_col):
        if col not in df.columns:
            raise ValueError(f"df must contain column '{col}'")

    df = df.sort(ts_col)

    use_temp = _temperature_is_usable(df, temp_col)

    start_ts = df.select(pl.col(ts_col).min()).item()
    end_ts = df.select(pl.col(ts_col).max()).item()

    if start_ts is None or end_ts is None:
        params = _default_params(used_temperature=use_temp)
        return {
            "df": df,
            "params": params,
            "epochs": pl.DataFrame(),
        }

    dur_hours = (end_ts - start_ts).total_seconds() / 3600.0

    agg_exprs = [
        pl.col(x_col).mean().alias("x_mean"),
        pl.col(y_col).mean().alias("y_mean"),
        pl.col(z_col).mean().alias("z_mean"),
        pl.col(x_col).std().alias("x_sd"),
        pl.col(y_col).std().alias("y_sd"),
        pl.col(z_col).std().alias("z_sd"),
    ]

    required_epoch_cols = [
        "x_mean",
        "y_mean",
        "z_mean",
        "x_sd",
        "y_sd",
        "z_sd",
    ]

    if use_temp:
        agg_exprs.append(
            pl.col(temp_col).cast(pl.Float64, strict=False).mean().alias("t_mean")
        )
        required_epoch_cols.append("t_mean")

    epochs = (
        df.group_by_dynamic(
            index_column=ts_col,
            every="10s",
            period="10s",
            closed="left",
            label="left",
        )
        .agg(agg_exprs)
        .drop_nulls(required_epoch_cols)
        .rename({ts_col: "epoch_start"})
        .sort("epoch_start")
    )

    if dur_hours < min_required_hours:
        params = _default_params(
            n_epochs_used=0,
            used_temperature=use_temp,
        )
        return {
            "df": df,
            "params": params,
            "epochs": epochs,
        }

    still_epochs = epochs.filter(
        (pl.col("x_sd") < sd_crit)
        & (pl.col("y_sd") < sd_crit)
        & (pl.col("z_sd") < sd_crit)
        & (pl.col("x_mean").abs() < 2.0)
        & (pl.col("y_mean").abs() < 2.0)
        & (pl.col("z_mean").abs() < 2.0)
    )

    if still_epochs.height < 10:
        params = _default_params(
            n_epochs_used=int(still_epochs.height),
            used_temperature=use_temp,
        )
        return {
            "df": df,
            "params": params,
            "epochs": epochs,
        }

    use_hours = min(float(min_hours), float(dur_hours))

    valid = False
    best_params: Optional[CalibrationParams] = None

    while True:
        cutoff = start_ts + timedelta(hours=float(use_hours))

        subset = still_epochs.filter(pl.col("epoch_start") < cutoff)

        acc_rm = subset.select(["x_mean", "y_mean", "z_mean"]).to_numpy()

        if use_temp and "t_mean" in subset.columns:
            tmp_rm = subset.select("t_mean").to_numpy().ravel()
        else:
            tmp_rm = None

        ok, params = _icp_fit_unit_sphere(
            acc_rm=acc_rm,
            tmp_rm=tmp_rm,
            sphere_crit=sphere_crit,
            max_iter=max_iter,
            tol=tol,
        )

        best_params = params
        valid = ok

        if valid or use_hours >= dur_hours:
            break

        use_hours = min(use_hours + float(extend_hours), float(dur_hours))

    if best_params is None:
        best_params = _default_params(used_temperature=use_temp)

    if apply and best_params.valid:
        ox, oy, oz = best_params.offset.tolist()
        sx, sy, sz = best_params.scale.tolist()
        mx, my, mz = best_params.temp_scale.tolist()
        tbar = best_params.temp_mean

        if best_params.used_temperature and temp_col is not None:
            df_out = df.with_columns(
                [
                    (
                        ((pl.col(x_col) + pl.lit(ox)) * pl.lit(sx))
                        + (pl.col(temp_col).cast(pl.Float64, strict=False) - pl.lit(tbar))
                        * pl.lit(mx)
                    ).alias(out_cols[0]),
                    (
                        ((pl.col(y_col) + pl.lit(oy)) * pl.lit(sy))
                        + (pl.col(temp_col).cast(pl.Float64, strict=False) - pl.lit(tbar))
                        * pl.lit(my)
                    ).alias(out_cols[1]),
                    (
                        ((pl.col(z_col) + pl.lit(oz)) * pl.lit(sz))
                        + (pl.col(temp_col).cast(pl.Float64, strict=False) - pl.lit(tbar))
                        * pl.lit(mz)
                    ).alias(out_cols[2]),
                ]
            )
        else:
            df_out = df.with_columns(
                [
                    ((pl.col(x_col) + pl.lit(ox)) * pl.lit(sx)).alias(out_cols[0]),
                    ((pl.col(y_col) + pl.lit(oy)) * pl.lit(sy)).alias(out_cols[1]),
                    ((pl.col(z_col) + pl.lit(oz)) * pl.lit(sz)).alias(out_cols[2]),
                ]
            )
    else:
        df_out = df

    return {
        "df": df_out,
        "params": best_params,
        "epochs": epochs,
    }
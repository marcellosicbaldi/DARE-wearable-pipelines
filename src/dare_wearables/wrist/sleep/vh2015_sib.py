import numpy as np
import pandas as pd
import polars as pl

from dare_wearables.wrist.sleep.angle_utils import build_anglez_5s_pd, build_anglez_5s_pl


def _sib_bouts_from_angle_5s_pl(
    angle_5s: pl.DataFrame,
    *,
    ts_col: str = "ts",
    angle_col: str = "z_angle_5s",
    change_thresh_deg: float = 5.0,
    min_bout_minutes: int = 5,
) -> pl.DataFrame:
    """
    Extract SIB bouts from a 5-second z-angle series.

    This helper is used by the polars wrapper after the angle series has already
    been constructed, which avoids the accidental recursion that would
    otherwise happen if `vh2015_sib()` called itself.
    """
    if ts_col not in angle_5s.columns or angle_col not in angle_5s.columns:
        raise ValueError(f"angle_5s must contain columns {ts_col!r} and {angle_col!r}.")

    angle_pd = angle_5s.select([ts_col, angle_col]).to_pandas().copy()
    angle_pd[ts_col] = pd.to_datetime(angle_pd[ts_col], errors="coerce")
    angle_pd = angle_pd.dropna(subset=[ts_col]).sort_values(ts_col).set_index(ts_col)

    d_angle = angle_pd[angle_col].diff().abs()
    no_change = d_angle <= float(change_thresh_deg)

    is_nan = angle_pd[angle_col].isna()
    no_change = no_change & (~is_nan) & (~is_nan.shift(1, fill_value=True))

    group_id = (no_change.ne(no_change.shift(1, fill_value=False))).cumsum()
    tmp = pd.DataFrame({"no_change": no_change, "group_id": group_id}, index=angle_pd.index)

    bouts = (
        tmp.groupby("group_id")
        .agg(value=("no_change", "first"), start=("no_change", lambda s: s.index.min()))
    )
    bouts["end"] = tmp.groupby("group_id").apply(lambda g: g.index.max())
    bouts = bouts.reset_index(drop=True)
    bouts["duration"] = bouts["end"] - bouts["start"]
    bouts = bouts[(bouts["value"]) & (bouts["duration"] >= pd.Timedelta(minutes=min_bout_minutes))]
    bouts = bouts[["start", "end", "duration"]].sort_values("start").reset_index(drop=True)

    if bouts.empty:
        return pl.DataFrame(
            schema={
                "start": pl.Datetime,
                "end": pl.Datetime,
                "duration": pl.Duration("ns"),
            }
        )
    return pl.from_pandas(bouts)

def label_epochs_from_bouts(
    epochs: pl.DataFrame,
    bouts: pl.DataFrame,
    *,
    ts_col: str = "ts",
    start_col: str = "start",
    end_col: str = "end",
    label_col: str = "sib_sleep",
) -> pl.DataFrame:
    """
    Given epochs (ts grid) and bouts (start/end), return epochs with boolean label_col
    indicating if ts is inside ANY bout.

    Uses join_asof on start times to avoid expensive interval checks.
    """
    if bouts.is_empty():
        return epochs.with_columns(pl.lit(False).alias(label_col))

    ts_dtype = epochs.schema[ts_col]
    epochs_sorted = epochs.sort(ts_col)
    bouts_sorted = bouts.with_columns(
        [
            pl.col(start_col).cast(ts_dtype),
            pl.col(end_col).cast(ts_dtype),
        ]
    ).sort(start_col)

    # as-of join: for each ts, attach the most recent bout start <= ts
    joined = epochs_sorted.join_asof(
        bouts_sorted,
        left_on=ts_col,
        right_on=start_col,
        strategy="backward",
    )

    # inside bout if we matched a start and ts <= end
    return joined.with_columns(
        (
            pl.col(end_col).is_not_null()
            & (pl.col(ts_col) <= pl.col(end_col))
        ).alias(label_col)
    )

def vh2015_sib(
    acc_df: pl.DataFrame,
    nonwear_df: pl.DataFrame | None = None,
    *,
    ts_col: str = "ts",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    # angle pipeline (match your VH2018 intermediates)
    rolling_5s: str = "5s",
    resample_5s: str = "5s",
    angle_col_out: str = "z_angle_5s",
    # SIB params (Van Hees 2015)
    change_thresh_deg: float = 5.0,
    min_bout_minutes: int = 5,
    # outputs
    return_epochs: bool = True,
    return_intermediates: bool = False,
) -> dict:
    """
    Compute Van Hees 2015 Sustained Inactivity Bouts (SIB) over the *entire recording*
    (not limited to SPT).

    Steps:
      1) rolling median (5s) on x/y/z
      2) compute z-angle (degrees)
      3) resample to 5s grid with mean
      4) (optional) mask non-wear by setting angle to null
      5) run vh2015_sib() on the full 5s angle series
      6) (optional) label epochs via label_epochs_from_bouts()

    Returns dict with:
      - 'sib_bouts': [start, end, duration]
      - 'sib_epochs': [ts, z_angle_5s, sib_sleep] (if return_epochs)
      - optionally intermediates: 'angle_5s'
    """
    # --- checks ---
    if ts_col not in acc_df.columns:
        raise ValueError(f"acc_df must contain a datetime column '{ts_col}'")
    for c in (x_col, y_col, z_col):
        if c not in acc_df.columns:
            raise ValueError(f"acc_df is missing column '{c}'")

    angle_5s = build_anglez_5s_pl(
        acc_df,
        ts_col=ts_col,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        rolling_5s=rolling_5s,
        resample_5s=resample_5s,
        angle_col_out=angle_col_out,
    )

    # 4) mask nonwear (set angle to null inside intervals)
    if nonwear_df is not None and nonwear_df.height > 0:
        if not {"start", "end"}.issubset(set(nonwear_df.columns)):
            raise ValueError("nonwear_df must have datetime columns ['start','end'].")

        intervals = nonwear_df.select(["start", "end"]).to_dicts()
        conds = [(pl.col(ts_col) >= it["start"]) & (pl.col(ts_col) <= it["end"]) for it in intervals]

        if conds:
            is_nonwear = pl.fold(pl.lit(False), lambda acc, x: acc | x, conds).alias("is_nonwear")
            angle_5s = angle_5s.with_columns(is_nonwear).with_columns(
                pl.when(pl.col("is_nonwear"))
                .then(pl.lit(None, dtype=pl.Float64))
                .otherwise(pl.col(angle_col_out))
                .alias(angle_col_out)
            ).drop("is_nonwear")

    # IMPORTANT: SIB logic expects a continuous series; nulls will create breaks.
    # We keep nulls and let diff() propagate nulls; that will naturally split runs.
    bouts = _sib_bouts_from_angle_5s_pl(
        angle_5s,
        ts_col=ts_col,
        angle_col=angle_col_out,
        change_thresh_deg=change_thresh_deg,
        min_bout_minutes=min_bout_minutes,
    )

    out = {"sib_bouts": bouts}

    if return_epochs:
        epochs_labeled = label_epochs_from_bouts(
            angle_5s.select([ts_col, angle_col_out]),
            bouts,
            ts_col=ts_col,
            label_col="sib_sleep",
        ).select([ts_col, angle_col_out, "sib_sleep"])
        out["sib_epochs"] = epochs_labeled

    if return_intermediates:
        out["angle_5s"] = angle_5s

    return out

def vh2015_sib_pd(
    acc: pd.DataFrame,
    nonwear: pd.DataFrame | None = None,
    *,
    # index is datetime
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    rolling_5s: str = "5s",
    resample_5s: str = "5s",
    change_thresh_deg: float = 5.0,
    min_bout_minutes: int = 5,
    return_epochs: bool = True,
    return_intermediates: bool = False,
) -> dict:
    """
    Van Hees 2015 Sustained Inactivity Bouts (SIB) over the whole recording (pandas).

    Input
    -----
    acc: DataFrame indexed by DatetimeIndex, with x/y/z columns.
    nonwear (optional): DataFrame with datetime columns ['start','end'] (any tz-naive/aware
                        is fine as long as it matches acc.index).

    Returns
    -------
    dict with:
      - sib_bouts: DataFrame [start, end, duration]
      - sib_epochs: DataFrame indexed by ts with columns [z_angle_5s, sib_sleep] (if return_epochs)
      - optionally: angle_5s: DataFrame with z_angle_5s
    """
    if not isinstance(acc.index, pd.DatetimeIndex):
        raise ValueError("acc must have a DatetimeIndex.")
    for c in (x_col, y_col, z_col):
        if c not in acc.columns:
            raise ValueError(f"acc is missing column '{c}'")

    angle_5s = build_anglez_5s_pd(
        acc.sort_index(),
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        rolling_5s=rolling_5s,
        resample_5s=resample_5s,
        angle_col_out="z_angle_5s",
    )

    # 4) mask nonwear by setting angle to NaN inside intervals
    if nonwear is not None and len(nonwear) > 0:
        if not {"start", "end"}.issubset(nonwear.columns):
            raise ValueError("nonwear must have columns ['start','end']")
        for s, e in nonwear[["start", "end"]].itertuples(index=False, name=None):
            angle_5s.loc[(angle_5s.index >= s) & (angle_5s.index <= e), "z_angle_5s"] = np.nan

    # --- Van Hees 2015 SIB on the full 5s angle series ---
    d_angle = angle_5s["z_angle_5s"].diff().abs()
    no_change = d_angle <= change_thresh_deg

    # Break runs at NaNs (nonwear) explicitly
    is_nan = angle_5s["z_angle_5s"].isna()
    no_change = no_change & (~is_nan) & (~is_nan.shift(1, fill_value=True))

    # run ids when no_change changes
    group_id = (no_change.ne(no_change.shift(1, fill_value=False))).cumsum()

    tmp = pd.DataFrame({"no_change": no_change, "group_id": group_id}, index=angle_5s.index)

    bouts = (
        tmp.groupby("group_id")
        .agg(value=("no_change", "first"), start=("no_change", lambda s: s.index.min()))
    )
    bouts["end"] = tmp.groupby("group_id").apply(lambda g: g.index.max())
    bouts = bouts.reset_index(drop=True)

    bouts["duration"] = bouts["end"] - bouts["start"]
    bouts = bouts[(bouts["value"]) & (bouts["duration"] >= pd.Timedelta(minutes=min_bout_minutes))]
    sib_bouts = bouts[["start", "end", "duration"]].sort_values("start").reset_index(drop=True)

    out = {"sib_bouts": sib_bouts}

    if return_epochs:   
        # label epochs efficiently: merge_asof on start, then check <= end
        epochs = angle_5s.copy()
        if sib_bouts.empty:
            epochs["sib_sleep"] = False
            out["sib_epochs"] = epochs[["z_angle_5s", "sib_sleep"]]
        else:
            e = epochs.reset_index(names="ts").sort_values("ts")
            b = sib_bouts.sort_values("start")
            joined = pd.merge_asof(e, b, left_on="ts", right_on="start", direction="backward")
            joined["sib_sleep"] = joined["end"].notna() & (joined["ts"] <= joined["end"])
            joined = joined.set_index("ts")
            out["sib_epochs"] = joined[["z_angle_5s", "sib_sleep"]]

    if return_intermediates:
        out["angle_5s"] = angle_5s

    return out

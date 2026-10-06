import numpy as np
import pandas as pd

def sleep_metrics(
    spt_window: pd.DataFrame,
    sib_bouts: pd.DataFrame,
    *,
    day_col: str = "night_id",
    spt_start_col: str = "start",
    spt_end_col: str = "end",
    bout_start_col: str = "start",
    bout_end_col: str = "end",
) -> pd.DataFrame:
    """
    Pandas version of sleep_metrics:
    - spt_window: one row per night_id with start/end of SPT
    - sib_bouts: 0+ rows per night_id with bouts within SPT, must include 'duration' (timedelta)

    Returns one row per night_id with:
      n_bouts, n_awakenings, TST, WASO, sleep_onset_proxy, wake_time_proxy,
      sol_proxy, terminal_wake_proxy, sleep_midpoint, sleep_efficiency,
      awakenings_per_hour_spt, fragmentation_awakenings_per_hour_tst, longest_bout
    """

    # --- validate ---
    required_spt = {day_col, spt_start_col, spt_end_col}
    if not required_spt.issubset(spt_window.columns):
        raise ValueError(f"spt_window must contain columns {required_spt}")

    required_bouts = {day_col, bout_start_col, bout_end_col, "duration"}
    if not required_bouts.issubset(sib_bouts.columns):
        raise ValueError(f"sib_bouts must contain columns {required_bouts}")

    spt = spt_window[[day_col, spt_start_col, spt_end_col]].copy()
    spt = spt.rename(columns={spt_start_col: "spt_start", spt_end_col: "spt_end"})
    spt["spt_start"] = pd.to_datetime(spt["spt_start"], errors="coerce")
    spt["spt_end"]   = pd.to_datetime(spt["spt_end"], errors="coerce")
    spt["spt_dur"]   = spt["spt_end"] - spt["spt_start"]

    # If no bouts, return zeros/NULLs for all nights
    if sib_bouts.empty:
        out = spt.copy()
        out["n_bouts"] = 0
        out["n_awakenings"] = 0
        out["TST"] = pd.Timedelta(0)
        out["WASO"] = pd.Timedelta(0)
        out["sleep_midpoint"] = out["spt_start"] + out["spt_dur"] / 2
        out["sleep_efficiency"] = 0.0
        out["awakenings_per_hour_spt"] = 0.0
        out["fragmentation_awakenings_per_hour_tst"] = 0.0
        out["longest_bout"] = pd.NaT
        return out.sort_values(day_col).reset_index(drop=True)

    bouts = sib_bouts[[day_col, bout_start_col, bout_end_col, "duration"]].copy()
    bouts = bouts.rename(columns={bout_start_col: "bout_start", bout_end_col: "bout_end", "duration": "bout_dur"})
    bouts["bout_start"] = pd.to_datetime(bouts["bout_start"], errors="coerce")
    bouts["bout_end"]   = pd.to_datetime(bouts["bout_end"], errors="coerce")
    bouts["bout_dur"]   = pd.to_timedelta(bouts["bout_dur"], errors="coerce")
    bouts = bouts.dropna(subset=["bout_start", "bout_end", "bout_dur"])
    bouts = bouts.sort_values([day_col, "bout_start"]).reset_index(drop=True)

    # --- per-night aggregates ---
    agg = (
        bouts.groupby(day_col, as_index=False)
        .agg(
            n_bouts=("bout_dur", "size"),
            TST=("bout_dur", "sum"),
            longest_bout=("bout_dur", "max"),
            sleep_onset_proxy=("bout_start", "min"),
            wake_time_proxy=("bout_end", "max"),
        )
    )

    # --- gaps between bouts within night ---
    bouts["prev_end"] = bouts.groupby(day_col)["bout_end"].shift(1)
    bouts["gap"] = bouts["bout_start"] - bouts["prev_end"]

    gaps = (
        bouts.dropna(subset=["gap"])
        .groupby(day_col, as_index=False)
        .agg(
            n_awakenings=("gap", "size"),
            WASO=("gap", "sum"),
            max_wake_gap=("gap", "max"),
        )
    )

    # --- join to SPT ---
    out = spt.merge(agg, on=day_col, how="left").merge(gaps, on=day_col, how="left")

    # Fill nulls for nights with SPT but no bouts (rare here, but safe)
    out["n_bouts"] = out["n_bouts"].fillna(0).astype(int)
    out["n_awakenings"] = out["n_awakenings"].fillna(0).astype(int)
    out["TST"] = out["TST"].fillna(pd.Timedelta(0))
    out["WASO"] = out["WASO"].fillna(pd.Timedelta(0))
    out["sleep_midpoint"] = out["spt_start"] + out["spt_dur"] / 2

    # Sleep efficiency
    spt_sec = out["spt_dur"].dt.total_seconds()
    tst_sec = out["TST"].dt.total_seconds()

    out["sleep_efficiency"] = np.where(spt_sec > 0, tst_sec / spt_sec, np.nan)

    # Awakenings per hour of SPT
    out["awakenings_per_hour_spt"] = np.where(
        spt_sec > 0, out["n_awakenings"] / (spt_sec / 3600.0), 0.0
    )

    # Fragmentation: awakenings per hour of TST
    out["fragmentation_awakenings_per_hour_tst"] = np.where(
        tst_sec > 0, out["n_awakenings"] / (tst_sec / 3600.0), 0.0
    )

    # Clamp WASO to not exceed (SPT - TST)
    max_waso = (out["spt_dur"] - out["TST"]).clip(lower=pd.Timedelta(0))
    out["WASO"] = np.minimum(out["WASO"].values.astype("timedelta64[ns]"),
                             max_waso.values.astype("timedelta64[ns]")).astype("timedelta64[ns]")

    return out.sort_values(day_col).reset_index(drop=True)


def _normalize_interval_table(
    df: pd.DataFrame,
    *,
    start_col: str,
    end_col: str,
    start_name: str,
    end_name: str,
    id_col: str | None = None,
    id_name: str = "night_id",
) -> pd.DataFrame:
    cols = [start_col, end_col]
    if id_col is not None and id_col in df.columns:
        cols.append(id_col)

    out = df[cols].copy()
    out = out.rename(columns={start_col: start_name, end_col: end_name})
    if id_col is not None and id_col in df.columns:
        out = out.rename(columns={id_col: id_name})

    out[start_name] = pd.to_datetime(out[start_name], errors="coerce")
    out[end_name] = pd.to_datetime(out[end_name], errors="coerce")
    out = out.dropna(subset=[start_name, end_name])
    out = out[out[end_name] >= out[start_name]].sort_values(start_name).reset_index(drop=True)

    if id_name not in out.columns:
        out[id_name] = np.arange(len(out), dtype=int)

    return out


def _check_non_overlapping_windows(
    guider: pd.DataFrame,
    *,
    start_col: str = "guider_start",
    end_col: str = "guider_end",
) -> None:
    if guider.empty:
        return
    prev_end = guider[end_col].shift(1)
    overlap = guider[start_col] < prev_end
    if overlap.fillna(False).any():
        raise ValueError(
            "Guider windows overlap in time. This would make SIB-to-guider assignment ambiguous."
        )


def compute_sleep_from_guider(
    guider_df: pd.DataFrame,
    sib_bouts: pd.DataFrame,
    nonwear_periods_final: pd.DataFrame,
    *,
    guider_start_col: str = "start",
    guider_end_col: str = "end",
    guider_id_col: str | None = "night_id",
    sib_start_col: str = "start",
    sib_end_col: str = "end",
    nw_start_col: str = "start",
    nw_end_col: str = "end",
    sib_assignment_mode: str = "overlap",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute nightly sleep metrics from a generic sleep guider plus SIB bouts.

    The guider can be any interval source that approximately defines the sleep
    period, for example:
    - diary / TIB
    - lower-back derived lying interval
    - HDCZA-derived SPT

    GGIR-like edge handling:
    - with `sib_assignment_mode="overlap"`:
      a SIB is assigned to a guider night if it overlaps that guider at all
    - with `sib_assignment_mode="contained"`:
      a SIB is assigned only if it is fully contained within the guider window
    - sleep onset is the start of the first assigned SIB
    - waking time is the end of the last assigned SIB
    - therefore, with the default overlap mode, an edge SIB may extend outside
      the guider and still define the final timing rather than the guider boundary

    Returns
    -------
    guider_out:
      one row per guider interval with the original guider timing, derived SPT,
      non-wear inside guider, and sleep metrics.
    sib_in_guider:
      SIB bouts that overlap each guider interval, with night_id attached.
    """
    guider = _normalize_interval_table(
        guider_df,
        start_col=guider_start_col,
        end_col=guider_end_col,
        start_name="guider_start",
        end_name="guider_end",
        id_col=guider_id_col,
        id_name="night_id",
    )
    _check_non_overlapping_windows(guider, start_col="guider_start", end_col="guider_end")

    if sib_assignment_mode not in {"overlap", "contained"}:
        raise ValueError(
            f"Unsupported sib_assignment_mode {sib_assignment_mode!r}. "
            "Use 'overlap' or 'contained'."
        )

    sib = _normalize_interval_table(
        sib_bouts,
        start_col=sib_start_col,
        end_col=sib_end_col,
        start_name="start",
        end_name="end",
        id_col=None,
    )
    sib["duration"] = sib["end"] - sib["start"]

    nw = _normalize_interval_table(
        nonwear_periods_final,
        start_col=nw_start_col,
        end_col=nw_end_col,
        start_name="start",
        end_name="end",
        id_col=None,
    )

    guider["guider_dur"] = guider["guider_end"] - guider["guider_start"]

    if sib.empty:
        sib_in_guider = pd.DataFrame(columns=["night_id", "start", "end", "duration"])
    else:
        guider_cross = guider[["night_id", "guider_start", "guider_end"]].assign(_k=1)
        sib_cross = sib[["start", "end", "duration"]].assign(_k=1)
        sib_join = guider_cross.merge(sib_cross, on="_k").drop(columns="_k")
        if sib_assignment_mode == "overlap":
            overlap = (sib_join["start"] < sib_join["guider_end"]) & (sib_join["end"] > sib_join["guider_start"])
        else:
            overlap = (sib_join["start"] >= sib_join["guider_start"]) & (sib_join["end"] <= sib_join["guider_end"])
        sib_in_guider = (
            sib_join.loc[overlap, ["night_id", "start", "end", "duration"]]
            .sort_values(["night_id", "start", "end"])
            .reset_index(drop=True)
        )

    if sib_in_guider.empty:
        spt_window = pd.DataFrame(columns=["night_id", "spt_start", "spt_end", "spt_dur"])
        metrics = pd.DataFrame(columns=["night_id"])
    else:
        spt_window = (
            sib_in_guider.groupby("night_id", as_index=False)
            .agg(
                spt_start=("start", "min"),
                spt_end=("end", "max"),
            )
            .sort_values("night_id")
            .reset_index(drop=True)
        )
        spt_window["spt_dur"] = spt_window["spt_end"] - spt_window["spt_start"]

        metrics = sleep_metrics(
            spt_window=spt_window,
            sib_bouts=sib_in_guider,
            day_col="night_id",
            spt_start_col="spt_start",
            spt_end_col="spt_end",
            bout_start_col="start",
            bout_end_col="end",
        ).drop(columns=["spt_start", "spt_end", "spt_dur"], errors="ignore")

    if nw.empty:
        guider_nw = pd.DataFrame(columns=["night_id", "nonwear_dur_in_guider"])
    else:
        guider_cross = guider[["night_id", "guider_start", "guider_end"]].assign(_k=1)
        nw_cross = nw[["start", "end"]].assign(_k=1)
        nw_join = guider_cross.merge(nw_cross, on="_k").drop(columns="_k")

        ov_start = nw_join[["start", "guider_start"]].max(axis=1)
        ov_end = nw_join[["end", "guider_end"]].min(axis=1)
        ov_dur = (ov_end - ov_start).clip(lower=pd.Timedelta(0))
        nw_join["ov_dur"] = ov_dur

        guider_nw = (
            nw_join.loc[nw_join["ov_dur"] > pd.Timedelta(0), ["night_id", "ov_dur"]]
            .groupby("night_id", as_index=False)["ov_dur"]
            .sum()
            .rename(columns={"ov_dur": "nonwear_dur_in_guider"})
        )

    guider_out = (
        guider.merge(spt_window, on="night_id", how="left")
        .merge(guider_nw, on="night_id", how="left")
        .merge(metrics, on="night_id", how="left")
    )
    guider_out["nonwear_dur_in_guider"] = guider_out["nonwear_dur_in_guider"].fillna(pd.Timedelta(0))
    guider_out["spt_found"] = guider_out["spt_start"].notna() & guider_out["spt_end"].notna()

    return guider_out.sort_values("guider_start").reset_index(drop=True), sib_in_guider


def compute_sleep_from_tib(
    tib_df: pd.DataFrame,
    sib_bouts: pd.DataFrame,
    nonwear_periods_final: pd.DataFrame,
    *,
    tib_start_col: str = "start",
    tib_end_col: str = "end",
    sib_start_col: str = "start",
    sib_end_col: str = "end",
    nw_start_col: str = "start",
    nw_end_col: str = "end",
    merge_gap: str | pd.Timedelta = "10 min",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backward-compatible wrapper around `compute_sleep_from_guider()` for diary/TIB windows.

    The `merge_gap` argument is retained for API compatibility and is currently
    unused.
    """
    guider_out, sib_in_guider = compute_sleep_from_guider(
        guider_df=tib_df,
        sib_bouts=sib_bouts,
        nonwear_periods_final=nonwear_periods_final,
        guider_start_col=tib_start_col,
        guider_end_col=tib_end_col,
        guider_id_col="night_id" if "night_id" in tib_df.columns else None,
        sib_start_col=sib_start_col,
        sib_end_col=sib_end_col,
        nw_start_col=nw_start_col,
        nw_end_col=nw_end_col,
    )

    tib_out = guider_out.rename(
        columns={
            "guider_start": "tib_start",
            "guider_end": "tib_end",
            "guider_dur": "tib_dur",
            "nonwear_dur_in_guider": "nonwear_dur_in_tib",
        }
    )
    return tib_out.sort_values("tib_start").reset_index(drop=True), sib_in_guider


def compute_sleep_from_spt(
    spt_df: pd.DataFrame,
    sib_bouts: pd.DataFrame,
    nonwear_periods_final: pd.DataFrame,
    *,
    spt_start_col: str = "spt_start",
    spt_end_col: str = "spt_end",
    sib_start_col: str = "start",
    sib_end_col: str = "end",
    nw_start_col: str = "start",
    nw_end_col: str = "end",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backward-compatible wrapper around `compute_sleep_from_guider()` for an
    already supplied SPT-like guider.
    """
    return compute_sleep_from_guider(
        guider_df=spt_df,
        sib_bouts=sib_bouts,
        nonwear_periods_final=nonwear_periods_final,
        guider_start_col=spt_start_col,
        guider_end_col=spt_end_col,
        guider_id_col="night_id" if "night_id" in spt_df.columns else None,
        sib_start_col=sib_start_col,
        sib_end_col=sib_end_col,
        nw_start_col=nw_start_col,
        nw_end_col=nw_end_col,
    )

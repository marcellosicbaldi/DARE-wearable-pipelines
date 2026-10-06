import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dare_wearables.lower_back.utils.bool_to_bouts import bool_to_bouts

def detect_lying_lowerback(
    df: pd.DataFrame,
    acc_cols=("acc_x", "acc_y", "acc_z"),
    gyr_cols=("gyr_x", "gyr_y", "gyr_z"),
    wear_col="wear",
    wear_1hz_override: pd.Series | None = None,
    out_fs_hz=1,
    grav_win_s=7,
    act_win_s=15,
    gyro_active_thr=8.0,
    vm_std_active_thr=0.25,
    lie_on_deg=65.0,
    lie_off_deg=45.0,
    lie_max_deg=130.0,   # NEW: upper threshold for valid lying
    min_bout_s=60,
    merge_gap_s=10,
):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex.")

    d = df.copy()

    rule = f"{int(1000/out_fs_hz)}ms" if out_fs_hz > 1 else "1s"

    acc_1 = d.loc[:, acc_cols].resample(rule).mean()
    gyr_1 = d.loc[:, gyr_cols].resample(rule).mean()

    # --- wear at 1 Hz ---
    if wear_1hz_override is not None:
        wear_1 = (
            wear_1hz_override.sort_index()
            .reindex(acc_1.index, method="ffill")
            .fillna(False)
            .astype(bool)
        )
    else:
        if wear_col in d.columns:
            wear_1 = d[wear_col].astype(float).resample(rule).mean() >= 0.5
        else:
            wear_1 = pd.Series(True, index=acc_1.index)

    acc_1 = acc_1.where(wear_1)
    gyr_1 = gyr_1.where(wear_1)

    # --- gravity direction ---
    grav_win = max(3, int(round(grav_win_s * out_fs_hz)))
    gvec = acc_1.rolling(grav_win, center=True, min_periods=grav_win // 2).mean()

    gnorm = np.sqrt((gvec**2).sum(axis=1))
    u = gvec.div(gnorm, axis=0)

    # --- activity proxies ---
    gyro_rms = np.sqrt((gyr_1**2).mean(axis=1))
    vm = np.sqrt((acc_1**2).sum(axis=1))
    act_win = max(5, int(round(act_win_s * out_fs_hz)))
    vm_std = vm.rolling(act_win, center=True, min_periods=act_win // 2).std()

    active = (gyro_rms > gyro_active_thr) | (vm_std > vm_std_active_thr)
    active = active & wear_1 & gnorm.notna()

    if active.sum() < 60:
        u_ref = u.median(skipna=True).to_numpy(dtype=float)
    else:
        u_ref = u.loc[active].median(skipna=True).to_numpy(dtype=float)

    nr = np.linalg.norm(u_ref)
    if not np.isfinite(nr) or nr == 0:
        raise RuntimeError("Could not estimate a valid upright reference vector.")
    u_ref = u_ref / nr

    dot = (u * u_ref).sum(axis=1).clip(-1.0, 1.0)
    theta = np.degrees(np.arccos(dot)).where(wear_1)

    # --- hysteresis with upper threshold ---
    th = theta.to_numpy()
    wear_np = wear_1.to_numpy()

    lying = np.zeros(len(theta), dtype=bool)
    in_lie = False

    for i in range(len(th)):
        if not wear_np[i] or not np.isfinite(th[i]):
            in_lie = False
            lying[i] = False
            continue

        over_max = (lie_max_deg is not None) and (th[i] > lie_max_deg)

        if not in_lie:
            if (th[i] >= lie_on_deg) and (not over_max):
                in_lie = True
        else:
            if (th[i] <= lie_off_deg) or over_max:
                in_lie = False

        lying[i] = in_lie

    labels = pd.Series(lying, index=theta.index, name="lying")

    # --- bouts ---
    bouts = bool_to_bouts(labels, state=True).copy()
    if bouts.empty:
        bouts = pd.DataFrame(columns=["start", "end", "duration_s"])
        return labels, theta.rename("theta_deg"), bouts

    bouts["duration_s"] = (bouts["end"] - bouts["start"]).dt.total_seconds()

    # merge short gaps between lying bouts
    if merge_gap_s and merge_gap_s > 0:
        merged = []
        cur_s = bouts.loc[0, "start"]
        cur_e = bouts.loc[0, "end"]

        for i in range(1, len(bouts)):
            s, e = bouts.loc[i, "start"], bouts.loc[i, "end"]
            gap = (s - cur_e).total_seconds()
            if gap <= merge_gap_s:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e

        merged.append((cur_s, cur_e))
        bouts = pd.DataFrame(merged, columns=["start", "end"])
        bouts["duration_s"] = (bouts["end"] - bouts["start"]).dt.total_seconds()

    # drop short bouts
    bouts = bouts[bouts["duration_s"] >= float(min_bout_s)].reset_index(drop=True)

    return labels, theta.rename("theta_deg"), bouts

def fill_short_nonwear_between_lying(
    wear_1hz: pd.Series,
    lying_1hz: pd.Series,
    max_nonwear="40min",
    edge_tol="2min",
):
    """
    If a nonwear bout is <= max_nonwear and there is lying within edge_tol
    immediately before and after, flip that nonwear interval to wear=True.

    Returns:
      wear_fixed_1hz, nonwear_bouts, nonwear_bouts_filled
    """
    wear = wear_1hz.sort_index().fillna(False).astype(bool)
    lying = lying_1hz.reindex(wear.index).fillna(False).astype(bool)

    nonwear_bouts = bool_to_bouts(~wear, state=True)  # bouts where nonwear=True
    if nonwear_bouts.empty:
        return wear, nonwear_bouts, nonwear_bouts

    max_s = pd.Timedelta(max_nonwear).total_seconds()
    tol = pd.Timedelta(edge_tol)

    wear_fixed = wear.copy()
    filled = []

    for _, r in nonwear_bouts.iterrows():
        s, e = r["start"], r["end"]
        dur_s = (e - s).total_seconds()
        if dur_s > max_s:
            continue

        # check for lying immediately before and after the gap (with tolerance windows)
        pre = lying.loc[max(lying.index[0], s - tol): s]
        post = lying.loc[e: min(lying.index[-1], e + tol)]

        if pre.any() and post.any():
            wear_fixed.loc[s:e] = True
            filled.append((s, e))

    nonwear_filled = pd.DataFrame(filled, columns=["start", "end"])
    return wear_fixed, nonwear_bouts, nonwear_filled

def vanhees_tib_from_bouts(
    lying_bouts: pd.DataFrame,
    min_block="30min",
    gap_merge="30min",
    day_offset_hours=12,
    tiny_gap_merge="2min",   # NEW: pre-merge tiny gaps before Step 7
    merge_before_filter=False, # NEW: whether to merge before or after filtering by min_block (default: filter first, then merge)
):
    """
    lying_bouts: DataFrame with columns ['start','end'] (datetime64), optionally duration_s.

    Workflow:
      - Pre-merge tiny gaps (e.g., <2 min) between all lying bouts
      - Merge candidate lying bouts across gaps <= gap_merge
      - Keep merged blocks >= min_block
      - Step 9: longest block in each noon->noon day

    Returns:
      blocks_merged: merged candidate blocks after pre-merge + filtering
      tib_df: per noon-noon day, longest block clipped to that day window (step 9)
    """
    empty_tib = pd.DataFrame(columns=[
        "day_start", "day_end", "tib_start", "tib_end", "tib_duration_s",
        "block_start", "block_end"
    ])

    if lying_bouts.empty:
        return lying_bouts.copy(), empty_tib

    b = lying_bouts.copy()
    b["start"] = pd.to_datetime(b["start"])
    b["end"] = pd.to_datetime(b["end"])
    b = b.sort_values("start").reset_index(drop=True)

    # Remove invalid rows if any
    b = b[b["end"] > b["start"]].copy().reset_index(drop=True)
    if b.empty:
        return b, empty_tib

    def _merge_by_gap(df_blocks: pd.DataFrame, max_gap_s: float) -> pd.DataFrame:
        """Merge consecutive blocks if gap <= max_gap_s (also merges overlaps)."""
        if df_blocks.empty:
            return df_blocks.copy()

        df_blocks = df_blocks.sort_values("start").reset_index(drop=True)

        merged = []
        cur_start = df_blocks.loc[0, "start"]
        cur_end = df_blocks.loc[0, "end"]

        for i in range(1, len(df_blocks)):
            s = df_blocks.loc[i, "start"]
            e = df_blocks.loc[i, "end"]
            gap = (s - cur_end).total_seconds()

            if gap <= max_gap_s:
                cur_end = max(cur_end, e)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e

        merged.append((cur_start, cur_end))

        out = pd.DataFrame(merged, columns=["start", "end"])
        out["duration_s"] = (out["end"] - out["start"]).dt.total_seconds()
        return out

    # --- NEW: pre-merge tiny gaps before Step 7 ---
    if tiny_gap_merge is not None:
        tiny_gap_s = pd.Timedelta(tiny_gap_merge).total_seconds()
        b = _merge_by_gap(b, tiny_gap_s)
    else:
        b["duration_s"] = (b["end"] - b["start"]).dt.total_seconds()

    min_block_s = pd.Timedelta(min_block).total_seconds()
    gap_s = pd.Timedelta(gap_merge).total_seconds()

    if merge_before_filter:
        # For lower-back TIB, we first merge plausible within-night interruptions
        # and only then enforce the final minimum duration on the resulting block.
        blocks_merged = _merge_by_gap(b, gap_s).reset_index(drop=True)
        blocks_merged = blocks_merged[blocks_merged["duration_s"] >= min_block_s].copy().reset_index(drop=True)
    else:
        # Legacy behavior: keep only blocks >= min_block, then merge the survivors.
        b = b[b["duration_s"] >= min_block_s].copy().reset_index(drop=True)
        if b.empty:
            return b, empty_tib
        blocks_merged = _merge_by_gap(b, gap_s).reset_index(drop=True)

    if blocks_merged.empty:
        return blocks_merged, empty_tib

    # Step 9: for each noon-noon day, pick longest block overlap
    offset = pd.Timedelta(hours=day_offset_hours)

    # analysis-day label = date of (timestamp - offset)
    day_min = (blocks_merged["start"] - offset).dt.floor("D").min()
    day_max = (blocks_merged["end"] - offset).dt.floor("D").max()

    days = pd.date_range(day_min, day_max, freq="D")
    rows = []

    for day in days:
        day_start = day + offset
        day_end = day_start + pd.Timedelta(days=1)

        # overlap with each merged block
        ov_start = np.maximum(
            blocks_merged["start"].values.astype("datetime64[ns]"),
            np.datetime64(day_start)
        )
        ov_end = np.minimum(
            blocks_merged["end"].values.astype("datetime64[ns]"),
            np.datetime64(day_end)
        )

        ov_dur_s = (ov_end - ov_start) / np.timedelta64(1, "s")
        ov_dur_s = np.where(ov_dur_s > 0, ov_dur_s, 0)

        if len(ov_dur_s) == 0 or ov_dur_s.max() <= 0:
            continue

        j = int(np.argmax(ov_dur_s))
        tib_start = pd.Timestamp(ov_start[j])
        tib_end = pd.Timestamp(ov_end[j])

        rows.append({
            "day_start": day_start,
            "day_end": day_end,
            "tib_start": tib_start,
            "tib_end": tib_end,
            "tib_duration_s": float(ov_dur_s[j]),
            "block_start": blocks_merged.loc[j, "start"],
            "block_end": blocks_merged.loc[j, "end"],
        })

    tib_df = pd.DataFrame(rows)
    return blocks_merged, tib_df

def detect_naps(
    lying_bouts: pd.DataFrame,
    tib_df: pd.DataFrame,
    min_nap="20min",
    nap_gap_merge="10min",
    include_edge_intervals=True,
    recording_start=None,
    recording_end=None,
):
    """
    Detect nap candidates as lying bouts outside the main TIB windows.

    By default, naps are searched in:
      - before the first TIB (recording start -> first TIB start)
      - between consecutive TIBs
      - after the last TIB (last TIB end -> recording end)

    Parameters
    ----------
    lying_bouts : pd.DataFrame
        DataFrame with columns ['start', 'end'] (datetime-like).
        These should be corrected lying bouts (ideally after non-wear correction).
    tib_df : pd.DataFrame
        DataFrame with columns ['tib_start', 'tib_end'] (datetime-like),
        containing one main TIB interval per day.
    min_nap : str or Timedelta-like, default "20min"
        Minimum duration required for a nap candidate.
    nap_gap_merge : str or Timedelta-like, default "10min"
        Maximum gap allowed to merge adjacent lying bouts within a nap-search interval.
    include_edge_intervals : bool, default True
        If True, also search before the first TIB and after the last TIB.
    recording_start : datetime-like or None
        Optional explicit recording start. If None, inferred from lying_bouts['start'].min().
    recording_end : datetime-like or None
        Optional explicit recording end. If None, inferred from lying_bouts['end'].max().

    Returns
    -------
    naps_df : pd.DataFrame
        One row per nap candidate with columns:
        ['nap_start', 'nap_end', 'nap_duration_s', 'nap_duration_min',
         'interval_type', 'interval_start', 'interval_end',
         'between_tib_end', 'and_next_tib_start', 'is_ge_30min']
    """
    empty = pd.DataFrame(columns=[
        "nap_start", "nap_end", "nap_duration_s", "nap_duration_min",
        "interval_type", "interval_start", "interval_end",
        "between_tib_end", "and_next_tib_start", "is_ge_30min"
    ])

    if lying_bouts.empty or tib_df.empty:
        return empty

    # --- Prepare lying bouts ---
    lb = lying_bouts.copy()
    lb["start"] = pd.to_datetime(lb["start"])
    lb["end"] = pd.to_datetime(lb["end"])
    lb = lb.sort_values("start").reset_index(drop=True)
    lb = lb[lb["end"] > lb["start"]].copy().reset_index(drop=True)
    if lb.empty:
        return empty

    # --- Prepare TIB ---
    t = tib_df.copy()
    t["tib_start"] = pd.to_datetime(t["tib_start"])
    t["tib_end"] = pd.to_datetime(t["tib_end"])
    t = t.sort_values("tib_start").reset_index(drop=True)
    t = t[t["tib_end"] > t["tib_start"]].copy().reset_index(drop=True)
    if t.empty:
        return empty

    # Recording bounds (optional explicit, otherwise inferred from lying bouts)
    rec_start = pd.to_datetime(recording_start) if recording_start is not None else lb["start"].min()
    rec_end = pd.to_datetime(recording_end) if recording_end is not None else lb["end"].max()

    min_nap_s = pd.Timedelta(min_nap).total_seconds()
    nap_gap_s = pd.Timedelta(nap_gap_merge).total_seconds()

    def _merge_by_gap(df_blocks: pd.DataFrame, max_gap_s: float) -> pd.DataFrame:
        if df_blocks.empty:
            return df_blocks.copy()

        df_blocks = df_blocks.sort_values("start").reset_index(drop=True)

        merged = []
        cur_s = df_blocks.loc[0, "start"]
        cur_e = df_blocks.loc[0, "end"]

        for i in range(1, len(df_blocks)):
            s = df_blocks.loc[i, "start"]
            e = df_blocks.loc[i, "end"]
            gap = (s - cur_e).total_seconds()

            if gap <= max_gap_s:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e

        merged.append((cur_s, cur_e))
        return pd.DataFrame(merged, columns=["start", "end"])

    # --- Build nap-search intervals ---
    intervals = []

    if include_edge_intervals and pd.notna(rec_start):
        # Before first TIB
        first_tib_start = t.loc[0, "tib_start"]
        if first_tib_start > rec_start:
            intervals.append({
                "interval_type": "before_first_tib",
                "interval_start": rec_start,
                "interval_end": first_tib_start,
                "between_tib_end": pd.NaT,
                "and_next_tib_start": first_tib_start,
            })

    # Between consecutive TIBs
    for i in range(len(t) - 1):
        gap_start = t.loc[i, "tib_end"]
        gap_end = t.loc[i + 1, "tib_start"]
        if gap_end <= gap_start:
            continue

        intervals.append({
            "interval_type": "between_tib",
            "interval_start": gap_start,
            "interval_end": gap_end,
            "between_tib_end": gap_start,
            "and_next_tib_start": gap_end,
        })

    if include_edge_intervals and pd.notna(rec_end):
        # After last TIB
        last_tib_end = t.loc[len(t) - 1, "tib_end"]
        if rec_end > last_tib_end:
            intervals.append({
                "interval_type": "after_last_tib",
                "interval_start": last_tib_end,
                "interval_end": rec_end,
                "between_tib_end": last_tib_end,
                "and_next_tib_start": pd.NaT,
            })

    if len(intervals) == 0:
        return empty

    rows = []

    # --- Search naps within each interval ---
    for itv in intervals:
        gap_start = itv["interval_start"]
        gap_end = itv["interval_end"]

        # Lying bouts overlapping this interval
        cand = lb[(lb["end"] > gap_start) & (lb["start"] < gap_end)].copy()
        if cand.empty:
            continue

        # Clip to interval
        cand["start"] = cand["start"].clip(lower=gap_start, upper=gap_end)
        cand["end"] = cand["end"].clip(lower=gap_start, upper=gap_end)
        cand = cand[cand["end"] > cand["start"]].copy()
        if cand.empty:
            continue

        # Merge brief interruptions
        cand = _merge_by_gap(cand[["start", "end"]], nap_gap_s)

        # Keep only long enough
        cand["nap_duration_s"] = (cand["end"] - cand["start"]).dt.total_seconds()
        cand = cand[cand["nap_duration_s"] >= min_nap_s].copy()
        if cand.empty:
            continue

        for _, r in cand.iterrows():
            rows.append({
                "nap_start": r["start"],
                "nap_end": r["end"],
                "nap_duration_s": float(r["nap_duration_s"]),
                "nap_duration_min": float(r["nap_duration_s"] / 60.0),
                "interval_type": itv["interval_type"],
                "interval_start": itv["interval_start"],
                "interval_end": itv["interval_end"],
                "between_tib_end": itv["between_tib_end"],
                "and_next_tib_start": itv["and_next_tib_start"],
                "is_ge_30min": bool(r["nap_duration_s"] >= 30 * 60),
            })

    naps_df = pd.DataFrame(rows)
    if naps_df.empty:
        return empty

    return naps_df.sort_values("nap_start").reset_index(drop=True)

def plot_week_tib(theta_1hz, labels_1hz, tib_df, subject_id = None, naps_df=None, save_path=None,
                  tib_y=170, nap_y=155, line_width=16):
    """
    Single plot over the whole recording:
      - theta line
      - light lying shading
      - TIB as red horizontal lines
      - Naps as magenta horizontal lines (optional)
      - day boundaries at tib_df['day_start']

    Parameters
    ----------
    theta_1hz : pd.Series
        Angle-to-upright signal (degrees), indexed by datetime.
    labels_1hz : pd.Series
        Boolean lying label, indexed by datetime.
    tib_df : pd.DataFrame
        Must contain ['tib_start', 'tib_end'] and ideally ['day_start'].
    save_path : str or None
        If provided, saves figure; otherwise shows it.
    naps_df : pd.DataFrame or None
        Optional. Must contain ['nap_start', 'nap_end'].
    tib_y : float
        Y-position for TIB horizontal lines.
    nap_y : float
        Y-position for nap horizontal lines (set equal to tib_y if you want same row).
    line_width : float
        Width of TIB/nap horizontal lines.
    """
    th = theta_1hz.dropna()
    if th.empty:
        print("theta_1hz is empty.")
        return

    lab = labels_1hz.reindex(th.index).fillna(False).astype(bool)

    fig, ax = plt.subplots(1, 1, figsize=(21, 6))
    ax.plot(th.index, th.values, label="Theta")
    ax.set_ylabel("theta (deg)")
    if subject_id is not None:
        ax.set_title(f"Subject {subject_id}: Theta angle + lying + TIB + naps")
    else:
        ax.set_title("Theta angle + lying + TIB + naps")
    ax.grid(True, alpha=0.3)

    # --- Shade lying segments ---
    in_seg = False
    seg_start = None
    lying_label_added = False

    for t, is_lie in lab.items():
        if is_lie and not in_seg:
            in_seg = True
            seg_start = t
        elif (not is_lie) and in_seg:
            ax.axvspan(
                seg_start, t,
                alpha=0.25,
                label="Lying" if not lying_label_added else None
            )
            lying_label_added = True
            in_seg = False

    if in_seg:
        ax.axvspan(
            seg_start, lab.index[-1],
            alpha=0.25,
            label="Lying" if not lying_label_added else None
        )

    # --- TIB windows as red hlines ---
    if tib_df is not None and not tib_df.empty:
        for i, r in tib_df.iterrows():
            ax.hlines(
                y=tib_y,
                xmin=pd.Timestamp(r["tib_start"]),
                xmax=pd.Timestamp(r["tib_end"]),
                color="red",
                linewidth=line_width,
                label="Time in Bed" if i == 0 else None
            )

        # day boundary lines (if available)
        if "day_start" in tib_df.columns:
            for w0 in tib_df["day_start"].dropna().sort_values().unique():
                ax.axvline(pd.Timestamp(w0), linewidth=1, alpha=0.3)

    # --- Nap windows as magenta hlines ---
    if naps_df is not None and not naps_df.empty:
        for i, r in naps_df.iterrows():
            ax.hlines(
                y=nap_y,
                xmin=pd.Timestamp(r["nap_start"]),
                xmax=pd.Timestamp(r["nap_end"]),
                color="magenta",
                linewidth=line_width,
                label="Nap" if i == 0 else None
            )

    # Optional: make sure line bars are visible even if theta range is small
    y_min = min(float(th.min()), nap_y - 10 if naps_df is not None and not naps_df.empty else tib_y - 10)
    y_max = max(float(th.max()), tib_y + 10)
    ax.set_ylim(y_min, y_max)

    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def qc_tib_night_wear(
    tib_df: pd.DataFrame,
    wear_1hz: pd.Series,
    *,
    night_start_h=18,
    night_end_h=11,
    min_wear_hours_night=8.0,
    max_cont_nonwear_hours_night=3.0,
    min_wear_frac_in_tib=0.85,
    tib_min_h=3.0,
    tib_max_h=14.0,
    min_overlap_night_h=4.0,
):
    """
    Quality-control daily TIB windows to handle cases where the sensor is removed during the night.

    Parameters
    ----------
    tib_df : pd.DataFrame
        Must contain ['day_start','day_end','tib_start','tib_end'] (datetime-like).
        Typically one row per noon->noon day.
    wear_1hz : pd.Series (bool)
        Wear mask at 1 Hz indexed by datetime.
        Recommended: use wear_fixed_1hz (after filling short false nonwear gaps),
        because it avoids rejecting nights due to spurious nonwear artifacts.
    night_start_h, night_end_h : int
        Night interval in clock hours: [18:00, next day 11:00] by default.
    Thresholds : floats
        See variable names.

    Returns
    -------
    tib_qc : pd.DataFrame
        tib_df plus QC metrics + columns:
        ['wear_hours_night','max_cont_nonwear_h_night','wear_frac_in_tib',
         'tib_duration_h','overlap_night_h','tib_valid','invalid_reasons']
    tib_valid : pd.DataFrame
        Subset of tib_qc where tib_valid == True
    """

    if tib_df is None or tib_df.empty:
        tib_qc = pd.DataFrame(columns=list(tib_df.columns) if tib_df is not None else [])
        tib_qc["tib_valid"] = pd.Series(dtype=bool)
        tib_qc["invalid_reasons"] = pd.Series(dtype=str)
        return tib_qc, tib_qc.copy()

    w = wear_1hz.sort_index().fillna(False).astype(bool)

    tib = tib_df.copy()
    for c in ["day_start", "day_end", "tib_start", "tib_end"]:
        tib[c] = pd.to_datetime(tib[c])

    # Helper: compute overlap (hours) between [a0,a1] and [b0,b1]
    def overlap_hours(a0, a1, b0, b1):
        s = max(a0, b0)
        e = min(a1, b1)
        return max(0.0, (e - s).total_seconds() / 3600.0)

    # Helper: maximum continuous nonwear bout overlap with interval
    def max_continuous_nonwear_hours(interval_start, interval_end):
        # slice to interval for efficiency
        ww = w.loc[interval_start:interval_end]
        if ww.empty:
            return 0.0

        nonwear = (~ww).astype(int)

        # run-length encode nonwear
        change = nonwear.ne(nonwear.shift())
        change.iloc[0] = True
        run_id = change.cumsum()

        # get starts for each run
        starts = nonwear.index[change]
        dt = (nonwear.index[1] - nonwear.index[0]) if len(nonwear.index) > 1 else pd.Timedelta(seconds=1)
        ends = starts[1:].append(pd.DatetimeIndex([nonwear.index[-1] + dt]))

        states = nonwear.groupby(run_id).first().to_numpy()

        # compute max duration where state==1
        max_h = 0.0
        for st, s, e in zip(states, starts, ends):
            if st != 1:
                continue
            max_h = max(max_h, overlap_hours(s, e, interval_start, interval_end))
        return max_h

    out = []
    for _, r in tib.iterrows():
        day_start = r["day_start"]
        day_end = r["day_end"]
        tib_start = r["tib_start"]
        tib_end = r["tib_end"]

        # Define night window within the day (18:00 -> next day 11:00)
        base = day_start.normalize()  # midnight of day_start date
        night_start = base + pd.Timedelta(hours=night_start_h)
        night_end = base + pd.Timedelta(days=1, hours=night_end_h)

        # guard: clip to available wear index
        # (not strictly necessary, but avoids empty slices if day edges exceed recording)
        rec_start, rec_end = w.index.min(), w.index.max()
        ns = max(night_start, rec_start)
        ne = min(night_end, rec_end)

        # --- Metrics ---
        # Wear hours in night
        ww_night = w.loc[ns:ne]
        wear_hours_night = float(ww_night.sum() / 3600.0) if len(ww_night) else 0.0

        # Max continuous nonwear in night (hours)
        max_nonwear_h = max_continuous_nonwear_hours(ns, ne) if ns < ne else 0.0

        # Wear fraction inside TIB
        ts = max(tib_start, rec_start)
        te = min(tib_end, rec_end)
        ww_tib = w.loc[ts:te]
        wear_frac_in_tib = float(ww_tib.mean()) if len(ww_tib) else 0.0

        # TIB duration
        tib_duration_h = float((tib_end - tib_start).total_seconds() / 3600.0)

        # Overlap of TIB with night (hours)
        overlap_night_h = overlap_hours(tib_start, tib_end, night_start, night_end)

        # --- QC decision ---
        reasons = []
        if wear_hours_night < min_wear_hours_night:
            reasons.append(f"wear_hours_night<{min_wear_hours_night}")
        if max_nonwear_h >= max_cont_nonwear_hours_night:
            reasons.append(f"max_cont_nonwear_night>={max_cont_nonwear_hours_night}")
        if wear_frac_in_tib < min_wear_frac_in_tib:
            reasons.append(f"wear_frac_in_tib<{min_wear_frac_in_tib}")
        if tib_duration_h < tib_min_h or tib_duration_h > tib_max_h:
            reasons.append(f"tib_duration_outside[{tib_min_h},{tib_max_h}]")
        if overlap_night_h < min_overlap_night_h:
            reasons.append(f"overlap_night_h<{min_overlap_night_h}")

        tib_valid = (len(reasons) == 0)

        row = r.to_dict()
        row.update({
            "night_start": night_start,
            "night_end": night_end,
            "wear_hours_night": wear_hours_night,
            "max_cont_nonwear_h_night": max_nonwear_h,
            "wear_frac_in_tib": wear_frac_in_tib,
            "tib_duration_h": tib_duration_h,
            "overlap_night_h": overlap_night_h,
            "tib_valid": tib_valid,
            "invalid_reasons": ";".join(reasons),
        })
        out.append(row)

    tib_qc = pd.DataFrame(out)
    tib_valid_df = tib_qc[tib_qc["tib_valid"]].reset_index(drop=True)
    return tib_qc, tib_valid_df

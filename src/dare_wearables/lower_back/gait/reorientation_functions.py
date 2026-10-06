import numpy as np
import pandas as pd

def _unit_vector(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        return None
    return v / n

def detect_orientation_flips(
    df: pd.DataFrame,
    cols=("acc_x", "acc_y", "acc_z"),
    fs=100,
    window="10min",
    # static detection uses rolling STD of vector magnitude
    static_roll_s=2.0,          # rolling window length for "stillness" (seconds)
    static_std_thr=0.02,        # threshold on rolling std of VM (tune if needed)
    min_static_frac=0.10,       # need at least this fraction of samples "static" in a window
    reference_from_first="3h",  # determine reference orientation from first 3 hours (majority)
    dot_flip_thr=-0.3,          # dot < this => upside down (conservative, not too strict)
    dot_correct_thr=0.3,        # dot > this => correct
    min_segment_windows=2       # remove tiny segments shorter than this many windows
):
    """
    Detect correct vs upside-down orientation changes for long recordings.

    Returns:
      orientation_windows: pd.Series indexed by window start, values in {"correct","upside_down","unknown"}
      segments: pd.DataFrame with columns [start, end, orientation]
      flip_times: list of timestamps where orientation changes
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")

    xcol, ycol, zcol = cols
    needed = [xcol, ycol, zcol]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    d = df[needed].copy()

    # Ensure numeric
    d = d.apply(pd.to_numeric, errors="coerce")
    d = d.dropna(how="any")
    if d.empty:
        raise ValueError("No valid samples after dropping NaNs.")

    # Vector magnitude (unit-agnostic)
    vm = np.sqrt(d[xcol]**2 + d[ycol]**2 + d[zcol]**2)

    # Rolling std of VM to detect "static" samples
    static_roll = max(1, int(round(static_roll_s * fs)))
    vm_std = vm.rolling(static_roll, center=True, min_periods=static_roll//2).std()

    # Window loop
    orient_records = []

    for t0, block in d.resample(window):
        if len(block) < fs * 10:  # need at least 10 s of data in the window
            orient_records.append((t0, "unknown", np.nan))
            continue

        vm_b = np.sqrt(block[xcol]**2 + block[ycol]**2 + block[zcol]**2)
        vm_std_b = vm_b.rolling(static_roll, center=True, min_periods=static_roll//2).std()

        static_mask = (vm_std_b < static_std_thr).fillna(False)
        static_frac = static_mask.mean()

        if static_frac < min_static_frac:
            orient_records.append((t0, "unknown", static_frac))
            continue

        # Robust gravity estimate from static samples (median is robust to outliers)
        gx = block.loc[static_mask, xcol].median()
        gy = block.loc[static_mask, ycol].median()
        gz = block.loc[static_mask, zcol].median()

        g = _unit_vector([gx, gy, gz])
        if g is None:
            orient_records.append((t0, "unknown", static_frac))
            continue

        # Save the unit gravity vector for later reference comparison
        orient_records.append((t0, g, static_frac))

    # Build a series of gravity vectors per window (or "unknown")
    idx = [r[0] for r in orient_records]
    g_list = [r[1] for r in orient_records]
    static_fracs = pd.Series([r[2] for r in orient_records], index=idx, name="static_frac")

    # Determine reference gravity direction using first N hours: majority of valid vectors
    ref_end = (d.index.min() + pd.Timedelta(reference_from_first))
    valid_ref = []
    for t0, g in zip(idx, g_list):
        if t0 <= ref_end and isinstance(g, np.ndarray):
            valid_ref.append(g)

    if not valid_ref:
        # fallback: first valid window
        for g in g_list:
            if isinstance(g, np.ndarray):
                valid_ref = [g]
                break

    if not valid_ref:
        # nothing valid at all
        orientation_windows = pd.Series(["unknown"] * len(idx), index=idx, name="orientation")
        segments = pd.DataFrame(columns=["start", "end", "orientation"])
        flip_times = []
        return orientation_windows, segments, flip_times

    # Reference = normalized mean of unit vectors in the reference period
    ref = _unit_vector(np.mean(np.vstack(valid_ref), axis=0))
    if ref is None:
        ref = valid_ref[0]

    # Classify each window by dot product with reference
    labels = []
    dots = []
    for g in g_list:
        if not isinstance(g, np.ndarray):
            labels.append("unknown")
            dots.append(np.nan)
            continue
        dot = float(np.dot(g, ref))  # in [-1, 1]
        dots.append(dot)
        if dot <= dot_flip_thr:
            labels.append("upside_down")
        elif dot >= dot_correct_thr:
            labels.append("correct")
        else:
            labels.append("unknown")

    orientation_windows = pd.Series(labels, index=idx, name="orientation")
    dot_series = pd.Series(dots, index=idx, name="ref_dot")

    # Optional: fill short "unknown" gaps by forward/backward fill only between same labels
    # (keeps it conservative; doesn’t invent flips)
    ow = orientation_windows.copy()
    # forward fill unknowns
    ow_ffill = ow.replace("unknown", np.nan).ffill()
    ow_bfill = ow.replace("unknown", np.nan).bfill()
    # fill only when ffill == bfill (same surrounding label)
    ow = ow.where(ow != "unknown", np.where(ow_ffill == ow_bfill, ow_ffill, "unknown"))
    ow = pd.Series(ow, index=orientation_windows.index, name="orientation")

    # Remove tiny segments (debounce)
    # Run-length encoding
    segs = []
    current_label = ow.iloc[0]
    start = ow.index[0]
    count = 1
    for t, lab in zip(ow.index[1:], ow.iloc[1:]):
        if lab == current_label:
            count += 1
        else:
            segs.append((start, t, current_label, count))
            start = t
            current_label = lab
            count = 1
    segs.append((start, ow.index[-1] + pd.Timedelta(window), current_label, count))

    # Merge / relabel short segments as unknown
    cleaned = []
    for s, e, lab, nwin in segs:
        if lab in ("correct", "upside_down") and nwin < min_segment_windows:
            lab = "unknown"
        cleaned.append((s, e, lab))
    segments = pd.DataFrame(cleaned, columns=["start", "end", "orientation"])

    # Recreate window labels from cleaned segments
    ow2 = ow.copy()
    for s, e, lab in cleaned:
        mask = (ow2.index >= s) & (ow2.index < e)
        ow2.loc[mask] = lab
    ow2.name = "orientation"

    # Flip times = boundaries where label changes between correct <-> upside_down
    flip_times = []
    prev = ow2.iloc[0]
    for t, lab in zip(ow2.index[1:], ow2.iloc[1:]):
        if prev != lab and {prev, lab} == {"correct", "upside_down"}:
            flip_times.append(t)
        prev = lab

    # Return useful extras too (optional)
    # You can keep dot_series/static_fracs if you want debugging.
    return ow2, segments, flip_times

def build_flipped_mask(df, segments):
    """
    df: original accel dataframe with DatetimeIndex
    segments: DataFrame with columns ['start','end','orientation']
    returns: boolean Series indexed like df, True where flipped
    """
    flipped = pd.Series(False, index=df.index)

    for s, e, lab in segments[["start", "end", "orientation"]].itertuples(index=False):
        if lab == "upside_down":
            flipped.loc[(flipped.index >= s) & (flipped.index < e)] = True

    return flipped
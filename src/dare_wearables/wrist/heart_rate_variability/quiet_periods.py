from __future__ import annotations

import pandas as pd


def build_quiet_periods(start_sleep, end_sleep, bursts):
    """
    Build quiet periods within [start_sleep, end_sleep], including the start/end
    of the night and the no-burst case.
    """
    if bursts.empty:
        return pd.DataFrame([{
            "start": start_sleep,
            "end": end_sleep
        }])

    intervals = []

    if bursts["start"].iloc[0] > start_sleep:
        intervals.append((start_sleep, bursts["start"].iloc[0]))

    for prev_end, next_start in zip(bursts["end"].iloc[:-1], bursts["start"].iloc[1:]):
        if next_start > prev_end:
            intervals.append((prev_end, next_start))

    if bursts["end"].iloc[-1] < end_sleep:
        intervals.append((bursts["end"].iloc[-1], end_sleep))

    quiet_periods = pd.DataFrame(intervals, columns=["start", "end"])
    quiet_periods["duration_s"] = (quiet_periods["end"] - quiet_periods["start"]).dt.total_seconds()

    return quiet_periods[quiet_periods["duration_s"] > 0].reset_index(drop=True)


def build_variable_windows(qstart, qend, min_window, max_window, step):
    """
    Rules:
    - if the quiet period is shorter than 1 min, discard it
    - if it is between 1 and 5 min, use the whole quiet period
    - if it is longer than 5 min, use 5-min windows with a fixed 1-min step
    """
    seg_len = qend - qstart

    if seg_len < min_window:
        return []

    win_len = min(seg_len, max_window)

    if seg_len <= max_window:
        return [(qstart, qend)]

    starts = []
    current_start = qstart
    last_start = qend - win_len

    while current_start <= last_start:
        starts.append(current_start)
        current_start += step

    if len(starts) == 0 or starts[-1] < last_start:
        starts.append(last_start)

    return [(s, s + win_len) for s in starts]

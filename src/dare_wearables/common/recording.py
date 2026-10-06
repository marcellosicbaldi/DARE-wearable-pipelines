"""Timestamp boundaries shared by wrist processing stages within this package."""
import numpy as np
import pandas as pd


def continuous_intersections(acc, ppg, *, gap_seconds=1.0, min_seconds=0.0,
                             acc_frequency=None, ppg_frequency=None):
    """Intersect continuous runs in BOTH streams; never compress recording gaps."""
    if gap_seconds <= 0 or min_seconds < 0:
        raise ValueError("Gap threshold must be positive and minimum duration nonnegative.")
    def runs(frame, frequency):
        if frame.empty:
            return []
        idx = frame.index
        if not isinstance(idx, pd.DatetimeIndex) or idx.hasnans or not idx.is_monotonic_increasing or idx.has_duplicates:
            raise ValueError("Recording timestamps must be unique, increasing datetimes.")
        delta = np.diff(idx.to_numpy(dtype="datetime64[ns]").astype(np.int64)) / 1e9
        threshold = gap_seconds
        if frequency is not None:
            if not np.isfinite(frequency) or frequency <= 0:
                raise ValueError("Recording sample frequencies must be positive and finite.")
            threshold = min(threshold, 1.5 / frequency)
            within_run = delta[delta <= threshold]
            if len(delta) and not len(within_run):
                raise ValueError("No continuous samples match the configured sample frequency.")
            if len(within_run) and not np.allclose(within_run, 1 / frequency, rtol=0.02, atol=1e-6):
                raise ValueError("Recording timestamps do not match the configured sample frequency.")
        cuts = np.flatnonzero(delta > threshold) + 1
        edges = np.r_[0, cuts, len(frame)]
        return [(idx[a], idx[b-1]) for a, b in zip(edges[:-1], edges[1:])]
    aa, pp = runs(acc, acc_frequency), runs(ppg, ppg_frequency)
    result = []
    i = j = 0
    while i < len(aa) and j < len(pp):
        start, end = max(aa[i][0], pp[j][0]), min(aa[i][1], pp[j][1])
        if start <= end and (end-start).total_seconds() >= min_seconds:
            a, p = acc.loc[start:end], ppg.loc[start:end]
            if len(a) and len(p):
                result.append((a, p))
        if aa[i][1] <= pp[j][1]:
            i += 1
        else:
            j += 1
    return result


def in_timezone(frame, timezone):
    """Interpret naive clock times in the configured zone; preserve aware instants."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Recording requires a DatetimeIndex.")
    result = frame.copy()
    result.index = (frame.index.tz_localize(timezone) if frame.index.tz is None
                    else frame.index.tz_convert(timezone))
    return result


def timestamp_in_timezone(value, timezone):
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize(timezone) if timestamp.tz is None else timestamp.tz_convert(timezone)

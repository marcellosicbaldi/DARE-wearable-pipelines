"""Conservative, opt-in quarantine of recorded McRoberts rollover anchors.

OMX fields follow OpenMovement's omx_data_t: timestamp plus fractional time,
with a signed sample-index offset. Raw files and retained timestamps are never
rewritten. Unknown anomalies fail closed; only the verified short, single-anchor
minute/hour excursion is eligible. See docs/lower_back_timestamp_quality.md.
"""
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from dare_wearables.common.output_state import signature

POLICY = "quarantine_bad_anchors"
HEADER = np.dtype({
    "names": ["type", "stream", "length", "sequence", "timestamp", "fraction", "offset", "rate", "modifier", "packing", "count"],
    "formats": ["S1", "S1", "<u2", "<u4", "<u4", "<u2", "<i2", "<u2", "i1", "u1", "<u2"],
    "offsets": [0, 1, 2, 4, 8, 12, 14, 16, 18, 21, 22], "itemsize": 512})


@dataclass(frozen=True)
class TimestampOptions:
    policy: str = "strict"

    def validate(self):
        if self.policy not in ("strict", POLICY):
            raise ValueError(f"timestamps.policy must be 'strict' or '{POLICY}'.")

    @classmethod
    def from_mapping(cls, values=None):
        values = dict(values or {})
        if values.keys() - {"policy"}:
            raise ValueError(f"Unknown timestamp options: {sorted(values.keys() - {'policy'})}")
        result = cls(**values)
        result.validate()
        return result


def packet_bounds(h):
    """Vectorized packed-date bounds, matching the production decoder in ns."""
    packed = h["timestamp"].astype(np.int64)
    year = (packed >> 26) & 63
    month, day = (packed >> 22) & 15, (packed >> 17) & 31
    hour, minute, second = (packed >> 12) & 31, (packed >> 6) & 63, packed & 63
    if (np.any(packed & 0x80000000) or np.any((month < 1) | (month > 12))
            or np.any((day < 1) | (day > 31)) or np.any(hour > 23)
            or np.any(minute > 59) or np.any(second > 59)):
        raise ValueError("Unsupported or invalid OMX packed timestamp; no recovery applied.")
    months = np.datetime64("2000-01", "M") + (year * 12 + month - 1).astype("timedelta64[M]")
    days_in_month = ((months + np.timedelta64(1, "M")).astype("datetime64[D]") - months.astype("datetime64[D]")).astype(int)
    if np.any(day > days_in_month):
        raise ValueError("Invalid OMX calendar date; no recovery applied.")
    date = months.astype("datetime64[D]") + (day - 1).astype("timedelta64[D]")
    base = date.astype("datetime64[ns]").astype(np.int64) + (hour*3600 + minute*60 + second)*1_000_000_000
    rate, modifier = h["rate"].astype(float), h["modifier"].astype(float)
    if np.any(rate == 0):
        raise ValueError("Invalid zero sample rate in sensor packet")
    fs = rate.copy()
    fs[modifier > 1] /= modifier[modifier > 1]
    fs[modifier < -1] *= -modifier[modifier < -1]
    fs[modifier == 0] = 1 / rate[modifier == 0]
    fs[modifier == -1] = 1 / (rate[modifier == -1] * 60)
    starts = base + (h["fraction"].astype(float)*1e9/65536).astype(np.int64) - (h["offset"]*1e9/fs).astype(np.int64)
    ends = starts + ((h["count"]-1)*1e9/fs).astype(np.int64)
    return starts, ends, fs


def _iso(ns):
    return str(np.datetime64(int(ns), "ns"))


def quarantine_plan(source):
    """Return eligible file-packet IDs and a JSON-safe QC record before decoding."""
    source = Path(source)
    identity = signature({}, [source])
    if source.stat().st_size % 512:
        raise ValueError("Incomplete OMX sector; refusing timestamp recovery.")
    packets = np.memmap(source, mode="r", dtype=HEADER)
    drop, events, restarts, streams = set(), [], [], []
    for sid in (b"a", b"g"):
        ids = np.flatnonzero((packets["stream"] == sid) & (packets["count"] > 0) & (packets["count"] <= 80))
        if not len(ids):
            raise ValueError("Both accelerometer and gyroscope streams are required.")
        h = {name: packets[name][ids].copy() for name in HEADER.names}
        starts, ends, fs = packet_bounds(h)
        if np.unique(fs).size != 1:
            raise ValueError("Sensor streams have inconsistent sampling rates.")
        period = 1e9 / fs[0]
        positions = np.r_[0, np.cumsum(h["count"][:-1].astype(np.int64))]
        anchors = positions + h["offset"].astype(np.int64)
        same = ((h["timestamp"][1:] == h["timestamp"][:-1]) &
                (h["fraction"][1:] == h["fraction"][:-1]) & (anchors[1:] == anchors[:-1]))
        group_starts = np.r_[0, np.flatnonzero(~same) + 1]
        seq_delta = np.diff(h["sequence"].astype(np.int64))
        for i in np.flatnonzero(seq_delta != 1) + 1:
            if starts[i] <= ends[i-1]:
                raise ValueError("OMX restart overlaps prior time; refusing timestamp recovery.")
            restarts.append({"stream": sid.decode(), "file_packet": int(ids[i]),
                "previous_sequence": int(h["sequence"][i-1]), "sequence": int(h["sequence"][i]),
                "sequence_delta": int(seq_delta[i-1]), "start": _iso(ends[i-1] + period),
                "end": _iso(starts[i]), "gap_seconds": float((starts[i]-ends[i-1]-period)/1e9)})
        keep = np.ones(len(ids), dtype=bool)
        for i in np.flatnonzero(starts[1:] <= ends[:-1]) + 1:
            a = int(group_starts[np.searchsorted(group_starts, i-1, side="right")-1])
            def reject():
                raise ValueError(f"Unrecognized OMX timestamp anomaly in stream {sid.decode()}, packet {int(ids[i])}; no recovery applied.")
            if a == 0 or not 2 <= i-a <= 40 or fs[0] != 100.:
                reject()
            forward = (starts[a]-ends[a-1]-period)/1e9
            backward = (starts[i]-ends[i-1]-period)/1e9
            unit = min((60., 3600.), key=lambda value: abs(forward-value))
            if (abs(forward-unit) > 5 or abs(backward+unit) > 5 or abs(forward+backward) > 5
                    or (int(h["timestamp"][a]) & 63) != 59 or int(h["fraction"][a]) < 64000
                    or starts[i] <= ends[a-1] or not np.all(seq_delta[a-1:i] == 1)
                    or not np.all(same[a:i-1]) or i-a > 40 or (positions[i]-positions[a])/fs[0] > 32):
                reject()
            check = slice(a-1, i+1)
            if (np.any(h["type"][check] != b"d") or np.any(h["length"][check] != 508)
                    or np.any(h["packing"][check] != 0x32)):
                reject()
            for packet_id in ids[check]:
                raw = np.memmap(source, mode="r", dtype="<u2", offset=int(packet_id)*512, shape=(256,))
                if int(raw.sum()) % 65536:
                    raise ValueError(f"OMX checksum failure near packet {int(packet_id)}; no recovery applied.")
            keep[a:i] = False
            drop.update(map(int, ids[a:i]))
            events.append({"stream": sid.decode(), "first_file_packet": int(ids[a]),
                "last_file_packet": int(ids[i-1]), "next_file_packet": int(ids[i]),
                "first_sequence": int(h["sequence"][a]), "next_sequence": int(h["sequence"][i]),
                "excluded_packets": int(i-a), "excluded_samples": int(h["count"][a:i].sum()),
                "excluded_nominal_seconds": float(h["count"][a:i].sum()/fs[0]),
                "start": _iso(ends[a-1]+period), "end": _iso(starts[i]),
                "anchor_time": _iso(starts[a]+int(h["offset"][a]*period)),
                "rollover_seconds": unit, "forward_step_seconds": float(forward),
                "backward_step_seconds": float(backward), "checksums_verified": int(i-a+2)})
        retained_starts, retained_ends = starts[keep], ends[keep]
        if np.any(retained_starts[1:] <= retained_ends[:-1]):
            raise ValueError("OMX timestamps still overlap after candidate quarantine; refusing recovery.")
        streams.append({"stream": sid.decode(), "input_packets": len(ids), "input_samples": int(h["count"].sum()),
                        "excluded_packets": int((~keep).sum()), "excluded_samples": int(h["count"][~keep].sum()),
                        "retained_samples": int(h["count"][keep].sum()), "sample_rate_hz": float(fs[0])})
    if streams[0]["sample_rate_hz"] != streams[1]["sample_rate_hz"]:
        raise ValueError("Sensor streams have inconsistent sampling rates.")
    if identity != signature({}, [source]):
        raise ValueError("OMX source changed during timestamp preflight.")
    return drop, {"policy": POLICY, "revision": 1, "input_identity": identity,
                  "events": events, "recording_discontinuities": restarts, "streams": streams,
                  "retained_timestamps_changed": False, "invalid_samples_inserted": False}


def processing_segments(frame, fs, quality):
    """Respect gaps/restarts in files that required timing recovery."""
    if not quality.get("events") and not quality.get("recording_discontinuities"):
        return None
    times = pd.DatetimeIndex(frame.datetime).as_unit("ns")
    if not len(times):
        return None
    delta = np.diff(times.asi8)
    if np.any(delta <= 0):
        raise ValueError("Processing timestamps must be strictly increasing.")
    breaks = np.flatnonzero((delta > 1.5e9/fs) | (delta < .5e9/fs)) + 1
    restart_times = pd.to_datetime([r["end"] for r in quality.get("recording_discontinuities", [])])
    if len(restart_times):
        breaks = np.r_[breaks, times.searchsorted(restart_times)]
    breaks = np.unique(breaks[(breaks > 0) & (breaks < len(times))])
    return np.column_stack((np.r_[0, breaks], np.r_[breaks, len(times)])).tolist()


def invalid_intervals(frame, fs, quality):
    """Missing observed time, distinct from wear/nonwear; half-open recording-local naive times."""
    columns = ["start", "end", "duration_seconds", "reason"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    times = pd.DatetimeIndex(frame.datetime).as_unit("ns")
    gaps = np.flatnonzero(np.diff(times.asi8) > 1.5e9/fs)
    rows = []
    events = quality.get("events", [])
    restarts = quality.get("recording_discontinuities", [])
    for i in gaps:
        first, stop = times[i]+pd.Timedelta(seconds=1/fs), times[i+1]
        reason = "unobserved_or_unaligned"
        for name, intervals in (("recording_discontinuity", restarts), ("quarantined_anchor", events)):
            if any(pd.Timestamp(row["start"]) < stop and pd.Timestamp(row["end"]) > first for row in intervals):
                reason = name
        rows.append({"start": first, "end": stop, "duration_seconds": (stop-first).total_seconds(), "reason": reason})
    return pd.DataFrame(rows, columns=columns)


QUALITY_FILES = ("timestamp_quality.json", "time_invalid_intervals.csv", "recording_discontinuities.csv")


def write_quality(folder, quality, intervals):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/QUALITY_FILES[0]).write_text(json.dumps(quality, indent=2, default=str)+"\n")
    intervals.to_csv(folder/QUALITY_FILES[1], index=False)
    pd.DataFrame(quality.get("recording_discontinuities", []), columns=[
        "stream", "file_packet", "previous_sequence", "sequence", "sequence_delta", "start", "end", "gap_seconds"]
    ).to_csv(folder/QUALITY_FILES[2], index=False)

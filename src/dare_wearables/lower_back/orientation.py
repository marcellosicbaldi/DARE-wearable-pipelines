"""Conservative belt-inversion correction shared by gait and posture.

The belt is assumed to preserve the face toward the body. Its allowed states
are identity and a 180-degree rotation about PA: diag(-1, -1, +1). There is no
tilt alignment. Only walking during original wear supplies orientation votes;
sitting, lying and nonwear never supply a mounting reference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

ACC = ["acc_x", "acc_y", "acc_z"]
GYR = ["gyr_x", "gyr_y", "gyr_z"]
FLIPPED_COLUMNS = ["acc_x", "acc_y", "gyr_x", "gyr_y"]
EPISODE_COLUMNS = ["episode", "wear_start", "wear_end", "correction_start", "correction_end",
                   "wear_seconds", "status", "inverted", "eligible_sequences", "upright_sequences",
                   "inverted_sequences", "weighted_evidence_seconds", "consensus_fraction"]
EVIDENCE_COLUMNS = ["episode", "start_time", "end_time", "duration_s", "gravity_x", "gravity_y",
                    "gravity_z", "gravity_norm_mps2", "resultant_ratio", "vertical_fraction", "vote"]


@dataclass(frozen=True)
class OrientationOptions:
    method: str = "inversion"
    min_sequences: int = 3
    min_evidence_seconds: float = 60.
    min_sequence_seconds: float = 10.
    weight_cap_seconds: float = 60.
    min_consensus_fraction: float = .9
    min_vertical_fraction: float = float(np.cos(np.deg2rad(45.)))
    detection_chunk_seconds: float = 3600.

    def validate(self):
        if self.method not in ("inversion", "none"):
            raise ValueError("Orientation method must be 'inversion' or 'none'; tilt alignment is not a production option.")
        if isinstance(self.min_sequences, bool) or not isinstance(self.min_sequences, int) or self.min_sequences < 2:
            raise ValueError("Orientation requires at least two independent walking sequences.")
        for name in ("min_evidence_seconds", "min_sequence_seconds", "weight_cap_seconds", "detection_chunk_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Orientation {name} must be positive and finite.")
        if self.detection_chunk_seconds < self.min_sequence_seconds:
            raise ValueError("Orientation detection chunks cannot be shorter than walking evidence.")
        if not .5 < self.min_consensus_fraction <= 1 or not .5 < self.min_vertical_fraction <= 1:
            raise ValueError("Orientation consensus and vertical fractions must lie in (0.5, 1].")

    @classmethod
    def from_mapping(cls, values=None):
        result = cls(**(values or {}))
        result.validate()
        return result


def _walking_sequences(data, sampling_rate_hz):
    try:
        from mobgap.gait_sequences import GsdIonescu
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Walking-based inversion detection requires mobgap; install it or explicitly set orientation.method='none'.") from exc
    body = data[ACC+GYR].rename(columns=dict(zip(ACC+GYR,
        ["acc_is", "acc_ml", "acc_pa", "gyr_is", "gyr_ml", "gyr_pa"])))
    return GsdIonescu().detect(body, sampling_rate_hz=sampling_rate_hz).gs_list_


def _vote(data, fs, options):
    values = data[ACC].to_numpy(dtype=float)
    info = dict.fromkeys(["gravity_x", "gravity_y", "gravity_z", "gravity_norm_mps2",
                          "resultant_ratio", "vertical_fraction"], np.nan)
    info["vote"] = "unreliable"
    if not np.isfinite(values).all():
        return info
    block = int(round(fs))
    starts = np.arange(0, len(values), block)
    means = np.add.reduceat(values, starts, axis=0)/np.minimum(block, len(values)-starts)[:, None]
    gravity = np.median(means, axis=0)
    norm = np.linalg.norm(gravity)
    denominator = np.linalg.norm(values, axis=1).mean()
    resultant = np.linalg.norm(values.mean(axis=0))/denominator if denominator else 0.
    vertical = gravity[0]/norm if norm else 0.
    info.update(zip(("gravity_x", "gravity_y", "gravity_z"), gravity))
    info.update(gravity_norm_mps2=float(norm), resultant_ratio=float(resultant), vertical_fraction=float(vertical))
    if .5*9.80665 <= norm <= 1.5*9.80665 and resultant >= .8:
        if vertical >= options.min_vertical_fraction:
            info["vote"] = "upright"
        elif vertical <= -options.min_vertical_fraction:
            info["vote"] = "inverted"
    return info


def _episode_decision(evidence, options):
    eligible = evidence.loc[evidence.vote.isin(["upright", "inverted"])]
    weights = eligible.duration_s.clip(upper=options.weight_cap_seconds)
    counts = eligible.vote.value_counts()
    sums = weights.groupby(eligible.vote).sum()
    total = float(weights.sum())
    info = {"status": "insufficient_walking_evidence", "inverted": False,
            "eligible_sequences": len(eligible), "upright_sequences": int(counts.get("upright", 0)),
            "inverted_sequences": int(counts.get("inverted", 0)),
            "weighted_evidence_seconds": total, "consensus_fraction": np.nan}
    if len(eligible) < options.min_sequences or total < options.min_evidence_seconds:
        return info
    winner = sums.idxmax()
    other = "inverted" if winner == "upright" else "upright"
    fraction = float(sums[winner]/total)
    info["consensus_fraction"] = fraction
    if counts.get(winner, 0) < options.min_sequences:
        return info
    # Sustained evidence for both states can indicate an unrecognised removal
    # inside final wear. Do not apply the majority state to the whole episode.
    opposite_supported = (counts.get(other, 0) >= options.min_sequences
                          and sums.get(other, 0.) >= options.min_evidence_seconds)
    if fraction < options.min_consensus_fraction or opposite_supported:
        info["status"] = "conflicting_walking_evidence"
    else:
        info.update(status=winner, inverted=winner == "inverted")
    return info


def correct_belt_inversion(data, *, options=None, walking_detector=None):
    """Return corrected signals, episode QC, walking evidence and a summary.

    Fit each continuous final-wear episode independently, respecting timestamp
    gaps and time_segment boundaries. Unknown/conflicting episodes keep their
    input axes. Nonwear uses the preceding episode's state (the first episode
    before initial wear); this supplies a consistent choice for the existing
    optional TIB gap filling, without relabelling wear or calibrating from it.
    A confident state is never propagated across a recording discontinuity.
    """
    options = options or OrientationOptions()
    options.validate()
    index = data.index
    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.has_duplicates
            or not index.is_monotonic_increasing):
        raise ValueError("Inversion correction requires unique increasing datetimes.")
    if "wear" not in data or data.wear.isna().any() or not data.wear.isin([True, False]).all():
        raise ValueError("Inversion correction requires a complete boolean wear mask.")
    if any(column not in data for column in ACC+GYR):
        raise ValueError("Inversion correction requires both accelerometer and gyroscope triads.")
    fs = 100
    wear = data.wear.to_numpy(dtype=bool)
    continuous = np.r_[False, np.isclose(np.diff(index.as_unit("ns").asi8)/1e9, 1/fs, rtol=.02, atol=.00005)] if len(data) else np.array([], dtype=bool)
    if "time_segment" in data:
        continuous[1:] &= data.time_segment.to_numpy()[1:] == data.time_segment.to_numpy()[:-1]
    previous_wear = np.r_[False, wear[:-1]] if len(data) else wear
    same_episode = wear & previous_wear & continuous
    starts = np.flatnonzero(wear & ~same_episode)
    ends = np.flatnonzero(wear & ~np.r_[same_episode[1:], False]) + 1 if len(data) else np.array([], dtype=int)
    chunk_size = int(round(options.detection_chunk_seconds*fs))
    detector = walking_detector or _walking_sequences
    records, evidence_rows = [], []
    for episode, (start, end) in enumerate(zip(starts, ends)):
        local_rows = []
        if options.method == "inversion":
            for first in range(start, end, chunk_size):
                last = min(first+chunk_size, end)
                if last-first < options.min_sequence_seconds*fs:
                    continue
                chunk = data.iloc[first:last]
                for sequence in detector(chunk, fs).itertuples():
                    a, b = int(sequence.start), int(sequence.end)
                    if not 0 <= a < b <= len(chunk):
                        raise ValueError("Orientation walking detector returned invalid boundaries.")
                    if (b-a)/fs < options.min_sequence_seconds:
                        continue
                    local_rows.append({"episode": episode, "start_time": chunk.index[a],
                        "end_time": chunk.index[b-1]+pd.Timedelta(seconds=1/fs),
                        "duration_s": (b-a)/fs, **_vote(chunk.iloc[a:b], fs, options)})
        evidence = pd.DataFrame(local_rows, columns=EVIDENCE_COLUMNS)
        decision = _episode_decision(evidence, options)
        if options.method == "none":
            decision["status"] = "disabled"
        evidence_rows.extend(local_rows)
        records.append({"episode": episode, "wear_start": index[start],
                        "wear_end": index[end-1]+pd.Timedelta(seconds=1/fs),
                        "correction_start": index[start],
                        "correction_end": index[end-1]+pd.Timedelta(seconds=1/fs),
                        "wear_seconds": (end-start)/fs, **decision})
    out = data.copy()
    positions = out.columns.get_indexer(FLIPPED_COLUMNS)
    if (positions < 0).any():
        raise ValueError("Inversion correction requires both accelerometer and gyroscope triads.")
    for i, record in enumerate(records):
        first, stop = starts[i], ends[i]
        # Carry only through adjacent, observed nonwear; never across a gap.
        next_start = starts[i+1] if i+1 < len(starts) else len(data)
        interrupted = np.flatnonzero(~continuous[stop:next_start])
        stop = stop+int(interrupted[0]) if len(interrupted) else next_start
        if i == 0 and continuous[1:first+1].all():
            first = 0
        record["correction_start"] = index[first]
        record["correction_end"] = index[stop-1]+pd.Timedelta(seconds=1/fs)
        if record["inverted"]:
            out.iloc[first:stop, positions] = -out.iloc[first:stop, positions]
    episodes = pd.DataFrame(records, columns=EPISODE_COLUMNS)
    evidence = pd.DataFrame(evidence_rows, columns=EVIDENCE_COLUMNS)
    known = episodes.status.isin(["upright", "inverted"])
    summary = {"method": options.method, "parameters": asdict(options),
        "axis_correction": {"upright": [1, 1, 1], "inverted": [-1, -1, 1]},
        "assumption": "Belt preserves the face toward the body; PA is unchanged by inversion.",
        "tilt_alignment": False, "wear_mask_unchanged": out.wear.equals(data.wear),
        "wear_episodes": len(episodes), "walking_sequences": len(evidence),
        "inverted_wear_hours": float(episodes.loc[episodes.inverted.astype(bool), "wear_seconds"].sum()/3600),
        "uncertain_wear_hours": float(episodes.loc[~known, "wear_seconds"].sum()/3600),
        "status_counts": {str(k): int(v) for k, v in episodes.status.value_counts().items()}}
    return out, episodes, evidence, summary


def write_orientation_qc(folder, episodes, evidence, summary):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(folder/"inversion_episodes.csv", index=False)
    evidence.to_csv(folder/"walking_evidence.csv", index=False)
    (folder/"inversion_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")


def annotate_tib_orientation(tib, episodes):
    """Expose uncertain calibration coverage without changing existing TIB QC."""
    tib = tib.copy()
    unknown = episodes.loc[~episodes.status.isin(["upright", "inverted"])]
    seconds = []
    for row in tib.itertuples():
        overlap = [(min(pd.Timestamp(row.tib_end), r.wear_end)-max(pd.Timestamp(row.tib_start), r.wear_start)).total_seconds()
                   for r in unknown.itertuples()]
        seconds.append(sum(max(0., value) for value in overlap))
    tib["orientation_uncertain_wear_hours_in_tib"] = np.asarray(seconds)/3600
    tib["orientation_review_required"] = tib.orientation_uncertain_wear_hours_in_tib.gt(0)
    return tib

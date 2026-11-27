"""Experimental stable mounting correction for posture at its one-Hz grid.

Estimate mounting only from trusted walking evidence, never from lying windows.
Hold one proper rotation over each wear episode and the following nonwear gap.
The forward-axis projection is assumed anterior, as in gait alignment. A change
of wearing state or a recording discontinuity can start a new mounting episode;
ambiguous episodes are reported explicitly rather than aligned from lying data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ACC = ["acc_x", "acc_y", "acc_z"]
GYR = ["gyr_x", "gyr_y", "gyr_z"]


def proper_rotation_from_gravity(gravity):
    """Map gravity to +x, preserving projected +z and right handedness."""
    up = np.asarray(gravity, dtype=float)
    magnitude = np.linalg.norm(up)
    if not np.isfinite(up).all() or magnitude <= 0:
        raise ValueError("Gravity must be finite and nonzero.")
    up = up / magnitude
    forward = np.array([0., 0., 1.]) - up[2] * up
    if np.linalg.norm(forward) < .2:
        raise ValueError("The forward axis is ambiguous.")
    forward /= np.linalg.norm(forward)
    return np.vstack((up, np.cross(forward, up), forward))


def estimate_episode_rotation(evidence, *, min_sequences=3, min_evidence_seconds=60.,
                              consensus_angle_deg=30., min_consensus_fraction=.8):
    """Use a duration-capped dominant cluster of >=10-second walking GSs.

    Weights are capped at 60 seconds per GS so a single long sequence cannot
    dominate mounting inference. Consensus is weighted, and a second mounting
    state inside one wear episode is flagged instead of switching at posture
    transitions. These are comparison parameters, not validated study cutoffs.
    """
    duration = (evidence.end_time - evidence.start_time).dt.total_seconds()
    gravity = evidence[["gravity_is", "gravity_ml", "gravity_pa"]].to_numpy(dtype=float)
    norms = np.linalg.norm(gravity, axis=1)
    eligible = (evidence.status.eq("aligned").to_numpy() & duration.ge(10).to_numpy()
                & np.isfinite(gravity).all(axis=1) & (norms > 0))
    vectors = gravity[eligible] / norms[eligible, None]
    weights = np.minimum(duration.to_numpy()[eligible], 60.)
    info = {"status": "insufficient_walking_evidence", "eligible_sequences": len(vectors),
            "weighted_evidence_seconds": float(weights.sum()), "consensus_fraction": np.nan,
            "rotation_deg": 0., "upright_x": np.nan, "upright_y": np.nan, "upright_z": np.nan}
    if len(vectors) < min_sequences or weights.sum() < min_evidence_seconds:
        return np.eye(3), info
    threshold = np.cos(np.deg2rad(consensus_angle_deg))
    support = (vectors @ vectors.T >= threshold) @ weights
    selected = vectors @ vectors[np.argmax(support)] >= threshold
    fraction = float(weights[selected].sum()/weights.sum())
    info["consensus_fraction"] = fraction
    if fraction < min_consensus_fraction or selected.sum() < min_sequences:
        info["status"] = "ambiguous_walking_orientation"
        return np.eye(3), info
    up = np.average(vectors[selected], weights=weights[selected], axis=0)
    up /= np.linalg.norm(up)
    try:
        rotation = proper_rotation_from_gravity(up)
    except ValueError:
        info["status"] = "ambiguous_forward_axis"
        return np.eye(3), info
    info.update(status="walking_supported", upright_x=float(up[0]), upright_y=float(up[1]),
                upright_z=float(up[2]), rotation_deg=float(np.degrees(np.arccos(
                    np.clip((np.trace(rotation)-1)/2, -1., 1.)))))
    return rotation, info


def stable_posture_correction(frame_1hz, walking_qc, **estimation_kwargs):
    """Rotate both sensor triads, holding a fixed frame across posture changes.

    Nonwear remains nonwear. It receives the preceding episode's rotation so
    the existing optional lying-gap filling sees a continuous frame. Before
    the first wear episode the first episode rotation is used. Unsupported
    episodes use identity and retain explicit QC; no reference is inferred
    from their sitting/lying measurements.
    """
    if not isinstance(frame_1hz.index, pd.DatetimeIndex) or not frame_1hz.index.is_monotonic_increasing or frame_1hz.index.has_duplicates:
        raise ValueError("Posture mounting correction requires increasing unique datetimes.")
    out = frame_1hz.copy()
    if out.empty:
        return out, pd.DataFrame()
    wear = out.wear.fillna(False).astype(bool)
    continuous = out.index.to_series().diff().eq(pd.Timedelta(seconds=1))
    if "time_segment" in out:
        continuous &= out.time_segment.eq(out.time_segment.shift())
    same_episode = wear & wear.shift(fill_value=False) & continuous
    starts = np.flatnonzero((wear & ~same_episode).to_numpy())
    ends = np.flatnonzero((wear & ~same_episode.shift(-1, fill_value=False)).to_numpy()) + 1
    records = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        t0, t1 = out.index[start], out.index[end-1]+pd.Timedelta(seconds=1)
        evidence = walking_qc.loc[walking_qc.start_time.ge(t0) & walking_qc.end_time.le(t1)]
        rotation, info = estimate_episode_rotation(evidence, **estimation_kwargs)
        first = 0 if i == 0 else start
        stop = starts[i+1] if i+1 < len(starts) else len(out)
        for columns in (ACC, GYR):
            out.iloc[first:stop, out.columns.get_indexer(columns)] = (
                frame_1hz.iloc[first:stop][columns].to_numpy(dtype=float) @ rotation.T)
        records.append({"episode": i, "wear_start": t0, "wear_end": t1,
                        "correction_start": out.index[first],
                        "correction_end": out.index[stop-1]+pd.Timedelta(seconds=1), **info,
                        **{f"r{a}{b}": float(rotation[a,b]) for a in range(3) for b in range(3)}})
    return out, pd.DataFrame(records)

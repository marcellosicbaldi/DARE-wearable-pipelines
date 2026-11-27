"""Constant, right-handed gravity alignment within a detected gait sequence.

Input follows MobGap's body frame: IS is up, ML is right and PA is forward.
Gravity cannot identify heading. We therefore preserve the projection of the
existing PA axis onto the horizontal plane; the loader's mounting convention
still needs to establish anterior versus posterior. In particular, an upside
down sensor with anterior preserved is corrected by diag(-1, -1, +1), not -I.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ACC_COLUMNS = ["acc_is", "acc_ml", "acc_pa"]
GYR_COLUMNS = ["gyr_is", "gyr_ml", "gyr_pa"]
GRAV_MS2 = 9.80665


def rotate_body_frame(data: pd.DataFrame, rotation: np.ndarray) -> pd.DataFrame:
    """Apply the same proper rotation to acceleration and angular velocity."""
    rotation = np.asarray(rotation, dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8)):
        raise ValueError("Orientation correction requires a proper rotation (determinant +1).")
    out = data.copy()
    for columns in (ACC_COLUMNS, GYR_COLUMNS):
        out[columns] = data[columns].to_numpy(dtype=float) @ rotation.T
    return out


def align_gait_sequence(data: pd.DataFrame, sampling_rate_hz: float):
    """Estimate one gravity vector and rotation, without switching inside a GS.

    Component medians of one-second acceleration means reduce step-related
    oscillations. Short, nonfinite or implausible sequences remain unchanged
    and receive an explicit QC status. No static/nonwear or early-recording
    reference is used. Acceleration must be in m/s² and gyroscope in deg/s.
    """
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive and finite.")
    rotation = np.eye(3)
    info = {"status": "insufficient_data", "rotation_deg": 0.0,
            "gravity_is": np.nan, "gravity_ml": np.nan, "gravity_pa": np.nan,
            "gravity_norm_mps2": np.nan, "resultant_ratio": np.nan,
            "vertical_reversed": False}
    acc = data[ACC_COLUMNS].to_numpy(dtype=float)
    if len(acc) < 2 * sampling_rate_hz or not np.isfinite(acc).all():
        return data.copy(), rotation, info
    block = max(1, int(round(sampling_rate_hz)))
    starts = np.arange(0, len(acc), block)
    counts = np.minimum(block, len(acc) - starts)
    means = np.add.reduceat(acc, starts, axis=0) / counts[:, None]
    gravity = np.median(means, axis=0)
    magnitude = float(np.linalg.norm(gravity))
    resultant = float(np.linalg.norm(acc.mean(axis=0)) / np.linalg.norm(acc, axis=1).mean())
    info.update(zip(("gravity_is", "gravity_ml", "gravity_pa"), gravity.tolist()))
    info.update(gravity_norm_mps2=magnitude, resultant_ratio=resultant,
                vertical_reversed=bool(gravity[0] < 0))
    if not 0.5 * GRAV_MS2 <= magnitude <= 1.5 * GRAV_MS2 or resultant < 0.8:
        info["status"] = "implausible_gravity"
        return data.copy(), rotation, info
    up = gravity / magnitude
    forward = np.array([0.0, 0.0, 1.0]) - up[2] * up
    if np.linalg.norm(forward) < 0.2:
        info["status"] = "ambiguous_forward_axis"
        return data.copy(), rotation, info
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    rotation = np.vstack((up, right, forward))
    info.update(status="aligned", rotation_deg=float(np.degrees(
        np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))))
    return rotate_body_frame(data, rotation), rotation, info

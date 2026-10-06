from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import polars as pl

from dare_wearables.common.autocalibrate import autocalibrate
from dare_wearables.wrist.data_io.geneactiv import as_polars_dataframe, read_geneactiv_bin
from dare_wearables.wrist.nonwear.vanhees2013 import vanhees2013


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _format_candidates(paths: list[Path], *, limit: int = 12) -> str:
    shown = "\n".join(f"  - {path}" for path in paths[:limit])
    if len(paths) > limit:
        shown += f"\n  ... and {len(paths) - limit} more"
    return shown


def _filter_subject_prefixed_bins(paths: list[Path], participant: str) -> list[Path]:
    prefix = str(participant).lower()
    return [path for path in paths if path.name.lower().startswith(prefix)]


def find_geneactiv_bin_file(
    input_root: str | Path,
    participant: str,
    visit: str = "T0",
    sensor: str = "GENEActiv",
    geneactiv_file_path: str | Path | None = None,
    require_subject_id_prefix: bool = True,
) -> Path:
    """
    Locate one GENEActiv .bin file for a participant/visit.

    The DARE-FALLSPREDICT data were not present when this adapter was written, so this
    finder accepts a direct file path and also searches the common layouts used
    by the GP pipeline.
    """
    root = Path(input_root).expanduser()

    if geneactiv_file_path is not None:
        direct = Path(geneactiv_file_path).expanduser()
        if not direct.is_absolute():
            direct = root / direct
        direct = direct.resolve()
        if not direct.exists():
            raise FileNotFoundError(f"GENEActiv file not found: {direct}")
        if direct.suffix.lower() != ".bin":
            raise ValueError(f"Expected a GENEActiv .bin file, got: {direct}")
        if require_subject_id_prefix and not direct.name.lower().startswith(str(participant).lower()):
            raise ValueError(
                "GENEActiv file does not start with the participant ID. "
                f"Expected filename prefix {participant!r}, got: {direct.name}"
            )
        return direct

    if root.is_file():
        if root.suffix.lower() != ".bin":
            raise ValueError(f"Expected input_root to be a .bin file or directory, got: {root}")
        if require_subject_id_prefix and not root.name.lower().startswith(str(participant).lower()):
            raise ValueError(
                "GENEActiv file does not start with the participant ID. "
                f"Expected filename prefix {participant!r}, got: {root.name}"
            )
        return root.resolve()

    if not root.exists():
        raise FileNotFoundError(f"GENEActiv input root not found: {root}")

    participant = str(participant)
    visit = str(visit)
    sensor = str(sensor)

    candidate_dirs = [
        root / participant / visit / sensor,
        root / participant / visit,
        root / participant / sensor / visit,
        root / sensor / participant / visit,
        root / participant,
        root,
    ]

    candidates: list[Path] = []
    for directory in candidate_dirs:
        if directory.exists() and directory.is_dir():
            candidates.extend(path for path in directory.iterdir() if path.suffix.lower() == ".bin")

    if not candidates:
        all_bins = list(root.rglob("*.bin")) + list(root.rglob("*.BIN"))
        candidates = [
            path
            for path in all_bins
            if participant in {part.name for part in path.parents}
            and visit in {part.name for part in path.parents}
        ]

    candidates = _dedupe_paths(candidates)
    unfiltered_candidates = candidates
    if require_subject_id_prefix:
        candidates = _filter_subject_prefixed_bins(candidates, participant)
    if not candidates:
        prefix_note = (
            f" with filename starting with {participant!r}"
            if require_subject_id_prefix
            else ""
        )
        rejected_note = (
            "\nFiles found before participant-prefix filtering:\n"
            f"{_format_candidates(unfiltered_candidates)}"
            if unfiltered_candidates
            else ""
        )
        raise FileNotFoundError(
            f"Could not find a GENEActiv .bin file{prefix_note}. "
            f"Searched under {root} for participant={participant!r}, visit={visit!r}, sensor={sensor!r}."
            f"{rejected_note}"
        )
    if len(candidates) > 1:
        raise FileExistsError(
            "More than one GENEActiv .bin file matched. Set `geneactiv_file_path` in the TOML "
            "or pass it explicitly.\n"
            f"{_format_candidates(candidates)}"
        )

    return candidates[0]


def _infer_sampling_frequency_from_time(time: pd.Series) -> float:
    diffs = pd.to_datetime(time).diff().dt.total_seconds().dropna()
    diffs = diffs.loc[diffs > 0]
    if diffs.empty:
        raise ValueError("Could not infer sampling frequency from GENEActiv timestamps.")
    return float(1.0 / diffs.median())


def load_geneactiv_recording(geneactiv_bin_path: str | Path) -> tuple[pl.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Load a GENEActiv .bin file, autocalibrate acceleration, and return objects
    compatible with the existing sleep/circadian/activity pipeline stages.
    """
    geneactiv_bin_path = Path(geneactiv_bin_path).expanduser().resolve()
    info, data = read_geneactiv_bin(str(geneactiv_bin_path))
    raw_df = as_polars_dataframe(info, data)

    calibrated = autocalibrate(
        raw_df,
        ts_col="time",
        x_col="x",
        y_col="y",
        z_col="z",
        temp_col="temperature",
    )
    calibrated_df = calibrated["df"]
    cal_params = calibrated["params"]
    if not {"x_cal", "y_cal", "z_cal"}.issubset(set(calibrated_df.columns)):
        calibrated_df = calibrated_df.with_columns(
            [
                pl.col("x").alias("x_cal"),
                pl.col("y").alias("y_cal"),
                pl.col("z").alias("z_cal"),
            ]
        )

    raw_pd = raw_df.select(["time", "x", "y", "z", "temperature", "light"]).to_pandas()
    raw_pd["time"] = pd.to_datetime(raw_pd["time"])
    raw_pd = raw_pd.sort_values("time")

    acc_df = raw_pd.set_index("time")[["x", "y", "z"]].copy()
    temp_df = raw_pd.set_index("time")[["temperature"]].rename(columns={"temperature": "temp"}).copy()

    accel_freq_hz = float(info.fs) if np.isfinite(info.fs) and info.fs > 0 else _infer_sampling_frequency_from_time(raw_pd["time"])
    temperature_freq_hz = accel_freq_hz / 300.0

    metadata = {
        "device": "GENEActiv",
        "accel_freq_hz": accel_freq_hz,
        "temperature_freq_hz": temperature_freq_hz,
        "geneactiv_bin_path": str(geneactiv_bin_path),
        "start_time": acc_df.index.min(),
        "end_time": acc_df.index.max(),
        "n_acc_samples": int(len(acc_df)),
        "n_temp_samples": int(len(temp_df)),
        "geneactiv_npages": int(info.npages),
        "calibration_valid": bool(cal_params.valid) if cal_params is not None else False,
        "calibration_n_epochs_used": int(cal_params.n_epochs_used) if cal_params is not None else 0,
        "calibration_error_start": float(cal_params.cal_error_start)
        if cal_params is not None and cal_params.cal_error_start is not None
        else np.nan,
        "calibration_error_end": float(cal_params.cal_error_end)
        if cal_params is not None and cal_params.cal_error_end is not None
        else np.nan,
    }

    return calibrated_df, acc_df, temp_df, metadata


def detect_geneactiv_nonwear_intervals(
    calibrated_df: pl.DataFrame,
    *,
    fs: float,
    nonwear_method: str = "vanhees2013",
) -> pd.DataFrame:
    method = nonwear_method.lower()
    if method not in {"vanhees2013", "vanhees", "ggir"}:
        raise ValueError(
            f"Unsupported GENEActiv nonwear_method {nonwear_method!r}. "
            "Use 'vanhees2013', 'vanhees', or 'ggir'."
        )

    out = vanhees2013(
        calibrated_df,
        ts_col="time",
        x_col="x_cal",
        y_col="y_cal",
        z_col="z_cal",
        freq=float(fs),
        quiet=True,
        return_debug=True,
    )

    if out.nonwear_df.height == 0:
        return pd.DataFrame(columns=["start", "end"])

    nonwear_df = out.nonwear_df.select(["start", "end"]).to_pandas()
    nonwear_df["start"] = pd.to_datetime(nonwear_df["start"])
    nonwear_df["end"] = pd.to_datetime(nonwear_df["end"])
    return nonwear_df.sort_values("start").reset_index(drop=True)


def preprocess_geneactiv_recording(
    geneactiv_bin_path: str | Path,
    *,
    fs: Optional[float] = None,
    nonwear_method: str = "vanhees2013",
) -> Dict[str, object]:
    calibrated_df, acc_df, temp_df, info = load_geneactiv_recording(geneactiv_bin_path)
    if fs is None:
        fs = float(info["accel_freq_hz"])

    nonwear_df = detect_geneactiv_nonwear_intervals(
        calibrated_df,
        fs=float(fs),
        nonwear_method=nonwear_method,
    )

    return {
        "calibrated_df": calibrated_df,
        "acc_df": acc_df,
        "temp_df": temp_df,
        "info": info,
        "fs": float(fs),
        "nonwear_df": nonwear_df,
    }

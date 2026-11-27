"""Exploratory, plot-oriented comparison; never called by the production runners.

Run from notebooks/compare_lower_back_nonwear.ipynb, or pass --config to this
script. GGIR's original R functions are downloaded, checksum-verified and run
only when run_comparison is called. Rscript and the locked Python environment
are sufficient; installing or upgrading the GGIR R package is unnecessary.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from urllib.request import urlopen
from uuid import uuid4

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dare_wearables.lower_back.nonwear.ggir2023 import NonwearOptions
from dare_wearables.lower_back.nonwear.vanhees2013 import vanhees2013
from dare_wearables.lower_back.preprocessing import (
    GRAV_MS2,
    _build_wear_mask_from_nonwear,
    preprocess_lowerback_file,
)

FS = 100.0
EXPECTED_HOURS = 7 * 24.0
GGIR_VERSION = "3.3-9"
GGIR_COMMIT = "5acd50f9283e02fe0efc98057978dc8e9d5629ab"
GGIR_BASE_URL = f"https://raw.githubusercontent.com/cran/GGIR/{GGIR_COMMIT}/R"
GGIR_SOURCES = {
    "load_params.R": "389d9db65d523ea10e40326a5422e636c8fb0ff30484e11666772de33a1bafc1",
    "detect_nonwear_clipping.R": "6fd2e93cf3b30779970d1c3d79858396ac5d0248ea004add52032cdb0bf857d4",
    "g.weardec.R": "08c0f07ccc76f4ef881038c95eb7476c54a181a8192c6e87b666f5b55e95bdee",
}
METHODS = ("current", "ggir_2023_matched", "ggir_2023_matched_3axes")
AXES_REQUIRED = dict(zip(METHODS, (2, 2, 3)))
LABELS = {
    "current": "Legacy Python · ≥2 axes",
    "ggir_2023_matched": "GGIR 2023 · matched · ≥2 axes",
    "ggir_2023_matched_3axes": "GGIR 2023 · matched · 3 axes (experimental)",
}
COLORS = dict(zip(METHODS, ("#c0392b", "#19715b", "#315fc4")))

# This bridge calls upstream functions with either 2 or 3 required axes, using
# matched SD/range thresholds. The 3-axis choice is an experimental departure
# from GGIR's standard. Its postprocessing can still add nonwear. Only base R
# is used; clipping is exported separately and excluded from the mask (r1 + r3).
R_BRIDGE = r'''
args = commandArgs(trailingOnly = TRUE)
source_dir = args[1]
input_path = args[2]
n = as.numeric(args[3])
fs = as.numeric(args[4])
output_dir = args[5]
matched_sd = as.numeric(args[6]) / 1000
matched_range = as.numeric(args[7]) / 1000
for (name in c("load_params.R", "detect_nonwear_clipping.R", "g.weardec.R")) {
  source(file.path(source_dir, name))
}
defaults = load_params()
stopifnot(defaults$params_cleaning$nonwear_approach == "2023")
windows = defaults$params_general$windowsizes
if (length(args) >= 8) windows[3] = as.numeric(args[8]) # optional validation window override
medium_samples = windows[2] * fs
stopifnot(n >= medium_samples, medium_samples == round(medium_samples))
connection = file(input_path, "rb")
values = readBin(connection, "double", n = n * 3, endian = "little")
close(connection)
stopifnot(length(values) == n * 3)
data = as.data.frame(matrix(values, ncol = 3, byrow = TRUE))
rm(values)
names(data) = c("x", "y", "z")
# Compute identical acceleration scores once. The sole difference between the
# two GGIR variants is the number of required axes at the initial decision.
detected = detect_nonwear_clipping(
  data = data, windowsizes = windows, sf = fs,
  sdcriter = matched_sd, racriter = matched_range, nonwear_approach = "2023"
)
metalong = data.frame(nonwearscore = detected$NWav,
                      clippingscore = detected$CWav)
profiles = c(ggir_2023_matched = 2L, ggir_2023_matched_3axes = 3L)
settings = list()
for (name in names(profiles)) {
  axes_required = profiles[[name]]
  decision = g.weardec(metalong, wearthreshold = axes_required, ws2 = windows[2],
                       params_cleaning = defaults$params_cleaning)
  count = detected$nmin
  debug = data.frame(
    start_sample = (seq_len(count) - 1) * medium_samples,
    stop_sample = seq_len(count) * medium_samples,
    nonwear_score = detected$NWav, axes_required = axes_required,
    is_nonwear_window = as.logical(decision$r1),
    is_added_nonwear = as.logical(decision$r3),
    is_final_nonwear = as.logical(decision$r1 + decision$r3),
    is_clipping = as.logical(decision$r2)
  )
  write.csv(debug, file.path(output_dir, paste0(name, "_windows.csv")), row.names = FALSE)
  settings[[name]] = data.frame(
    method = name, sd_thresh_mg = matched_sd * 1000,
    range_thresh_mg = matched_range * 1000, axes_required = axes_required,
    window_minutes = windows[3] / 60, step_minutes = windows[2] / 60,
    statistics_hz = fs / ceiling(fs / 5),
    edge_correction = defaults$params_cleaning$nonWearEdgeCorrection,
    nonwear_approach = "2023", r_version = R.version.string
  )
}
write.csv(do.call(rbind, settings), file.path(output_dir, "ggir_settings.csv"), row.names = FALSE)
'''


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def load_settings(config_path: str | Path) -> dict:
    """Read private settings; participant selection is always explicit."""
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("rb") as stream:
        settings = tomllib.load(stream)["comparison"]
    for key in ("bronze_root", "output_root"):
        path = Path(settings[key]).expanduser()
        settings[key] = (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
    participants = settings.get("participant_ids", [])
    visits = settings.get("visits", ["T0"])
    if not participants:
        raise ValueError("Set participant_ids explicitly; an empty list never starts a batch.")
    if not visits:
        raise ValueError("Select at least one visit.")
    for value in [*participants, *visits, settings.get("sensor", "McRoberts")]:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("Participant IDs, visits and sensor must be quoted, simple folder names.")
    settings["participant_ids"] = list(dict.fromkeys(participants))
    settings["visits"] = list(dict.fromkeys(visits))
    settings.setdefault("sensor", "McRoberts")
    settings.setdefault("rscript", "Rscript")
    settings["config_path"] = config_path
    bronze, output = settings["bronze_root"], settings["output_root"]
    if output == bronze or bronze in output.parents:
        raise ValueError("Choose an output_root outside bronze_root.")
    return settings


def _ggir_sources(cache: Path) -> Path:
    """Fetch the pinned upstream files on demand, verifying exact bytes."""
    cache.mkdir(parents=True, exist_ok=True)
    for name, expected in GGIR_SOURCES.items():
        path = cache / name
        payload = path.read_bytes() if path.exists() else urlopen(
            f"{GGIR_BASE_URL}/{name}", timeout=60
        ).read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"GGIR source checksum mismatch: {name}; comparison stopped.")
        if not path.exists():
            path.write_bytes(payload)
    return cache


def _mask(windows: pd.DataFrame, column: str, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for row in windows.loc[windows[column]].itertuples(index=False):
        start, stop = int(row.start_sample), int(row.stop_sample)
        if not 0 <= start < stop <= n:
            raise ValueError("Detector window exceeds its recording.")
        mask[start:stop] = True
    return mask


def _bouts(mask: np.ndarray, times: pd.DatetimeIndex, gap_after: np.ndarray) -> pd.DataFrame:
    """End intervals at missing time; durations count observed samples only."""
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1] | gap_after])
    stops = np.flatnonzero(mask & np.r_[~mask[1:] | gap_after, True]) + 1
    return pd.DataFrame({
        "start_sample": starts, "stop_sample": stops,
        "start": times[starts],
        "end": times[stops - 1] + pd.Timedelta(seconds=1 / FS),
        "duration_seconds": (stops - starts) / FS,
    })


def _python_windows(windows: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Recover sample-defined epochs without searching lossy timestamps.

    The detector exports timestamps at microsecond precision, while OMX samples
    can retain nanoseconds. Its 15-minute epochs are defined by sample count,
    so use that same grid and check the exported timestamps at their resolution.
    """
    windows = windows.copy()
    epoch_samples = int(15 * 60 * FS)
    if len(windows) != len(times) // epoch_samples:
        raise ValueError("Current diagnostic windows do not match the expected sample grid.")
    starts = np.arange(len(windows), dtype=np.int64) * epoch_samples
    stops = starts + epoch_samples
    for column, indices in (("win_start", starts), ("win_end", stops - 1)):
        exported = pd.DatetimeIndex(windows[column]).as_unit("us")
        expected = times[indices].as_unit("us")
        if not np.array_equal(exported.asi8, expected.asi8):
            raise ValueError(f"Current diagnostic {column} timestamps do not match their samples.")
    windows["start_sample"] = starts
    windows["stop_sample"] = stops
    return windows


def _production_mask_from_windows(windows: pd.DataFrame, times: pd.DatetimeIndex,
                                  column: str = "is_final_nonwear") -> np.ndarray:
    """Merge labelled epochs, then apply production's timestamp-bout masking.

    Preserve the exported microsecond boundaries here: production compares
    those inclusive bout ends with the original nanosecond sample timestamps.
    Applying each epoch separately would introduce false holes inside a bout.
    """
    flags = windows[column].to_numpy(dtype=bool)
    if len(flags) == 0:
        return np.zeros(len(times), dtype=bool)
    starts = np.flatnonzero(flags & np.r_[True, ~flags[:-1]])
    ends = np.flatnonzero(flags & np.r_[~flags[1:], True])
    bouts = pd.DataFrame({
        "start": windows.iloc[starts].win_start.to_numpy(),
        "end": windows.iloc[ends].win_end.to_numpy(),
    })
    return ~_build_wear_mask_from_nonwear(times, bouts)


def _ggir_windows(path: Path, times: pd.DatetimeIndex) -> pd.DataFrame:
    windows = pd.read_csv(path)
    for column in ("is_nonwear_window", "is_added_nonwear", "is_final_nonwear", "is_clipping"):
        values = windows[column].astype(str).str.lower()
        if not values.isin(["true", "false"]).all():
            raise ValueError(f"Invalid GGIR flags: {path.name}, {column}")
        windows[column] = values.eq("true")
    starts = windows.start_sample.to_numpy(dtype=int)
    stops = windows.stop_sample.to_numpy(dtype=int)
    if len(windows) == 0 or np.any(starts < 0) or np.any(stops > len(times)):
        raise ValueError("GGIR returned invalid window bounds.")
    windows["win_start"] = times[starts]
    windows["win_end"] = times[stops - 1]
    return windows


def _plot_timeline(frame: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    timeline = pd.DataFrame()
    for column, axis in zip(("acc_x", "acc_y", "acc_z"), ("x", "y", "z")):
        signal = pd.Series(frame[column].to_numpy() / GRAV_MS2, index=times)
        stats = signal.resample("1s").agg(["mean", "min", "max"])
        for stat in ("mean", "min", "max"):
            timeline[f"{axis}_{stat}"] = stats[stat]
    timeline["observed_samples"] = pd.Series(1, index=times).resample("1s").count()
    return timeline


def _compare_recording(path: Path, folder: Path, sources: Path, rscript: str,
                       subject: str, visit: str) -> pd.DataFrame:
    # Use the existing loader, calibration, sample cap and current detector once.
    prepared = preprocess_lowerback_file(str(path), verbose=False, nonwear=NonwearOptions(method="current"))
    frame = prepared.pop("df")
    calibration = prepared.pop("calibration")
    times = pd.DatetimeIndex(frame.datetime).as_unit("ns")
    n = len(times)
    if len(times) < 15 * 60 * FS:
        raise ValueError("Comparison needs at least 15 minutes of observed samples.")
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Comparison needs unique, increasing timestamps.")
    if any(not np.isfinite(frame[column].to_numpy()).all() for column in ("acc_x", "acc_y", "acc_z")):
        raise ValueError("Calibrated acceleration contains nonfinite values.")
    calibrated = calibration["df"]
    calibrated_columns = ["x_cal", "y_cal", "z_cal"] if {"x_cal", "y_cal", "z_cal"}.issubset(
        calibrated.columns
    ) else ["x", "y", "z"]
    if calibrated.height != n:
        raise ValueError("Calibrated samples and current detector input do not align.")
    signature = inspect.signature(vanhees2013).parameters
    if (signature["non_wear_window"].default, signature["window_step_size"].default,
        signature["num_axes_required"].default) != (60.0, 15.0, 2):
        raise ValueError("Current detector window/axis defaults changed; review the matched comparison.")
    sd_mg = signature["std_thresh_mg"].default
    range_mg = signature["value_range_thresh_mg"].default
    folder.mkdir(parents=True, exist_ok=False)
    current = _python_windows(prepared["nonwear_windows"], times)
    current["axes_required"] = AXES_REQUIRED["current"]
    current.to_csv(folder / "current_windows.csv", index=False)
    current_mask = ~frame.wear.to_numpy(dtype=bool)
    if not np.array_equal(current_mask, _production_mask_from_windows(current, times)):
        raise ValueError("Merged current diagnostic bouts and production wear mask disagree.")

    with tempfile.TemporaryDirectory(prefix="r-input-", dir=folder) as temporary:
        temporary = Path(temporary)
        binary = temporary / "acceleration_g.bin"
        # Read the exact calibrated g values used by Python, before its final
        # conversion to m/s². Chunking avoids another full recording in RAM.
        with binary.open("wb") as stream:
            for start in range(0, n, 1_000_000):
                chunk = calibrated.slice(start, 1_000_000).select(calibrated_columns)
                chunk.to_numpy().astype("<f8", copy=False).tofile(stream)
        del chunk, calibrated, calibration
        timeline = _plot_timeline(frame, times)
        del frame  # release full-resolution ACC/gyro before the R process starts
        bridge = temporary / "compare.R"
        bridge.write_text(R_BRIDGE)
        completed = subprocess.run(
            [rscript, "--vanilla", str(bridge), str(sources), str(binary), str(n),
             str(FS), str(folder), str(sd_mg), str(range_mg)],
            capture_output=True, text=True,
        )
        (folder / "r_stdout.log").write_text(completed.stdout)
        (folder / "r_stderr.log").write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"GGIR comparison failed; inspect {folder / 'r_stderr.log'}")

    # Missing time is reported separately from observed nonwear for all methods.
    delta = np.diff(times.asi8) / 1e9
    gap_after = delta > 1.5 / FS
    gap_indices = np.flatnonzero(gap_after)
    gaps = pd.DataFrame({
        "start": times[gap_indices] + pd.Timedelta(seconds=1 / FS),
        "end": times[gap_indices + 1],
        "duration_seconds": delta[gap_indices] - 1 / FS,
    })
    gaps.to_csv(folder / "recording_gaps.csv", index=False)
    recorded_hours = n / FS / 3600
    missing_hours = max(EXPECTED_HOURS - recorded_hours, 0)

    summaries = []
    for method in METHODS:
        windows = current if method == "current" else _ggir_windows(folder / f"{method}_windows.csv", times)
        if method != "current":
            windows.to_csv(folder / f"{method}_windows.csv", index=False)
        nonwear = current_mask if method == "current" else _mask(windows, "is_final_nonwear", n)
        base = (_production_mask_from_windows(windows, times, "is_nonwear_window")
                if method == "current" else _mask(windows, "is_nonwear_window", n))
        rounding_difference = np.count_nonzero(_mask(windows, "is_final_nonwear", n) ^ nonwear)
        bouts = _bouts(nonwear, times, gap_after)
        bouts.to_csv(folder / f"{method}_bouts.csv", index=False)
        timeline[f"{method}_nonwear_fraction"] = pd.Series(nonwear, index=times).resample("1s").mean()
        disagreement = nonwear ^ current_mask
        timeline[f"{method}_disagreement_fraction"] = pd.Series(disagreement, index=times).resample("1s").mean()
        hours = nonwear.sum() / FS / 3600
        summaries.append({
            "subject": subject, "visit": visit, "method": method,
            "axes_required": AXES_REQUIRED[method],
            "recorded_hours": recorded_hours,
            "detected_nonwear_hours": hours,
            "detected_nonwear_percent": 100 * nonwear.mean(),
            "base_nonwear_hours": base.sum() / FS / 3600,
            "added_nonwear_hours": (nonwear & ~base).sum() / FS / 3600,
            "missing_recording_hours": missing_hours,
            "missing_recording_percent": 100 * missing_hours / EXPECTED_HOURS,
            "total_unavailable_percent": 100 * (hours + missing_hours) / EXPECTED_HOURS,
            "internal_gap_hours": gaps.duration_seconds.sum() / 3600,
            "disagreement_hours": disagreement.sum() / FS / 3600,
            "additional_nonwear_vs_current_hours": (nonwear & ~current_mask).sum() / FS / 3600,
            "additional_wear_vs_current_hours": (~nonwear & current_mask).sum() / FS / 3600,
            "unclassified_tail_minutes": (n % int(15 * 60 * FS)) / FS / 60,
            "boundary_rounding_difference_samples": rounding_difference,
        })
    timeline.index.name = "datetime"
    timeline.to_parquet(folder / "plot_timeline.parquet")
    covered = n // int(15 * 60 * FS) * int(15 * 60 * FS)
    _write_json(folder / "recording.json", {
        "subject": subject, "visit": visit, "input": str(path),
        "input_size_bytes": path.stat().st_size, "input_mtime_ns": path.stat().st_mtime_ns,
        "sample_rate_hz": FS, "n_samples": n,
        "start": times[0], "end": times[-1] + pd.Timedelta(seconds=1 / FS),
        "unclassified_tail_start": times[covered] if covered < n else None,
        "current_sd_thresh_mg": sd_mg, "current_range_thresh_mg": range_mg,
        "expected_hours": EXPECTED_HOURS,
    })
    return pd.DataFrame(summaries)


def run_comparison(config_path: str | Path) -> Path:
    """Run explicitly selected recordings into a new, never-reused directory."""
    settings = load_settings(config_path)
    rscript = shutil.which(str(Path(settings["rscript"]).expanduser()))
    if rscript is None:
        raise FileNotFoundError("Rscript was not found; install R or set comparison.rscript.")
    recordings = []
    for subject in settings["participant_ids"]:
        for visit in settings["visits"]:
            folder = settings["bronze_root"] / subject / visit / settings["sensor"]
            files = sorted(p for p in folder.glob("*") if p.is_file() and p.suffix.lower() == ".omx")
            if len(files) != 1:
                raise ValueError(f"Expected exactly one OMX recording for {subject}/{visit}; found {len(files)}.")
            recordings.append((subject, visit, files[0]))
    sources = _ggir_sources(settings["output_root"] / "ggir_source" / GGIR_COMMIT)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = settings["output_root"] / f"run_{stamp}_{uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "created_utc": stamp, "settings": settings,
        "comparison_schema_version": 2, "methods": METHODS,
        "axes_required": AXES_REQUIRED,
        "ggir_version": GGIR_VERSION, "ggir_commit": GGIR_COMMIT,
        "ggir_source_sha256": GGIR_SOURCES,
        "comparison_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "current_detector_sha256": hashlib.sha256(Path(inspect.getfile(vanhees2013)).read_bytes()).hexdigest(),
        "scope": "Detector functions on shared calibrated samples; not a full GGIR reader/imputation run.",
    }
    _write_json(run / "manifest.json", manifest)
    results = []
    try:
        for subject, visit, path in recordings:
            print(f"Comparing {subject}/{visit}", flush=True)
            results.append(_compare_recording(path, run / subject / visit, sources, rscript, subject, visit))
        pd.concat(results, ignore_index=True).to_csv(run / "subject_summary.csv", index=False)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    else:
        manifest["status"] = "completed"
    finally:
        _write_json(run / "manifest.json", manifest)
    return run


def load_summary(run: str | Path) -> pd.DataFrame:
    run = Path(run).expanduser().resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("This run did not complete; inspect manifest.json.")
    summary = pd.read_csv(run / "subject_summary.csv", dtype={"subject": "string", "visit": "string"})
    return _active_summary(summary)


def _active_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Exclude retired variants and refuse to mislabel an older comparison."""
    missing = set(METHODS) - set(summary.method.dropna())
    if missing:
        raise ValueError(
            "This run lacks the matched-threshold two-/three-axis comparison. "
            "Reload the helper and run a new comparison; old results are preserved. "
            f"Missing methods: {', '.join(sorted(missing))}."
        )
    return summary.loc[summary.method.isin(METHODS)].copy()


def plot_summary(summary: pd.DataFrame, upper_quantile: float = 1.0):
    """Reference-notebook-style histograms; trimming affects drawing only."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    summary = _active_summary(summary)
    if not 0 < upper_quantile <= 1:
        raise ValueError("upper_quantile must be in (0, 1].")
    sns.set_context("notebook")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    columns = ("detected_nonwear_hours", "detected_nonwear_percent",
               "missing_recording_hours", "total_unavailable_percent")
    titles = ("Detected nonwear (hours)", "Detected nonwear (% of observed samples)",
              "Missing recording until 7 days (hours)", "Nonwear + missing (% of 7 days)")
    for ax, column, title in zip(axes.flat, columns, titles):
        methods = ("current",) if column == "missing_recording_hours" else METHODS
        distributions = {}
        for method in methods:
            full = summary.loc[summary.method.eq(method), column].dropna()
            if not full.empty:
                distributions[method] = (full, full[full <= full.quantile(upper_quantile)])
        if not distributions:
            ax.set_title(title)
            continue
        pooled = pd.concat([shown for _, shown in distributions.values()])
        bins = np.histogram_bin_edges(pooled, bins=25)
        for method, (full, shown) in distributions.items():
            label = "All methods" if len(methods) == 1 else LABELS[method]
            q1, median, q3 = full.quantile([.25, .5, .75])
            color = "#666666" if len(methods) == 1 else COLORS[method]
            sns.histplot(shown, bins=bins, element="step", fill=False, color=color, ax=ax)
            ax.axvline(median, color=color, ls="--", label=f"{label}: {median:.2f} [{q1:.2f}–{q3:.2f}] (n={len(full)})")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.legend(fontsize=8)
    fig.suptitle(f"Median [IQR] uses all recordings; plot upper quantile = {upper_quantile:g}")
    fig.tight_layout()
    return fig


def plot_paired(summary: pd.DataFrame):
    """Compare current vs GGIR, then isolate GGIR's two-/three-axis change."""
    import matplotlib.pyplot as plt

    summary = _active_summary(summary)
    paired = summary.pivot(index=["subject", "visit"], columns="method", values="detected_nonwear_percent")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    pairs = (("current", "ggir_2023_matched"),
             ("ggir_2023_matched", "ggir_2023_matched_3axes"))
    short_labels = dict(zip(METHODS, ("Current Python", "GGIR matched ≥2 axes", "GGIR matched 3 axes")))
    for ax, (x_method, y_method) in zip(axes, pairs):
        ax.scatter(paired[x_method], paired[y_method], color=COLORS[y_method], alpha=.75)
        ax.plot([0, 100], [0, 100], "--", color="#777777", lw=1)
        ax.set(title=f"{short_labels[x_method]} vs {short_labels[y_method]}",
               xlabel=f"{short_labels[x_method]} nonwear (%)", ylabel=f"{short_labels[y_method]} nonwear (%)",
               xlim=(-2, 102), ylim=(-2, 102))
        ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_recording(run: str | Path, subject: str, visit: str = "T0", *, start=None, end=None,
                   max_plot_points: int = 12_000):
    """Aligned acceleration envelopes, nonwear shading and disagreement.

    start/end control display only. Detection always uses the full selected
    recording. Wide views aggregate the saved 1-second trace for drawing only,
    retaining the full sample min/max envelope in each display bin.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    run = Path(run).expanduser().resolve()
    load_summary(run)  # refuse incomplete runs
    folder = run / str(subject) / str(visit)
    info = json.loads((folder / "recording.json").read_text())
    timeline = pd.read_parquet(folder / "plot_timeline.parquet")
    left = pd.Timestamp(start) if start is not None else timeline.index[0]
    right = pd.Timestamp(end) if end is not None else timeline.index[-1] + pd.Timedelta(seconds=1)
    if left >= right:
        raise ValueError("Plot start must precede plot end.")
    view = timeline.loc[(timeline.index >= left) & (timeline.index < right)]
    if view.empty:
        raise ValueError("The requested interval does not overlap the recording.")
    if max_plot_points < 1:
        raise ValueError("max_plot_points must be positive.")
    display_seconds = max(1, int(np.ceil(len(view) / max_plot_points)))
    aggregations = {f"{axis}_{stat}": stat for axis in ("x", "y", "z") for stat in ("mean", "min", "max")}
    signal_view = view[list(aggregations)].resample(f"{display_seconds}s", origin=left).agg(aggregations)
    gaps = pd.read_csv(folder / "recording_gaps.csv", parse_dates=["start", "end"])
    fig, axes = plt.subplots(4, 1, figsize=(17, 11), sharex=True, gridspec_kw={"height_ratios": [2, 2, 2, 1]})
    for ax, method in zip(axes[:3], METHODS):
        for axis, color in zip(("x", "y", "z"), ("#a64d79", "#26846e", "#476aa0")):
            ax.fill_between(signal_view.index, signal_view[f"{axis}_min"], signal_view[f"{axis}_max"], color=color, alpha=.13)
            ax.plot(signal_view.index, signal_view[f"{axis}_mean"], color=color, lw=.6, label=axis)
        bouts = pd.read_csv(folder / f"{method}_bouts.csv", parse_dates=["start", "end"])
        overlap = bouts.loc[(bouts.end > left) & (bouts.start < right)]
        for i, row in enumerate(overlap.itertuples(index=False)):
            ax.axvspan(max(left, row.start), min(right, row.end), color=COLORS[method], alpha=.20,
                       label="Nonwear" if i == 0 else None)
        ax.set_title(LABELS[method], loc="left")
        ax.set_ylabel("Acceleration (g)")
        ax.legend(loc="upper right", ncol=4, fontsize=8)
    for method in METHODS[1:]:
        axes[-1].plot(view.index, view[f"{method}_disagreement_fraction"], color=COLORS[method],
                      lw=.8, label=f"{LABELS[method]} vs current")
    axes[-1].set(ylabel="Disagreement\n(fraction / s)", ylim=(-.05, 1.05))
    axes[-1].legend(fontsize=8, loc="upper right")
    for ax in axes:
        for row in gaps.loc[(gaps.end > left) & (gaps.start < right)].itertuples(index=False):
            ax.axvspan(max(left, row.start), min(right, row.end), color="#666666", alpha=.35)
        if info["unclassified_tail_start"] is not None:
            tail = pd.Timestamp(info["unclassified_tail_start"])
            stop = min(right, pd.Timestamp(info["end"]))
            if max(left, tail) < stop:
                ax.axvspan(max(left, tail), stop, color="#e4b44f", alpha=.3, hatch="//")
        ax.set_xlim(left, right)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    fig.suptitle(f"{subject} · {visit} — acceleration display bins: {display_seconds} s, sample min/max; gray = missing; amber = unclassified tail")
    fig.tight_layout()
    return fig


def plot_window_decisions(run: str | Path, subject: str, visit: str = "T0", *, start=None, end=None):
    """Separate initial threshold labels from postprocessing additions."""
    import matplotlib.pyplot as plt

    run = Path(run).expanduser().resolve()
    load_summary(run)
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True, sharey=True)
    for ax, method in zip(axes, METHODS):
        windows = pd.read_csv(run / str(subject) / str(visit) / f"{method}_windows.csv",
                              parse_dates=["win_start", "win_end"])
        for row, column, color, label in (
            (0, "is_nonwear_window", "#666666", "Initial nonwear"),
            (1, "is_added_nonwear", "#ca8a19", "Added by postprocessing"),
            (2, "is_final_nonwear", COLORS[method], "Final nonwear"),
        ):
            flags = windows[column].astype(str).str.lower().eq("true")
            for i, window in enumerate(windows.loc[flags].itertuples(index=False)):
                ax.plot([window.win_start, window.win_end], [row, row], color=color, lw=6,
                        solid_capstyle="butt", label=label if i == 0 else None)
        ax.set_yticks([0, 1, 2], ["Initial", "Added", "Final"])
        ax.set_title(LABELS[method], loc="left")
        ax.set_ylim(-.5, 2.5)
        if start is not None or end is not None:
            ax.set_xlim(left=pd.Timestamp(start) if start is not None else None,
                        right=pd.Timestamp(end) if end is not None else None)
    fig.suptitle("15-minute detector decisions on observed samples; inspect gaps in the acceleration plot")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run_comparison(args.config))

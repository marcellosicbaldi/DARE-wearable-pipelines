"""Auditable outputs for respiration-assisted nonwear (no raw data or PSD)."""
from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from dare_wearables.common.output_state import OutputRun, _digest, signature
from .ggir2023 import NonwearOptions, detector_provenance
from dare_wearables.lower_back.io import timestamp_quality
from dare_wearables.lower_back.io.timestamp_quality import TimestampOptions, QUALITY_FILES, write_quality

TABLES = {
    "baseline_bouts": "accelerometer_nonwear_intervals.csv",
    "evidence_bouts": "respiration_evidence_intervals.csv",
    "rescued_bouts": "respiration_reassigned_intervals.csv",
    "nonwear_bouts_before_duration_filter": "respiration_refined_nonwear_intervals.csv",
    "short_nonwear_bouts": "short_nonwear_reassigned_intervals.csv",
    "nonwear_bouts": "final_nonwear_intervals.csv",
}


def source_hashes():
    from dare_wearables.lower_back import preprocessing
    from dare_wearables.lower_back.io import mcroberts_loader
    from dare_wearables.common import autocalibrate
    modules = [Path(__file__), Path(preprocessing.__file__), Path(mcroberts_loader.__file__),
               Path(autocalibrate.__file__), Path(__file__).with_name("respiration.py"),
               Path(__file__).with_name("vanhees2013.py"), Path(__file__).with_name("assessment.py"),
               Path(__file__).with_name("ggir2023.py"), Path(timestamp_quality.__file__)]
    modules.append(Path(__file__).with_name("minute_grid.py"))
    return {str(path.name): _digest(path) for path in modules}


def output_record(folder, options):
    folder = Path(folder)
    names = [*TABLES.values(), *QUALITY_FILES, "accelerometer_nonwear_windows.parquet", "nonwear_daily.csv", "nonwear_summary.json"]
    if options.save_window_features:
        names.append("respiration_windows.parquet")
    return OutputRun(folder / "respiration_manifest.json", [folder / name for name in names])


def audit_signature(source, options, nonwear=None, timestamps=None):
    import scipy
    import polars
    source = Path(source).resolve()
    stat = source.stat()
    nonwear = nonwear or NonwearOptions()
    timestamps = timestamps or TimestampOptions()
    timestamps.validate()
    parameters = {"method": "accelerometer_plus_gyro_respiration_v3", "options": asdict(options),
                  "baseline_detector": detector_provenance(nonwear),
                  "timestamps": asdict(timestamps),
                  "input": {"path": str(source), "size_bytes": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns},
                  "sources": source_hashes(), "dependencies": {
                      "numpy": np.__version__, "pandas": pd.__version__,
                      "scipy": scipy.__version__, "polars": polars.__version__}}
    return signature(parameters, [source]), parameters


def daily_summary(frame, fs):
    """Count observed samples, avoiding allocation of one Python date per sample."""
    columns = ["date", "observed_hours", "baseline_nonwear_hours", "revised_nonwear_hours",
               "reassigned_to_wear_hours", "reassigned_09_21_hours",
               "respiration_reassigned_to_wear_hours", "short_nonwear_reassigned_to_wear_hours"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    times = pd.DatetimeIndex(frame.datetime)
    base = ~frame.wear_accelerometer.to_numpy(dtype=bool)
    final = ~frame.wear.to_numpy(dtype=bool)
    rescued = frame.wear_respiration_override.to_numpy(dtype=bool)
    short = (frame.wear_short_nonwear_override.to_numpy(dtype=bool)
             if "wear_short_nonwear_override" in frame else np.zeros(len(frame), dtype=bool))
    reassigned = rescued | short
    rows = []
    for day in pd.date_range(times[0].normalize(), times[-1].normalize(), freq="D"):
        first, stop = times.searchsorted([day, day + pd.DateOffset(days=1)])
        left, right = times.searchsorted([day + pd.Timedelta(hours=9), day + pd.Timedelta(hours=21)])
        counts = [stop - first, base[first:stop].sum(), final[first:stop].sum(),
                  reassigned[first:stop].sum(), reassigned[left:right].sum(),
                  rescued[first:stop].sum(), short[first:stop].sum()]
        rows.append([str(day.date()), *(float(value) / fs / 3600 for value in counts)])
    return pd.DataFrame(rows, columns=columns)


def write_audit(prepared, folder, *, source, options):
    """Publish a completion marker last; incomplete exports cannot be reused."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    record = output_record(folder, options)
    record.clear()
    diagnostics = prepared["respiration"]
    result = diagnostics["result"]
    nonwear = prepared.get("nonwear_options", NonwearOptions())
    timestamps = prepared.get("timestamp_options", TimestampOptions())
    key, provenance = audit_signature(source, options, nonwear, timestamps)
    intervals = prepared.get("time_invalid_intervals", pd.DataFrame(columns=["start", "end", "duration_seconds", "reason"]))
    quality = prepared.get("timestamp_quality", {"policy": timestamps.policy, "events": [], "recording_discontinuities": []})
    write_quality(folder, quality, intervals)
    for name, filename in TABLES.items():
        diagnostics[name].to_csv(folder / filename, index=False)
    prepared.get("nonwear_windows", pd.DataFrame()).to_parquet(
        folder / "accelerometer_nonwear_windows.parquet", index=False)
    daily = daily_summary(prepared["df"], result.fs)
    daily["time_invalid_hours"] = 0.
    for row in intervals.itertuples():
        for index, day in enumerate(pd.to_datetime(daily.date)):
            first, stop = max(row.start, day), min(row.end, day+pd.Timedelta(days=1))
            if stop > first:
                daily.loc[index, "time_invalid_hours"] += (stop-first).total_seconds()/3600
    daily.to_csv(folder / "nonwear_daily.csv", index=False)
    if options.save_window_features:
        result.windows.to_parquet(folder / "respiration_windows.parquet", index=False)
    summary = {**diagnostics["summary"], "baseline_method": nonwear.method,
               "baseline_detector": prepared.get("nonwear_provenance", provenance["baseline_detector"]),
               "observed_hours": result.n_samples / result.fs / 3600,
               "sample_rate_hz": result.fs, "n_samples": result.n_samples,
               "start": str(prepared["df"].datetime.iloc[0]) if result.n_samples else None,
               "last_sample": str(prepared["df"].datetime.iloc[-1]) if result.n_samples else None,
               "n_windows": len(result.windows),
               "positive_windows": int(result.windows.respiration_present.sum()),
               "timestamp_policy": timestamps.policy,
               "quarantined_anchor_count": len(quality.get("events", [])),
               "recording_discontinuity_count": len(quality.get("recording_discontinuities", [])),
               "time_invalid_hours": float(intervals.duration_seconds.sum()/3600),
               "reassigned_09_21_hours": float(daily.reassigned_09_21_hours.sum()),
               "provenance": provenance,
               "interval_contract": "half-open sample bounds [start_sample, stop_sample); timestamp end exclusive",
               "daytime_hours": "09:00–21:00, QC reporting only; detection runs at all hours"}
    (folder / "nonwear_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    record.complete(key)
    return summary

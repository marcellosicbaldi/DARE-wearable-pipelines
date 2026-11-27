"""Read-only inspection of saved GGIR/respiration/duration-filter results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dare_wearables.common.output_state import _digest, signature
from dare_wearables.lower_back.io.mcroberts_loader import read_dp7_as_dataframe_fast
from dare_wearables.lower_back.io.timestamp_quality import TimestampOptions
from dare_wearables.lower_back.nonwear.respiration import _bounds, _merge, _segment_bounds

SHADING = {
    "final_nonwear": ("Final nonwear", "#315fc4"),
    "respiration": ("Wear: respiration", "#19715b"),
    "short_bouts": ("Wear: nonwear <15 min", "#8e44ad"),
}


def find_saved_run(root, *, run=None, subject=None, visit="T0", fallback_roots=()):
    """Latest completed assessment/refinement containing the chosen participant."""
    if run is not None:
        candidates = [Path(run).expanduser().resolve()]
    else:
        candidates = []
        for base in (root, *fallback_roots):
            base = Path(base).expanduser()
            candidates.extend([*base.glob("duration_*"), *base.glob("assessment_*")])
        candidates.sort(key=lambda p: p.name.split("_", 2)[1], reverse=True)
    for candidate in candidates:
        path = candidate / "assessment.json"
        if not path.is_file():
            path = candidate / "duration_filter_manifest.json"
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text())
        if manifest.get("status") not in ("completed", "completed_with_failures"):
            continue
        if subject is None or any(item["subject"] == str(subject) and item["visit"] == str(visit)
                                  and item.get("status", "completed") == "completed"
                                  for item in manifest["recordings"]):
            return candidate.resolve()
    raise FileNotFoundError("No completed results for this participant; set INSPECTION_RUN to a saved assessment or duration-filter folder.")


def _verified(path, expected):
    if not expected or _digest(path) != expected:
        raise ValueError(f"Saved result checksum failed: {path}")
    return path


def _read_bouts(path, expected, n, fs):
    frame = pd.read_csv(_verified(path, expected))
    for column in ("start_sample", "stop_sample", "duration_seconds"):
        frame[column] = pd.to_numeric(frame[column])
    bounds = _bounds(frame, n)
    for column in ("start", "end"):
        frame[column] = pd.to_datetime(frame[column], format="mixed")
    if (frame.start.isna().any() or frame.end.isna().any()
            or (frame.end <= frame.start).any()
            or not np.allclose((bounds[:, 1] - bounds[:, 0]) / fs, frame.duration_seconds,
                               rtol=0, atol=1e-8)):
        raise ValueError(f"Invalid saved bout timestamps or durations: {path}")
    return frame


def load_saved_results(run, subject="900001", visit="T0", sensor="McRoberts"):
    """Load exact saved label intervals and metadata, without reading raw signals.

    Keep the respiration and duration-rule reasons separate. Their disjoint union
    with final nonwear must reproduce the original accelerometer nonwear exactly.
    """
    run = Path(run).expanduser().resolve()
    if (run / "assessment.json").is_file():
        return _load_assessment_results(run, subject, visit, sensor)
    manifest = json.loads((run / "duration_filter_manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError("Choose a completed duration-filter run.")
    matches = [item for item in manifest["recordings"] if
               (item["subject"], item["visit"], item["sensor"]) == (str(subject), str(visit), sensor)]
    if len(matches) != 1:
        raise ValueError(f"Expected one saved result for {subject}/{visit}/{sensor}; found {len(matches)}.")
    item = matches[0]
    reference = item["reference"]
    relative = Path(str(subject)) / str(visit) / sensor / "nonwear"
    folder = run / relative
    original = Path(reference["run"]) / relative
    marker = _verified(original / "respiration_manifest.json", reference["completion_sha256"])
    source_hashes = json.loads(marker.read_text())["outputs"]
    summary = json.loads(_verified(original / "nonwear_summary.json",
                                   source_hashes["nonwear_summary.json"]).read_text())
    final_summary = json.loads(_verified(folder / "nonwear_summary.json",
        manifest["output_sha256"].get(str(relative / "nonwear_summary.json"))).read_text())
    n, fs = int(summary["n_samples"]), float(summary["sample_rate_hz"])
    if n < 1 or not np.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid saved recording length or sample rate.")
    if n != final_summary["n_samples"] or fs != final_summary["sample_rate_hz"]:
        raise ValueError("Source and final recording metadata disagree.")
    tables = {}
    for name, filename in (("baseline", "accelerometer_nonwear_intervals.csv"),
                           ("respiration", "respiration_reassigned_intervals.csv")):
        tables[name] = _read_bouts(original / filename, source_hashes.get(filename), n, fs)
    for name, filename in (("final_nonwear", "final_nonwear_intervals.csv"),
                           ("short_bouts", "short_nonwear_reassigned_intervals.csv")):
        tables[name] = _read_bouts(folder / filename, manifest["output_sha256"].get(str(relative / filename)), n, fs)
    return _inspection_metadata(run,subject,visit,sensor,tables,summary,final_summary,
                                Path(reference["source"]).expanduser(),reference["input_identity"],
                                float(manifest["min_nonwear_bout_seconds"]))


def _load_assessment_results(run, subject, visit, sensor):
    """Native production assessments already contain all reassignment reasons."""
    manifest=json.loads((run/'assessment.json').read_text())
    if manifest.get('status') not in ('completed','completed_with_failures'):
        raise ValueError('Choose a finished assessment.')
    matches=[item for item in manifest['recordings'] if
             (item['subject'],item['visit'],item['sensor'])==(str(subject),str(visit),sensor)]
    if len(matches)!=1 or matches[0].get('status')!='completed':
        raise ValueError(f'No completed result for {subject}/{visit}/{sensor}.')
    item=matches[0]
    folder=run/str(subject)/str(visit)/sensor/'nonwear'
    hashes=json.loads((folder/'respiration_manifest.json').read_text())['outputs']
    summary=json.loads(_verified(folder/'nonwear_summary.json',hashes.get('nonwear_summary.json')).read_text())
    n,fs=int(summary['n_samples']),float(summary['sample_rate_hz'])
    tables={name:_read_bouts(folder/filename,hashes.get(filename),n,fs) for name,filename in (
        ('baseline','accelerometer_nonwear_intervals.csv'),('respiration','respiration_reassigned_intervals.csv'),
        ('final_nonwear','final_nonwear_intervals.csv'),('short_bouts','short_nonwear_reassigned_intervals.csv'))}
    return _inspection_metadata(run,subject,visit,sensor,tables,summary,summary,
                                Path(item['source']).expanduser(),item['identity'],
                                float(summary.get('min_nonwear_bout_seconds',0)))


def _inspection_metadata(run,subject,visit,sensor,tables,summary,final_summary,source,identity,cutoff):
    n,fs=int(summary['n_samples']),float(summary['sample_rate_hz'])
    if n < 1 or not np.isfinite(fs) or fs <= 0:
        raise ValueError('Invalid saved recording length or sample rate.')
    pieces = np.concatenate([_bounds(tables[name], n) for name in SHADING])
    pieces = pieces[np.argsort(pieces[:, 0], kind="stable")]
    if (np.any(pieces[1:, 0] < pieces[:-1, 1]) or
            not np.array_equal(_merge(pieces), _merge(_bounds(tables["baseline"], n)))):
        raise ValueError("Final/respiration/short-bout labels do not partition baseline nonwear.")
    if ((tables["final_nonwear"].duration_seconds < cutoff).any()
            or (tables["short_bouts"].duration_seconds >= cutoff).any()):
        raise ValueError("Saved bouts disagree with the duration cutoff.")
    if not np.isclose(tables["final_nonwear"].duration_seconds.sum() / 3600,
                      final_summary["revised_nonwear_hours"], rtol=0, atol=1e-8):
        raise ValueError("Final bouts and subject summary disagree.")
    policy = summary.get("timestamp_policy", summary.get("provenance", {}).get("timestamps", {}).get("policy", "strict"))
    TimestampOptions(policy=policy).validate()
    review = pd.concat([frame.assign(category=name, bout_index=np.arange(len(frame)))
                        for name, frame in tables.items() if name != "baseline"], ignore_index=True)
    review = review.sort_values(["start", "category"]).reset_index(drop=True)
    return {"run": run, "subject": str(subject), "visit": str(visit), "sensor": sensor,
            "source": source, "input_identity": identity,
            "sample_rate_hz": fs, "n_samples": n, "timestamp_policy": policy,
            "min_nonwear_bout_seconds": cutoff, "source_summary": summary,
            "baseline_window_seconds": summary.get("baseline_detector", {}).get("window_seconds", 3600.),
            "end_of_recording_correction": summary.get("baseline_detector", {}).get("end_of_recording_correction",
                                                                                   summary.get("baseline_detector", {}).get("edge_correction", True)),
            "summary": final_summary, "intervals": tables,
            "review_bouts": review[["category", "bout_index", "start", "end", "duration_seconds"]]}


def load_saved_raw(inspection):
    """Decode raw ACC/gyro with the saved timestamp policy and verify alignment.

    No calibration or detection is rerun. The initial full-file OMX decode can
    require substantial RAM; the notebook caches it when changing the plot view.
    """
    source, expected = inspection["source"], inspection["input_identity"]
    if signature({}, [source]) != expected:
        raise ValueError("The raw OMX changed since the saved assessment.")
    raw = read_dp7_as_dataframe_fast(source, verbose=0,
                                   timestamps=TimestampOptions(policy=inspection["timestamp_policy"]))
    fs, n = inspection["sample_rate_hz"], inspection["n_samples"]
    if not np.isclose(raw.attrs.get("sample_rate_hz", np.nan), fs) or len(raw) < n:
        raise ValueError("Raw recording sample rate or length differs from the saved assessment.")
    raw = raw.iloc[:n].set_index("datetime")
    raw.index = pd.DatetimeIndex(raw.index).as_unit("ns")
    times = raw.index
    summary = inspection["source_summary"]
    if times[0] != pd.Timestamp(summary["start"]) or times[-1] != pd.Timestamp(summary["last_sample"]):
        raise ValueError("Raw and saved recording boundaries do not align.")
    period = pd.Timedelta(seconds=1 / fs)
    for table in inspection["intervals"].values():
        bounds = _bounds(table, n)
        if (not np.array_equal(times.take(bounds[:, 0]).to_numpy(), table.start.to_numpy())
                or not np.array_equal((times.take(bounds[:, 1] - 1) + period).to_numpy(), table.end.to_numpy())):
            raise ValueError("Raw samples and saved nonwear bout boundaries do not align.")
    if signature({}, [source]) != expected:
        raise ValueError("Raw input changed during inspection loading.")
    return raw


def plot_saved_raw(raw, inspection, *, start=None, end=None):
    """Full-rate acceleration above gyro, with three disjoint saved label colors.

    Plot limits crop display only. Both None means the full analysed recording;
    no samples are downsampled. Lines are broken across missing recording time.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fs = inspection["sample_rate_hz"]
    left = pd.Timestamp(start) if start is not None else raw.index[0]
    right = pd.Timestamp(end) if end is not None else raw.index[-1] + pd.Timedelta(seconds=1 / fs)
    if left >= right:
        raise ValueError("Plot start must precede plot end.")
    first, stop = raw.index.searchsorted([left, right])
    view = raw.iloc[first:stop]
    if view.empty:
        raise ValueError("Plot interval contains no observed samples.")
    segments = _segment_bounds(view.index, fs)
    # Retain the neighbouring sample on each side so a view starting/ending
    # inside a gap still shades the complete visible missing-time portion.
    context = raw.index[max(0, first-1):min(len(raw), stop+1)]
    context_segments = _segment_bounds(context, fs)
    fig, axes = plt.subplots(2, 1, figsize=(17, 8), sharex=True, layout="constrained")
    colors = ("#c75489", "#008f7a", "#4776ad")
    for ax, names, ylabel, title in zip(axes, (("ax", "ay", "az"), ("gx", "gy", "gz")),
                                       ("Acceleration (g)", "Angular velocity (°/s)"),
                                       ("Raw acceleration", "Raw gyroscope")):
        for i, (a, b) in enumerate(segments):
            segment = view.iloc[a:b]
            for name, color in zip(names, colors):
                ax.plot(segment.index, segment[name], lw=.65, color=color, label=name if i == 0 else None)
        for name, (label, color) in SHADING.items():
            bouts = inspection["intervals"][name]
            visible = bouts.loc[(bouts.end > left) & (bouts.start < right)]
            for bout in visible.itertuples():
                ax.axvspan(max(left, bout.start), min(right, bout.end), color=color, alpha=.2, lw=0)
        for (_, a), (b, _) in zip(context_segments[:-1], context_segments[1:]):
            gap_start = max(left, context[a-1] + pd.Timedelta(seconds=1/fs))
            gap_end = min(right, context[b])
            if gap_end > gap_start:
                ax.axvspan(gap_start, gap_end, facecolor="#999999", alpha=.2, hatch="//", lw=0)
        ax.set(ylabel=ylabel, title=f"{title} · {fs:g} Hz", xlim=(left, right))
        ax.legend(loc="upper left", ncol=3)
    shade_handles = [Patch(facecolor=color, alpha=.3, label=label if name != "short_bouts" else
                           f"Wear: nonwear <{inspection['min_nonwear_bout_seconds']/60:g} min")
                     for name, (label, color) in SHADING.items()]
    shade_handles.append(Patch(facecolor="#999999", alpha=.2, hatch="//", label="Missing/time-invalid"))
    # Keep gyro axis colors identifiable alongside the interval legend.
    axes[-1].add_artist(axes[-1].get_legend())
    axes[-1].legend(handles=shade_handles, loc="upper right", ncol=2)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M:%S"))
    axes[-1].set_xlabel("Recording-local time")
    window=inspection.get('baseline_window_seconds',3600)/60
    end_correction=inspection.get('end_of_recording_correction',True)
    fig.suptitle(f"{inspection['subject']} / {inspection['visit']} — saved {window:g}-min / 15-min nonwear results · end extension={end_correction}")
    return fig

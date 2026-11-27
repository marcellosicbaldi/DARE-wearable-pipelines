"""Frozen, descriptive nonwear benchmark on completely reviewed recordings.

Standalone research runner: never changes production defaults or saved labels.
Prepare the immutable plan first, then run all declared candidates on all subjects.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import html
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dare_wearables.common.autocalibrate import autocalibrate
from dare_wearables.common.output_state import _digest, signature
from dare_wearables.lower_back.annotation.data import AnnotationConfig, RawCache, atomic_json
from dare_wearables.lower_back.nonwear import ggir2023
from dare_wearables.lower_back.nonwear.respiration import (
    RespirationSettings, _merge, _subtract, analyze_gyro,
)
from dare_wearables.lower_back.nonwear.vanhees2013 import vanhees2013


@dataclass(frozen=True)
class Candidate:
    id: str
    window_minutes: int
    step_minutes: int
    axes: int = 2
    respiration_seconds: int = 60
    minimum_minutes: int = 15
    postprocessing: bool = True
    method: str = "ggir2023"


def candidate(window, step, axes=2, respiration=60, minimum=15, post=True):
    return Candidate(f"ggir_w{window}_s{step}_a{axes}_r{respiration}_m{minimum}"
                     + ("" if post else "_no_post"),
                     window, step, axes, respiration, minimum, post)


def candidates():
    """Declared before scoring; no selection or tuning during the run."""
    windows = ((60, 15), (30, 15), (30, 1), (15, 1))
    result = [candidate(w, s, a, r) for w, s in windows
              for a in (2, 3) for r in (0, 60)]
    result += [candidate(w, s, respiration=30) for w, s in windows]
    result += [candidate(w, 1, minimum=m) for w in (30, 15) for m in (0, 5)]
    result += [candidate(w, 1, post=False) for w in (30, 15)]
    result += [Candidate("legacy2013_original", 60, 15, 2, 0, 0, True, "legacy2013")]
    assert len(result) == 27 and len({c.id for c in result}) == 27
    return result


BASELINE = candidate(60, 15).id


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def source_hashes():
    paths = [Path(__file__), ROOT / "src/dare_wearables/common/autocalibrate.py",
             Path(ggir2023.__file__), ROOT / "src/dare_wearables/lower_back/nonwear/respiration.py",
             ROOT / "src/dare_wearables/lower_back/nonwear/vanhees2013.py"]
    return {str(p.relative_to(ROOT)): _digest(p) for p in paths}


def prepare(config_path, output_root):
    config = AnnotationConfig.from_toml(config_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(output_root).resolve() / ("benchmark_" + stamp)
    run.mkdir(parents=True, exist_ok=False)
    caches = {}
    for p in config.cache_root.glob("*/metadata.json"):
        meta = json.loads(p.read_text())
        caches[meta["input_identity"]] = p.parent
    items = []
    for p in sorted(config.output_root.glob("*/T0/annotations.json")):
        session = json.loads(p.read_text())
        reviewed = [r for r in session["rows"] if r["status"] == "reviewed"]
        if not reviewed:
            continue
        if any(r["status"] == "pending" for r in session["rows"]):
            raise ValueError(f"Unfinished candidate annotations in {p}")
        cache_folder = caches[session["raw_input_identity"]]
        cache = RawCache(cache_folder)
        if signature({}, [session["raw_source"]]) != session["raw_input_identity"]:
            raise ValueError(f"Raw input identity changed for {session['subject']}")
        if any(not 0 <= r["start_sample"] < r["stop_sample"] <= len(cache.times) for r in reviewed):
            raise ValueError("Annotation sample bounds outside cache")
        folder = run / session["subject"] / session["visit"]
        folder.mkdir(parents=True)
        for name in ("annotations.json", "reviewed_intervals.csv"):
            shutil.copyfile(p.with_name(name), folder / name)
        items.append({"subject": session["subject"], "visit": session["visit"],
                      "cache": str(cache_folder), "n_samples": len(cache.times),
                      "fs": cache.fs, "raw_input_identity": session["raw_input_identity"],
                      "cache_metadata_sha256": _digest(cache_folder / "metadata.json"),
                      "labels_sha256": _digest(folder / "annotations.json"),
                      "export_sha256": _digest(folder / "reviewed_intervals.csv"),
                      "cache_files": {name: {"size": (cache_folder / name).stat().st_size,
                                               "mtime_ns": (cache_folder / name).stat().st_mtime_ns}
                                      for name in ("signals.npy", "timestamps.npy")},
                      "reference_annotations": len(reviewed)})
    if len(items) != 20 or sum(i["reference_annotations"] for i in items) != 62:
        raise ValueError("Expected the agreed 20 participants and 62 reviewed annotations")
    plan = {"schema": 1, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidates": [asdict(c) for c in candidates()], "baseline": BASELINE,
            "participants": items, "source_sha256": source_hashes(),
            "evaluation": {"scope": "all 20 participants; descriptive; no held-out subjects or tuning",
                "wear_reference": "User confirmed complete recordings were reviewed. All observed time outside confirmed nonwear is wear.",
                "signal_units": "calibrated acceleration in g; raw gyroscope in degrees/s",
                "acceleration_sd_mg": 13, "acceleration_range_mg": 50,
                "statistics_hz": 5, "respiration": clean(asdict(RespirationSettings())),
                "respiration_overlap": 0.5, "end_of_recording_extension": False,
                "postprocessing": "GGIR generic gap filling and start correction; physical-hour thresholds preserved at each step size",
                "legacy": "Original 2013 implementation including original edge rules, applied separately to continuous segments",
                "missing_data": "No synthetic signal. Detectors restart at cache segment boundaries. Missing time excluded from accuracy.",
                "partial_epochs": "Remain wear as in production; unassessed tails reported for each acceleration grid",
                "episode_reference": "One logical episode per reviewed annotation ID, even if exported as several observed pieces",
                "episode_prediction": "Adjacent retained-sample intervals form one episode across missing time, but never across observed wear",
                "episode_matching": "One-to-one maximum-cardinality matching at observed-sample IoU >=0.5, then maximize IoU",
                "boundaries": "Signed and absolute elapsed-time start/end errors on matched episodes only",
                "night": "22:00-08:00 recorded local clock", "uncertainty": "Participant bootstrap, fixed seed, 2000 resamples",
                "gap_spanning_reference": "Keep each annotation as one episode across internal gaps. Export timestamp-only interpolation ledgers; synthetic rows are never scored."}}
    atomic_json(run / "plan.json", clean(plan))
    (run / "plan.sha256").write_text(_digest(run / "plan.json") + "\n")
    pd.DataFrame(plan["candidates"]).to_csv(run / "candidates.csv", index=False)
    atomic_json(run / "status.json", {"status": "prepared", "completed": [], "failures": {}})
    print(f"FROZEN {run} ({len(items)} participants, {len(candidates())} candidates)", flush=True)
    return run


def fast_scores(axes, fs, window_minutes, step_minutes):
    """GGIR forward axis-flag propagation with reusable one-minute summaries.

    Extrema are exact on the same 5 Hz subsample as production. Centered prefix
    moments accelerate variance; threshold-adjacent values use the original R-
    compatible SD calculation. Every 15-minute grid is checked against production.
    """
    medium = int(step_minutes * 60 * fs)
    count = len(axes[0]) // medium
    if not count:
        return np.empty(0, dtype=int)
    width = window_minutes // step_minutes
    flags = np.zeros((count, 3), dtype=bool)
    stride = int(np.ceil(fs / 5))
    block = int(60 * fs) // stride
    for col, axis in enumerate(axes):
        x = axis[::stride]
        offsets = np.arange(0, len(x), block)
        mins = np.minimum.reduceat(x, offsets)
        maxs = np.maximum.reduceat(x, offsets)
        lo = np.lib.stride_tricks.sliding_window_view(
            np.pad(mins, (0, window_minutes - 1), constant_values=np.inf), window_minutes).min(axis=1)
        hi = np.lib.stride_tricks.sliding_window_view(
            np.pad(maxs, (0, window_minutes - 1), constant_values=-np.inf), window_minutes).max(axis=1)
        minute = np.arange(count) * step_minutes
        left = minute * block
        right = np.minimum(left + window_minutes * block, len(x))
        centered = x - x.mean()
        sums = np.r_[0., np.cumsum(centered, dtype=np.float64)]
        squares = np.r_[0., np.cumsum(centered * centered, dtype=np.float64)]
        n = right - left
        total = sums[right] - sums[left]
        variance = np.maximum(0., ((squares[right] - squares[left]) - total * total / n) / (n - 1))
        std = np.sqrt(variance)
        for i in np.flatnonzero(np.abs(std - .013) < 1e-7):
            std[i] = ggir2023._sample_sd(x[left[i]:right[i]])
        passing = (hi[minute] - lo[minute] < .050) & (std < .013)
        starts = np.flatnonzero(passing)
        changes = np.zeros(count + 1, dtype=int)
        np.add.at(changes, starts, 1)
        np.add.at(changes, np.minimum(starts + width, count), -1)
        flags[:, col] = np.cumsum(changes[:-1]) > 0
    scores = flags.sum(axis=1)
    for i in np.flatnonzero(scores == 1):
        if 0 < i < count - 1 and scores[i - 1] > 1 and scores[i + 1] > 1:
            scores[i] = 2
    return scores


def decisions(scores, axes_required, step_minutes, postprocessing=True):
    initial = np.asarray(scores) >= axes_required
    if not postprocessing:
        return initial
    added = np.zeros(len(initial), dtype=bool)
    three = 180 / step_minutes
    six = 360 / step_minutes
    for _ in range(3):
        signal = np.r_[False, initial | added, False]
        padded = np.r_[False, added, False]
        starts = np.flatnonzero(np.diff(signal.astype(np.int8)) == 1) + 1
        stops = np.flatnonzero(np.diff(signal.astype(np.int8)) == -1) + 1
        for i in range(len(starts) - 1):
            gap = starts[i + 1] - stops[i]
            ratio = gap / (stops[i] - starts[i] + stops[i + 1] - starts[i + 1])
            if (gap < six and ratio < .3) or (gap < three and ratio < .8):
                padded[stops[i]:starts[i + 1]] = True
        if len(starts) and 1 < starts[0] + 1 < three:
            padded[:starts[0]] = True
        added = padded[1:-1]
    return initial | added


def mask_bouts(mask, medium, offset=0):
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return np.column_stack((np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))) * medium + offset


def split_bounds(bounds, segments):
    pieces = []
    for a, b in bounds:
        for first, stop in segments:
            if stop > a and first < b:
                pieces.append((max(int(a), int(first)), min(int(b), int(stop))))
    return np.asarray(pieces, dtype=np.int64).reshape(-1, 2)


def length(bounds):
    bounds = np.asarray(bounds, dtype=np.int64).reshape(-1, 2)
    return int(np.sum(bounds[:, 1] - bounds[:, 0]))


def intersection(a, b):
    a, b = np.asarray(a).reshape(-1, 2), np.asarray(b).reshape(-1, 2)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i, 1], b[j, 1]) - max(a[i, 0], b[j, 0]))
        if a[i, 1] <= b[j, 1]:
            i += 1
        else:
            j += 1
    return int(total)


def reference(session, cache, folder=None):
    rows = [r for r in session["rows"] if r["status"] == "reviewed" and r["label"] == "nonwear"]
    rows = sorted(rows, key=lambda r: r["start_sample"])
    bounds = np.asarray([(r["start_sample"], r["stop_sample"]) for r in rows], dtype=np.int64).reshape(-1, 2)
    if len(bounds) and np.any(bounds[1:, 0] < bounds[:-1, 1]):
        raise ValueError("Overlapping reference episodes")
    if folder is not None:
        exported = []
        for r in rows:
            a, b = r["start_sample"], r["stop_sample"]
            exported.append({"annotation_id": r["id"], "start_sample": a, "stop_sample": b,
                "start": pd.Timestamp(cache.boundary_ns(a)).isoformat(),
                "end": pd.Timestamp(cache.boundary_ns(b, "end")).isoformat(),
                "observed_seconds": (b-a)/cache.fs,
                "elapsed_seconds": (cache.boundary_ns(b, "end")-cache.boundary_ns(a))/1e9,
                "observed_parts": len(split_bounds([[a, b]], cache.meta["segments"]))})
        pd.DataFrame(exported).to_csv(folder / "reference_episodes.csv", index=False)
        gaps = []
        for target in rows:
            for (_, stop), (first, _) in zip(cache.meta["segments"][:-1], cache.meta["segments"][1:]):
                if target["start_sample"] < first < target["stop_sample"]:
                    left, right = cache.boundary_ns(stop, "end"), cache.boundary_ns(first)
                    count = round((right-left)/1e9*cache.fs)
                    for t in left + np.arange(count, dtype=np.int64) * (right-left) // count:
                        gaps.append({"annotation_id": target["id"], "timestamp": pd.Timestamp(t).isoformat(),
                                     "reference_label": "nonwear", "observed": False, "scorable": False})
        pd.DataFrame(gaps, columns=["annotation_id", "timestamp", "reference_label", "observed", "scorable"]).to_csv(
            folder / "reference_gap_timestamps.csv", index=False)
    return rows, bounds


def periods(times):
    day, night = [], []
    first, last = pd.Timestamp(times[0]), pd.Timestamp(times[-1])
    for date in pd.date_range(first.normalize(), last.normalize(), freq="D"):
        for a, b, target in ((date, date+pd.Timedelta(hours=8), night),
                              (date+pd.Timedelta(hours=8), date+pd.Timedelta(hours=22), day),
                              (date+pd.Timedelta(hours=22), date+pd.Timedelta(days=1), night)):
            left, right = np.searchsorted(times, [a.value, b.value])
            if right > left:
                target.append((left, right))
    return {"day": _merge(day), "night": _merge(night), "all": np.array([[0, len(times)]])}


def confusion(predicted, truth, scope, fs):
    ref = intersection(truth, scope)
    pred = intersection(predicted, scope)
    # Interval intersection as actual spans, rather than an epoch majority label.
    shared = []
    for a, b in predicted:
        for c, d in truth:
            if d > a and c < b:
                shared.append((max(a, c), min(b, d)))
    tp = intersection(shared, scope)
    return {"tp_seconds": tp/fs, "fp_seconds": (pred-tp)/fs,
            "fn_seconds": (ref-tp)/fs, "tn_seconds": (length(scope)-pred-ref+tp)/fs}


def rates(row):
    tp, fp, fn, tn = (row[k+"_seconds"] for k in ("tp", "fp", "fn", "tn"))
    divide = lambda a, b: a/b if b else np.nan
    precision, recall = divide(tp, tp+fp), divide(tp, tp+fn)
    specificity = divide(tn, tn+fp)
    return {"precision": precision, "recall": recall, "f1": divide(2*tp, 2*tp+fp+fn),
            "specificity": specificity, "accuracy": divide(tp+tn, tp+fp+fn+tn),
            "balanced_accuracy": (recall+specificity)/2,
            "mcc": divide(tp*tn-fp*fn, np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))),
            "false_nonwear_minutes_per_day": divide(fp/60, (tp+fp+fn+tn)/86400),
            "reference_nonwear_hours": (tp+fn)/3600, "predicted_nonwear_hours": (tp+fp)/3600,
            "nonwear_bias_minutes": (fp-fn)/60}


def event_metrics(predicted, truth, rows, cache):
    predicted = _merge(predicted)
    ious = np.zeros((len(truth), len(predicted)))
    overlaps = np.zeros_like(ious)
    for i, ref in enumerate(truth):
        for j, pred in enumerate(predicted):
            shared = max(0, min(ref[1], pred[1])-max(ref[0], pred[0]))
            overlaps[i, j] = shared
            ious[i, j] = shared / ((ref[1]-ref[0])+(pred[1]-pred[0])-shared)
    matches = {}
    if ious.size:
        a, b = linear_sum_assignment(-((ious >= .5)*1000 + ious))
        matches = {int(i): int(j) for i, j in zip(a, b) if ious[i, j] >= .5}
    details = []
    for i, (a, b) in enumerate(truth):
        j = matches.get(i)
        detail = {"annotation_id": rows[i]["id"], "reference_duration_minutes": (b-a)/cache.fs/60,
                  "covered_fraction": overlaps[i].sum()/(b-a),
                  "any_overlap": bool(overlaps[i].sum()), "matched_iou50": j is not None,
                  "overlapping_predicted_fragments": int(np.count_nonzero(overlaps[i]))}
        if j is not None:
            p, q = predicted[j]
            detail.update(iou=ious[i, j],
                start_error_seconds=(cache.boundary_ns(int(p))-cache.boundary_ns(int(a)))/1e9,
                end_error_seconds=(cache.boundary_ns(int(q), "end")-cache.boundary_ns(int(b), "end"))/1e9)
        details.append(detail)
    return {"reference_events": len(truth), "predicted_events": len(predicted), "matched_events_iou50": len(matches),
            "any_overlap_events": int(np.count_nonzero(overlaps.sum(axis=1))),
            "half_covered_events": int(np.count_nonzero(overlaps.sum(axis=1) >= (truth[:, 1]-truth[:, 0])*.5))}, details


def frame_bounds(bounds, cache):
    return pd.DataFrame([{"start_sample": int(a), "stop_sample": int(b),
        "start": pd.Timestamp(cache.boundary_ns(int(a))).isoformat(),
        "end": pd.Timestamp(cache.boundary_ns(int(b), "end")).isoformat(),
        "observed_seconds": (b-a)/cache.fs} for a, b in bounds],
        columns=["start_sample", "stop_sample", "start", "end", "observed_seconds"])


def process_subject(item, run, declared):
    started = time.monotonic()
    folder = run / item["subject"] / item["visit"]
    cache = RawCache(item["cache"])
    if _digest(cache.folder / "metadata.json") != item["cache_metadata_sha256"]:
        raise ValueError("Cached metadata changed since plan freeze")
    for name, stat in item["cache_files"].items():
        actual = (cache.folder / name).stat()
        if actual.st_size != stat["size"] or actual.st_mtime_ns != stat["mtime_ns"]:
            raise ValueError("Raw cache changed since plan freeze")
    if _digest(folder / "annotations.json") != item["labels_sha256"]:
        raise ValueError("Frozen annotations changed")
    if _digest(folder / "reviewed_intervals.csv") != item["export_sha256"]:
        raise ValueError("Frozen observed exports changed")
    session = json.loads((folder / "annotations.json").read_text())
    rows, truth = reference(session, cache, folder)
    fs, n = cache.fs, len(cache.times)
    segments = np.asarray(cache.meta["segments"], dtype=int)
    scope = periods(cache.times)
    print(f"{item['subject']}: calibration ({n/fs/3600:.1f} observed hours)", flush=True)
    raw = pl.DataFrame({"ts": cache.times.view("datetime64[ns]"),
                        **{name: cache.values[:, i] for i, name in enumerate("xyz")}})
    calibration = autocalibrate(raw, apply=False)
    params = calibration["params"]
    atomic_json(folder / "calibration.json", clean(asdict(params)))
    axes = [np.asarray(cache.values[:, i]).copy() for i in range(3)]
    if params.valid:
        for i in range(3):
            axes[i] += params.offset[i]
            axes[i] *= params.scale[i]
    del raw, calibration
    gc.collect()
    acceleration = {}
    parity = []
    for window, step in ((60, 15), (30, 15), (30, 1), (15, 1)):
        scores_by_segment = []
        for first, stop in segments:
            part = [x[first:stop] for x in axes]
            scores = fast_scores(part, fs, window, step)
            scores_by_segment.append(scores)
            if step == 15:
                original, clipping = ggir2023._epoch_scores(part, fs, int(900*fs), len(scores), window_epochs=window//15)
                if not np.array_equal(scores, original):
                    raise AssertionError(f"Production score mismatch {item['subject']} {window}/{step}")
                actual = decisions(scores, 2, 15)
                expected = ggir2023._wear_decisions(original, clipping, end_of_recording_correction=False)[2]
                if not np.array_equal(actual, expected):
                    raise AssertionError("Production postprocessing mismatch")
        for required in (2, 3):
            for post in (True, False):
                pieces = [mask_bouts(decisions(score, required, step, post), int(step*60*fs), int(first))
                          for score, (first, stop) in zip(scores_by_segment, segments)]
                acceleration[(window, step, required, post)] = np.concatenate(pieces)
        parity.append({"window_minutes": window, "step_minutes": step,
                       "production_parity": "exact scores and final decisions" if step == 15 else "experimental; synthetic parity checks",
                       "unassessed_tail_seconds": sum((b-a) % int(step*60*fs) for a, b in segments)/fs})
    print(f"{item['subject']}: legacy reference", flush=True)
    legacy = []
    for first, stop in segments:
        frame = pl.DataFrame({"ts": cache.times[first:stop].view("datetime64[ns]"),
                              **{name: axes[i][first:stop] for i, name in enumerate("xyz")}})
        result = vanhees2013(frame, freq=fs, quiet=True, return_debug=True)
        flags = result.window_df["is_final_nonwear"].to_numpy() if len(result.window_df) else np.empty(0, dtype=bool)
        legacy.append(mask_bouts(flags, int(900*fs), int(first)))
    acceleration["legacy2013"] = np.concatenate(legacy)
    del axes, frame, result
    gc.collect()
    evidence = {0: np.empty((0, 2), dtype=np.int64)}
    gyro = pd.DataFrame({name: cache.values[:, i+3] for i, name in enumerate(("gx", "gy", "gz"))},
                        index=pd.DatetimeIndex(cache.times.view("datetime64[ns]")), copy=False)
    for window in (30, 60):
        print(f"{item['subject']}: gyro {window}-second windows", flush=True)
        result = analyze_gyro(gyro, fs, window_seconds=window, overlap=.5, settings=RespirationSettings(),
                              store_psd=False, segment_bounds=segments)
        result.windows.to_parquet(folder / f"respiration_{window}s.parquet", index=False)
        selected = result.windows.loc[result.windows.respiration_present, ["start_sample", "stop_sample"]].to_numpy(dtype=np.int64)
        evidence[window] = _merge(selected)
        del result
    del gyro
    gc.collect()
    metrics, events, predictions = [], [], []
    for c in declared:
        base = acceleration["legacy2013"] if c.method == "legacy2013" else acceleration[(c.window_minutes, c.step_minutes, c.axes, c.postprocessing)]
        remaining = _subtract(base, evidence[c.respiration_seconds])
        remaining = split_bounds(remaining, segments)
        if len(remaining):
            remaining = remaining[(remaining[:, 1]-remaining[:, 0])/fs >= c.minimum_minutes*60]
        # Score all actual samples, retaining nanosecond-aligned manual boundaries.
        merged = _merge(remaining)
        event_summary, details = event_metrics(merged, truth, rows, cache)
        for period, support in scope.items():
            stats = confusion(merged, truth, support, fs)
            metrics.append({"subject": item["subject"], "visit": item["visit"], "candidate": c.id,
                            "period": period, "observed_hours": length(support)/fs/3600,
                            **stats, **rates(stats), **(event_summary if period == "all" else {})})
        events.extend({"subject": item["subject"], "candidate": c.id, **d} for d in details)
        table = frame_bounds(remaining, cache)
        table.insert(0, "candidate", c.id)
        predictions.append(table)
    pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
    pd.DataFrame(events).to_csv(folder / "events.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(folder / "predictions.csv", index=False)
    atomic_json(folder / "verification.json", {"production_parity": parity,
        "reference_annotations": len(rows), "reference_observed_hours": length(truth)/fs/3600,
        "observed_hours": n/fs/3600, "missing_clock_hours": (cache.end_ns-cache.first_ns)/1e9/3600-n/fs/3600,
        "elapsed_seconds": time.monotonic()-started,
        "outputs_sha256": {name: _digest(folder/name) for name in
            ("metrics.csv", "events.csv", "predictions.csv", "calibration.json", "reference_episodes.csv",
             "respiration_30s.parquet", "respiration_60s.parquet")}})
    print(f"{item['subject']}: completed in {time.monotonic()-started:.1f}s", flush=True)


def aggregate(run, plan):
    metrics, events = [], []
    for item in plan["participants"]:
        folder = run / item["subject"] / item["visit"]
        if not (folder / "verification.json").is_file():
            continue
        verification = json.loads((folder / "verification.json").read_text())
        for name, expected in verification["outputs_sha256"].items():
            if _digest(folder/name) != expected:
                raise ValueError(f"Changed completed output: {folder/name}")
        metrics.append(pd.read_csv(folder / "metrics.csv", dtype={"subject": str}))
        events.append(pd.read_csv(folder / "events.csv", dtype={"subject": str}))
    data, episodes = pd.concat(metrics, ignore_index=True), pd.concat(events, ignore_index=True)
    data.to_csv(run / "participant_metrics.csv", index=False)
    episodes.to_csv(run / "episode_metrics.csv", index=False)
    rng = np.random.default_rng(20261126)
    bootstrap_ids = rng.integers(0, len(metrics), (2000, len(metrics)))
    summary = []
    for (name, period), frame in data.groupby(["candidate", "period"], sort=False):
        pooled = frame[[k+"_seconds" for k in ("tp", "fp", "fn", "tn")]].sum().to_dict()
        row = {"candidate": name, "period": period, "participants": len(frame),
               "observed_hours": frame.observed_hours.sum(), **pooled, **rates(pooled)}
        for key in ("precision", "recall", "f1", "balanced_accuracy", "false_nonwear_minutes_per_day"):
            values = frame.sort_values("subject")[key].to_numpy()
            estimates = np.nanmean(values[bootstrap_ids], axis=1)
            row["macro_"+key] = np.nanmean(values)
            row["macro_"+key+"_ci_low"], row["macro_"+key+"_ci_high"] = np.nanquantile(estimates, [.025, .975])
        if period == "all":
            for key in ("reference_events", "predicted_events", "matched_events_iou50", "any_overlap_events", "half_covered_events"):
                row[key] = int(frame[key].sum())
            row["event_recall_iou50"] = row["matched_events_iou50"]/row["reference_events"]
            row["event_precision_iou50"] = row["matched_events_iou50"]/row["predicted_events"] if row["predicted_events"] else np.nan
            detail = episodes.loc[(episodes.candidate == name) & episodes.matched_iou50]
            for key in ("start", "end"):
                value = detail[key+"_error_seconds"]
                row[key+"_median_absolute_error_minutes"] = value.abs().median()/60
                row[key+"_median_signed_error_minutes"] = value.median()/60
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(run / "summary.csv", index=False)
    episodes["duration_group"] = pd.cut(episodes.reference_duration_minutes, [0, 15, 30, 60, np.inf],
                                         right=False, labels=["<15 min", "15-30 min", "30-60 min", ">=60 min"])
    strata = episodes.groupby(["candidate", "duration_group"], observed=False).agg(
        episodes=("annotation_id", "count"), mean_coverage=("covered_fraction", "mean"),
        any_overlap_recall=("any_overlap", "mean"), iou50_recall=("matched_iou50", "mean"))
    strata.to_csv(run / "duration_strata.csv")
    report(run, plan, data, episodes, summary)
    return summary


def report(run, plan, data, episodes, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    all_results = summary.loc[summary.period == "all"].sort_values("macro_f1", ascending=False)
    names = all_results.candidate.tolist()
    fig, ax = plt.subplots(figsize=(11, 8))
    for _, row in all_results.iterrows():
        ax.scatter(row["recall"], row["precision"], s=65 if row.candidate == BASELINE else 32,
                   marker="*" if row.candidate == BASELINE else "o")
        ax.annotate(str(names.index(row.candidate)+1), (row["recall"], row["precision"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Nonwear recall (observed time)", ylabel="Nonwear precision (observed time)",
           title="All 20 reviewed participants — fixed configurations", xlim=(0, 1.02), ylim=(0, 1.02))
    ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(run / "precision_recall.png", dpi=160); plt.close(fig)
    chosen = list(dict.fromkeys([BASELINE] + names[:5]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(axes, ("f1", "false_nonwear_minutes_per_day"), ("Participant nonwear F1", "False nonwear minutes / observed day")):
        values = [data.loc[(data.period == "all") & (data.candidate == c), metric].to_numpy() for c in chosen]
        ax.boxplot(values, labels=[str(names.index(c)+1) for c in chosen], showfliers=False)
        for i, v in enumerate(values):
            ax.scatter(np.full(len(v), i+1), v, s=12, alpha=.5)
        ax.set(title=title, xlabel="Configuration number in table"); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(run / "participant_comparison.png", dpi=160); plt.close(fig)
    # Full-week annotation/prediction overview for every participant: no display downsampling of labels.
    show = [BASELINE, names[0], candidate(30, 1).id, candidate(15, 1).id]
    show = list(dict.fromkeys(show))
    fig, axes = plt.subplots(20, 1, figsize=(15, 31))
    for ax, item in zip(axes, plan["participants"]):
        folder = run / item["subject"] / item["visit"]
        if not (folder/"predictions.csv").exists():
            continue
        cache = RawCache(item["cache"])
        session = json.loads((folder/"annotations.json").read_text())
        rows, truth = reference(session, cache)
        pred = pd.read_csv(folder/"predictions.csv")
        for level, (name, bounds) in enumerate([("Reference", truth)] + [(c, pred.loc[pred.candidate == c, ["start_sample", "stop_sample"]].to_numpy()) for c in show]):
            spans = [((cache.times[int(a)]-cache.first_ns)/1e9/3600,
                      (cache.boundary_ns(int(b), "end")-cache.times[int(a)])/1e9/3600) for a, b in bounds]
            ax.broken_barh(spans, (level-.32, .64), facecolors="black" if level == 0 else f"C{level-1}")
        for a, b in cache.meta["gaps"]:
            ax.axvspan((a-cache.first_ns)/1e9/3600, (b-cache.first_ns)/1e9/3600, color="grey", alpha=.3)
        ax.set(xlim=(0, 168), yticks=range(len(show)+1), yticklabels=["Ref"]+[str(names.index(c)+1) for c in show], ylabel=item["subject"])
        ax.grid(axis="x", alpha=.2)
    axes[-1].set_xlabel("Hours since recording start; grey indicates missing time")
    fig.suptitle("Reference and candidate nonwear across all 20 participants", y=.998)
    fig.tight_layout(); fig.savefig(run / "all_participants.png", dpi=130); plt.close(fig)
    table = all_results[["candidate", "precision", "recall", "f1", "macro_f1", "false_nonwear_minutes_per_day",
                         "event_recall_iou50", "start_median_absolute_error_minutes", "end_median_absolute_error_minutes"]].copy()
    table.insert(0, "No.", np.arange(1, len(table)+1))
    table = table.rename(columns={"false_nonwear_minutes_per_day": "False NW min/day", "macro_f1": "Participant mean F1",
                                  "event_recall_iou50": "Event recall IoU≥0.5", "start_median_absolute_error_minutes": "Start error min",
                                  "end_median_absolute_error_minutes": "End error min"})
    baseline = all_results.loc[all_results.candidate == BASELINE].iloc[0]
    best = all_results.iloc[0]
    period = summary.loc[(summary.candidate.isin([BASELINE, best.candidate])) & (summary.period != "all"),
                         ["candidate", "period", "precision", "recall", "f1", "false_nonwear_minutes_per_day"]]
    duration = pd.read_csv(run/"duration_strata.csv")
    duration = duration.loc[duration.candidate.isin([BASELINE, best.candidate])]
    reference_hours = baseline.reference_nonwear_hours
    text = f"""<!doctype html><html><head><meta charset="utf-8"><title>Lower-back nonwear benchmark</title>
    <style>body{{font:16px system-ui;max-width:1500px;margin:40px auto;padding:0 24px;color:#192734}}h1,h2{{color:#123b4a}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{padding:8px;border-bottom:1px solid #dde5e9;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{max-width:100%;height:auto}}code{{font-size:12px}}.note{{background:#edf5f7;padding:18px;border-radius:8px}}</style></head><body>
    <h1>Lower-back nonwear benchmark</h1>
    <p>20 completely reviewed T0 recordings; {int(baseline.reference_events)} reference episodes; {baseline.observed_hours:.2f} observed hours, including {reference_hours:.2f} nonwear hours.
    27 configurations declared before scoring. All configurations evaluated on all participants. No held-out split, model fitting, or tuning after results.</p>
    <p class="note">Highest participant-mean F1: <b>{html.escape(best.candidate)}</b>, {best.macro_f1:.3f}; current production baseline {baseline.macro_f1:.3f}.
    These are descriptive results on the reviewed dataset. The rank uses mean participant F1, with pooled time-level metrics and episode metrics shown separately.</p>
    <h2>Results for every fixed configuration</h2>{table.to_html(index=False, float_format=lambda v: f'{v:.3f}', escape=True)}
    <p>Configuration names: w = ACC window minutes, s = ACC step minutes, a = required axes, r = gyro window seconds (0 = off), m = final minimum nonwear minutes.
    All GGIR variants use SD &lt;13 mg AND range &lt;50 mg. Respiration uses 10–30 brpm (strict endpoints), any positive axis, 50% overlap, RMS ≥0.01°/s, band fraction ≥0.5, concentration ≥0.5, peak/background ≥6 dB.</p>
    <img src="precision_recall.png" alt="Precision and recall"><img src="participant_comparison.png" alt="Participant comparison">
    <h2>Day and night</h2><p>Night is 22:00–08:00 in the stored local clock.</p>{period.to_html(index=False,float_format=lambda v:f'{v:.3f}')}
    <h2>Episode duration</h2>{duration.to_html(index=False,float_format=lambda v:f'{v:.3f}')}
    <h2>All participant timelines</h2><p>Numbers refer to the configuration table. Reference episodes remain separate logical annotations.</p><img src="all_participants.png" alt="All participant timelines">
    <h2>Scoring and verification</h2><ul>
    <li>Confirmed nonwear is the reference; the user confirmed that the full recordings were reviewed, so remaining observed time is wear. Rejected diary proposals do not establish extra nonwear.</li>
    <li>Time-level scores use exact half-open sample intervals, not majority-vote epoch labels. Missing data are excluded. Individual participants contribute equally to the mean participant F1; pooled scores weight observed duration.</li>
    <li>Each annotation ID is one reference episode, including its gap-separated observed pieces. Predicted intervals adjacent in retained sample order are one event. One-to-one event matches require IoU ≥0.5 on observed samples. Boundary errors are conditional on matching.</li>
    <li>Gap-spanning reference annotations remain one logical episode. Timestamp-only interpolation ledgers contain no synthetic ACC/gyro, and no missing samples enter accuracy calculations. Detector windows still restart at gaps; minimum durations use observed, gap-split bouts as production does.</li>
    <li>All variants share the same cached raw observations and production autocalibration. The 60/15 and 30/15 acceleration scores and two-axis final decisions were checked exactly against the existing production implementation in every continuous segment.</li>
    <li>One-minute steps preserve GGIR's forward axis-flag propagation, three postprocessing passes and physical-hour thresholds. They are experimental adaptations. Final incomplete epochs remain wear as production does; unassessed seconds are exported per grid.</li>
    <li>Participant bootstrap 95% intervals (2000 resamples, fixed seed) are in summary.csv; they describe variability in this reviewed sample. No external validation is claimed.</li>
    <li>Production source defaults, source sensor files and original manual labels were not changed.</li></ul>
    <h2>Downloads</h2><p><a href="summary.csv">Summary and confidence intervals</a> · <a href="participant_metrics.csv">Participant/day/night metrics</a> · <a href="episode_metrics.csv">Every reference episode</a> · <a href="duration_strata.csv">Duration strata</a> · <a href="candidates.csv">All candidate parameters</a> · <a href="plan.json">Frozen plan and provenance</a></p>
    </body></html>"""
    (run/"report.html").write_text(text)


def execute(run):
    run = Path(run).resolve()
    if _digest(run/"plan.json") != (run/"plan.sha256").read_text().strip():
        raise ValueError("Frozen plan checksum mismatch")
    plan = json.loads((run/"plan.json").read_text())
    if plan["source_sha256"] != source_hashes():
        raise ValueError("Code changed after plan freeze; prepare a fresh run")
    declared = [Candidate(**v) for v in plan["candidates"]]
    status = json.loads((run/"status.json").read_text())
    status["status"] = "running"
    atomic_json(run/"status.json", status)
    for item in plan["participants"]:
        subject = item["subject"]
        if subject in status["completed"]:
            continue
        try:
            process_subject(item, run, declared)
            status["completed"].append(subject)
            status["failures"].pop(subject, None)
        except Exception as exc:
            import traceback
            status["failures"][subject] = str(exc)
            (run/subject/item["visit"]/"error.txt").write_text(traceback.format_exc())
            print(f"FAILED {subject}: {exc}", flush=True)
        atomic_json(run/"status.json", status)
        gc.collect()
    if status["failures"]:
        status["status"] = "failed"
        atomic_json(run/"status.json", status)
        raise RuntimeError(status["failures"])
    summary = aggregate(run, plan)
    status["status"] = "completed"
    status["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    status["summary_sha256"] = _digest(run/"summary.csv")
    atomic_json(run/"status.json", status)
    print(summary.loc[summary.period == "all", ["candidate", "macro_f1", "precision", "recall", "f1", "false_nonwear_minutes_per_day"]]
          .sort_values("macro_f1", ascending=False).to_string(index=False), flush=True)
    print(f"REPORT {run/'report.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT/"configs/nonwear_annotation.local.toml"))
    parser.add_argument("--output-root", default=str(ROOT/"outputs/nonwear_benchmark"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.run_dir and args.prepare_only:
        parser.error("Choose either an existing run or prepare-only")
    run = args.run_dir or prepare(args.config, args.output_root)
    if not args.prepare_only:
        execute(run)


if __name__ == "__main__":
    main()

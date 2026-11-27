"""Rescore saved predictions with an explicitly reviewed reference correction.

Supply a private JSON correction through --correction. Detector settings and
predictions are reused unchanged; original annotations and reports are preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import html
import re
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_lower_back_nonwear as bench
from dare_wearables.common.output_state import _digest
from dare_wearables.lower_back.annotation.data import RawCache, atomic_json


def load_correction(path):
    correction = json.loads(Path(path).read_text())
    required = {"subject", "visit", "annotation_id", "requested_start", "requested_end", "reason", "authority"}
    if (not isinstance(correction, dict) or set(correction) != required
            or any(not isinstance(v, str) or not v.strip() for v in correction.values())):
        raise ValueError("Correction must supply exactly the nonempty text fields: " + ", ".join(sorted(required)))
    if not re.fullmatch(r"[0-9]+", correction["subject"]) or not re.fullmatch(r"T[0-9]+", correction["visit"]):
        raise ValueError("Correction requires a numeric subject and a protocol visit such as T0.")
    start, end = (pd.Timestamp(correction[k]) for k in ("requested_start", "requested_end"))
    if pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None or end <= start:
        raise ValueError("Correction requires increasing, timezone-naive stored timestamps.")
    return {**correction, "algorithm_changes": "none; rescore existing predictions only"}


def rescore(parent, correction):
    parent = Path(parent).resolve()
    if _digest(parent / "plan.json") != (parent / "plan.sha256").read_text().strip():
        raise ValueError("Parent frozen plan changed")
    old = json.loads((parent / "plan.json").read_text())
    if old["source_sha256"] != bench.source_hashes():
        raise ValueError("Parent detector/scoring source changed")
    if json.loads((parent / "status.json").read_text())["status"] != "completed":
        raise ValueError("Finish the original benchmark before correcting the reference")
    run = parent / "reference_corrected"
    plan = json.loads(json.dumps(old))
    plan["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    plan["parent_run"] = str(parent)
    plan["parent_plan_sha256"] = _digest(parent / "plan.json")
    plan["reference_correction_script_sha256"] = _digest(__file__)
    correction = dict(correction)
    matches = [item for item in plan["participants"]
               if (item["subject"], item["visit"]) == (correction["subject"], correction["visit"])]
    if len(matches) != 1:
        raise ValueError("Correction must identify exactly one participant/visit in the frozen benchmark.")
    run.mkdir(exist_ok=False)
    plan["reference_correction"] = correction
    for item in plan["participants"]:
        relative = Path(item["subject"]) / item["visit"]
        source, folder = parent / relative, run / relative
        folder.mkdir(parents=True)
        verification = json.loads((source / "verification.json").read_text())
        for filename, expected in verification["outputs_sha256"].items():
            if _digest(source / filename) != expected:
                raise ValueError(f"Parent predictions/features changed: {source / filename}")
        for name in ("annotations.json", "reviewed_intervals.csv", "predictions.csv", "calibration.json",
                     "respiration_30s.parquet", "respiration_60s.parquet"):
            shutil.copy2(source / name, folder / name)
        if (item["subject"], item["visit"]) == (correction["subject"], correction["visit"]):
            cache = RawCache(item["cache"])
            start_ns, end_ns = (pd.Timestamp(correction[k]).value for k in ("requested_start", "requested_end"))
            if not cache.first_ns <= start_ns < end_ns <= cache.end_ns:
                raise ValueError("Reference correction lies outside the observed recording span.")
            a = cache.nearest(start_ns)
            b = cache.nearest(end_ns, edge="end")
            if a >= b:
                raise ValueError("Reference correction contains no retained samples.")
            session = json.loads((folder / "annotations.json").read_text())
            for row in session["rows"]:
                if row["id"] == correction["annotation_id"]:
                    raise ValueError("Correction annotation ID already exists.")
                if row["status"] == "reviewed" and row["start_sample"] < b and row["stop_sample"] > a:
                    raise ValueError("Correction overlaps an existing confirmed label")
            row = {"id": correction["annotation_id"], "status": "reviewed", "label": "nonwear",
                   "start_sample": a, "stop_sample": b, "notes": correction["reason"]}
            session["rows"].append(row)
            session["benchmark_reference_correction"] = correction
            atomic_json(folder / "annotations.json", session)
            export = pd.read_csv(folder / "reviewed_intervals.csv", dtype={"subject": str})
            parts = bench.split_bounds([[a, b]], cache.meta["segments"])
            added = []
            for part, (left, right) in enumerate(parts, 1):
                added.append({"subject": item["subject"], "visit": item["visit"],
                    "annotation_id": correction["annotation_id"], "label": "nonwear", "part": part,
                    "start": pd.Timestamp(cache.boundary_ns(int(left))).isoformat(),
                    "end": pd.Timestamp(cache.boundary_ns(int(right), "end")).isoformat(),
                    "start_sample": int(left), "stop_sample": int(right),
                    "duration_seconds": (right-left)/cache.fs, "reviewer": correction["authority"],
                    "notes": correction["reason"], "diary_start": "", "diary_end": "",
                    "raw_input_identity": item["raw_input_identity"]})
            pd.concat([export, pd.DataFrame(added)], ignore_index=True).to_csv(folder / "reviewed_intervals.csv", index=False)
            item["reference_annotations"] += 1
            correction.update(start_sample=a, stop_sample=b,
                snapped_start=pd.Timestamp(cache.boundary_ns(a)).isoformat(),
                snapped_end=pd.Timestamp(cache.boundary_ns(b, "end")).isoformat(),
                observed_seconds=(b-a)/cache.fs)
        item["labels_sha256"] = _digest(folder / "annotations.json")
        item["export_sha256"] = _digest(folder / "reviewed_intervals.csv")
    # Freeze corrected reference before recomputing any accuracy values.
    atomic_json(run / "plan.json", plan)
    (run / "plan.sha256").write_text(_digest(run / "plan.json") + "\n")
    shutil.copy2(parent / "candidates.csv", run / "candidates.csv")
    atomic_json(run / "status.json", {"status": "rescoring", "completed": [], "failures": {}})
    for item in plan["participants"]:
        relative = Path(item["subject"]) / item["visit"]
        folder = run / relative
        cache = RawCache(item["cache"])
        session = json.loads((folder / "annotations.json").read_text())
        rows, truth = bench.reference(session, cache, folder)
        predicted = pd.read_csv(folder / "predictions.csv")
        metrics, events = [], []
        for spec in plan["candidates"]:
            name = spec["id"]
            bounds = bench._merge(predicted.loc[predicted.candidate == name, ["start_sample", "stop_sample"]].to_numpy(dtype=np.int64))
            event_summary, details = bench.event_metrics(bounds, truth, rows, cache)
            for period, support in bench.periods(cache.times).items():
                counts = bench.confusion(bounds, truth, support, cache.fs)
                metrics.append({"subject": item["subject"], "visit": item["visit"], "candidate": name,
                    "period": period, "observed_hours": bench.length(support)/cache.fs/3600,
                    **counts, **bench.rates(counts), **(event_summary if period == "all" else {})})
            events.extend({"subject": item["subject"], "candidate": name, **d} for d in details)
        pd.DataFrame(metrics).to_csv(folder / "metrics.csv", index=False)
        pd.DataFrame(events).to_csv(folder / "events.csv", index=False)
        verify = json.loads((parent / relative / "verification.json").read_text())
        verify.update(reference_annotations=len(rows), reference_observed_hours=bench.length(truth)/cache.fs/3600,
                      predictions_reused_from=str(parent / relative / "predictions.csv"))
        verify["outputs_sha256"] = {name: _digest(folder/name) for name in
            ("metrics.csv", "events.csv", "predictions.csv", "calibration.json", "reference_episodes.csv",
             "respiration_30s.parquet", "respiration_60s.parquet")}
        atomic_json(folder / "verification.json", verify)
        print(f"Rescored {item['subject']}", flush=True)
    summary = bench.aggregate(run, plan)
    report_path = run / "report.html"
    report = report_path.read_text()
    note = ("<p class=\"note\"><b>Reference correction:</b> "
            + html.escape(correction["reason"]) + " Authority: " + html.escape(correction["authority"])
            + ". One additional nonwear episode was snapped to retained sample boundaries. "
            "All predictions were reused unchanged from the frozen benchmark; no detector was rerun or tuned. "
            "Original manual annotations and parent reports were preserved.</p>")
    report = report.replace("<h2>Results for every fixed configuration</h2>", note+"<h2>Results for every fixed configuration</h2>")
    data = pd.read_csv(run / "participant_metrics.csv", dtype={"subject": str})
    all_results = summary.loc[summary.period == "all"].sort_values("macro_f1", ascending=False)
    names = {bench.BASELINE: "Baseline F1", all_results.iloc[0].candidate: "Highest mean F1",
             "ggir_w30_s1_a2_r60_m15_no_post": "30/1 gyro60 no-post F1",
             "ggir_w15_s1_a2_r60_m15_no_post": "15/1 gyro60 no-post F1"}
    comparison = data.loc[(data.period == "all") & data.candidate.isin(names)].pivot(index="subject", columns="candidate", values="f1").rename(columns=names)
    ref = data.loc[(data.period == "all") & (data.candidate == bench.BASELINE)].set_index("subject")
    comparison.insert(0, "Reference NW hours", ref.reference_nonwear_hours)
    comparison.to_csv(run / "participant_comparison.csv")
    report = report.replace("<h2>Day and night</h2>", f"<h2>All {len(plan['participants'])} participants</h2>" + comparison.to_html(float_format=lambda x: f"{x:.3f}") + "<h2>Day and night</h2>")
    # Make the predominance of one long-nonwear participant explicit.
    largest = ref.reference_nonwear_hours.idxmax()
    share = ref.loc[largest, "reference_nonwear_hours"] / ref.reference_nonwear_hours.sum()
    report = report.replace("<h2>Scoring and verification</h2>",
        f"<p class=\"note\">The largest contributor supplies {share:.1%} of reference nonwear time. "
        "Pooled metrics are consequently dominated by that recording; participant-mean F1 and per-episode results are essential for interpreting short-removal performance.</p><h2>Scoring and verification</h2>")
    report_path.write_text(report)
    atomic_json(run / "status.json", {"status": "completed", "completed": [i["subject"] for i in plan["participants"]],
        "failures": {}, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_sha256": _digest(run/"summary.csv"), "reference_events": int(all_results.iloc[0].reference_events),
        "predictions_unchanged": True})
    print(all_results[["candidate", "macro_f1", "precision", "recall", "false_nonwear_minutes_per_day", "event_recall_iou50"]].head(8).to_string(index=False), flush=True)
    print(f"FINAL REPORT {report_path}", flush=True)
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_run", type=Path)
    parser.add_argument("--correction", required=True, type=Path, help="Private reviewed reference correction JSON; see docs/lower_back_validation.md.")
    args = parser.parse_args()
    rescore(args.parent_run, load_correction(args.correction))

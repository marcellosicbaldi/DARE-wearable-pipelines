"""Apply a final minimum-duration rule to completed nonwear assessment exports.

Reads saved intervals only, verifies their completion checksums, combines
recovery runs without duplicating controls, and writes a separate derived run.
This does not recompute raw features, gait, TIB, or production silver outputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dare_wearables.common.output_state import _digest
from dare_wearables.lower_back.nonwear.respiration import filter_short_nonwear_bouts


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def _read_reference(run, item):
    folder = run / item["subject"] / item["visit"] / item["sensor"] / "nonwear"
    marker = folder / "respiration_manifest.json"
    completion = json.loads(marker.read_text())
    names = ["final_nonwear_intervals.csv", "nonwear_summary.json"]
    if "time_invalid_intervals.csv" in completion["outputs"]:
        names.append("time_invalid_intervals.csv")
    digests = {}
    for name in names:
        digests[name] = _digest(folder / name)
        if digests[name] != completion["outputs"][name]:
            raise ValueError(f"Saved output checksum failed: {folder / name}")
    summary = json.loads((folder / "nonwear_summary.json").read_text())
    if summary.get("baseline_method") != "ggir_2023_matched":
        raise ValueError("This refinement requires the GGIR 2023 matched baseline.")
    if summary.get("min_nonwear_bout_seconds", 0) != 0:
        raise ValueError("Use the original respiration-only results, before duration filtering.")
    frame = pd.read_csv(folder / "final_nonwear_intervals.csv")
    for name in ("start_sample", "stop_sample", "duration_seconds"):
        frame[name] = pd.to_numeric(frame[name])
    for name in ("start", "end"):
        frame[name] = pd.to_datetime(frame[name], format="mixed")
    fs = summary["sample_rate_hz"]
    if (not np.isfinite(fs) or fs <= 0
            or not np.isfinite(frame[["start_sample", "stop_sample", "duration_seconds"]]).all().all()
            or (frame.start_sample < 0).any()
            or (frame.stop_sample > summary["n_samples"]).any()
            or (frame.stop_sample <= frame.start_sample).any()
            or (frame.start.isna() | frame.end.isna()).any()
            or (frame.end <= frame.start).any()):
        raise ValueError(f"Invalid saved interval bounds: {folder}")
    if any(stamp.tzinfo is not None for stamp in frame.start):
        raise ValueError("Expected recording-local timestamps without a timezone conversion.")
    samples = frame[["start_sample", "stop_sample"]].to_numpy()
    if not np.equal(samples, np.floor(samples)).all():
        raise ValueError("Expected whole-sample bounds.")
    if len(frame) > 1 and (np.any(samples[1:, 0] < samples[:-1, 1])
            or np.any(frame.start.to_numpy()[1:] < frame.end.to_numpy()[:-1])):
        raise ValueError("Saved nonwear intervals overlap or are unsorted.")
    if not np.allclose((frame.stop_sample - frame.start_sample) / fs,
                       frame.duration_seconds, rtol=0, atol=1e-8):
        raise ValueError("Saved sample bounds and durations disagree.")
    wall = (frame.end - frame.start).dt.total_seconds()
    if ((wall - frame.duration_seconds).abs() > 1 / fs + 1e-6).any():
        raise ValueError("An exported bout bridges missing time.")
    if not np.isclose(frame.duration_seconds.sum() / 3600,
                      summary["revised_nonwear_hours"], rtol=0, atol=1e-8):
        raise ValueError("Saved bouts and subject summary disagree.")
    if "time_invalid_intervals.csv" in names:
        invalid = pd.read_csv(folder / "time_invalid_intervals.csv")
        for gap in invalid.itertuples():
            if ((frame.start < pd.Timestamp(gap.end)) & (frame.end > pd.Timestamp(gap.start))).any():
                raise ValueError("Nonwear overlaps time-invalid data.")
    return {"run": run, "item": item, "summary": summary, "bouts": frame,
            "reference": {"run": str(run), "source": item.get("source"), "input_identity": item.get("identity"),
                          "completion_sha256": _digest(marker),
                          "used_output_sha256": digests}}


def classify_episodes(frame):
    """Count each bout once; crossing bouts go to the longer day/night portion.

    Night is [22:00, 08:00) in the recording's local clock. The periodic integral
    handles midnight and bouts spanning multiple days without splitting episodes.
    """
    table = frame.copy()
    hour = 3600 * 10**9
    def integral(stamp):
        days, remainder = divmod(pd.Timestamp(stamp).value, 24 * hour)
        return days * 10 * hour + min(remainder, 8 * hour) + max(0, remainder - 22 * hour)
    night = np.array([integral(e) - integral(s) for s, e in zip(table.start, table.end)], dtype=np.int64)
    wall = (table.end - table.start).to_numpy(dtype="timedelta64[ns]").astype(np.int64)
    day = wall - night
    table["period"] = np.where(night > day, "night", np.where(day > night, "day", "tie"))
    table["crosses_day_night_boundary"] = (day > 0) & (night > 0)
    table["day_wall_seconds"] = day / 1e9
    table["night_wall_seconds"] = night / 1e9
    return table


def refine_assessments(runs, output_root, *, minimum_seconds=900., expected_recordings=None):
    # Validate even if the input contains no nonwear bouts.
    filter_short_nonwear_bouts(pd.DataFrame({"duration_seconds": []}), minimum_seconds)
    selected, required, run_digests, duplicates = {}, set(), {}, 0
    for run in map(lambda p: Path(p).expanduser().resolve(), runs):
        manifest_path = run / "assessment.json"
        manifest = json.loads(manifest_path.read_text())
        run_digests[str(run)] = _digest(manifest_path)
        for item in manifest["recordings"]:
            key = (item["subject"], item["visit"], item["sensor"])
            if item["status"] != "skipped":
                required.add(key)
            if item["status"] != "completed":
                continue
            reference = _read_reference(run, item)
            if key in selected:
                pd.testing.assert_frame_equal(selected[key]["bouts"], reference["bouts"])
                for field in ("baseline_nonwear_hours", "revised_nonwear_hours", "observed_hours"):
                    if selected[key]["summary"][field] != reference["summary"][field]:
                        raise ValueError(f"Duplicate control summaries disagree: {key}")
                duplicates += 1
            else:
                selected[key] = reference
    if set(selected) != required:
        raise ValueError(f"Missing successful recordings: {sorted(required - set(selected))}")
    if not selected or (expected_recordings is not None and len(selected) != expected_recordings):
        raise ValueError(f"Unexpected recording count: {len(selected)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(output_root).expanduser().resolve() / f"duration_{stamp}_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "operation": "filter_saved_respiration_refined_nonwear_bouts",
                "min_nonwear_bout_seconds": minimum_seconds, "comparison": "duration_seconds < cutoff becomes wear",
                "night": "22:00–08:00 recording-local time", "episode_assignment": "majority duration; ties separate",
                "input_assessment_sha256": run_digests, "duplicate_controls_not_counted": duplicates,
                "processing_sha256": {str(p): _digest(p) for p in
                    (Path(__file__), Path(__file__).parents[1] / "src/dare_wearables/lower_back/nonwear/respiration.py")},
                "downstream_gait_tib_rerun": False, "recordings": []}
    _json(output / "duration_filter_manifest.json", manifest)
    cohort, all_before, all_after, all_short = [], [], [], []
    for (subject, visit, sensor), reference in sorted(selected.items()):
        before = reference["bouts"]
        kept, short = filter_short_nonwear_bouts(before, minimum_seconds)
        summary = reference["summary"]
        previous = float(before.duration_seconds.sum() / 3600)
        removed = float(short.duration_seconds.sum() / 3600)
        final = float(kept.duration_seconds.sum() / 3600)
        if not np.isclose(previous, final + removed, rtol=0, atol=1e-10):
            raise AssertionError("Duration partition failed.")
        row = {"subject": subject, "visit": visit, "sensor": sensor, "status": "completed",
               "observed_hours": summary["observed_hours"],
               "sample_rate_hz": summary["sample_rate_hz"], "n_samples": summary["n_samples"],
               "baseline_method": summary["baseline_method"],
               "baseline_nonwear_hours": summary["baseline_nonwear_hours"],
               "nonwear_before_duration_filter_hours": previous,
               "respiration_reassigned_to_wear_hours": summary["reassigned_to_wear_hours"],
               "short_nonwear_reassigned_to_wear_hours": removed, "revised_nonwear_hours": final,
               "reassigned_to_wear_hours": summary["reassigned_to_wear_hours"] + removed,
               "min_nonwear_bout_seconds": minimum_seconds,
               "episodes_before": len(before), "episodes_after": len(kept), "episodes_removed": len(short)}
        classified = []
        for frame in (before, kept, short):
            result = classify_episodes(frame)
            result.insert(0, "subject", subject)
            result.insert(1, "visit", visit)
            classified.append(result)
        before_table, after_table, short_table = classified
        row.update({f"{period}_episodes": int(after_table.period.eq(period).sum()) for period in ("day", "night", "tie")})
        cohort.append(row)
        all_before.append(before_table)
        all_after.append(after_table)
        all_short.append(short_table)
        folder = output / subject / visit / sensor / "nonwear"
        folder.mkdir(parents=True)
        kept.to_csv(folder / "final_nonwear_intervals.csv", index=False)
        short.to_csv(folder / "short_nonwear_reassigned_intervals.csv", index=False)
        _json(folder / "nonwear_summary.json", row)
        manifest["recordings"].append({"subject": subject, "visit": visit, "sensor": sensor,
                                        "reference": reference["reference"]})
    table = pd.DataFrame(cohort)
    table.to_csv(output / "cohort_summary.csv", index=False)
    stages = {}
    for name, frames in (("before_filter", all_before), ("after_filter", all_after), ("reassigned_to_wear", all_short)):
        bouts = pd.concat(frames, ignore_index=True)
        bouts.to_csv(output / f"episodes_{name}.csv", index=False)
        hours = table[{"before_filter": "nonwear_before_duration_filter_hours", "after_filter": "revised_nonwear_hours",
                       "reassigned_to_wear": "short_nonwear_reassigned_to_wear_hours"}[name]]
        stages[name] = {"episodes": len(bouts), "day_episodes": int(bouts.period.eq("day").sum()),
                        "night_episodes": int(bouts.period.eq("night").sum()), "tied_episodes": int(bouts.period.eq("tie").sum()),
                        "boundary_crossing_episodes": int(bouts.crosses_day_night_boundary.sum()),
                        "nonwear_hours_total": float(hours.sum()), "nonwear_hours_mean_per_recording": float(hours.mean()),
                        "nonwear_hours_median_per_recording": float(hours.median()),
                        "participants_with_episodes": int(bouts.subject.nunique())}
    stats = {"recordings": len(table), "participants": int(table.subject.nunique()), "visits": sorted(table.visit.unique()),
             "minimum_nonwear_seconds": minimum_seconds, "stages": stages,
             "mean_final_nonwear_hours_per_observed_day": float((table.revised_nonwear_hours / table.observed_hours * 24).mean()),
             "participants_with_zero_final_nonwear": int(table.revised_nonwear_hours.eq(0).sum())}
    _json(output / "statistics.json", stats)
    # Confirm the frozen references and processing code stayed unchanged.
    for run, digest in run_digests.items():
        if _digest(Path(run) / "assessment.json") != digest:
            raise ValueError("Input assessment changed during refinement.")
    for reference in selected.values():
        item = reference["item"]
        folder = reference["run"] / item["subject"] / item["visit"] / item["sensor"] / "nonwear"
        if _digest(folder / "respiration_manifest.json") != reference["reference"]["completion_sha256"]:
            raise ValueError("Reference completion marker changed during refinement.")
        for name, digest in reference["reference"]["used_output_sha256"].items():
            if _digest(folder / name) != digest:
                raise ValueError("Reference output changed during refinement.")
    for path, digest in manifest["processing_sha256"].items():
        if _digest(path) != digest:
            raise ValueError("Processing code changed during refinement.")
    manifest["status"] = "completed"
    manifest["output_sha256"] = {str(p.relative_to(output)): _digest(p) for p in output.rglob("*")
                                  if p.is_file() and p.name != "duration_filter_manifest.json"}
    _json(output / "duration_filter_manifest.json", manifest)
    print(f"Results: {output}")
    print(json.dumps(stats, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assessment_runs", nargs="+")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--minimum-seconds", type=float, default=900.)
    parser.add_argument("--expected-recordings", type=int)
    args = parser.parse_args()
    refine_assessments(args.assessment_runs, args.output_root, minimum_seconds=args.minimum_seconds,
                       expected_recordings=args.expected_recordings)


if __name__ == "__main__":
    main()

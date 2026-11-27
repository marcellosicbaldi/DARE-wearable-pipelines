"""Compare frozen legacy gait, count fixes, and per-sequence proper rotations.

Example (private paths supplied by the caller)::

    .venv/bin/python scripts/compare_lower_back_gait_orientation.py \
        --baseline-run /absolute/full_run_folder --output-root /absolute/new_folder \
        --participant 900001 --participant 900002

The baseline runs in a separate interpreter with the frozen source snapshot.
Preprocessing is cached once from that same snapshot, then shared by all arms.
No files in the baseline run or original silver tree are changed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
VARIANTS = ("legacy", "counts_fixed", "uncorrected", "sequence_aligned")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    temp.replace(path)


def worker(args):
    # PYTHONPATH, selected by the parent, determines frozen versus current code.
    import numpy as np
    import pandas as pd
    from dare_wearables.lower_back.batch import _decode
    from dare_wearables.lower_back.preprocessing import preprocess_lowerback_file
    from dare_wearables.lower_back.pipeline import apply_orientation_correction
    from dare_wearables.lower_back.gait import functions as gait
    from dare_wearables.lower_back.gait.daily import run_daily_gait

    output = args.output_root
    manifest = json.loads((output / "manifest.json").read_text())
    baseline = Path(manifest["baseline_run"])
    plan = json.loads((baseline / "run_plan.json").read_text())
    config = _decode(plan["config"])
    subject = args.participant[0]
    folder = output / subject
    folder.mkdir(exist_ok=True)
    if args.stage == "preprocess":
        item = next(r for r in plan["recordings"] if r["subject"] == subject and r["visit"] == args.visit)
        result = preprocess_lowerback_file(item["source"], protocol_hours=config.protocol_hours,
                                          nonwear=config.nonwear, respiration=config.respiration,
                                          timestamps=config.timestamps)
        frame = result["df"]
        frame.to_parquet(folder / "preprocessed.parquet", index=False)
        write_json(folder / "preprocessing.json", {"input_identity": item["identity"],
                   "n_samples": len(frame), "wear_samples": int(frame.wear.sum()),
                   "timestamp_quality": result["timestamp_quality"]})
        return

    variant = args.variant
    target = output / variant / "silver" / subject / args.visit / config.sensor / "gait"
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "comparison_complete.json"
    marker.unlink(missing_ok=True)
    clinical = pd.read_csv(config.redcap_csv, dtype={"patient_id": "string"}, low_memory=False)
    clinical = clinical.rename(columns={"obs_pt_height": "Height"})
    cohort = gait.get_cohort(clinical, subject)
    sensor_height = gait.get_sensor_height(clinical, subject=subject)
    if variant == "legacy":
        rows = gait.get_sensor_height(clinical)
        rows = gait._participant_baseline(rows, subject)
        height = float(rows.Height.unique().item())
    else:
        height = gait.get_participant_height(clinical, subject)
    print(subject, variant, "cohort", cohort, "height", height, "sensor height", sensor_height, flush=True)
    frame = pd.read_parquet(folder / "preprocessed.parquet").set_index("datetime")
    # Pandas may attach a large quality dictionary; it is identical in all arms.
    if variant in ("legacy", "counts_fixed"):
        frame = apply_orientation_correction(frame)
    tables = defaultdict(list)
    segment_id = 0
    started = time.monotonic()
    original_gsd = gait.GsdIonescu.detect
    original_lrc = gait.LrcUllrich.predict
    original_turn = gait.TdElGohary.detect

    def processor(data, **kwargs):
        nonlocal segment_id
        segment_id += 1
        detected, contacts, turns = [], [], []

        def detect(instance, *a, **kw):
            result = original_gsd(instance, *a, **kw)
            detected.append(instance.gs_list_.copy())
            return result

        def predict(instance, *a, **kw):
            result = original_lrc(instance, *a, **kw)
            contacts.append(instance.ic_lr_list_.copy())
            return result

        def turning(instance, *a, **kw):
            result = original_turn(instance, *a, **kw)
            turns.append(instance.turn_list_.copy())
            return result

        with patch.object(gait.GsdIonescu, "detect", detect), \
                patch.object(gait.LrcUllrich, "predict", predict), \
                patch.object(gait.TdElGohary, "detect", turning):
            if variant != "legacy":
                kwargs.update(participant_height_m=height,
                              orientation_mode="sequence" if variant == "sequence_aligned" else "none")
            result = gait.process_gait_data(data, **kwargs)
        gs = detected[0].copy()
        gs.index.name = "gs_id"
        gs = gs.reset_index()
        gs["segment_id"] = segment_id
        gs["segment_start"] = data.datetime.iloc[0]
        if len(gs):
            gs["start_time"] = data.datetime.iloc[gs.start.to_numpy(dtype=int)].to_numpy()
            gs["end_time"] = data.datetime.iloc[gs.end.to_numpy(dtype=int)-1].to_numpy() + np.timedelta64(10, "ms")
        tables["gait_sequences"].append(gs)
        for name, events in (("initial_contacts", contacts), ("turns", turns)):
            for (_, region), event in zip(gs.iterrows(), events):
                event = event.reset_index()
                for column in ("ic", "start", "end"):
                    if column in event:
                        event[column] += int(region.start)
                if "ic" in event:
                    event["time"] = data.datetime.iloc[event.ic.to_numpy(dtype=int)].to_numpy()
                event["gs_id"], event["segment_id"] = region.gs_id, segment_id
                tables[name].append(event)
        for name in ("orientation", "final_strides", "stride_list", "per_wb_params"):
            value = result.get(name)
            if value is None or value.empty:
                continue
            value = value.copy().reset_index()
            value["segment_id"] = segment_id
            value["segment_start"] = data.datetime.iloc[0]
            tables[name].append(value)
        print(subject, variant, "segment", segment_id, "date", str(data.datetime.iloc[0].date()),
              "hours", round(len(data)/360000, 2), "GS", len(gs), flush=True)
        return result

    height_kwargs = {"participant_height_m": height} if variant != "legacy" else {}
    run_daily_gait(frame, subject=subject, cohort=cohort, sensor_height_m=sensor_height,
                   save_folder=target, processor=processor, **height_kwargs)
    for name, values in tables.items():
        table = pd.concat(values, ignore_index=True)
        if "rule_obj" in table:
            table["rule_obj"] = table.rule_obj.map(str)
        table.to_parquet(target / f"diagnostic_{name}.parquet", index=False)
    source_root = baseline / "source_snapshot" if variant == "legacy" else REPO / "src"
    sources = {str(p.relative_to(source_root)): digest(p)
               for p in (source_root / "dare_wearables/lower_back/gait").glob("*.py")}
    write_json(marker, {"subject": subject, "visit": args.visit, "variant": variant,
               "cohort": cohort, "participant_height_m": height, "sensor_height_m": sensor_height,
               "elapsed_seconds": time.monotonic()-started, "gait_sources": sources,
               "preprocessed_sha256": digest(folder / "preprocessed.parquet"),
               "baseline_plan_sha256": digest(baseline / "run_plan.json"),
               "outputs": {p.name: digest(p) for p in target.iterdir() if p.is_file() and p != marker}})
    print("COMPLETE", subject, variant, flush=True)


def contact_differences(reference, candidate):
    """Same-GS nearest-reference distances; these are sensitivity, not accuracy."""
    import numpy as np
    distances, label_matches = [], []
    old_groups = {key: group.sort_values("ic") for key, group in reference.groupby(["segment_id", "gs_id"])}
    for key, group in candidate.groupby(["segment_id", "gs_id"]):
        old = old_groups.get(key)
        if old is None or old.empty:
            distances.extend([np.inf] * len(group))
            continue
        previous = old.ic.to_numpy()
        current = group.ic.to_numpy()
        right = np.searchsorted(previous, current).clip(0, len(previous)-1)
        left = (np.searchsorted(previous, current)-1).clip(0, len(previous)-1)
        nearest = np.where(np.abs(previous[left]-current) <= np.abs(previous[right]-current), left, right)
        delta = np.abs(previous[nearest]-current)
        distances.extend(delta.tolist())
        comparable = delta <= 10  # 100 ms at 100 Hz; no claim of true IC correspondence.
        label_matches.extend((old.lr_label.to_numpy()[nearest][comparable]
                              == group.lr_label.to_numpy()[comparable]).tolist())
    distances = np.asarray(distances)
    finite = distances[np.isfinite(distances)]
    return {"candidate_contacts": len(candidate), "reference_contacts": len(reference),
            "fraction_exact_ic": float((distances == 0).mean()) if len(distances) else np.nan,
            "median_nearest_ic_ms": float(np.median(finite)*10) if len(finite) else np.nan,
            "p90_nearest_ic_ms": float(np.quantile(finite, .9)*10) if len(finite) else np.nan,
            "fraction_within_100ms": float((distances <= 10).mean()) if len(distances) else np.nan,
            "lr_disagreement_within_100ms": 1-float(np.mean(label_matches)) if label_matches else np.nan}


def summarize(output, subjects, visit):
    import numpy as np
    import pandas as pd
    from dare_wearables.aggregation.gait import aggregate_gait
    rows, daily, sequence_checks, contact_checks, stride_checks = [], [], [], [], []
    for variant in VARIANTS:
        root = output / variant / "silver"
        aggregated, days = aggregate_gait(root, visit=visit)
        aggregated["variant"] = variant
        aggregated.to_csv(output / f"{variant}_subject_aggregation.csv", index=False)
        for subject in subjects:
            folder = root / subject / visit / "McRoberts/gait"
            if not (folder / "comparison_complete.json").exists():
                raise ValueError(f"Incomplete comparison arm: {variant}/{subject}")
            day = pd.read_csv(folder / f"{subject}_day_aggregation.csv")
            bouts = pd.read_csv(folder / f"{subject}_per_wb.csv")
            contacts = pd.read_parquet(folder / "diagnostic_initial_contacts.parquet")
            turns = pd.read_parquet(folder / "diagnostic_turns.parquet")
            gs = pd.read_parquet(folder / "diagnostic_gait_sequences.parquet")
            baseline = pd.read_parquet(output / "legacy/silver" / subject / visit / "McRoberts/gait/diagnostic_gait_sequences.parquet")
            keys = ["segment_id", "gs_id", "start", "end"]
            sequence_checks.append({"subject": subject, "variant": variant,
                                    "n_gait_sequences": len(gs),
                                    "identical_to_legacy": gs[keys].equals(baseline[keys])})
            if variant != "legacy":
                reference_folder = output / "legacy/silver" / subject / visit / "McRoberts/gait"
                old_contacts = pd.read_parquet(reference_folder / "diagnostic_initial_contacts.parquet")
                contact_checks.append({"subject": subject, "variant": variant,
                                       **contact_differences(old_contacts, contacts)})
                def by_sequence(path):
                    strides = pd.read_parquet(path / "diagnostic_final_strides.parquet")
                    return strides.groupby(["segment_id", "original_gs_id"]).agg(
                        retained_strides=("start", "size"), stride_duration_s=("stride_duration_s", "mean"),
                        cadence_spm=("cadence_spm", "mean"), stride_length_m=("stride_length_m", "mean"),
                        walking_speed_mps=("walking_speed_mps", "mean"))
                paired = by_sequence(reference_folder).join(by_sequence(folder), how="outer", lsuffix="_legacy", rsuffix="_revised")
                paired["subject"], paired["variant"] = subject, variant
                stride_checks.append(paired.reset_index())
            row = {"subject": subject, "visit": visit, "variant": variant,
                   "recorded_hours": day.hours.sum(), "wear_hours": day.hours.sum()-day.nonwear_time_minutes.sum()/60,
                   "n_valid_days": int(((day.hours-day.nonwear_time_minutes/60)>16).sum()),
                   "gait_sequences": len(gs), "initial_contacts": len(contacts),
                   "gait_sequence_minutes": ((gs.end-gs.start)/6000).sum(),
                   "detected_turns": len(turns),
                   "walking_bouts": len(bouts), "walking_minutes": bouts.duration_s.sum()/60,
                   "reported_steps": day.step_count.sum(), "retained_stride_intervals": bouts.n_strides.sum(),
                   "turns": bouts.n_turns.sum(), "mean_bout_cadence_spm": bouts.cadence_spm.mean(),
                   "mean_bout_stride_length_m": bouts.stride_length_m.mean(),
                   "mean_bout_speed_mps": bouts.walking_speed_mps.mean(),
                   "mean_bout_stride_duration_s": bouts.stride_duration_s.mean()}
            if variant == "sequence_aligned":
                orientation = pd.read_parquet(folder / "diagnostic_orientation.parquet")
                supported = orientation.status.eq("aligned")
                row.update(negative_vertical_candidates=int(orientation.vertical_reversed.sum()),
                           aligned_vertical_reversals=int((orientation.vertical_reversed & supported).sum()),
                           median_rotation_deg=orientation.loc[supported, "rotation_deg"].median(),
                           unaligned_sequences=int(orientation.status.ne("aligned").sum()))
            rows.append(row)
            day["subject"], day["variant"] = subject, variant
            daily.append(day)
    pd.DataFrame(rows).to_csv(output / "recording_comparison.csv", index=False)
    pd.concat(daily, ignore_index=True).to_csv(output / "daily_comparison.csv", index=False)
    pd.DataFrame(sequence_checks).to_csv(output / "sequence_detection_checks.csv", index=False)
    pd.DataFrame(contact_checks).to_csv(output / "contact_sensitivity.csv", index=False)
    pd.concat(stride_checks, ignore_index=True).to_csv(output / "paired_sequence_parameters.csv", index=False)
    full = pd.concat([pd.read_csv(output / f"{v}_subject_aggregation.csv", dtype={"subject": str}) for v in VARIANTS])
    full.to_csv(output / "subject_aggregation_comparison.csv", index=False)
    differences = []
    for subject in subjects:
        reference = full.loc[full.subject.eq(subject) & full.variant.eq("legacy")].iloc[0]
        for variant in VARIANTS[1:]:
            candidate = full.loc[full.subject.eq(subject) & full.variant.eq(variant)].iloc[0]
            for column in full.select_dtypes(include="number"):
                old, new = float(reference[column]), float(candidate[column])
                differences.append({"subject": subject, "variant": variant, "metric": column,
                                    "legacy": old, "revised": new, "difference": new-old,
                                    "percent_change": 100*(new-old)/abs(old) if old else np.nan})
    pd.DataFrame(differences).to_csv(output / "subject_metric_differences.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(subjects), 2, figsize=(13, 4*len(subjects)), squeeze=False)
    colors = {"legacy": "#777777", "sequence_aligned": "#14816f"}
    labels = {"legacy": "Your pipeline", "sequence_aligned": "Per-sequence rotation"}
    combined = pd.concat(daily)
    for i, subject in enumerate(subjects):
        for variant in ("legacy", "sequence_aligned"):
            data = combined.loc[combined.subject.eq(subject) & combined.variant.eq(variant)]
            for j, (column, ylabel) in enumerate((("total_walking_duration_min", "Walking minutes / observed day"),
                                                  ("wb_30__walking_speed_mps__avg", "Mean speed in bouts >30 s (m/s)"))):
                axes[i, j].plot(data.day, data[column], marker="o", label=labels[variant], color=colors[variant])
                axes[i, j].set(title=f"{subject} / {visit}", xlabel="Recording day", ylabel=ylabel)
                axes[i, j].grid(alpha=.18)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Gait comparison — shared calibration, timestamps and wear mask", fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "daily_comparison.png", dpi=160)
    plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Detection checks:", sequence_checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--participant", action="append", required=True)
    parser.add_argument("--visit", default="T0")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--stage", choices=("preprocess", "gait", "summarize"), default="gait")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--no-summary", action="store_true", help="Run arms only; useful for separate subject workers.")
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    sys.path.insert(0, str(REPO / "src"))
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        if args.baseline_run is None:
            parser.error("A new comparison requires --baseline-run.")
        baseline = args.baseline_run.resolve()
        plan = json.loads((baseline / "run_plan.json").read_text())
        write_json(manifest_path, {"baseline_run": str(baseline), "baseline_plan_sha256": digest(baseline / "run_plan.json"),
                   "subjects": args.participant, "visit": args.visit, "profile": plan["config"]})
    manifest = json.loads(manifest_path.read_text())
    baseline = Path(manifest["baseline_run"])
    if args.baseline_run is not None and args.baseline_run.resolve() != baseline:
        raise ValueError("This output directory belongs to a different baseline run.")
    if digest(baseline / "run_plan.json") != manifest["baseline_plan_sha256"]:
        raise ValueError("Frozen baseline plan changed.")
    from importlib import metadata
    from dare_wearables.common.output_state import signature
    plan = json.loads((baseline / "run_plan.json").read_text())
    if any(metadata.version(name) != version for name, version in plan["dependencies"].items()):
        raise ValueError("The comparison environment must match the frozen baseline dependencies.")
    if digest(plan["config"]["redcap_csv"]) != plan["metadata_sha256"]:
        raise ValueError("Clinical metadata changed after the baseline was frozen.")
    for subject in args.participant:
        item = next(r for r in plan["recordings"] if r["subject"] == subject and r["visit"] == args.visit)
        if signature({}, [item["source"]]) != item["identity"]:
            raise ValueError(f"Raw input changed after the baseline was frozen: {subject}/{args.visit}")
    if args.stage == "summarize":
        return summarize(output, args.participant, args.visit)
    variants = [args.variant] if args.variant else list(VARIANTS)
    for subject in args.participant:
        stages = [] if args.stage == "preprocess" else [("gait", variant) for variant in variants]
        if not (output / subject / "preprocessed.parquet").exists():
            stages.insert(0, ("preprocess", "legacy"))
        for stage, variant in stages:
            source = baseline / "source_snapshot" if variant == "legacy" or stage == "preprocess" else REPO / "src"
            marker = output / variant / "silver" / subject / args.visit / "McRoberts/gait/comparison_complete.json"
            if stage == "gait" and marker.exists():
                previous = json.loads(marker.read_text())
                source_hashes = {str(p.relative_to(source)): digest(p)
                                 for p in (source / "dare_wearables/lower_back/gait").glob("*.py")}
                if (previous["gait_sources"] == source_hashes
                        and previous["preprocessed_sha256"] == digest(output / subject / "preprocessed.parquet")
                        and previous.get("outputs")
                        and all((marker.parent / name).exists() and digest(marker.parent / name) == value
                                for name, value in previous["outputs"].items())):
                    print("REUSE", subject, variant, flush=True)
                    continue
            environment = dict(os.environ, PYTHONPATH=str(source), MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
                               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
            log = output / f"{subject}_{stage}_{variant}.log"
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--stage", stage,
                       "--output-root", str(output), "--participant", subject, "--visit", args.visit, "--variant", variant]
            print("RUN", subject, stage, variant, "log:", log, flush=True)
            with log.open("w") as stream:
                subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    if args.stage == "gait" and not args.variant and not args.no_summary:
        summarize(output, args.participant, args.visit)


if __name__ == "__main__":
    main()

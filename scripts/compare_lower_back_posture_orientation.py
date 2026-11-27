"""Compare posture mounting corrections using an existing frozen gait experiment.

All arms share the original 100-Hz preprocessing cache and unchanged posture,
nonwear-filling and TIB functions. Only the six sensor channels change. Nothing
is written to the original run. This is a sensitivity experiment, not a change
to the production posture default or an accuracy validation.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

from compare_lower_back_gait_orientation import digest, write_json

REPO = Path(__file__).resolve().parents[1]
VARIANTS = ("legacy", "proper_legacy_mask", "uncorrected", "stable_walking")
SOURCES = (
    "src/dare_wearables/lower_back/pipeline.py",
    "src/dare_wearables/lower_back/posture/lying_functions.py",
    "src/dare_wearables/lower_back/posture/orientation.py",
    "src/dare_wearables/lower_back/utils/bool_to_bouts.py",
    "src/dare_wearables/lower_back/gait/reorientation_functions.py",
    "scripts/compare_lower_back_posture_orientation.py",
    "scripts/compare_lower_back_gait_orientation.py",
)


def function_ast(path, name):
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        node.body.pop(0)
    return ast.dump(node)


def subject_root(output, variant, subject, visit):
    return output / variant / "silver" / subject / visit / "McRoberts"


def worker(args):
    import numpy as np
    import pandas as pd
    from dare_wearables.lower_back import pipeline
    from dare_wearables.lower_back.gait.reorientation_functions import (
        build_flipped_mask, detect_orientation_flips,
    )
    from dare_wearables.lower_back.posture.orientation import (
        ACC, GYR, stable_posture_correction,
    )

    manifest = json.loads((args.output_root / "manifest.json").read_text())
    subject = args.participant[0]
    gait_root = Path(manifest["gait_comparison"])
    folder = args.output_root / subject
    folder.mkdir(exist_ok=True)
    cache = gait_root / subject / "preprocessed.parquet"
    print("READ", subject, cache, flush=True)
    data = pd.read_parquet(cache, columns=ACC+GYR+["datetime", "wear", "time_segment"])
    data["datetime"] = pd.to_datetime(data.datetime)
    data = data.set_index("datetime").sort_index()
    # Legacy orientation inference must run at 100 Hz, before downsampling.
    windows, segments, _ = detect_orientation_flips(data)
    windows.to_csv(folder / "legacy_orientation_windows.csv")
    segments.to_csv(folder / "legacy_orientation_segments.csv", index=False)
    frame = data[ACC+GYR].resample("1s").mean()
    frame["wear"] = data.wear.astype(float).resample("1s").mean().ge(.5)
    frame["time_segment"] = data.time_segment.resample("1s").first()
    # Flip boundaries are whole seconds: rotation commutes with this resampling.
    for boundary in list(segments.start)+list(segments.end):
        if pd.Timestamp(boundary) != pd.Timestamp(boundary).floor("s"):
            raise ValueError("Legacy correction boundary falls inside a 1-Hz bin.")
    del data
    prep = manifest["preprocessing"][subject]
    quality = prep["timestamp_quality"]
    if (quality.get("events") or quality.get("recording_discontinuities")
            or quality["analysis"]["segment_count"] != 1):
        raise ValueError("This comparison requires contiguous recordings; preserve observed bins before extending it.")
    frame.to_parquet(folder / "signals_1hz.parquet")
    flipped = build_flipped_mask(frame, segments)
    walking_path = gait_root / "sequence_aligned/silver" / subject / args.visit / "McRoberts/gait" / f"{subject}_orientation_qc.csv"
    walking = pd.read_csv(walking_path, parse_dates=["start_time", "end_time"])
    stable, episode_qc = stable_posture_correction(frame, walking)
    episode_qc.to_csv(folder / "mounting_episode_qc.csv", index=False)
    print("EPISODES", subject, episode_qc[["wear_start", "wear_end", "status", "eligible_sequences", "consensus_fraction", "rotation_deg"]].to_string(index=False), flush=True)

    for variant in VARIANTS:
        corrected = stable.copy() if variant == "stable_walking" else frame.copy()
        if variant == "legacy":
            corrected.loc[flipped, ACC+GYR] *= -1
        elif variant == "proper_legacy_mask":
            corrected.loc[flipped, ["acc_x", "acc_y", "gyr_x", "gyr_y"]] *= -1
        assert corrected.wear.equals(frame.wear)
        target = subject_root(args.output_root, variant, subject, args.visit)
        target.mkdir(parents=True, exist_ok=True)
        (target / "comparison_complete.json").unlink(missing_ok=True)
        captured = []
        detector = pipeline.detect_lying_lowerback

        def capture(df, **kwargs):
            result = detector(df, **kwargs)
            labels, theta, _ = result
            wear = kwargs.get("wear_1hz_override", df.wear)
            captured.append(pd.DataFrame({"lying": labels, "theta_deg": theta, "wear": wear}))
            return result

        with patch.object(pipeline, "detect_lying_lowerback", capture):
            pipeline.run_time_in_bed_pipeline(corrected, save_root=target)
        assert len(captured) == 2
        diagnostics = captured[0].add_suffix("_before_fill").join(captured[1].add_suffix("_after_fill"))
        diagnostics["legacy_flip"] = flipped
        diagnostics.to_parquet(target / "diagnostic_posture_1hz.parquet")
        write_json(target / "comparison_complete.json", {
            "subject": subject, "visit": args.visit, "variant": variant,
            "wear_unchanged": corrected.wear.equals(frame.wear),
            "preprocessed_sha256": manifest["inputs"][subject]["preprocessed_sha256"],
            "walking_qc_sha256": digest(walking_path),
            "legacy_flipped_seconds": int(flipped.sum()),
            "outputs": {str(p.relative_to(target)): digest(p)
                        for p in sorted(target.rglob("*")) if p.is_file()},
        })
        print("DONE", subject, variant, flush=True)


def summarize(output, subjects, visit):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((output / "manifest.json").read_text())
    baseline = Path(manifest["baseline_run"])
    rows, nights, changes, episode_frames = [], [], [], []
    validation = {"frozen_posture_functions_match": True, "shared_wear_masks": True,
                  "legacy_matches_completed_original": {}, "output_hashes_verified": True}
    for subject in subjects:
        reference_root = subject_root(output, "legacy", subject, visit)
        reference = pd.read_parquet(reference_root / "diagnostic_posture_1hz.parquet")
        ref_nights = pd.read_csv(reference_root / "posture/tib_qc_df.csv", parse_dates=["day_start", "tib_start", "tib_end"])
        episode = pd.read_csv(output / subject / "mounting_episode_qc.csv")
        episode["subject"] = subject
        episode_frames.append(episode)
        for variant in VARIANTS:
            root = subject_root(output, variant, subject, visit)
            marker = json.loads((root / "comparison_complete.json").read_text())
            assert marker["wear_unchanged"]
            assert all(digest(root / name) == value for name, value in marker["outputs"].items())
            diagnostic = pd.read_parquet(root / "diagnostic_posture_1hz.parquet")
            assert diagnostic.wear_before_fill.equals(reference.wear_before_fill)
            bouts = pd.read_csv(root / "posture/lying_bouts.csv")
            naps = pd.read_csv(root / "posture/naps_df.csv")
            night = pd.read_csv(root / "posture/tib_qc_df.csv", parse_dates=["day_start", "tib_start", "tib_end"])
            filled = pd.read_csv(root / "nonwear/nonwear_filled_between_lying.csv", parse_dates=["start", "end"])
            mask = diagnostic.wear_before_fill
            first_diff = diagnostic.lying_before_fill.ne(reference.lying_before_fill)
            final_diff = diagnostic.lying_after_fill.ne(reference.lying_after_fill)
            common = diagnostic.theta_deg_before_fill.notna() & reference.theta_deg_before_fill.notna()
            angle_delta = diagnostic.loc[common, "theta_deg_before_fill"] - reference.loc[common, "theta_deg_before_fill"]
            rows.append({"subject": subject, "visit": visit, "variant": variant,
                         "observed_1hz_seconds": len(diagnostic), "wear_1hz_seconds": int(mask.sum()),
                         "lying_before_fill_minutes": diagnostic.lying_before_fill.sum()/60,
                         "lying_after_fill_minutes": diagnostic.lying_after_fill.sum()/60,
                         "retained_lying_bouts": len(bouts), "retained_lying_minutes": bouts.duration_s.sum()/60,
                         "nonwear_filled_minutes": (diagnostic.wear_after_fill & ~mask).sum()/60,
                         "nonwear_filled_bouts": len(filled),
                         "first_pass_changed_wear_minutes": (first_diff & mask).sum()/60,
                         "final_changed_minutes": final_diff.sum()/60,
                         "final_added_lying_minutes": (diagnostic.lying_after_fill & ~reference.lying_after_fill).sum()/60,
                         "final_removed_lying_minutes": (~diagnostic.lying_after_fill & reference.lying_after_fill).sum()/60,
                         "median_abs_angle_difference_deg": angle_delta.abs().median(),
                         "p95_abs_angle_difference_deg": angle_delta.abs().quantile(.95),
                         "tib_candidates": len(night), "valid_tib_nights": int(night.tib_valid.sum()),
                         "mean_candidate_tib_hours": night.tib_duration_h.mean(),
                         "mean_valid_tib_hours": night.loc[night.tib_valid, "tib_duration_h"].mean(),
                         "nap_candidates": len(naps), "nap_candidate_minutes": naps.nap_duration_min.sum(),
                         "legacy_flipped_minutes": diagnostic.legacy_flip.sum()/60})
            night["subject"], night["variant"] = subject, variant
            nights.append(night)
            paired = ref_nights.merge(night, on="day_start", how="outer", suffixes=("_legacy", "_candidate"), indicator=True)
            paired["subject"], paired["variant"] = subject, variant
            for which in ("start", "end"):
                paired[f"tib_{which}_difference_minutes"] = (
                    paired[f"tib_{which}_candidate"]-paired[f"tib_{which}_legacy"]).dt.total_seconds()/60
            paired["tib_duration_difference_minutes"] = (paired.tib_duration_s_candidate-paired.tib_duration_s_legacy)/60
            changes.append(paired)
        original = baseline / "silver" / subject / visit / "McRoberts"
        if (original / "full_run_complete.json").exists():
            checks = {}
            for name in ("posture/lying_bouts.csv", "posture/tib_blocks_merged.csv", "posture/tib_df.csv",
                         "posture/naps_df.csv", "posture/tib_qc_df.csv", "posture/tib_valid_df.csv",
                         "nonwear/nonwear_bouts.csv", "nonwear/nonwear_filled_between_lying.csv"):
                pd.testing.assert_frame_equal(pd.read_csv(original / name), pd.read_csv(reference_root / name))
                checks[name] = True
            validation["legacy_matches_completed_original"][subject] = checks
    recordings = pd.DataFrame(rows)
    recordings.to_csv(output / "recording_comparison.csv", index=False)
    night_frame = pd.concat(nights, ignore_index=True)
    night_frame.to_csv(output / "nightly_comparison.csv", index=False)
    paired_frame = pd.concat(changes, ignore_index=True)
    paired_frame.to_csv(output / "nightly_differences.csv", index=False)
    pd.concat(episode_frames, ignore_index=True).to_csv(output / "mounting_episode_qc.csv", index=False)
    write_json(output / "validation.json", validation)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"legacy": "#6b7280", "proper_legacy_mask": "#c87d25", "uncorrected": "#6378bc", "stable_walking": "#14816f"}
    labels = {"legacy": "Your pipeline", "proper_legacy_mask": "Proper rotation, old mask", "uncorrected": "Uncorrected", "stable_walking": "Stable walking reference"}
    fig, axes = plt.subplots(len(subjects), 2, figsize=(13, 3.8*len(subjects)), squeeze=False)
    for i, subject in enumerate(subjects):
        for variant in VARIANTS:
            data = night_frame.loc[night_frame.subject.eq(subject) & night_frame.variant.eq(variant)]
            axes[i, 0].plot(data.day_start, data.tib_duration_h, marker="o", ms=4, label=labels[variant], color=colors[variant], alpha=.85)
            rejected = data.loc[~data.tib_valid]
            axes[i, 0].scatter(rejected.day_start, rejected.tib_duration_h, marker="x", s=65, color=colors[variant])
        axes[i, 0].set(title=f"{subject} / {visit}: candidate TIB (× fails QC)", ylabel="Hours", xlabel="Noon-to-noon analysis day")
        for variant in VARIANTS[1:]:
            data = paired_frame.loc[paired_frame.subject.eq(subject) & paired_frame.variant.eq(variant)]
            axes[i, 1].plot(data.day_start, data.tib_duration_difference_minutes, marker="o", ms=4, label=labels[variant], color=colors[variant])
        axes[i, 1].axhline(0, color="#aaaaaa", lw=.8)
        axes[i, 1].set(title=f"{subject}: change from your pipeline", ylabel="TIB duration change (minutes)", xlabel="Noon-to-noon analysis day")
        for ax in axes[i]:
            ax.grid(alpha=.18)
            ax.tick_params(axis="x", labelrotation=25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Posture orientation sensitivity — identical wear mask and TIB rules", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "nightly_comparison.png", dpi=160)
    plt.close(fig)
    print(recordings.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gait-comparison", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--participant", action="append", required=True)
    parser.add_argument("--visit", default="T0")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(REPO / "src"))
    args.output_root = args.output_root.resolve()
    if args.worker:
        return worker(args)
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        if args.gait_comparison is None:
            parser.error("A new comparison requires --gait-comparison.")
        gait_root = args.gait_comparison.resolve()
        gait_manifest = json.loads((gait_root / "manifest.json").read_text())
        baseline = Path(gait_manifest["baseline_run"])
        if output == gait_root or gait_root in output.parents or baseline in output.parents:
            raise ValueError("Choose a separate comparison output directory.")
        frozen = baseline / "source_snapshot"
        for relative in ("lower_back/posture/lying_functions.py", "lower_back/utils/bool_to_bouts.py", "lower_back/gait/reorientation_functions.py"):
            assert digest(REPO / "src/dare_wearables" / relative) == digest(frozen / "dare_wearables" / relative), relative
        for name in ("run_time_in_bed_pipeline", "apply_orientation_correction"):
            assert function_ast(REPO / "src/dare_wearables/lower_back/pipeline.py", name) == function_ast(frozen / "dare_wearables/lower_back/pipeline.py", name), name
        inputs = {}
        for subject in args.participant:
            cache = gait_root / subject / "preprocessed.parquet"
            prior = gait_root / "sequence_aligned/silver" / subject / args.visit / "McRoberts/gait/comparison_complete.json"
            expected = json.loads(prior.read_text())["preprocessed_sha256"]
            assert digest(cache) == expected, f"Preprocessing cache changed: {subject}"
            walking = prior.parent / f"{subject}_orientation_qc.csv"
            expected_walking = json.loads(prior.read_text())["outputs"][walking.name]
            assert digest(walking) == expected_walking, f"Walking evidence changed: {subject}"
            inputs[subject] = {"preprocessed_sha256": expected, "walking_qc_sha256": digest(walking)}
        write_json(manifest_path, {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "gait_comparison": str(gait_root), "baseline_run": str(baseline),
                   "gait_manifest_sha256": digest(gait_root / "manifest.json"),
                   "baseline_plan_sha256": digest(baseline / "run_plan.json"),
                   "subjects": args.participant, "visit": args.visit, "variants": VARIANTS,
                   "preprocessing": gait_manifest["preprocessing"], "inputs": inputs,
                   "sources": {name: digest(REPO / name) for name in SOURCES},
                   "comparison_parameters": {"min_sequences": 3, "min_evidence_seconds": 60,
                       "min_sequence_seconds": 10, "weight_cap_seconds": 60,
                       "consensus_angle_deg": 30, "min_consensus_fraction": .8}})
        for name in SOURCES:
            target = output / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, target)
    manifest = json.loads(manifest_path.read_text())
    assert args.participant == manifest["subjects"] and args.visit == manifest["visit"]
    assert all(digest(REPO / name) == value for name, value in manifest["sources"].items()), "Comparison source changed. Use a new output folder."
    baseline = Path(manifest["baseline_run"])
    gait_root = Path(manifest["gait_comparison"])
    plan = json.loads((baseline / "run_plan.json").read_text())
    assert digest(baseline / "run_plan.json") == manifest["baseline_plan_sha256"]
    assert digest(gait_root / "manifest.json") == manifest["gait_manifest_sha256"]
    assert all(metadata.version(name) == version for name, version in plan["dependencies"].items())
    from dare_wearables.common.output_state import signature
    for subject in args.participant:
        source = next(r for r in plan["recordings"] if r["subject"] == subject and r["visit"] == args.visit)
        assert signature({}, [source["source"]]) == source["identity"], "Original input changed."
        walking = gait_root / "sequence_aligned/silver" / subject / args.visit / "McRoberts/gait" / f"{subject}_orientation_qc.csv"
        assert digest(walking) == manifest["inputs"][subject]["walking_qc_sha256"]
    if not args.summarize_only:
        environment = dict(os.environ, PYTHONPATH=str(REPO / "src"), MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
                           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        workers = []
        for subject in args.participant:
            log = output / f"{subject}.log"
            stream = log.open("w")
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--output-root", str(output),
                       "--participant", subject, "--visit", args.visit]
            workers.append((subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT), stream, log))
            print("RUN", subject, "log:", log, flush=True)
        failed = []
        for process, stream, log in workers:
            status = process.wait()
            stream.close()
            if status:
                failed.append(str(log))
        if failed:
            raise RuntimeError(f"Comparison workers failed; inspect {failed}")
    summarize(output, args.participant, args.visit)


if __name__ == "__main__":
    main()

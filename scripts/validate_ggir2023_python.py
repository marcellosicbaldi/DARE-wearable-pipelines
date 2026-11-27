"""Optional R-vs-Python validation on a reproducible random subject sample.

Production never imports this script. Rscript is needed only for this reference
check, which reruns the unchanged, checksum-verified upstream R functions on
exactly the same calibrated full-rate acceleration as the Python detector.
Use --assessment for an existing R-backed nonwear-only assessment. Subjects are
sampled without replacement from its successfully processed recordings; reader
failures are listed as ineligible, and the sample is frozen before processing.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dare_wearables.common.output_state import signature
from dare_wearables.lower_back.nonwear.audit import source_hashes, write_audit
from dare_wearables.lower_back.nonwear.ggir2023 import GGIR_COMMIT, GGIR_SOURCES, NonwearOptions
from dare_wearables.lower_back.nonwear.respiration import RespirationOptions, RespirationSettings
from dare_wearables.lower_back import preprocessing


def _reference_module():
    # Keep the reference bridge independent of the Python implementation.
    import importlib.util
    spec = importlib.util.spec_from_file_location("ggir_reference", PROJECT_ROOT / "scripts/compare_lower_back_nonwear.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.GGIR_COMMIT == GGIR_COMMIT and module.GGIR_SOURCES == GGIR_SOURCES
    return module


def reference_sources(cache=None):
    root = Path(cache).expanduser() if cache else Path.home() / ".cache/dare_wearables/ggir" / GGIR_COMMIT
    return _reference_module()._ggir_sources(root)


def run_reference(acc_df, *, freq=100., sources=None, rscript="Rscript", window_seconds=3600.):
    """Run the independent comparison bridge; no detector port code is reused."""
    executable = shutil.which(rscript)
    if executable is None:
        raise FileNotFoundError("Rscript is required only for this optional reference validation.")
    sources = reference_sources(sources)
    with tempfile.TemporaryDirectory(prefix="dare-ggir-reference-") as temporary:
        folder = Path(temporary)
        binary = folder / "acc.bin"
        with binary.open("wb") as stream:
            for start in range(0, len(acc_df), 1_000_000):
                acc_df.slice(start, 1_000_000).select(["x", "y", "z"]).to_numpy().astype("<f8", copy=False).tofile(stream)
        script = folder / "reference.R"
        script.write_text(_reference_module().R_BRIDGE)
        result = subprocess.run([executable, "--vanilla", str(script), str(sources), str(binary),
                                 str(len(acc_df)), str(freq), str(folder), "13", "50", str(window_seconds)],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Upstream GGIR reference failed: {result.stderr.strip()}")
        return (pd.read_csv(folder / "ggir_2023_matched_windows.csv"),
                pd.read_csv(folder / "ggir_settings.csv").iloc[0].to_dict())


def _bounds(windows):
    flag = windows.is_final_nonwear.to_numpy(dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[False, flag, False]))
    if not len(edges):
        return np.empty((0, 2), dtype=np.int64)
    epochs = edges.reshape(-1, 2)
    return np.column_stack((windows.start_sample.to_numpy()[epochs[:, 0]],
                            windows.stop_sample.to_numpy()[epochs[:, 1] - 1])).astype(np.int64)


def _validate_recording(item, previous, run, options, sources, expected_sources):
    started = time.monotonic()
    source = Path(item["source"])
    if signature({}, [source]) != item["identity"] or source_hashes() != expected_sources:
        raise ValueError("Input or processing source changed since the frozen selection.")
    folder = Path(run) / item["subject"] / item["visit"] / item["sensor"]
    folder.mkdir(parents=True)
    old = Path(previous) / item["subject"] / item["visit"] / item["sensor"] / "nonwear"
    parity = {}
    python_detect = preprocessing.detect_nonwear

    def compare(acc_df, *, freq, options):
        tick = time.monotonic()
        actual = python_detect(acc_df, freq=freq, options=options)
        parity["python_detector_seconds"] = time.monotonic() - tick
        tick = time.monotonic()
        expected, settings = run_reference(acc_df, freq=freq, sources=sources)
        parity["r_reference_seconds_including_io"] = time.monotonic() - tick
        observed = actual.window_df.to_pandas()
        observed.to_parquet(folder / "python_epochs.parquet", index=False)
        expected.to_csv(folder / "r_epochs.csv", index=False)
        parity["epoch_mismatches"] = int(np.any(observed[expected.columns].to_numpy() != expected.to_numpy(), axis=1).sum())
        pd.testing.assert_frame_equal(observed[expected.columns], expected, check_dtype=False, check_exact=True)
        # Also agree with the preserved R-backed assessment, including ns times.
        saved = pd.read_parquet(old / "accelerometer_nonwear_windows.parquet")
        pd.testing.assert_frame_equal(observed, saved, check_dtype=False, check_exact=True)
        times = pd.DatetimeIndex(acc_df["ts"].to_numpy()).as_unit("ns")
        bounds = _bounds(expected)
        reference_bouts = pd.DataFrame({"start": times[bounds[:, 0]], "end": times[bounds[:, 1] - 1]})
        pd.testing.assert_frame_equal(actual.nonwear_df.to_pandas(), reference_bouts, check_exact=True)
        py_mask = ~preprocessing._build_wear_mask_from_nonwear(times, actual.nonwear_df.to_pandas())
        r_mask = np.zeros(len(times), dtype=bool)
        for first, stop in bounds:
            r_mask[first:stop] = True
        parity["sample_mismatches"] = int(np.count_nonzero(py_mask != r_mask))
        np.testing.assert_array_equal(py_mask, r_mask)
        parity.update(n_samples=len(times), n_epochs=len(expected), n_bouts=len(bounds),
                      r_nonwear_samples=int(r_mask.sum()), r_settings=settings,
                      checks="exact epoch scores/flags, full-rate mask, inclusive ns bout bounds, saved R assessment")
        return actual

    with patch.object(preprocessing, "detect_nonwear", side_effect=compare):
        prepared = preprocessing.preprocess_lowerback_file(str(source), verbose=False, respiration=options,
            nonwear=NonwearOptions(window_seconds=3600., end_of_recording_correction=True))
    summary = write_audit(prepared, folder / "nonwear", source=source, options=options)
    for name in ("accelerometer_nonwear_intervals.csv", "respiration_reassigned_intervals.csv", "final_nonwear_intervals.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(folder / "nonwear" / name), pd.read_csv(old / name), check_exact=True)
    pd.testing.assert_frame_equal(prepared["respiration"]["result"].windows,
                                  pd.read_parquet(old / "respiration_windows.parquet"), check_exact=True)
    parity["respiration_features_and_final_bouts_identical_to_saved_r_run"] = True
    if signature({}, [source]) != item["identity"] or source_hashes() != expected_sources:
        raise ValueError("Input or processing source changed during validation.")
    parity.update(subject=item["subject"], visit=item["visit"], status="passed",
                  observed_hours=summary["observed_hours"], baseline_nonwear_hours=summary["baseline_nonwear_hours"],
                  revised_nonwear_hours=summary["revised_nonwear_hours"], elapsed_seconds=time.monotonic() - started)
    (folder / "equivalence.json").write_text(json.dumps(parity, indent=2) + "\n")
    return parity


def validate(assessment, output_root, *, count=10, seed=20261102, workers=2, cache=None):
    if count < 1 or workers < 1:
        raise ValueError("count and workers must be positive")
    previous = Path(assessment).expanduser().resolve()
    manifest = json.loads((previous / "assessment.json").read_text())
    eligible = sorted((item for item in manifest["recordings"] if item["status"] == "completed"),
                      key=lambda item: (item["subject"], item["visit"]))
    if count > len(eligible):
        raise ValueError("Not enough completed recordings for sampling without replacement.")
    if len({item["subject"] for item in eligible}) != len(eligible):
        raise ValueError("Select a single-visit reference assessment to sample unique subjects.")
    for item in eligible:
        detector = item["summary"]["baseline_detector"]
        if detector.get("ggir_commit") != GGIR_COMMIT or "r_version" not in detector:
            raise ValueError("The reference assessment must use the pinned original R implementation.")
    options_dict = dict(manifest["parameters"]["config"]["respiration"])
    options = RespirationOptions(**{**options_dict, "settings": RespirationSettings(**options_dict["settings"])})
    options.validate(100.)
    current_sources = source_hashes()
    for name in ("mcroberts_loader.py", "autocalibrate.py", "respiration.py"):
        if current_sources[name] != manifest["parameters"]["sources"][name]:
            raise ValueError(f"Reference preprocessing or respiration differs: {name}")
    selected = random.Random(seed).sample(eligible, count)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(output_root).expanduser().resolve() / f"python_parity_{stamp}_{uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    sources = reference_sources(cache)
    selection = {"seed": seed, "sampling": "random.Random.sample, without replacement, sorted eligible subjects",
                 "reference_assessment": str(previous), "eligible_count": len(eligible),
                 "eligible_subjects": [item["subject"] for item in eligible],
                 "excluded": [{key: item.get(key) for key in ("subject", "visit", "status", "reason")}
                              for item in manifest["recordings"] if item["status"] != "completed"],
                 "selected": selected, "source_hashes": current_sources,
                 "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "respiration_options": asdict(options)}
    (run / "selection.json").write_text(json.dumps(selection, indent=2, default=str) + "\n")
    snapshot = run / "processing_sources"
    snapshot.mkdir()
    import inspect
    for path in (Path(preprocessing.__file__), Path(inspect.getfile(preprocessing.detect_nonwear))):
        shutil.copy2(path, snapshot)
    shutil.copy2(__file__, snapshot)
    upstream = run / "upstream_ggir_sources"
    upstream.mkdir()
    for name in GGIR_SOURCES:
        shutil.copy2(sources / name, upstream / name)
    print(f"Validation: {run}", flush=True)
    print(f"Selected (seed {seed}): " + ", ".join(item["subject"] for item in selected), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_validate_recording, item, previous, run, options, sources, current_sources): item for item in selected}
        for future in as_completed(futures):
            item = futures[future]
            try:
                row = future.result()
                print(f"PASS {item['subject']}/{item['visit']}: {row['n_epochs']} epochs, {row['n_samples']} samples, "
                      f"{row['sample_mismatches']} mismatches; {row['elapsed_seconds']:.1f} s", flush=True)
            except Exception as error:
                row = {"subject": item["subject"], "visit": item["visit"], "status": "failed", "error": str(error)}
                (run / f"error_{item['subject']}_{item['visit']}.txt").write_text(traceback.format_exc())
                print(f"FAIL {item['subject']}/{item['visit']}: {error}", flush=True)
            rows.append(row)
            pd.DataFrame(rows).drop(columns=["r_settings"], errors="ignore").to_csv(run / "equivalence_summary.csv", index=False)
    result = {"status": "passed" if all(row["status"] == "passed" for row in rows) else "failed",
              "requested": count, "passed": sum(row["status"] == "passed" for row in rows), "recordings": rows}
    (run / "validation.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    passed = [row for row in rows if row["status"] == "passed"]
    lines = ["# GGIR 2023 Python equivalence validation", "",
             f"Result: **{result['passed']}/{count} subjects passed**; seed **{seed}**.", "",
             f"Sampled without replacement from {len(eligible)} completed subjects in the previous R-backed T0 assessment. "
             f"Its {len(manifest['recordings']) - len(eligible)} unsuccessful recordings were ineligible; no replacement subjects "
             "were selected after seeing validation results. See selection.json for the frozen population and input identities.", "",
             f"Reference: unchanged GGIR 3.3-9 R functions, commit `{GGIR_COMMIT}`, matched SD <13 mg / range <50 mg, >=2 axes. "
             "Python and R receive identical full-rate calibrated acceleration. Python retains GGIR's overlapping-window scores, "
             "three postprocessing passes and edge rules. Clipping is compared separately and excluded from nonwear.", "",
             f"Compared **{sum(row['n_epochs'] for row in passed):,} complete 15-minute epochs** and "
             f"**{sum(row['n_samples'] for row in passed):,} full-rate samples** in passed subjects. "
             "Checks require exact equality of all epoch scores/flags, full-rate masks, bout sample boundaries and inclusive nanosecond timestamps. "
             "Respiration features and final bouts must also equal the preserved R-backed assessment (strict 10 < BR <30 brpm, all hours).", "",
             "| Subject | Epochs | Samples | Epoch differences | Sample differences | Nonwear before / after respiration (h) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in sorted(passed, key=lambda row: row["subject"]):
        lines.append(f"| {row['subject']} | {row['n_epochs']} | {row['n_samples']:,} | {row['epoch_mismatches']} | "
                     f"{row['sample_mismatches']} | {row['baseline_nonwear_hours']:.3f} / {row['revised_nonwear_hours']:.3f} |")
    lines.extend(["", "Production now uses only Python; R remains an optional validation dependency. "
                  "Gait, TIB and downstream outputs were not rerun. Equivalence on this sample is not a proof for every possible input "
                  "or a validation of wear accuracy against ground truth.", ""])
    (run / "equivalence_report.md").write_text("\n".join(lines))
    if result["status"] != "passed":
        raise RuntimeError(f"Equivalence validation failed; inspect {run}")
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--source-cache", type=Path)
    args = parser.parse_args()
    validate(args.assessment, args.output_root, count=args.count, seed=args.seed, workers=args.workers, cache=args.source_cache)

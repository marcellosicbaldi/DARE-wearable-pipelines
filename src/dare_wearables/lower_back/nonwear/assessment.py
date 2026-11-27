"""Resumable nonwear-only assessment, isolated from gait/TIB and production silver."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import gc
import json
import multiprocessing
import time
import traceback
import uuid

import pandas as pd

from dare_wearables.common.output_state import signature
from .audit import audit_signature, output_record, source_hashes, write_audit
from .ggir2023 import NonwearOptions, prepare_runtime
from dare_wearables.lower_back.io.timestamp_quality import TimestampOptions


def _json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def inventory(config):
    subjects = (list(map(str, config.participant_ids)) if config.participant_ids else
                sorted(path.name for path in config.bronze_root.iterdir() if path.is_dir()))
    records = []
    for subject in subjects:
        for visit in config.visits:
            folder = config.bronze_root / subject / visit / config.sensor
            files = sorted(p for p in folder.glob("*") if p.suffix.lower() == ".omx")
            item = {"subject": subject, "visit": visit, "sensor": config.sensor}
            if len(files) != 1:
                item.update(status="skipped", reason=f"Expected one OMX; found {len(files)}")
            elif files[0].stat().st_size < config.min_size_bytes:
                item.update(status="skipped", reason="Recording below configured minimum size")
            else:
                source = files[0].resolve()
                item.update(status="pending", source=str(source), identity=signature({}, [source]))
            records.append(item)
    return records


def _process(item, folder, options, expected_sources, nonwear=None, timestamps=None):
    from dare_wearables.lower_back.preprocessing import preprocess_lowerback_file
    started = time.monotonic()
    source = Path(item["source"])
    if source_hashes() != expected_sources:
        raise ValueError("Processing source changed after the assessment started.")
    nonwear = nonwear or NonwearOptions()
    timestamps = timestamps or TimestampOptions()
    initial_key, _ = audit_signature(source, options, nonwear, timestamps)
    if signature({}, [source]) != item["identity"]:
        raise ValueError("Input changed since the frozen inventory was created.")
    prepared = preprocess_lowerback_file(str(source), verbose=False, respiration=options, nonwear=nonwear, timestamps=timestamps)
    if signature({}, [source]) != item["identity"]:
        raise ValueError("Input changed during processing.")
    if audit_signature(source, options, nonwear, timestamps)[0] != initial_key:
        raise ValueError("Processing source or environment changed during analysis.")
    summary = write_audit(prepared, folder, source=source, options=options)
    del prepared
    gc.collect()
    return {name: value for name, value in summary.items() if name != "provenance"} | {
        "elapsed_seconds": time.monotonic() - started}


def _finish(run, items):
    rows = [{**{k: item[k] for k in ("subject", "visit", "status")},
             **item.get("summary", {}), "reason": item.get("reason", "")}
            for item in items]
    table = pd.DataFrame(rows)
    table.to_csv(run / "cohort_summary.csv", index=False)
    completed = table.loc[table.status.eq("completed")]
    if not completed.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        axes[0].scatter(completed.baseline_nonwear_hours, completed.revised_nonwear_hours,
                        s=18, alpha=.7)
        upper = max(float(completed.baseline_nonwear_hours.max()), 1)
        axes[0].plot([0, upper], [0, upper], "--", color="grey")
        axes[0].set(xlabel="Original nonwear (h)", ylabel="Remaining nonwear (h)")
        axes[1].scatter(completed.reassigned_to_wear_hours, completed.reassigned_09_21_hours,
                        s=18, alpha=.7)
        axes[1].set(xlabel="Total reassigned to wear (h)", ylabel="Reassigned 09:00–21:00 (h)")
        fig.savefig(run / "cohort_qc.png", dpi=160)
        plt.close(fig)
    return table


def run_assessment(config, *, resume=None, inventory_only=False, workers=1):
    """Process the configured recordings once, with per-recording completion hashes.

    This deliberately ignores gait/TIB switches: it exports masks/features only.
    Resume requires the same configuration, source revision and frozen inputs.
    """
    if not config.respiration.enabled:
        raise ValueError("Enable lower_back.respiration for this assessment.")
    config.respiration.validate(100)
    config.nonwear.validate()
    config.timestamps.validate()
    if workers < 1:
        raise ValueError("workers must be positive")
    parameters = {"config": asdict(config), "sources": source_hashes()}
    key = signature(parameters, [])
    if resume is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run = config.silver_root / f"assessment_{stamp}_{uuid.uuid4().hex[:8]}"
        run.mkdir(parents=True, exist_ok=False)
        manifest = {"signature": key, "parameters": parameters, "status": "inventoried",
                    "recordings": inventory(config)}
    else:
        run = Path(resume).expanduser().resolve()
        manifest = json.loads((run / "assessment.json").read_text())
        if manifest["signature"] != key:
            raise ValueError("Configuration or processing source changed; create a new assessment.")
    items = manifest["recordings"]
    _json(run / "assessment.json", manifest)
    print(f"Assessment: {run}", flush=True)
    print(f"Inventory: {len(items)} recordings; {sum(i['status'] == 'skipped' for i in items)} skipped", flush=True)
    if inventory_only:
        return run
    prepare_runtime(config.nonwear)
    pending = []
    for item in items:
        if item["status"] == "skipped":
            continue
        source = Path(item["source"])
        folder = run / item["subject"] / item["visit"] / item["sensor"] / "nonwear"
        record_state = output_record(folder, config.respiration)
        try:
            key, _ = audit_signature(source, config.respiration, config.nonwear, config.timestamps)
            reusable = signature({}, [source]) == item["identity"] and record_state.reusable(key)
        except OSError:
            reusable = False
        if reusable:
            item["status"] = "completed"
            item.pop("reason", None)
            item["summary"] = json.loads((folder / "nonwear_summary.json").read_text())
            item["summary"].pop("provenance", None)
            print(f"Reused {item['subject']}/{item['visit']}", flush=True)
        else:
            record_state.clear()
            item.pop("summary", None)
            item["status"] = "pending"
            pending.append((item, folder))
    manifest["status"] = "running"
    _json(run / "assessment.json", manifest)

    def record(item, summary=None, error=None):
        if error is None:
            item.update(status="completed", summary=summary)
            item.pop("reason", None)
            print(f"Completed {item['subject']}/{item['visit']}: "
                  f"{summary['baseline_nonwear_hours']:.2f} → {summary['revised_nonwear_hours']:.2f} h; "
                  f"{summary.get('elapsed_seconds', 0):.1f} s", flush=True)
        else:
            item.update(status="failed", reason=str(error))
            item.pop("summary", None)
            print(f"FAILED {item['subject']}/{item['visit']}: {error}", flush=True)
            (run / f"error_{item['subject']}_{item['visit']}.txt").write_text(traceback.format_exc())
        _json(run / "assessment.json", manifest)

    if workers == 1:
        for item, folder in pending:
            try:
                record(item, _process(item, folder, config.respiration, parameters["sources"], config.nonwear, config.timestamps))
            except Exception as error:
                record(item, error=error)
    else:
        # spawn avoids inheriting a full raw recording in child processes.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_process, item, folder, config.respiration, parameters["sources"], config.nonwear, config.timestamps): item
                       for item, folder in pending}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record(item, future.result())
                except Exception as error:
                    record(item, error=error)
    _finish(run, items)
    manifest["status"] = "completed_with_failures" if any(i["status"] == "failed" for i in items) else "completed"
    _json(run / "assessment.json", manifest)
    return run

"""Isolated, resumable full lower-back runs with a frozen input inventory.

Each recording runs in a separate Python process. Completion hashes cover gait,
posture, time-in-bed and nonwear outputs together, rather than nonwear alone.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import fcntl
from importlib import metadata, util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

import pandas as pd

from dare_wearables.common.output_state import _digest, signature
from dare_wearables.lower_back.annotation.data import atomic_json
from dare_wearables.lower_back.io.timestamp_quality import TimestampOptions
from dare_wearables.lower_back.nonwear.ggir2023 import NonwearOptions
from dare_wearables.lower_back.nonwear.respiration import RespirationOptions, RespirationSettings
from dare_wearables.lower_back.orientation import OrientationOptions


@dataclass
class _RunConfig:
    bronze_root: Path
    silver_root: Path
    redcap_csv: Path | None
    sensor: str
    visits: list[str]
    min_size_bytes: int
    participant_ids: list[str] | None
    run_gait: bool
    run_time_in_bed: bool
    save_omx_to_parquet: bool
    protocol_hours: float | None
    respiration: RespirationOptions
    nonwear: NonwearOptions
    timestamps: TimestampOptions
    orientation: OrientationOptions


def _decode(values):
    values = dict(values)
    for name in ("bronze_root", "silver_root", "redcap_csv"):
        values[name] = Path(values[name]) if values[name] is not None else None
    respiration = dict(values["respiration"])
    settings = dict(respiration.pop("settings"))
    for name in ("band_hz", "reference_band_hz"):
        settings[name] = tuple(settings[name])
    values["respiration"] = RespirationOptions(**respiration, settings=RespirationSettings(**settings))
    values["nonwear"] = NonwearOptions(**values["nonwear"])
    values["timestamps"] = TimestampOptions(**values["timestamps"])
    values["orientation"] = OrientationOptions.from_mapping(values.get("orientation"))
    return _RunConfig(**values)


def _source_files():
    files = {}
    for package in ("dare_wearables", "fallspredict_gp_pipeline", "fallspredict_pipeline"):
        spec = util.find_spec(package)
        if spec is None:
            continue
        root = Path(next(iter(spec.submodule_search_locations)))
        for path in sorted(root.rglob("*.py")):
            files[str(Path(package)/path.relative_to(root))] = path
    return files


def source_hashes():
    return {name: _digest(path) for name, path in _source_files().items()}


def dependencies():
    return {name: metadata.version(name) for name in
            ("DARE-wearable-pipelines", "numpy", "pandas", "polars", "scipy", "mobgap", "scikit-learn", "pyarrow")}


def _git_state():
    result = {}
    for name, args in (("branch", ["branch", "--show-current"]), ("commit", ["rev-parse", "HEAD"]),
                       ("working_tree", ["status", "--short"])):
        call = subprocess.run(["git", "-C", str(Path(__file__).parent), *args], capture_output=True, text=True)
        if call.returncode == 0:
            result[name] = call.stdout.strip()
    return result


def _validate(config):
    config.nonwear.validate()
    config.respiration.validate(100.)
    config.timestamps.validate()
    config.orientation.validate()
    if not config.respiration.enabled:
        raise ValueError("Enable respiration for this full nonwear-profile run.")
    if not config.run_gait or not config.run_time_in_bed:
        raise ValueError("The full-run launcher requires both gait and time-in-bed enabled.")
    if not config.bronze_root.is_dir():
        raise FileNotFoundError(f"Bronze root is unavailable: {config.bronze_root}")
    if config.redcap_csv is None or not config.redcap_csv.is_file():
        raise FileNotFoundError("A readable redcap_csv is required for gait metadata.")
    if not config.visits or len(set(config.visits)) != len(config.visits) or any(not re.fullmatch(r"T\d+", v) for v in config.visits):
        raise ValueError("Choose distinct protocol visits such as T0 and T1.")
    if config.sensor != "McRoberts" or config.min_size_bytes < 0:
        raise ValueError("Choose McRoberts and a nonnegative minimum recording size.")
    if config.protocol_hours is not None and (isinstance(config.protocol_hours, bool) or not 0 < config.protocol_hours <= 168):
        raise ValueError("protocol_hours must lie in (0, 168].")
    bronze, output = config.bronze_root.resolve(), config.silver_root.resolve()
    if bronze == output or bronze in output.parents or output in bronze.parents:
        raise ValueError("Choose an output root separate from bronze and its parents.")


def _inventory(config):
    subjects = config.participant_ids or sorted(p.name for p in config.bronze_root.iterdir()
                                               if p.is_dir() and re.fullmatch(r"\d+", p.name))
    if len(subjects) != len(set(subjects)) or any(not re.fullmatch(r"\d+", s) for s in subjects):
        raise ValueError("Participant IDs must be distinct numeric strings.")
    rows = []
    for subject in subjects:
        for visit in config.visits:
            folder = config.bronze_root / subject / visit / config.sensor
            files = sorted(p for p in folder.glob("*") if p.is_file() and p.suffix.lower() == ".omx")
            row = {"subject": subject, "visit": visit, "sensor": config.sensor}
            if len(files) != 1:
                row.update(status="skipped", reason=f"Expected one OMX; found {len(files)}")
            elif files[0].stat().st_size < config.min_size_bytes:
                row.update(status="skipped", reason="Recording below configured minimum size")
            else:
                row.update(status="pending", source=str(files[0].resolve()), identity=signature({}, [files[0]]))
            rows.append(row)
    if not any(r["status"] == "pending" for r in rows):
        raise ValueError("No eligible recordings in the requested inventory.")
    return rows


def prepare(config):
    _validate(config)
    records = _inventory(config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = config.silver_root / f"full_run_{stamp}_{uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    plan = {"schema": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": json.loads(json.dumps(asdict(config), default=str)),
            "recordings": records, "sources": source_hashes(), "dependencies": dependencies(),
            "python": list(sys.version_info[:3]), "git": _git_state(),
            "metadata_identity": signature({}, [config.redcap_csv]), "metadata_sha256": _digest(config.redcap_csv),
            "scope": "All eligible requested lower-back recordings; nonwear + gait + posture/time-in-bed. No benchmark labels injected."}
    atomic_json(run/"run_plan.json", plan)
    (run/"run_plan.sha256").write_text(_digest(run/"run_plan.json")+"\n")
    pd.DataFrame(records).to_csv(run/"recording_inventory.csv", index=False)
    for name, path in _source_files().items():
        target = run/"source_snapshot"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    atomic_json(run/"run_status.json", {"status": "inventoried", "recordings": records})
    print(f"Run folder: {run}", flush=True)
    print(f"Eligible: {sum(r['status']=='pending' for r in records)}; skipped: {sum(r['status']=='skipped' for r in records)}", flush=True)
    return run


def _plan(run):
    if _digest(run/"run_plan.json") != (run/"run_plan.sha256").read_text().strip():
        raise ValueError("Frozen run plan changed.")
    plan = json.loads((run/"run_plan.json").read_text())
    if plan["sources"] != source_hashes() or plan["dependencies"] != dependencies() or plan["python"] != list(sys.version_info[:3]):
        raise ValueError("Processing code or environment changed; restore the frozen revision or create a new run.")
    config = _decode(plan["config"])
    _validate(config)
    if signature({}, [config.redcap_csv]) != plan["metadata_identity"] or _digest(config.redcap_csv) != plan["metadata_sha256"]:
        raise ValueError("REDCap metadata changed after inventory.")
    return plan, config


def _folder(run, item):
    target = run/"silver"/item["subject"]/item["visit"]/item["sensor"]
    if not target.resolve().is_relative_to((run/"silver").resolve()):
        raise ValueError("Invalid recording output path.")
    return target


def _required(folder, item, save_parquet):
    names = ["nonwear/respiration_manifest.json", "nonwear/final_nonwear_intervals.csv",
             f"gait/{item['subject']}_day_aggregation.csv", f"gait/{item['subject']}_gait_status.csv",
             "posture/tib_df.csv", "posture/tib_valid_df.csv", "posture/tib_qc_df.csv", "posture/lying_bouts.csv"]
    names += ["orientation/inversion_summary.json", "orientation/inversion_episodes.csv", "orientation/walking_evidence.csv"]
    if save_parquet:
        names.append(f"{item['subject']}_{item['visit']}_preprocessed.parq")
    for name in names:
        if not (folder/name).is_file():
            raise ValueError(f"Full pipeline did not publish required output {name}")


def reusable(run, item):
    folder = _folder(run, item)
    try:
        completion = json.loads((folder/"full_run_complete.json").read_text())
        if completion["plan_sha256"] != _digest(run/"run_plan.json") or completion["input_identity"] != signature({}, [item["source"]]):
            return False
        outputs = {str(p.relative_to(folder)): _digest(p) for p in folder.rglob("*")
                   if p.is_file() and p.name != "full_run_complete.json"}
        return outputs == completion["outputs"] and bool(outputs)
    except (OSError, ValueError, KeyError):
        return False


def process_record(run, index):
    """Internal worker; called only by an explicit execution command."""
    run = Path(run).resolve()
    plan, config = _plan(run)
    item = plan["recordings"][index]
    if item["status"] != "pending" or signature({}, [item["source"]]) != item["identity"]:
        raise ValueError("Recording is ineligible or changed since inventory.")
    folder = _folder(run, item)
    if folder.exists():
        shutil.rmtree(folder)
    from dare_wearables.lower_back.pipeline import run_from_config
    worker_config = replace(config, silver_root=run/"silver", participant_ids=[item["subject"]], visits=[item["visit"]])
    run_from_config(worker_config)
    _plan(run)  # Recheck code, environment and metadata after all stages finish.
    if signature({}, [item["source"]]) != item["identity"]:
        raise ValueError("Raw recording changed during processing.")
    _required(folder, item, config.save_omx_to_parquet)
    from dare_wearables.lower_back.nonwear.audit import audit_signature, output_record
    key, _ = audit_signature(item["source"], config.respiration, config.nonwear, config.timestamps,
                             protocol_hours=config.protocol_hours)
    if not output_record(folder/"nonwear", config.respiration).reusable(key):
        raise ValueError("Nonwear audit completion failed verification.")
    atomic_json(folder/"full_run_complete.json", {"plan_sha256": _digest(run/"run_plan.json"),
        "input_identity": item["identity"], "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {str(p.relative_to(folder)): _digest(p) for p in folder.rglob("*") if p.is_file()}})


def finish(run, config, records):
    rows, tib = [], []
    for item in records:
        row = {k: item.get(k, "") for k in ("subject", "visit", "status", "reason")}
        if item["status"] == "completed":
            folder = _folder(run, item)
            summary = json.loads((folder/"nonwear/nonwear_summary.json").read_text())
            for key in ("observed_hours", "baseline_nonwear_hours", "revised_nonwear_hours",
                        "respiration_reassigned_to_wear_hours", "short_nonwear_reassigned_to_wear_hours"):
                row[key] = summary.get(key)
            orientation = json.loads((folder/"orientation/inversion_summary.json").read_text())
            row["orientation_inverted_wear_hours"] = orientation["inverted_wear_hours"]
            row["orientation_uncertain_wear_hours"] = orientation["uncertain_wear_hours"]
            nights = pd.read_csv(folder/"posture/tib_valid_df.csv")
            nights["subject"], nights["visit"] = item["subject"], item["visit"]
            tib.append(nights)
        rows.append(row)
    pd.DataFrame(rows).to_csv(run/"cohort_summary.csv", index=False)
    output = run/"aggregation"
    output.mkdir(exist_ok=True)
    if any(i["status"] == "completed" for i in records):
        from dare_wearables.aggregation.gait import write_gait_exports
        for visit in config.visits:
            write_gait_exports(run/"silver", visit=visit, output_dir=output, sensor=config.sensor,
                               min_valid_hours=16., allow_legacy_hours=False)
    if tib:
        pd.concat(tib, ignore_index=True).to_csv(output/"lower_back_tib_valid_all.csv", index=False)


def execute(run):
    run = Path(run).resolve()
    with (run/".lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This run is already executing in another process.") from exc
        plan, config = _plan(run)
        state = json.loads((run/"run_status.json").read_text())
        records = state["recordings"]
        state["status"] = "running"
        atomic_json(run/"run_status.json", state)
        for index, frozen in enumerate(plan["recordings"]):
            item = records[index]
            if frozen["status"] == "skipped":
                continue
            _plan(run)
            if signature({}, [frozen["source"]]) != frozen["identity"]:
                raise ValueError(f"Input changed since inventory: {frozen['subject']}/{frozen['visit']}")
            if reusable(run, frozen):
                item.update(status="completed", reason="")
                print(f"Reused {item['subject']}/{item['visit']}", flush=True)
            else:
                log = run/"logs"/f"{item['subject']}_{item['visit']}.log"
                log.parent.mkdir(exist_ok=True)
                item.update(status="running", reason="", log=str(log))
                atomic_json(run/"run_status.json", state)
                print(f"Processing {item['subject']}/{item['visit']} — {log}", flush=True)
                environment = dict(os.environ, MPLBACKEND="Agg")
                with log.open("a") as stream:
                    result = subprocess.run([sys.executable, "-u", "-m", "dare_wearables.lower_back.batch",
                        "--run-record", str(run), "--record-index", str(index)], stdout=stream,
                        stderr=subprocess.STDOUT, env=environment)
                if result.returncode == 0 and reusable(run, frozen):
                    item.update(status="completed", reason="")
                    print(f"Completed {item['subject']}/{item['visit']}", flush=True)
                else:
                    item.update(status="failed", reason=f"Worker exit {result.returncode}; see {log.name}")
                    # Keep logs, exclude incomplete stage outputs from aggregation.
                    folder = _folder(run, frozen)
                    if folder.exists():
                        shutil.rmtree(folder)
                    print(f"FAILED {item['subject']}/{item['visit']} — see log", flush=True)
            atomic_json(run/"run_status.json", state)
        finish(run, config, records)
        state["status"] = "completed_with_failures" if any(r["status"] == "failed" for r in records) else "completed"
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(run/"run_status.json", state)
        print(f"{state['status']}: {run}", flush=True)
        return not any(r["status"] == "failed" for r in records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal worker for a frozen lower-back full run.")
    parser.add_argument("--run-record", required=True, type=Path)
    parser.add_argument("--record-index", required=True, type=int)
    arguments = parser.parse_args()
    process_record(arguments.run_record, arguments.record_index)

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from dare_wearables.common.recording import continuous_intersections
from dare_wearables.common.output_state import OutputRun, signature, clear_outputs_on_failure

from dare_wearables.wrist.heart_rate.config import HeartRateConfig
from dare_wearables.wrist.heart_rate.io import find_empatica_ppg_acc_files, load_acc, load_ppg


def build_heart_rate_output_dir(
    output_root: str | Path,
    *,
    participant: str,
    visit: str,
    sensor: str,
    output_subdir: str = "beliefppg",
) -> Path:
    return Path(output_root) / str(participant) / str(visit) / str(sensor) / output_subdir


def _split_recording_portions(
    acc_df: pd.DataFrame,
    ppg_df: pd.DataFrame,
    *,
    empty_gap_seconds: float = 1.0,
    min_good_portion_minutes: float = 10.0,
    acc_frequency: float | None = None,
    ppg_frequency: float | None = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    portions = continuous_intersections(acc_df, ppg_df, gap_seconds=empty_gap_seconds,
                                        min_seconds=60 * min_good_portion_minutes,
                                        acc_frequency=acc_frequency, ppg_frequency=ppg_frequency)
    return [a for a, _ in portions], [p for _, p in portions]


@clear_outputs_on_failure({"output_subdir": ["hr_belief.csv", "hr_completion.json"]})
def run_heart_rate_pipeline(
    *,
    input_root: str | Path,
    output_root: str | Path,
    participant: str,
    visit: str = "T0",
    sensor: str = "Empatica",
    output_subdir: str = "beliefppg",
    ppg_frequency: int = 64,
    acc_frequency: int = 64,
    empty_gap_seconds: float = 1.0,
    min_good_portion_minutes: float = 10.0,
    min_inference_minutes: float = 5.0,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Run the BeliefPPG heart-rate workflow for one participant/visit."""
    if min_inference_minutes <= 0 or ppg_frequency <= 0 or acc_frequency <= 0:
        raise ValueError("Inference duration and sampling frequencies must be positive.")
    parameters = {k: v for k, v in locals().items() if k != "skip_existing"}
    output_dir = build_heart_rate_output_dir(
        output_root,
        participant=participant,
        visit=visit,
        sensor=sensor,
        output_subdir=output_subdir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hr_belief.csv"

    files = find_empatica_ppg_acc_files(
        input_root,
        participant=participant,
        visit=visit,
        sensor=sensor,
    )
    run = OutputRun(output_dir / "hr_completion.json", [output_path])
    key = signature(parameters, [files["ppg"], files["acc"]])
    if skip_existing and run.reusable(key):
        return {
            "participant": participant,
            "visit": visit,
            "status": "skipped",
            "reason": "BeliefPPG heart-rate output already exists",
            "output_dir": output_dir,
            "output_path": output_path,
        }

    run.clear()

    if files["ppg"] is None or files["acc"] is None:
        return {
            "participant": participant,
            "visit": visit,
            "status": "skipped",
            "reason": "Missing ppg.parquet or acc.parquet",
            "files": files,
            "output_dir": output_dir,
        }

    try:
        from beliefppg import infer_hr_uncertainty
    except ModuleNotFoundError as exc:
        if exc.name == "beliefppg":
            raise ModuleNotFoundError(
                "Heart-rate inference requires the heart-rate extra. "
                "From the repository, run: uv sync --locked --extra heart-rate"
            ) from exc
        raise

    ppg_df = load_ppg(files["ppg"])
    acc_df = load_acc(files["acc"])
    if ppg_df.shape[1] != 1 or acc_df.shape[1] != 3:
        raise ValueError("Heart-rate inference requires one PPG column and three accelerometer axes.")

    acc_df_portions, ppg_df_portions = _split_recording_portions(
        acc_df,
        ppg_df,
        empty_gap_seconds=empty_gap_seconds,
        acc_frequency=acc_frequency,
        ppg_frequency=ppg_frequency,
        min_good_portion_minutes=min_good_portion_minutes,
    )

    hr_all = []
    time_hr_all = []
    uncertainty_all = []
    min_samples = int(acc_frequency * 60 * min_inference_minutes)

    for acc, ppg in zip(acc_df_portions, ppg_df_portions):
        if len(acc) < min_samples or len(ppg) < int(ppg_frequency * 60 * min_inference_minutes):
            continue

        time = acc.index
        hr, uncertainty, time_intervals = infer_hr_uncertainty(
            ppg=ppg.values.reshape(-1, 1),
            ppg_freq=ppg_frequency,
            acc=acc.values,
            acc_freq=acc_frequency,
        )
        if len(hr) == 0:
            continue
        if len(hr) != len(uncertainty) or len(hr) != len(time_intervals):
            raise ValueError("BeliefPPG returned inconsistent output lengths.")
        # Inference interval bounds are seconds relative to the portion's origin.
        times = np.mean(time_intervals, axis=-1)
        if not np.isfinite(times).all() or (times < 0).any() or (times > (time[-1]-time[0]).total_seconds()).any():
            raise ValueError("BeliefPPG returned timestamps outside the recording portion.")
        hr_all.append(hr)
        uncertainty_all.append(uncertainty)
        time_hr_all.append(time[0] + pd.to_timedelta(times, unit="s"))

    if len(hr_all) == 0:
        empty_output = pd.DataFrame(columns=["hr", "uncertainty"])
        empty_output.to_csv(output_path)
        return {
            "participant": participant,
            "visit": visit,
            "status": "skipped",
            "reason": "No valid PPG/ACC portions long enough for BeliefPPG inference",
            "output_dir": output_dir,
            "output_path": output_path,
            "n_portions": len(acc_df_portions),
            "n_valid_portions": 0,
        }

    hr_belief = np.concatenate(hr_all)
    uncertainty_belief = np.concatenate(uncertainty_all)
    t_hr_belief = np.concatenate(time_hr_all)

    hr_belief_df = pd.DataFrame({
        "hr": hr_belief,
        "uncertainty": uncertainty_belief,
    }, index=t_hr_belief)

    hr_belief_df.to_csv(output_path)
    run.complete(key)

    return {
        "participant": participant,
        "visit": visit,
        "status": "completed",
        "output_dir": output_dir,
        "output_path": output_path,
        "n_portions": len(acc_df_portions),
        "n_valid_portions": len(hr_all),
        "n_heart_rate_samples": len(hr_belief_df),
    }


def run_heart_rate_from_config(
    config: HeartRateConfig,
    *,
    participant: str | None = None,
    visit: str | None = None,
) -> dict[str, Any]:
    participant_id = participant or config.participant
    if participant_id is None:
        raise ValueError("A participant must be provided via the config or function call.")

    visits = [visit] if visit is not None else config.visits
    if len(visits) != 1:
        raise ValueError("Use `run_heart_rate_batch_from_config` when processing multiple visits.")

    return run_heart_rate_pipeline(
        input_root=config.input_root,
        output_root=config.output_root,
        participant=participant_id,
        visit=visits[0],
        sensor=config.sensor,
        output_subdir=config.output_subdir,
        ppg_frequency=config.ppg_frequency,
        acc_frequency=config.acc_frequency,
        empty_gap_seconds=config.empty_gap_seconds,
        min_good_portion_minutes=config.min_good_portion_minutes,
        min_inference_minutes=config.min_inference_minutes,
        skip_existing=config.skip_existing,
    )


def run_heart_rate_batch_from_config(
    config: HeartRateConfig,
    *,
    participants: Iterable[str] | None = None,
    visits: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    participant_ids = list(participants) if participants is not None else None
    if participant_ids is None:
        participant_ids = [p.name for p in Path(config.input_root).iterdir() if not p.name.startswith(".")]
        participant_ids = sorted(participant_ids)

    visit_ids = list(visits) if visits is not None else config.visits
    results = []
    for visit in visit_ids:
        for participant in participant_ids:
            results.append(
                run_heart_rate_pipeline(
                    input_root=config.input_root,
                    output_root=config.output_root,
                    participant=str(participant),
                    visit=str(visit),
                    sensor=config.sensor,
                    output_subdir=config.output_subdir,
                    ppg_frequency=config.ppg_frequency,
                    acc_frequency=config.acc_frequency,
                    empty_gap_seconds=config.empty_gap_seconds,
                    min_good_portion_minutes=config.min_good_portion_minutes,
                    min_inference_minutes=config.min_inference_minutes,
                    skip_existing=config.skip_existing,
                )
            )
    return results

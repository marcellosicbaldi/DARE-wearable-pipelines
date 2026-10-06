from __future__ import annotations

from pathlib import Path

from ..config import EmpaticaSleepConfig
from .sleep.sleep_pipeline_gp import (
    run_sleep_and_circadian_pipeline_gp_from_silver,
    run_sleep_pipeline_gp_from_silver,
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="openpyxl")

def build_output_dir(
    output_root: str | Path,
    *,
    participant: str,
    visit: str,
    sensor: str,
) -> Path:
    return Path(output_root) / str(participant) / str(visit) / str(sensor) / "sleep_circadian"


def run_sleep_pipeline_from_config(
    config: EmpaticaSleepConfig,
    *,
    participant: str | None = None,
    visit: str | None = None,
) -> dict[str, object]:
    participant_id = participant or config.participant
    if participant_id is None:
        raise ValueError("A participant must be provided via the config or function call.")

    visit_id = visit or config.visit
    output_dir = build_output_dir(
        config.output_root,
        participant=participant_id,
        visit=visit_id,
        sensor=config.sensor,
    )

    return run_sleep_pipeline_gp_from_silver(
        silver_root=config.silver_root,
        participant=participant_id,
        visit=visit_id,
        sensor=config.sensor,
        parquet_engine=config.parquet_engine,
        recruitment_tracker_path=config.recruitment_tracker_path,
        nonwear_method=config.nonwear_method,
        timezone=config.timezone,
        diary_csv_path=config.diary_csv_path,
        lb_tib_csv_path=config.lb_tib_csv_path,
        return_epoch_outputs=config.return_epoch_outputs,
        return_intermediates=config.return_intermediates,
        save_folder=output_dir,
    )


def run_sleep_and_circadian_from_config(
    config: EmpaticaSleepConfig,
    *,
    participant: str | None = None,
    visit: str | None = None,
) -> dict[str, object]:
    participant_id = participant or config.participant
    if participant_id is None:
        raise ValueError("A participant must be provided via the config or function call.")

    visit_id = visit or config.visit
    output_dir = build_output_dir(
        config.output_root,
        participant=participant_id,
        visit=visit_id,
        sensor=config.sensor,
    )

    return run_sleep_and_circadian_pipeline_gp_from_silver(
        silver_root=config.silver_root,
        participant=participant_id,
        recruitment_tracker_path=config.recruitment_tracker_path,
        visit=visit_id,
        sensor=config.sensor,
        parquet_engine=config.parquet_engine,
        nonwear_method=config.nonwear_method,
        timezone=config.timezone,
        diary_csv_path=config.diary_csv_path,
        lb_tib_csv_path=config.lb_tib_csv_path,
        return_epoch_outputs=config.return_epoch_outputs,
        return_intermediates=config.return_intermediates,
        save_folder=output_dir,
    )

from __future__ import annotations

from pathlib import Path

from ..config import RavennaSleepConfig
from .sleep.sleep_pipeline_ravenna import (
    run_sleep_and_circadian_pipeline_ravenna_from_input_root,
    run_sleep_pipeline_ravenna_from_input_root,
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
    config: RavennaSleepConfig,
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

    return run_sleep_pipeline_ravenna_from_input_root(
        input_root=config.input_root,
        participant=participant_id,
        visit=visit_id,
        sensor=config.sensor,
        geneactiv_file_path=config.geneactiv_file_path,
        require_subject_id_prefix=config.require_subject_id_prefix,
        nonwear_method=config.nonwear_method,
        timezone=config.timezone,
        diary_csv_path=config.diary_csv_path,
        lb_tib_csv_path=config.lb_tib_csv_path,
        return_epoch_outputs=config.return_epoch_outputs,
        return_intermediates=config.return_intermediates,
        save_folder=output_dir,
    )


def run_sleep_and_circadian_from_config(
    config: RavennaSleepConfig,
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

    return run_sleep_and_circadian_pipeline_ravenna_from_input_root(
        input_root=config.input_root,
        participant=participant_id,
        visit=visit_id,
        sensor=config.sensor,
        geneactiv_file_path=config.geneactiv_file_path,
        require_subject_id_prefix=config.require_subject_id_prefix,
        nonwear_method=config.nonwear_method,
        timezone=config.timezone,
        diary_csv_path=config.diary_csv_path,
        lb_tib_csv_path=config.lb_tib_csv_path,
        return_epoch_outputs=config.return_epoch_outputs,
        return_intermediates=config.return_intermediates,
        save_folder=output_dir,
    )

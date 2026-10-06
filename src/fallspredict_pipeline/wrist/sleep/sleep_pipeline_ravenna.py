from __future__ import annotations

from ...common.output_state import fresh_outputs

import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..circadian.activity_intensity_gp import (
    run_activity_intensity_pipeline_gp_from_preprocessed,
    write_activity_intensity_outputs,
)
from ..circadian.circadian_pipeline_gp import run_circadian_pipeline_gp_from_preprocessed
from ..circadian.geneactiv_preprocessing import (
    find_geneactiv_bin_file,
    preprocess_geneactiv_recording,
)
from dare_wearables.wrist.sleep.pipeline import (
    run_wrist_from_preprocessed,
    _is_min_valid_days_error,
    _select_activity_sleep_windows,
    _skipped_stage_reason,
    _skipped_stage_result,
    run_sleep_pipeline_gp_from_preprocessed,
)


def _infer_participant_from_geneactiv_path(geneactiv_bin_path: str | Path) -> str:
    path = Path(geneactiv_bin_path)
    digit_tokens = ["".join(ch for ch in part if ch.isdigit()) for part in path.parts]
    digit_tokens = [token for token in digit_tokens if token]
    if digit_tokens:
        return digit_tokens[-1].zfill(4) if len(digit_tokens[-1]) <= 4 else digit_tokens[-1]
    raise ValueError(f"Could not infer participant from path: {geneactiv_bin_path}")


@fresh_outputs(["sleep_*.csv", "recording_info.csv"])
def run_sleep_pipeline_ravenna(
    geneactiv_bin_path: str | Path,
    participant: Optional[str] = None,
    visit: str = "T0",
    nonwear_method: str = "vanhees2013",
    timezone: str = "Europe/Rome",
    diary_csv_path: Optional[str | Path] = None,
    diary_subject_col: str | None = None,
    diary_start_col: str | None = None,
    diary_end_col: str | None = None,
    diary_visit_col: str | None = None,
    diary_night_id_col: str | None = None,
    lb_tib_csv_path: Optional[str | Path] = None,
    lb_subject_col: str | None = None,
    lb_start_col: str | None = None,
    lb_end_col: str | None = None,
    lb_visit_col: str | None = None,
    lb_night_id_col: str | None = None,
    sib_change_thresh_deg: float = 5.0,
    sib_min_bout_minutes: int = 5,
    haspt_ignore_invalid: float | bool = False,
    hdcza_threshold: float | tuple[float, float] | list[float] | None = None,
    spt_min_block_dur_minutes: int = 30,
    spt_max_gap_dur_minutes: int = 60,
    spt_max_gap_ratio: float = 1.0,
    rerun_with_6pm_window: bool = True,
    return_epoch_outputs: bool = True,
    return_intermediates: bool = False,
    save_folder: Optional[str | Path] = None,
) -> Dict[str, object]:
    """
    DARE-FALLSPREDICT / GENEActiv sleep pipeline.

    This is the GP sleep workflow with the wrist preprocessing swapped from
    Empatica parquet + DETACH to GENEActiv .bin + Van Hees/GGIR non-wear.
    """
    if participant is None:
        participant = _infer_participant_from_geneactiv_path(geneactiv_bin_path)

    preprocessed = preprocess_geneactiv_recording(
        geneactiv_bin_path=geneactiv_bin_path,
        nonwear_method=nonwear_method,
    )
    return run_sleep_pipeline_gp_from_preprocessed(
        calibrated_df=preprocessed["calibrated_df"],
        acc_df=preprocessed["acc_df"],
        temp_df=preprocessed["temp_df"],
        info=preprocessed["info"],
        nonwear_df=preprocessed["nonwear_df"],
        participant=participant,
        visit=visit,
        timezone=timezone,
        diary_csv_path=diary_csv_path,
        diary_subject_col=diary_subject_col,
        diary_start_col=diary_start_col,
        diary_end_col=diary_end_col,
        diary_visit_col=diary_visit_col,
        diary_night_id_col=diary_night_id_col,
        lb_tib_csv_path=lb_tib_csv_path,
        lb_subject_col=lb_subject_col,
        lb_start_col=lb_start_col,
        lb_end_col=lb_end_col,
        lb_visit_col=lb_visit_col,
        lb_night_id_col=lb_night_id_col,
        sib_change_thresh_deg=sib_change_thresh_deg,
        sib_min_bout_minutes=sib_min_bout_minutes,
        haspt_ignore_invalid=haspt_ignore_invalid,
        hdcza_threshold=hdcza_threshold,
        spt_min_block_dur_minutes=spt_min_block_dur_minutes,
        spt_max_gap_dur_minutes=spt_max_gap_dur_minutes,
        spt_max_gap_ratio=spt_max_gap_ratio,
        rerun_with_6pm_window=rerun_with_6pm_window,
        return_epoch_outputs=return_epoch_outputs,
        return_intermediates=return_intermediates,
        save_folder=save_folder,
    )


@fresh_outputs(["sleep_*.csv", "circadian_*.csv", "activity_intensity_*.csv",
                "recording_info.csv", "pipeline_stage_status.csv"])
def run_sleep_and_circadian_pipeline_ravenna(
    geneactiv_bin_path: str | Path,
    participant: Optional[str] = None,
    visit: str = "T0",
    nonwear_method: str = "vanhees2013",
    timezone: str = "Europe/Rome",
    diary_csv_path: Optional[str | Path] = None,
    diary_subject_col: str | None = None,
    diary_start_col: str | None = None,
    diary_end_col: str | None = None,
    diary_visit_col: str | None = None,
    diary_night_id_col: str | None = None,
    lb_tib_csv_path: Optional[str | Path] = None,
    lb_subject_col: str | None = None,
    lb_start_col: str | None = None,
    lb_end_col: str | None = None,
    lb_visit_col: str | None = None,
    lb_night_id_col: str | None = None,
    sib_change_thresh_deg: float = 5.0,
    sib_min_bout_minutes: int = 5,
    haspt_ignore_invalid: float | bool = False,
    hdcza_threshold: float | tuple[float, float] | list[float] | None = None,
    spt_min_block_dur_minutes: int = 30,
    spt_max_gap_dur_minutes: int = 60,
    spt_max_gap_ratio: float = 1.0,
    rerun_with_6pm_window: bool = True,
    return_epoch_outputs: bool = True,
    return_intermediates: bool = False,
    dayborder_hours: float = 0.0,
    min_wear_hours: float = 18.0,
    max_nonwear_hours: Optional[float] = None,
    require_full_24h: bool = False,
    max_days: Optional[int] = 7,
    min_valid_days: Optional[int] = 3,
    activity_threshold_mg: float = 40.0,
    intensity_thresholds_mg: tuple[float, float, float] = (40.0, 100.0, 400.0),
    l5_m5_step_minutes: int = 10,
    activity_epoch_seconds: int = 5,
    require_activity_spt_overlap: bool = True,
    plot: bool = False,
    save_folder: Optional[str | Path] = None,
) -> Dict[str, object]:
    """
    DARE-FALLSPREDICT combined pipeline: sleep, circadian, and activity-intensity from
    one shared GENEActiv preprocessing step.
    """
    if participant is None:
        participant = _infer_participant_from_geneactiv_path(geneactiv_bin_path)

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    preprocessed = preprocess_geneactiv_recording(
        geneactiv_bin_path=geneactiv_bin_path,
        nonwear_method=nonwear_method,
    )

    return run_wrist_from_preprocessed(
        preprocessed=preprocessed,
        participant=participant,
        visit=visit,
        nonwear_method=nonwear_method,
        timezone=timezone,
        diary_csv_path=diary_csv_path,
        diary_subject_col=diary_subject_col,
        diary_start_col=diary_start_col,
        diary_end_col=diary_end_col,
        diary_visit_col=diary_visit_col,
        diary_night_id_col=diary_night_id_col,
        lb_tib_csv_path=lb_tib_csv_path,
        lb_subject_col=lb_subject_col,
        lb_start_col=lb_start_col,
        lb_end_col=lb_end_col,
        lb_visit_col=lb_visit_col,
        lb_night_id_col=lb_night_id_col,
        sib_change_thresh_deg=sib_change_thresh_deg,
        sib_min_bout_minutes=sib_min_bout_minutes,
        haspt_ignore_invalid=haspt_ignore_invalid,
        hdcza_threshold=hdcza_threshold,
        spt_min_block_dur_minutes=spt_min_block_dur_minutes,
        spt_max_gap_dur_minutes=spt_max_gap_dur_minutes,
        spt_max_gap_ratio=spt_max_gap_ratio,
        rerun_with_6pm_window=rerun_with_6pm_window,
        return_epoch_outputs=return_epoch_outputs,
        return_intermediates=return_intermediates,
        dayborder_hours=dayborder_hours,
        min_wear_hours=min_wear_hours,
        max_nonwear_hours=max_nonwear_hours,
        require_full_24h=require_full_24h,
        max_days=max_days,
        min_valid_days=min_valid_days,
        activity_threshold_mg=activity_threshold_mg,
        intensity_thresholds_mg=intensity_thresholds_mg,
        l5_m5_step_minutes=l5_m5_step_minutes,
        activity_epoch_seconds=activity_epoch_seconds,
        require_activity_spt_overlap=require_activity_spt_overlap,
        plot=plot,
        save_folder=save_folder,
    )


def run_sleep_pipeline_ravenna_from_input_root(
    input_root: str | Path,
    participant: str,
    visit: str = "T0",
    sensor: str = "GENEActiv",
    geneactiv_file_path: str | Path | None = None,
    require_subject_id_prefix: bool = True,
    **kwargs,
) -> Dict[str, object]:
    geneactiv_bin_path = find_geneactiv_bin_file(
        input_root=input_root,
        participant=participant,
        visit=visit,
        sensor=sensor,
        geneactiv_file_path=geneactiv_file_path,
        require_subject_id_prefix=require_subject_id_prefix,
    )
    return run_sleep_pipeline_ravenna(
        geneactiv_bin_path=geneactiv_bin_path,
        participant=participant,
        visit=visit,
        **kwargs,
    )


def run_sleep_and_circadian_pipeline_ravenna_from_input_root(
    input_root: str | Path,
    participant: str,
    visit: str = "T0",
    sensor: str = "GENEActiv",
    geneactiv_file_path: str | Path | None = None,
    require_subject_id_prefix: bool = True,
    **kwargs,
) -> Dict[str, object]:
    geneactiv_bin_path = find_geneactiv_bin_file(
        input_root=input_root,
        participant=participant,
        visit=visit,
        sensor=sensor,
        geneactiv_file_path=geneactiv_file_path,
        require_subject_id_prefix=require_subject_id_prefix,
    )
    return run_sleep_and_circadian_pipeline_ravenna(
        geneactiv_bin_path=geneactiv_bin_path,
        participant=participant,
        visit=visit,
        **kwargs,
    )


__all__ = [
    "find_geneactiv_bin_file",
    "run_sleep_pipeline_ravenna",
    "run_sleep_pipeline_ravenna_from_input_root",
    "run_sleep_and_circadian_pipeline_ravenna",
    "run_sleep_and_circadian_pipeline_ravenna_from_input_root",
]

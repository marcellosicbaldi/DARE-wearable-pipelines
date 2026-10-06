from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from dare_wearables.common.paths import project_root
from dare_wearables.wrist.circadian.activity_intensity import (
    annotate_sleep_period_window,
    build_epoch_table,
    classify_epoch_intensity,
    compare_activity_intensity_against_ggir,
    impute_invalid_epochs_by_clocktime,
    load_sleep_windows,
    select_valid_epoch_days,
    summarize_activity_intensity,
    summarize_epoch_days,
    write_activity_intensity_outputs,
)
from dare_wearables.wrist.circadian.empatica import (
    find_empatica_parquet_files,
    preprocess_empatica_recording,
)

DEFAULT_GP_SLEEP_CSV = project_root() / "data" / "gold" / "sleep" / "night_aggregation.csv"


def _infer_participant_from_acc_path(acc_parquet_path: str | Path) -> str:
    path = Path(acc_parquet_path)
    if len(path.parents) < 3:
        raise ValueError(f"Could not infer participant from path: {acc_parquet_path}")
    return path.parents[2].name


def run_activity_intensity_pipeline_gp_from_preprocessed(calibrated_df,
                                                         acc_df: pd.DataFrame,
                                                         temp_df: pd.DataFrame,
                                                         info: dict,
                                                         nonwear_df: pd.DataFrame,
                                                         sleep_windows: pd.DataFrame,
                                                         *,
                                                         participant: Optional[str] = None,
                                                         visit: str = "T0",
                                                         epoch_seconds: int = 5,
                                                         thresholds_mg: Tuple[float, float, float] = (40.0, 100.0, 400.0),
                                                         dayborder_hours: float = 0.0,
                                                         timezone: str = "Europe/Rome",
                                                         min_wear_hours: float = 18.0,
                                                         min_valid_days: int = 3,
                                                         max_days: Optional[int] = 7,
                                                         require_spt_overlap: bool = True,
                                                         ggir_ms5_rdata_path: Optional[str | Path] = None) -> Dict[str, object]:
    """
    Run the GP activity-intensity analysis from a shared preprocessed recording
    and an already selected SPT window table.
    """
    epoch_df = build_epoch_table(
        calibrated_df=calibrated_df,
        nonwear_df=nonwear_df,
        epoch_seconds=epoch_seconds,
        dayborder_hours=dayborder_hours,
        timezone=timezone,
    )
    day_summary = summarize_epoch_days(
        epoch_df=epoch_df,
        epoch_seconds=epoch_seconds,
        min_wear_hours=min_wear_hours,
    )
    epoch_selected, valid_day_summary = select_valid_epoch_days(
        epoch_df=epoch_df,
        day_summary=day_summary,
        max_days=max_days,
        min_valid_days=min_valid_days,
    )
    epoch_imputed, imputed_epochs_per_day = impute_invalid_epochs_by_clocktime(epoch_selected)

    epoch_segmented = annotate_sleep_period_window(
        epoch_df=epoch_imputed,
        sleep_windows=sleep_windows,
        epoch_seconds=epoch_seconds,
    )
    if require_spt_overlap and not bool(epoch_segmented["in_spt"].any()):
        raw_start = epoch_segmented.index.min()
        raw_end = epoch_segmented.index.max()
        sleep_start = sleep_windows["spt_start"].min()
        sleep_end = sleep_windows["spt_end"].max()
        raise ValueError(
            "The retained accelerometer epochs do not overlap any supplied SPT window. "
            f"Accelerometer range: {raw_start} to {raw_end}. "
            f"Sleep-window range: {sleep_start} to {sleep_end}."
        )
    epoch_classified = classify_epoch_intensity(
        epoch_df=epoch_segmented,
        thresholds_mg=thresholds_mg,
    )
    intensity_daily_df, intensity_summary_df = summarize_activity_intensity(
        epoch_df=epoch_classified,
        epoch_seconds=epoch_seconds,
    )

    out: Dict[str, object] = {
        "participant": participant,
        "visit": visit,
        "info": info,
        "acc_df": acc_df,
        "temp_df": temp_df,
        "sleep_windows": sleep_windows,
        "nonwear_df": nonwear_df,
        "epoch_df": epoch_df,
        "day_summary": day_summary,
        "valid_day_summary": valid_day_summary,
        "epoch_imputed": epoch_imputed,
        "epoch_classified": epoch_classified,
        "imputed_epochs_per_day": imputed_epochs_per_day,
        "intensity_daily_df": intensity_daily_df,
        "intensity_summary_df": intensity_summary_df,
    }

    if ggir_ms5_rdata_path is not None:
        out["ggir_comparison"] = compare_activity_intensity_against_ggir(
            intensity_daily_df=intensity_daily_df,
            ggir_ms5_rdata_path=ggir_ms5_rdata_path,
        )

    return out


def run_activity_intensity_pipeline_gp(acc_parquet_path: str | Path,
                                       temp_parquet_path: str | Path,
                                       sleep_csv_path: str | Path = DEFAULT_GP_SLEEP_CSV,
                                       participant: Optional[str] = None,
                                       visit: str = "T0",
                                       sleep_subject: Optional[str] = None,
                                       parquet_engine: str = "fastparquet",
                                       nonwear_method: str = "empatica_detach",
                                       epoch_seconds: int = 5,
                                       thresholds_mg: Tuple[float, float, float] = (40.0, 100.0, 400.0),
                                       dayborder_hours: float = 0.0,
                                       timezone: str = "Europe/Rome",
                                       min_wear_hours: float = 18.0,
                                       min_valid_days: int = 3,
                                       max_days: Optional[int] = 7,
                                       require_spt_overlap: bool = True,
                                       ggir_ms5_rdata_path: Optional[str | Path] = None) -> Dict[str, object]:
    """
    GP / EmbracePlus activity-intensity pipeline.

    This mirrors `activity_intensity.py`, but swaps in the GP-specific loading
    and non-wear preprocessing:
    - load Empatica silver `acc.parquet` and `temp.parquet`
    - mark charging gaps as non-wear
    - run DETACH on non-charging chunks
    - classify imputed 5-second ENMO epochs into IN/LIG/MOD/VIG using 40/100/400 mg
    - use sleep windows from the configured GP sleep summary CSV
    """
    if participant is None:
        participant = _infer_participant_from_acc_path(acc_parquet_path)

    preprocessed = preprocess_empatica_recording(
        acc_parquet_path=acc_parquet_path,
        temp_parquet_path=temp_parquet_path,
        parquet_engine=parquet_engine,
        nonwear_method=nonwear_method,
    )

    sleep_windows = load_sleep_windows(
        sleep_csv_path=sleep_csv_path,
        subject=participant,
        visit=visit,
        timezone=timezone,
        sleep_subject=sleep_subject,
    )
    return run_activity_intensity_pipeline_gp_from_preprocessed(
        calibrated_df=preprocessed["calibrated_df"],
        acc_df=preprocessed["acc_df"],
        temp_df=preprocessed["temp_df"],
        info=preprocessed["info"],
        nonwear_df=preprocessed["nonwear_df"],
        sleep_windows=sleep_windows,
        participant=participant,
        visit=visit,
        epoch_seconds=epoch_seconds,
        thresholds_mg=thresholds_mg,
        dayborder_hours=dayborder_hours,
        timezone=timezone,
        min_wear_hours=min_wear_hours,
        min_valid_days=min_valid_days,
        max_days=max_days,
        require_spt_overlap=require_spt_overlap,
        ggir_ms5_rdata_path=ggir_ms5_rdata_path,
    )


def run_activity_intensity_pipeline_gp_from_silver(silver_root: str | Path,
                                                   participant: str,
                                                   visit: str = "T0",
                                                   sensor: str = "Empatica",
                                                   sleep_csv_path: str | Path = DEFAULT_GP_SLEEP_CSV,
                                                   **kwargs) -> Dict[str, object]:
    """
    Convenience wrapper to run the GP activity-intensity pipeline directly from
    the silver folder layout.
    """
    paths = find_empatica_parquet_files(
        silver_root=silver_root,
        participant=participant,
        visit=visit,
        sensor=sensor,
    )
    return run_activity_intensity_pipeline_gp(
        acc_parquet_path=paths["acc_parquet_path"],
        temp_parquet_path=paths["temp_parquet_path"],
        sleep_csv_path=sleep_csv_path,
        participant=participant,
        visit=visit,
        **kwargs,
    )


__all__ = [
    "DEFAULT_GP_SLEEP_CSV",
    "run_activity_intensity_pipeline_gp_from_preprocessed",
    "run_activity_intensity_pipeline_gp",
    "run_activity_intensity_pipeline_gp_from_silver",
    "write_activity_intensity_outputs",
]

run_activity_from_preprocessed = run_activity_intensity_pipeline_gp_from_preprocessed
run_activity_from_empatica = run_activity_intensity_pipeline_gp
run_activity_from_empatica_silver = run_activity_intensity_pipeline_gp_from_silver

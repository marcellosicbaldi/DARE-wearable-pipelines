"""Legacy Empatica API retained for Ravenna callers of the former copy."""
from pathlib import Path
from typing import Dict, Optional
from dare_wearables.wrist.sleep import pipeline as _implementation

# Algorithms and other entry points resolve directly to the shared core.
__all__ = [name for name in dir(_implementation) if not name.startswith("_")]


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))


def run_sleep_pipeline_gp(acc_parquet_path: str | Path,
                          temp_parquet_path: str | Path,
                          participant: Optional[str] = None,
                          visit: str = "T0",
                          parquet_engine: str = "fastparquet",
                          nonwear_method: str = "empatica_detach",
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
                          save_folder: Optional[str | Path] = None) -> Dict[str, object]:
    """Preserve the legacy positional order, including parquet_engine."""
    return _implementation.run_sleep_pipeline_gp(
        acc_parquet_path=acc_parquet_path,
        temp_parquet_path=temp_parquet_path,
        participant=participant,
        visit=visit,
        parquet_engine=parquet_engine,
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
        save_folder=save_folder,
    )

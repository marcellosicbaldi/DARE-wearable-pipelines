"""fallspredict_pipeline lower-back defaults over the shared processing runner."""
from pathlib import Path
from ..config import LowerBackConfig
from dare_wearables.lower_back import pipeline as _shared
from dare_wearables.lower_back.pipeline import (
    apply_orientation_correction, run_gait_pipeline, run_time_in_bed_pipeline,
    _load_gait_helpers, _list_subjects, acc_cols, gyro_cols,
)

MIN_SIZE = 300 * 1024 * 1024


def load_redcap_metadata(redcap_csv: str | Path, *, get_sensor_height_fn):
    return _shared.load_redcap_metadata(redcap_csv, get_sensor_height_fn=get_sensor_height_fn)


def run_from_config(config: LowerBackConfig) -> None:
    return _shared.run_from_config(config)


def main(*, bronze_root: str | Path, silver_root: str | Path,
         redcap_csv: str | Path | None = None, visits: list[str] | None = None,
         sensor: str = "McRoberts", min_size_bytes: int = MIN_SIZE,
         participant_ids: list[str] | None = None, run_gait: bool = True,
         run_time_in_bed: bool = True, save_omx_to_parquet: bool = True) -> None:
    return _shared.main(
        bronze_root=bronze_root, silver_root=silver_root, redcap_csv=redcap_csv,
        visits=visits, sensor=sensor, min_size_bytes=min_size_bytes,
        participant_ids=participant_ids, run_gait=run_gait,
        run_time_in_bed=run_time_in_bed, save_omx_to_parquet=save_omx_to_parquet,
    )

"""Configuration contracts accepted by shared runners."""
from pathlib import Path
from typing import Protocol


class LowerBackOptions(Protocol):
    bronze_root: Path
    silver_root: Path
    redcap_csv: Path | None
    visits: list[str]
    sensor: str
    min_size_bytes: int
    participant_ids: list[str] | None
    run_gait: bool
    run_time_in_bed: bool
    save_omx_to_parquet: bool

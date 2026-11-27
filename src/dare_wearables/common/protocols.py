"""Configuration contracts accepted by shared runners."""
from pathlib import Path
from dare_wearables.lower_back.nonwear.respiration import RespirationOptions
from dare_wearables.lower_back.nonwear.ggir2023 import NonwearOptions
from dare_wearables.lower_back.io.timestamp_quality import TimestampOptions
from dare_wearables.lower_back.orientation import OrientationOptions
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
    timestamps: TimestampOptions
    orientation: OrientationOptions
    nonwear: NonwearOptions
    respiration: RespirationOptions
    save_omx_to_parquet: bool
    protocol_hours: float | None

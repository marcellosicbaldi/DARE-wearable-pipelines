from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from .common.paths import project_root, resolve_path


def _load_toml(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = resolve_path(config_path)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with path.open("rb") as handle:
        return tomllib.load(handle), path


def _resolve_optional_path(value: str | Path | None, *, base: Path) -> Path | None:
    return resolve_path(value, base=base) if value else None

def _resolve_absolute_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None

    path = Path(value).expanduser()

    if not path.is_absolute():
        raise ValueError(f"Expected an absolute path, got: {value}")

    return path.resolve()


@dataclass(slots=True)
class EmpaticaSleepConfig:
    silver_root: Path
    output_root: Path
    recruitment_tracker_path: Path | None = None
    diary_csv_path: Path | None = None
    lb_tib_csv_path: Path | None = None
    sensor: str = "Empatica"
    visit: str = "T0"
    timezone: str = "Europe/Rome"
    parquet_engine: str = "fastparquet"
    nonwear_method: str = "empatica_detach"
    participant: str | None = None
    return_epoch_outputs: bool = True
    return_intermediates: bool = True

    @classmethod
    def from_toml(cls, config_path: str | Path) -> "EmpaticaSleepConfig":
        data, config_file = _load_toml(config_path)
        section = data.get("empatica_sleep", {})
        if "silver_root" not in section:
            raise KeyError("Missing required `empatica_sleep.silver_root` in configuration.")
        if "output_root" not in section:
            raise KeyError("Missing required `empatica_sleep.output_root` in configuration.")

        return cls(
            silver_root=_resolve_absolute_path(section["silver_root"]),
            output_root=_resolve_absolute_path(section["output_root"]),
            recruitment_tracker_path=_resolve_absolute_path(section.get("recruitment_tracker_path")),
            diary_csv_path=_resolve_absolute_path(section.get("diary_csv_path")),
            lb_tib_csv_path=_resolve_absolute_path(section.get("lb_tib_csv_path")),
            sensor=section.get("sensor", "Empatica"),
            visit=section.get("visit", "T0"),
            timezone=section.get("timezone", "Europe/Rome"),
            parquet_engine=section.get("parquet_engine", "fastparquet"),
            nonwear_method=section.get("nonwear_method", "empatica_detach"),
            participant=section.get("participant"),
            return_epoch_outputs=bool(section.get("return_epoch_outputs", True)),
            return_intermediates=bool(section.get("return_intermediates", True)),
        )


@dataclass(slots=True)
class LowerBackConfig:
    bronze_root: Path
    silver_root: Path
    redcap_csv: Path | None = None
    sensor: str = "McRoberts"
    visits: list[str] = field(default_factory=lambda: ["T0", "T1"])
    min_size_bytes: int = 300 * 1024 * 1024
    participant_ids: list[str] | None = None
    run_gait: bool = True
    run_time_in_bed: bool = True
    save_omx_to_parquet: bool = True

    @classmethod
    def from_toml(cls, config_path: str | Path) -> "LowerBackConfig":
        data, config_file = _load_toml(config_path)
        section = data.get("lower_back", {})
        if "bronze_root" not in section:
            raise KeyError("Missing required `lower_back.bronze_root` in configuration.")
        if "silver_root" not in section:
            raise KeyError("Missing required `lower_back.silver_root` in configuration.")

        return cls(
            bronze_root=_resolve_absolute_path(section["bronze_root"]),
            silver_root=_resolve_absolute_path(section["silver_root"]),
            redcap_csv=_resolve_absolute_path(section.get("redcap_csv")),
            sensor=section.get("sensor", "McRoberts"),
            visits=list(section.get("visits", ["T0", "T1"])),
            min_size_bytes=int(section.get("min_size_bytes", 300 * 1024 * 1024)),
            participant_ids=[str(value) for value in section["participant_ids"]]
            if section.get("participant_ids")
            else None,
            run_gait=bool(section.get("run_gait", True)),
            run_time_in_bed=bool(section.get("run_time_in_bed", True)),
            save_omx_to_parquet=bool(section.get("save_omx_to_parquet", True)),
        )


DEFAULT_PROJECT_ROOT = project_root()

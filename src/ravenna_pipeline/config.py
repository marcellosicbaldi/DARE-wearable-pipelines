from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]

from .common.paths import project_root, resolve_path


def _load_toml(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = resolve_path(config_path)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if tomllib is not None:
        with path.open("rb") as handle:
            return tomllib.load(handle), path
    return _load_simple_toml(path), path


def _load_simple_toml(path: Path) -> dict[str, Any]:
    """
    Small fallback for the simple config files used by this project.

    It supports sections, strings, booleans, numbers, and one-line arrays. The
    normal path is still stdlib `tomllib` on Python 3.11+ or `tomli`.
    """
    data: dict[str, Any] = {}
    current: dict[str, Any] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            current = data.setdefault(section, {})
            continue
        if "=" not in line or current is None:
            raise ValueError(f"Unsupported TOML line in {path}: {raw_line!r}")
        key, value = line.split("=", 1)
        current[key.strip()] = _parse_simple_toml_value(value.strip())

    return data


def _parse_simple_toml_value(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_toml_value(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _resolve_optional_path(value: str | Path | None, *, base: Path) -> Path | None:
    return resolve_path(value, base=base) if value else None


def _resolve_absolute_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None

    path = Path(value).expanduser()

    if not path.is_absolute():
        raise ValueError(f"Expected an absolute path, got: {value}")

    return path.resolve()


@dataclass
class RavennaSleepConfig:
    input_root: Path
    output_root: Path
    geneactiv_file_path: str | Path | None = None
    diary_csv_path: Path | None = None
    lb_tib_csv_path: Path | None = None
    sensor: str = "GENEActiv"
    visit: str = "T0"
    timezone: str = "Europe/Rome"
    nonwear_method: str = "vanhees2013"
    participant: str | None = None
    return_epoch_outputs: bool = True
    return_intermediates: bool = True
    require_subject_id_prefix: bool = True

    @classmethod
    def from_toml(cls, config_path: str | Path) -> "RavennaSleepConfig":
        data, config_file = _load_toml(config_path)
        section = data.get("ravenna_sleep", {})
        if "input_root" not in section:
            raise KeyError("Missing required `ravenna_sleep.input_root` in configuration.")
        if "output_root" not in section:
            raise KeyError("Missing required `ravenna_sleep.output_root` in configuration.")

        geneactiv_file_path = section.get("geneactiv_file_path")

        return cls(
            input_root=_resolve_optional_path(section["input_root"], base=config_file.parent),
            output_root=_resolve_optional_path(section["output_root"], base=config_file.parent),
            geneactiv_file_path=geneactiv_file_path,
            diary_csv_path=_resolve_optional_path(section.get("diary_csv_path"), base=config_file.parent),
            lb_tib_csv_path=_resolve_optional_path(section.get("lb_tib_csv_path"), base=config_file.parent),
            sensor=section.get("sensor", "GENEActiv"),
            visit=section.get("visit", "T0"),
            timezone=section.get("timezone", "Europe/Rome"),
            nonwear_method=section.get("nonwear_method", "vanhees2013"),
            participant=section.get("participant"),
            return_epoch_outputs=bool(section.get("return_epoch_outputs", True)),
            return_intermediates=bool(section.get("return_intermediates", True)),
            require_subject_id_prefix=bool(section.get("require_subject_id_prefix", True)),
        )


@dataclass
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


@dataclass
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
            bronze_root=_resolve_optional_path(section["bronze_root"], base=config_file.parent),
            silver_root=_resolve_optional_path(section["silver_root"], base=config_file.parent),
            redcap_csv=_resolve_optional_path(section.get("redcap_csv"), base=config_file.parent),
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

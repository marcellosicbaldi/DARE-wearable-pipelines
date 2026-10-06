from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from dare_wearables.common.paths import resolve_path


def _load_toml(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = resolve_path(config_path)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with path.open("rb") as handle:
        return tomllib.load(handle), path


def _resolve_absolute_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None

    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Expected an absolute path, got: {value}")

    return path.resolve()


@dataclass(slots=True)
class HRVConfig:
    """Configuration for participant-level heart-rate and HRV processing."""

    input_root: Path
    output_root: Path
    sleep_output_root: Path | None = None
    sleep_windows_path: Path | None = None
    participant: str | None = None
    visit: str = "T0"
    sensor: str = "Empatica"
    timezone: str = "Europe/Rome"
    output_subdir: str = "hrv"
    burst_output_subdir: str = "nocturnal_bursts"
    sleep_output_subdir: str = "sleep_circadian"
    sleep_output_filename: str = "sleep_output_all_guiders.csv"
    sleep_guider_priority: tuple[str, ...] = ("sleep_diary", "HDCZA", "lower_back_tib")
    sampling_frequency: int = 64
    threshold_bursts: float = 35 / 1000
    min_window: str = "1 min"
    max_window: str = "5 min"
    window_step: str = "1 min"
    min_beats_per_window: int = 30
    skip_existing: bool = True

    @classmethod
    def from_paths(
        cls,
        *,
        input_root: str | Path,
        output_root: str | Path,
        sleep_output_root: str | Path | None = None,
        sleep_windows_path: str | Path | None = None,
        participant: str | None = None,
        visit: str = "T0",
        sensor: str = "Empatica",
        timezone: str = "Europe/Rome",
    ) -> "HRVConfig":
        return cls(
            input_root=Path(input_root),
            output_root=Path(output_root),
            sleep_output_root=Path(sleep_output_root) if sleep_output_root else Path(output_root),
            sleep_windows_path=Path(sleep_windows_path) if sleep_windows_path else None,
            participant=participant,
            visit=visit,
            sensor=sensor,
            timezone=timezone,
        )

    @classmethod
    def from_toml(cls, config_path: str | Path) -> "HRVConfig":
        data, _ = _load_toml(config_path)
        section = data.get("heart_rate_variability", {})
        if "input_root" not in section:
            raise KeyError("Missing required `heart_rate_variability.input_root` in configuration.")
        if "output_root" not in section:
            raise KeyError("Missing required `heart_rate_variability.output_root` in configuration.")
        output_root = _resolve_absolute_path(section["output_root"])

        return cls(
            input_root=_resolve_absolute_path(section["input_root"]),
            output_root=output_root,
            sleep_output_root=(
                _resolve_absolute_path(section.get("sleep_output_root"))
                if section.get("sleep_output_root")
                else output_root
            ),
            sleep_windows_path=_resolve_absolute_path(section.get("sleep_windows_path")),
            participant=section.get("participant"),
            visit=section.get("visit", "T0"),
            sensor=section.get("sensor", "Empatica"),
            timezone=section.get("timezone", "Europe/Rome"),
            output_subdir=section.get("output_subdir", "hrv"),
            burst_output_subdir=section.get("burst_output_subdir", "nocturnal_bursts"),
            sleep_output_subdir=section.get("sleep_output_subdir", "sleep_circadian"),
            sleep_output_filename=section.get("sleep_output_filename", "sleep_output_all_guiders.csv"),
            sleep_guider_priority=tuple(
                section.get("sleep_guider_priority", ["sleep_diary", "HDCZA", "lower_back_tib"])
            ),
            sampling_frequency=int(section.get("sampling_frequency", 64)),
            threshold_bursts=float(section.get("threshold_bursts", 35 / 1000)),
            min_window=section.get("min_window", "1 min"),
            max_window=section.get("max_window", "5 min"),
            window_step=section.get("window_step", "1 min"),
            min_beats_per_window=int(section.get("min_beats_per_window", 30)),
            skip_existing=bool(section.get("skip_existing", True)),
        )

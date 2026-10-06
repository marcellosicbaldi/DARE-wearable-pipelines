from __future__ import annotations

from dataclasses import dataclass, field
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
class HeartRateConfig:
    """Configuration for Empatica PPG heart-rate processing with BeliefPPG."""

    input_root: Path
    output_root: Path
    participant: str | None = None
    visits: list[str] = field(default_factory=lambda: ["T0", "T1"])
    sensor: str = "Empatica"
    output_subdir: str = "beliefppg"
    ppg_frequency: int = 64
    acc_frequency: int = 64
    empty_gap_seconds: float = 1.0
    min_good_portion_minutes: float = 10.0
    min_inference_minutes: float = 5.0
    skip_existing: bool = True

    @classmethod
    def from_toml(cls, config_path: str | Path) -> "HeartRateConfig":
        data, _ = _load_toml(config_path)
        section = data.get("heart_rate", {})
        if "input_root" not in section:
            raise KeyError("Missing required `heart_rate.input_root` in configuration.")
        if "output_root" not in section:
            raise KeyError("Missing required `heart_rate.output_root` in configuration.")

        return cls(
            input_root=_resolve_absolute_path(section["input_root"]),
            output_root=_resolve_absolute_path(section["output_root"]),
            participant=section.get("participant"),
            visits=[str(visit) for visit in section.get("visits", ["T0", "T1"])],
            sensor=section.get("sensor", "Empatica"),
            output_subdir=section.get("output_subdir", "beliefppg"),
            ppg_frequency=int(section.get("ppg_frequency", 64)),
            acc_frequency=int(section.get("acc_frequency", 64)),
            empty_gap_seconds=float(section.get("empty_gap_seconds", 1.0)),
            min_good_portion_minutes=float(section.get("min_good_portion_minutes", 10.0)),
            min_inference_minutes=float(section.get("min_inference_minutes", 5.0)),
            skip_existing=bool(section.get("skip_existing", True)),
        )

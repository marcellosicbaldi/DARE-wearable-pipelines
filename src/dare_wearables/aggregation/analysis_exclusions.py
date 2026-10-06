"""Private-configured exclusions and column selection for final T0 datasets.

Apply after clinical/sensor joining, before the shared cohort schema and coverage
are calculated. Source exports and automatic per-window HRV filtering are not
modified. Participant identifiers are supplied separately from this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from dare_wearables.redcap.identity import normalize_subjects
from dare_wearables.aggregation.minimum_data import apply_minimum_observations


@dataclass(frozen=True)
class AnalysisExclusions:
    """Explicit manual decisions; an empty instance opts out of manual masks."""

    rmssd_sdnn_ids: tuple[str, ...] = ()
    sleep_circadian_ids: tuple[str, ...] = ()

    @classmethod
    def from_toml(cls, path: str | Path) -> "AnalysisExclusions":
        with Path(path).expanduser().open("rb") as handle:
            data = tomllib.load(handle)
        keys = {"rmssd_sdnn_ids", "sleep_circadian_ids"}
        section = data.get("manual_exclusions")
        if set(data) != {"manual_exclusions"} or not isinstance(section, dict) or set(section) != keys:
            raise ValueError("Expected [manual_exclusions] with rmssd_sdnn_ids and sleep_circadian_ids only.")
        values = {}
        for key in sorted(keys):
            items = section[key]
            if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
                raise ValueError(f"{key} must be an array of quoted participant IDs.")
            normalized = normalize_subjects(pd.Series(items, dtype="string"))
            if normalized.duplicated().any():
                raise ValueError(f"{key} contains duplicate normalized IDs.")
            values[key] = tuple(normalized)
        return cls(**values)


def resolve_analysis_exclusions(
    config_path: str | Path | None, *, no_manual_exclusions: bool = False,
) -> AnalysisExclusions:
    """Require an explicit file or opt-out; never silently lose study rules."""
    if not isinstance(no_manual_exclusions, bool):
        raise TypeError("no_manual_exclusions must be a boolean.")
    if (config_path is not None) == no_manual_exclusions:
        raise ValueError("Supply exclusions_config or explicitly set no_manual_exclusions=True, but not both.")
    return AnalysisExclusions() if no_manual_exclusions else AnalysisExclusions.from_toml(config_path)


SLEEP_CIRCADIAN_ACTIVITY_PREFIXES = (
    "sleep_diary_", "hdcza_", "lower_back_", "circadian_", "activity_",
)
EXCLUDED_ANALYSIS_COLUMNS = (
    "hdcza_spt_start_clock_h",
    "hdcza_spt_end_clock_h",
    "hdcza_guider_start_clock_h",
    "hdcza_guider_end_clock_h",
    "gait_wb_all__n_raw_initial_contacts__sum",
    "gait_wb_all__n_turns__sum",
    "hrv_window_length_s",
    "hrv_source_night_id",
    "hrv_priority_rank",
    "hrv_PIP_percent_discarded_median",
    "hrv_mean_hr_percent_discarded_median",
    "hrv_rmssd_percent_discarded_median",
    "hrv_sdnn_percent_discarded_median",
    "circadian_MESOR_log1p_mg",
    "circadian_Amplitude_log1p_mg",
    "circadian_Acrotime_hour",
    "circadian_Cosinor_n_days_used",
)


def apply_analysis_exclusions(frame: pd.DataFrame, *, exclusions: AnalysisExclusions) -> pd.DataFrame:
    """Mask the specified measurements, record exclusions, and drop 17 columns.

    A flag records the explicit exclusion decision even when the measurement
    was already missing. Other missing values do not imply visual exclusion.
    Participant rows and source-presence flags are preserved.
    """
    required = {"group", "subject", "visit"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Analysis exclusions require columns: {sorted(missing)}")
    out = frame.drop(columns=list(EXCLUDED_ANALYSIS_COLUMNS), errors="ignore").copy()
    subjects = normalize_subjects(out["subject"])
    t0 = out["visit"].eq("T0").fillna(False)
    hrv_mask = (
        t0 & out["group"].eq("BO").fillna(False)
        & subjects.isin(normalize_subjects(pd.Series(exclusions.rmssd_sdnn_ids, dtype="string")))
    )
    sleep_mask = (
        t0 & out["group"].eq("RA").fillna(False)
        & subjects.isin(normalize_subjects(pd.Series(exclusions.sleep_circadian_ids, dtype="string")))
    )
    for column in ("hrv_rmssd", "hrv_sdnn"):
        if column in out:
            out[column] = out[column].mask(hrv_mask)
    for column in out.columns:
        if column.startswith(SLEEP_CIRCADIAN_ACTIVITY_PREFIXES):
            out[column] = out[column].mask(sleep_mask)
    for flag, mask in (("rmssd_sdnn_exclusion", hrv_mask), ("sleep_circadian_exclusion", sleep_mask)):
        out[flag] = mask.astype("int8")
        out[f"{flag}_method"] = pd.Series("visual inspection", index=out.index, dtype="string").where(mask)
    return apply_minimum_observations(out)

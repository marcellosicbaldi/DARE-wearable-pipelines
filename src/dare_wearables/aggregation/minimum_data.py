"""Mask participant measurements with fewer than three valid observations.

Counts are domain-specific days/nights, never borrowed from another sensor.
Missing counts are not zero: populated measurements without coverage evidence
require a refreshed aggregation rather than silently passing the rule.
"""

from __future__ import annotations

import pandas as pd

from dare_wearables.redcap.schema import NON_SENSOR_COLUMNS


MIN_VALID_OBSERVATIONS = 3
GAIT_QC_COLUMNS = frozenset({"gait_n_days", "gait_n_valid_days", "gait_mean_valid_hours"})
HR_COUNT_COLUMNS = {
    "median_hr_day": "hr_n_valid_days",
    "median_hr_night": "hr_n_valid_nights",
    "hr_dip_pct": "hr_dip_n_valid_pairs",
}
HRV_METRICS = ("rmssd", "mean_hr", "sdnn", "PIP")


def _mask_measurements(out: pd.DataFrame, columns: list[str], count_column: str) -> None:
    # REDCap's clinic-measured gait_speed shares the sensor prefix. Clinical
    # measurements must never inherit wearable coverage requirements.
    columns = [column for column in columns if column in out and column not in NON_SENSOR_COLUMNS]
    if not columns:
        return
    counts = (
        pd.to_numeric(out[count_column], errors="coerce")
        if count_column in out else pd.Series(float("nan"), index=out.index)
    )
    if (counts.isna() & out[columns].notna().any(axis=1)).any():
        raise ValueError(
            f"Cannot enforce the three-day/night minimum: populated {columns[0]} "
            f"has no usable {count_column}. Refresh the source sensor aggregation "
            "before rebuilding the combined dataset (DARE-FALLSPREDICT GP: python -m "
            "fallspredict_gp_pipeline.aggregation.overall --sleep-method both)."
        )
    insufficient = counts.lt(MIN_VALID_OBSERVATIONS).fillna(False)
    for column in columns:
        out[column] = out[column].mask(insufficient)


def apply_minimum_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep QC counts and rows; mask only measurements for the failing domain."""
    out = frame.copy()
    _mask_measurements(out, [c for c in out if c.startswith("gait_") and c not in GAIT_QC_COLUMNS],
                       "gait_n_valid_days")
    for prefix in ("sleep_diary", "hdcza", "lower_back"):
        count = f"{prefix}_n_valid_nights"
        _mask_measurements(out, [c for c in out if c.startswith(f"{prefix}_") and c != count], count)
    for metric, count in HR_COUNT_COLUMNS.items():
        _mask_measurements(out, [metric], count)

    # The total HRV count describes nights with any retained metric. Each
    # physiological metric additionally needs three nights of its own data.
    hrv_qc = {"hrv_n_valid_nights", "hrv_n_windows"}
    hrv_qc.update(f"hrv_{metric}_n_valid_nights" for metric in HRV_METRICS)
    _mask_measurements(out, [c for c in out if c.startswith("hrv_") and c not in hrv_qc],
                       "hrv_n_valid_nights")
    for metric in HRV_METRICS:
        _mask_measurements(out, [f"hrv_{metric}"], f"hrv_{metric}_n_valid_nights")

    _mask_measurements(out, [c for c in out if c.startswith("activity_") and c != "activity_n_valid_days"],
                       "activity_n_valid_days")
    # Circadian's three-day requirement is enforced upstream. Its exported
    # cosinor_days is a duration (minutes / 1440), not a valid-day count.
    return out

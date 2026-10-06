"""Study-score compatibility with csv_generator_refactor_notes.ipynb.

PSQI and WFG intentionally reproduce the supplied notebook, including its clock
conversion and case-sensitive SI/NO severity items. These are compatibility
scores, not a new implementation or clinical validation of those instruments.
See docs/redcap_processing.md before changing these conventions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MMSE_ITEMS = (
    "draw_testf_b", "langauge_test_sentencef_b", "lanaguage_test_paperf_b",
    "right_handf_b", "foldf_b", "groundf_b", "language_2f_b", "watchf_b",
    "pencilf_b", "house_ynf_b", "bred_ynf_b", "cat_ynf_b", "best_scoref_b",
    "housef_b", "bredf_b", "catf_b", "floorf_yn_b", "locationf_yn_b",
    "cityf_yn_b", "cantonef_yn_b", "nationf_yn_b", "monthf_yn_b",
    "dayweekf_yn_b", "dayf_yn_b", "seasonf_yn_b", "yearf_yn_b",
)
FES_ITEMS = tuple(f"{name}f_b" for name in (
    "dress", "shower", "chair", "stairs", "take", "walk", "go_out",
))
CESD_REGULAR = tuple(f"{name}f_b" for name in (
    "preoccupation", "appetite", "melancony", "attention", "depressed",
    "effort", "failure", "fear", "agitated", "quiet", "alone", "grumpy",
    "cry", "sad", "insecurity", "tiredness",
))
CESD_REVERSED = tuple(f"{name}f_b" for name in ("value", "hope", "happy", "amused"))
PSQI_DISTURBANCES = tuple(f"{name}f_b" for name in (
    "wake_up", "toilet_get_up", "no_good_breathing", "coughing_snoring",
    "too_cold", "too_hot", "nightmares", "hurt", "sleep_problems_other_yn",
))
PSQI_ITEMS = (
    "sleep_qualityf_b", "falling_asleep_durationf_b", "min_no_sleepf_b",
    "hours_of_sleepf_b", "bed_timef_b", "get_up_timef_b",
    *PSQI_DISTURBANCES, "sleep_drugsf_b", "stay_awakef_b", "energyf_b",
)
WFG_ITEMS = (
    *FES_ITEMS, "previous_fallsf_b", "dizziness_or_unsteadinessf_b",
    "falls_lesionf_b", "inability_to_get_upf_b", "falls_numf_b", "cfsf_b",
    "temporary_consciousness_lossf_b", "walk_test_andata_b", "walk_test_ritorno_b",
)


def numeric(frame: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Coerce a requested block, treating negative REDCap sentinels as missing."""
    out = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").astype(float)
    return out.where(out >= 0)


def mmse_corrected_notebook(age: pd.Series, education_years: pd.Series) -> pd.Series:
    """Reproduce the commented cell-6 equation matching the May reference.

    This expression does not use the observed MMSE score. The historical column
    name is retained for compatibility, without asserting clinical validity.
    """
    age = pd.to_numeric(age, errors="coerce").astype(float)
    education = pd.to_numeric(education_years, errors="coerce").astype(float)
    valid = age.between(0, 93.9, inclusive="left") & education.ge(0) & np.isfinite(education)
    return 2.4 * np.log10((93.9 - age).where(valid)) * education.where(valid).pow(0.29) + 22.1


def psqi_notebook(frame: pd.DataFrame) -> pd.Series:
    """Reproduce the study notebook's seven-component PSQI total."""
    n = numeric(frame, PSQI_ITEMS)
    latency = n["falling_asleep_durationf_b"]
    latency_code = pd.Series(np.select(
        [latency <= 15, latency.between(16, 30), latency.between(31, 60)],
        [0, 1, 2], default=3,
    ), index=frame.index)

    def combined_score(s: pd.Series) -> pd.Series:
        return pd.Series(np.select(
            [s.between(1, 2), s.between(3, 4), s.between(5, 6)],
            [1, 2, 3], default=0,
        ), index=frame.index)

    c2 = combined_score(latency_code + n["min_no_sleepf_b"])
    hours = n["hours_of_sleepf_b"]
    c3 = np.select([hours < 5, hours.between(5, 6), (hours > 6) & (hours <= 7)],
                   [3, 2, 1], default=0)

    def legacy_wake_clock(value: float) -> str | None:
        if pd.isna(value) or not np.isfinite(value):
            return None
        # Retained verbatim in meaning from fix_time_float / float_to_hhmm.
        if value >= 24:
            value -= 10 * (int((value - 24) // 10) + 1)
        if abs(value - int(value) - 0.3) < 0.05:
            value = int(value) + 0.5
        minute = min(int(round((value - int(value)) * 100)), 59)
        return f"{int(value):02d}:{minute:02d}"

    wake = pd.to_datetime(n["get_up_timef_b"].map(legacy_wake_clock), format="%H:%M", errors="coerce")
    bed = pd.to_datetime(frame["bed_timef_b"], format="%H:%M", errors="coerce")
    duration = wake - bed
    duration = duration.where(duration >= pd.Timedelta(0), duration + pd.Timedelta(days=1))
    efficiency = hours / (duration.dt.total_seconds() / 3600) * 100
    c4 = np.select([efficiency < 65, efficiency.between(65, 74), efficiency.between(75, 84)],
                   [3, 2, 1], default=0)
    disturbance = n[list(PSQI_DISTURBANCES)].sum(axis=1)
    c5 = np.select([disturbance.between(1, 9), disturbance.between(10, 18), disturbance.between(19, 27)],
                   [1, 2, 3], default=0)
    c7 = combined_score(n[["stay_awakef_b", "energyf_b"]].sum(axis=1))
    return n["sleep_qualityf_b"] + c2 + c3 + c4 + c5 + n["sleep_drugsf_b"] + c7


def wfg_notebook(frame: pd.DataFrame) -> pd.Series:
    """Reproduce notebook WFG (CFS >= 4, faster walk <= 0.8 m/s)."""
    n = numeric(frame, WFG_ITEMS)
    fesi = n[list(FES_ITEMS)].sum(axis=1)
    key = (n["previous_fallsf_b"] == 1) | (n["dizziness_or_unsteadinessf_b"] == 1) | (fesi > 11)
    severity = (
        (n["falls_lesionf_b"] == 1) | (n["falls_numf_b"] >= 2) | (n["cfsf_b"] >= 4)
        | frame["inability_to_get_upf_b"].eq("SI").fillna(False)
        | frame["temporary_consciousness_lossf_b"].eq("SI").fillna(False)
    )
    walks = n[["walk_test_andata_b", "walk_test_ritorno_b"]]
    impaired = (4 / walks.min(axis=1).where(walks.gt(0).all(axis=1))) <= 0.8
    result = pd.Series("low", index=frame.index, dtype="string")
    result.loc[key & impaired] = "intermediate"
    result.loc[key & severity] = "high"
    result.loc[n["previous_fallsf_b"].isna()] = "ND"
    return result

"""Explicit cohort selection for local clinical and FRAT-up exports."""

from pathlib import Path


DATA_ROOT = Path("~/dare-data").expanduser()
COHORT_FOLDERS = {"BO": "Bologna", "RA": "Ravenna"}
COHORT_NAMES = {"BO": "DARE-FALLSPREDICT GP", "RA": "DARE-FALLSPREDICT"}


def validate_cohort(cohort: str) -> str:
    if cohort not in COHORT_NAMES:
        raise ValueError(f"cohort must be BO or RA; got {cohort!r}")
    return cohort


def default_redcap_csv(cohort: str) -> Path:
    return DATA_ROOT / "REDCap" / COHORT_FOLDERS[validate_cohort(cohort)] / "fallspredict_data.csv"

"""Participant identifiers shared by clinical and external CSV imports."""

import pandas as pd


def normalize_subjects(values: pd.Series) -> pd.Series:
    """Canonical sensor IDs: 7, 7.0 and 0007 become '0007'; reject invalid IDs."""
    text = values.astype("string").str.strip()
    valid = text.str.fullmatch(r"\d+(?:\.0+)?", na=False)
    if not valid.all():
        raise ValueError(f"Invalid or missing subject identifiers in {int((~valid).sum())} rows.")
    return text.str.replace(r"\.0+$", "", regex=True).str.lstrip("0").replace("", "0").str.zfill(4)

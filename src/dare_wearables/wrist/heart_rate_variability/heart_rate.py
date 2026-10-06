from __future__ import annotations

import pandas as pd


def build_heart_rate_epochs(
    hr_df: pd.DataFrame,
    *,
    epoch_seconds: int = 60,
    timezone: str = "Europe/Rome",
) -> pd.DataFrame:
    """Aggregate heart-rate samples into epoch-level HR features."""
    raise NotImplementedError("Heart-rate epoch aggregation has not been implemented yet.")


def summarize_heart_rate_daily(hr_epochs: pd.DataFrame) -> pd.DataFrame:
    """Summarize epoch-level heart rate by analysis day."""
    raise NotImplementedError("Daily heart-rate summaries have not been implemented yet.")

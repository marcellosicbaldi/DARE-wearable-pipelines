from __future__ import annotations

import pandas as pd


def compute_time_domain_hrv(ibi_df: pd.DataFrame) -> pd.DataFrame:
    """Compute time-domain HRV metrics from cleaned IBI data."""
    raise NotImplementedError("Time-domain HRV metrics have not been implemented yet.")


def compute_frequency_domain_hrv(ibi_df: pd.DataFrame) -> pd.DataFrame:
    """Compute frequency-domain HRV metrics from cleaned IBI data."""
    raise NotImplementedError("Frequency-domain HRV metrics have not been implemented yet.")


def compute_nonlinear_hrv(ibi_df: pd.DataFrame) -> pd.DataFrame:
    """Compute nonlinear HRV metrics from cleaned IBI data."""
    raise NotImplementedError("Nonlinear HRV metrics have not been implemented yet.")


def compute_hrv_windows(
    ibi_df: pd.DataFrame,
    *,
    window_seconds: int = 300,
    step_seconds: int | None = None,
) -> pd.DataFrame:
    """Compute HRV metrics over fixed windows."""
    raise NotImplementedError("Windowed HRV computation has not been implemented yet.")

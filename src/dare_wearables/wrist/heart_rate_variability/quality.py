from __future__ import annotations

import pandas as pd


def flag_valid_hr_windows(hr_df: pd.DataFrame, *, min_valid_window_seconds: int = 300) -> pd.DataFrame:
    """Return signal-quality flags for HR/HRV windows."""
    raise NotImplementedError("Heart-rate quality checks have not been implemented yet.")


def filter_valid_ibi(ibi_df: pd.DataFrame) -> pd.DataFrame:
    """Return IBI rows suitable for HRV metric computation."""
    raise NotImplementedError("IBI quality filtering has not been implemented yet.")

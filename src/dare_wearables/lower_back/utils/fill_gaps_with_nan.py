import numpy as np
import pandas as pd
    
def fill_gaps_with_nan(df, fs):
    # Ensure sorted
    df = df.sort_index()

    start = df.index[0]
    end = df.index[-1]

    # Compute total expected samples
    n_samples = int(np.round((end - start).total_seconds() * fs)) + 1

    # Create regular time grid
    full_index = start + pd.to_timedelta(
        np.arange(n_samples) / fs,
        unit="s"
    )

    # Map timestamps to nearest sample index
    sample_idx = np.round(
        (df.index - start).total_seconds() * fs
    ).astype(int)

    # Create empty regular dataframe
    df_full = pd.DataFrame(
        np.nan,
        index=full_index,
        columns=df.columns
    )

    # Insert existing data
    df_full.iloc[sample_idx] = df.values

    return df_full
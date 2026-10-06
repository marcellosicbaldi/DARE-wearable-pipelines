import pandas as pd

def bool_to_bouts(s: pd.Series, state=True) -> pd.DataFrame:
    """
    Convert boolean Series (DatetimeIndex) to bouts where s == state.
    Returns ['start','end'] where end is exclusive.
    """
    s = s.sort_index().fillna(False).astype(bool)
    if s.empty:
        return pd.DataFrame(columns=["start", "end"])

    x = (s == state)

    # Run boundaries: force first sample to start a run
    change = x.ne(x.shift())
    change.iloc[0] = True
    run_id = change.cumsum()

    starts = x.index[change]

    # estimate timestep (works for resampled 1S; otherwise falls back)
    if len(x.index) > 1:
        dt = (x.index[1] - x.index[0])
        if not isinstance(dt, pd.Timedelta) or dt <= pd.Timedelta(0):
            dt = pd.Timedelta(seconds=1)
    else:
        dt = pd.Timedelta(seconds=1)

    ends = starts[1:].append(pd.DatetimeIndex([x.index[-1] + dt]))

    states = x.groupby(run_id).first().to_numpy()
    # Now lengths match by construction
    runs = pd.DataFrame({"state": states, "start": starts, "end": ends})

    return runs[runs["state"]].loc[:, ["start", "end"]].reset_index(drop=True)
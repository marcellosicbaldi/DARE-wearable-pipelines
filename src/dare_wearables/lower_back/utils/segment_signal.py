import numpy as np
import pandas as pd

def segment_signal(signal, window_size, overlap):
    """
    Segment a signal into overlapping windows.

    Parameters
    ----------
    signal : numpy.ndarray or pd.Series
        Signal to segment.
    window_size : int
        Size of the window.
    overlap : float
        Overlap between consecutive windows.

    Returns
    -------
    segments : numpy.ndarray
        Segmented signal.

    Examples
    from utils.segment_signal import segment_signal

    acc_gen_segments = segment_signal(acc_gen_first_hour, window_size = 20, overlap = 0.5)
    """

    # check inputs
    if not isinstance(signal, (np.ndarray, pd.Series)):
        raise TypeError("`signal` must be a numpy.ndarray or pd.Series.")
    if not isinstance(window_size, int):
        raise TypeError("`window_size` must be an integer.")
    if not isinstance(overlap, float):
        raise TypeError("`overlap` must be a float.")
    if overlap < 0 or overlap >= 1:
        raise ValueError("`overlap` must be in [0, 1).")

    # get window step
    step = int(window_size * (1 - overlap))

    # segment signal
    if isinstance(signal, np.ndarray):
        segments = np.array(
            [signal[i : i + window_size] for i in range(0, len(signal) - window_size + 1, step)]
        )
    else:
        segments = np.array(
            [signal.iloc[i : i + window_size].values for i in range(0, len(signal) - window_size + 1, step)]
        )

    return segments


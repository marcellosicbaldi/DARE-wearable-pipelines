import numpy as np
from scipy.signal import cheby1, sosfiltfilt

def apply_resample(
    *, time, goal_fs=None, time_rs=None, data=(), indices=(), aa_filter=True, fs=None
):
    """
    Apply a resample to a set of data.

    Parameters
    ----------
    time : numpy.ndarray
        Array of original timestamps.
    goal_fs : float, optional
        Desired sampling frequency in Hz.  One of `goal_fs` or `time_rs` must be
        provided.
    time_rs : numpy.ndarray, optional
        Resampled time series to sample to. One of `goal_fs` or `time_rs` must be
        provided.
    data : tuple, optional
        Tuple of arrays to normally downsample using np.interpolation. Must match the
        size of `time`. Can handle `None` inputs, and will return an array of zeros
        matching the downsampled size.
    indices : tuple, optional
        Tuple of arrays of indices to downsample.
    aa_filter : bool, optional
        Apply an anti-aliasing filter before downsampling. Default is True. This
        is the same filter as used by :py:function:`scipy.signal.decimate`.
        See [1]_ for details. Ignored if upsampling.
    fs : {None, float}, optional
        Original sampling frequency in Hz. If `goal_fs` is an integer factor
        of `fs`, every nth sample will be taken, otherwise `np.np.interp` will be
        used. Leave blank to always use `np.np.interp`.

    Returns
    -------
    time_rs : numpy.ndarray
        Resampled time.
    data_rs : tuple, optional
        Resampled data, if provided.
    indices_rs : tuple, optional
        Resampled indices, if provided.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Downsampling_(signal_processing)
    """

    def resample(x, factor, t, t_rs):
        if (int(factor) == factor) and (factor > 1):
            # in case that t_rs is provided and ends earlier than t
            n = np.nonzero(t <= t_rs[-1])[0][-1] + 1
            return (x[: n : int(factor)],)
        else:
            if x.ndim == 1:
                return (np.interp(t_rs, t, x),)
            elif x.ndim == 2:
                xrs = np.zeros((t_rs.size, x.shape[1]), dtype=np.float64)
                for j in range(x.shape[1]):
                    xrs[:, j] = np.interp(t_rs, t, x[:, j])
                return (xrs,)

    if fs is None:
        # compute sampling frequency by hand
        fs = 1 / np.mean(np.diff(time[:5000]))

    if time_rs is None and goal_fs is None:
        raise ValueError("One of `time_rs` or `goal_fs` is required.")

    # get resampled time if necessary
    if time_rs is None:
        if int(fs / goal_fs) == fs / goal_fs and goal_fs < fs:
            time_rs = time[:: int(fs / goal_fs)]
        else:
            time_rs = np.arange(time[0], time[-1], 1 / goal_fs)
    else:
        goal_fs = 1 / np.mean(np.diff(time_rs[:5000]))
        # prevent t_rs from extrapolating
        time_rs = time_rs[time_rs <= time[-1]]

    # AA filter, if necessary
    if (fs / goal_fs) >= 1.0:
        sos = cheby1(8, 0.05, 0.8 / (fs / goal_fs), output="sos")
    else:
        aa_filter = False

    # resample data
    data_rs = ()

    for dat in data:
        if dat is None:
            data_rs += (None,)
        elif dat.ndim in [1, 2]:
            data_to_rs = sosfiltfilt(sos, dat, axis=0) if aa_filter else dat
            data_rs += resample(data_to_rs, fs / goal_fs, time, time_rs)
        else:
            raise ValueError("Data dimension exceeds 2, or data not understood.")

    # resampling indices
    indices_rs = ()
    for idx in indices:
        if idx is None:
            indices_rs += (None,)
        elif idx.ndim == 1:
            indices_rs += (
                np.around(np.interp(time[idx], time_rs, np.arange(time_rs.size))).astype(np.int_),
            )
        elif idx.ndim == 2:
            indices_rs += (np.zeros(idx.shape, dtype=np.int_),)
            for i in range(idx.shape[1]):
                indices_rs[-1][:, i] = np.around(
                    np.interp(
                        time[idx[:, i]], time_rs, np.arange(time_rs.size)
                    )  # cast to in on insert
                )

    ret = (time_rs,)
    if data_rs != ():
        ret += (data_rs,)
    if indices_rs != ():
        ret += (indices_rs,)

    return ret
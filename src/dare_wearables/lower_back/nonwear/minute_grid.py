"""Efficient one-minute GGIR grid used by the selected nonwear benchmark profile.

Same 5 Hz samples, axis propagation and threshold-adjacent R-compatible SD as
the frozen benchmark. No imports of notebooks, scripts or participant labels.
"""
import numpy as np


def minute_scores(axes, freq, medium, count, *, window_epochs):
    from .ggir2023 import _epoch_scores, _sample_sd
    stride = int(np.ceil(freq / 5))
    if medium % stride:
        # Preserve exact subsample alignment for uncommon non-divisible rates.
        return _epoch_scores(axes, freq, medium, count, window_epochs=window_epochs)
    if not count:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    block = medium // stride
    flags = np.zeros((count, 3), dtype=bool)
    clipping = np.zeros(count)
    starts = np.arange(count) * medium
    for col, axis in enumerate(axes):
        absolute = np.abs(axis[:count*medium])
        largest = np.maximum.reduceat(absolute, starts)
        fraction = np.add.reduceat((absolute > 7.5).astype(np.int32), starts) / medium
        clipping = np.maximum(clipping, np.where(largest > 11.25, 1., fraction))
        x = axis[::stride]
        offsets = np.arange(0, len(x), block)
        mins, maxs = np.minimum.reduceat(x, offsets), np.maximum.reduceat(x, offsets)
        lo = np.lib.stride_tricks.sliding_window_view(
            np.pad(mins, (0, window_epochs-1), constant_values=np.inf), window_epochs).min(axis=1)
        hi = np.lib.stride_tricks.sliding_window_view(
            np.pad(maxs, (0, window_epochs-1), constant_values=-np.inf), window_epochs).max(axis=1)
        left = np.arange(count) * block
        right = np.minimum(left + window_epochs*block, len(x))
        centered = x - x.mean()
        sums = np.r_[0., np.cumsum(centered, dtype=np.float64)]
        squares = np.r_[0., np.cumsum(centered*centered, dtype=np.float64)]
        n = right-left
        total = sums[right]-sums[left]
        variance = np.maximum(0., ((squares[right]-squares[left])-total*total/n)/(n-1))
        std = np.sqrt(variance)
        for i in np.flatnonzero(np.abs(std-.013) < 1e-7):
            std[i] = _sample_sd(x[left[i]:right[i]])
        passing = (hi[:count]-lo[:count] < .050) & (std < .013)
        first = np.flatnonzero(passing)
        changes = np.zeros(count+1, dtype=int)
        np.add.at(changes, first, 1)
        np.add.at(changes, np.minimum(first+window_epochs, count), -1)
        flags[:, col] = np.cumsum(changes[:-1]) > 0
    scores = flags.sum(axis=1)
    for i in np.flatnonzero(scores == 1):
        if 0 < i < count-1 and scores[i-1] > 1 and scores[i+1] > 1:
            scores[i] = 2
    return scores, clipping

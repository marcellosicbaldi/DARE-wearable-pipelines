import numpy as np
import pandas as pd

def signal_fixpeaks(
    env_diff,
    sampling_rate=1000,
    iterative=True,
    show=False,
    interval_min=None,
    interval_max=None,
    relative_interval_min=None,
    relative_interval_max=None,
    robust=False,
    method="Kubios",
    **kwargs,
):
    """**Correct Erroneous Peak Placements**

    Identify and correct erroneous peak placements based on outliers in peak-to-peak differences
    (period).

    Parameters
    ----------
    peaks : list or array or DataFrame or Series or dict
        The samples at which the peaks occur. If an array is passed in, it is assumed that it was
        obtained with :func:`.signal_findpeaks`. If a DataFrame is passed in, it is assumed to be
        obtained with :func:`.ecg_findpeaks` or :func:`.ppg_findpeaks` and to be of the same length
        as the input signal.
    sampling_rate : int
        The sampling frequency of the signal that contains the peaks (in Hz, i.e., samples/second).
    iterative : bool
        Whether or not to apply the artifact correction repeatedly (results in superior artifact
        correction).
    show : bool
        Whether or not to visualize artifacts and artifact thresholds.
    interval_min : float
        Only when ``method = "neurokit"``. The minimum interval between the peaks (in seconds).
    interval_max : float
        Only when ``method = "neurokit"``. The maximum interval between the peaks (in seconds).
    relative_interval_min : float
        Only when ``method = "neurokit"``. The minimum interval between the peaks as relative to
        the sample (expressed in standard deviation from the mean).
    relative_interval_max : float
        Only when ``method = "neurokit"``. The maximum interval between the peaks as relative to
        the sample (expressed in standard deviation from the mean).
    robust : bool
        Only when ``method = "neurokit"``. Use a robust method of standardization (see
        :func:`.standardize`) for the relative thresholds.
    method : str
        Either ``"Kubios"`` or ``"neurokit"``. ``"Kubios"`` uses the artifact detection and
        correction described in Lipponen, J. A., & Tarvainen, M. P. (2019). Note that ``"Kubios"``
        is only meant for peaks in ECG or PPG. ``"neurokit"`` can be used with peaks in ECG, PPG,
        or respiratory data.
    **kwargs
        Other keyword arguments.

    Returns
    -------
    peaks_clean : array
        The corrected peak locations.
    artifacts : dict
        Only if ``method="Kubios"``. A dictionary containing the indices of artifacts, accessible
        with the keys ``"ectopic"``, ``"missed"``, ``"extra"``, and ``"longshort"``.

    See Also
    --------
    signal_findpeaks, ecg_findpeaks, ecg_peaks, ppg_findpeaks, ppg_peaks

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      # Simulate ECG data and add noisy period
      ecg = nk.ecg_simulate(duration=240, sampling_rate=250, noise=2, random_state=42)
      ecg[20000:30000] += np.random.uniform(size=10000)
      ecg[40000:43000] = 0

      # Identify and Correct Peaks using "Kubios" Method
      rpeaks_uncorrected = nk.ecg_findpeaks(ecg, method="pantompkins", x=250)

      @savefig p_signal_fixpeaks1.png scale=100%
      info, rpeaks_corrected = nk.signal_fixpeaks(
          rpeaks_uncorrected, sampling_rate=250, iterative=True, method="Kubios", show=True
      )
      @suppress
      plt.close()

    .. ipython:: python

      # Visualize Artifact Correction
      rate_corrected = nk.signal_rate(rpeaks_corrected, desired_length=len(ecg))
      rate_uncorrected = nk.signal_rate(rpeaks_uncorrected, desired_length=len(ecg))

      @savefig p_signal_fixpeaks2.png scale=100%
      nk.signal_plot(
          [rate_uncorrected, rate_corrected],
          labels=["Heart Rate Uncorrected", "Heart Rate Corrected"]
      )
      @suppress
      plt.close()

    .. ipython:: python

      import numpy as np

      # Simulate Abnormal Signals
      signal = nk.signal_simulate(duration=4, sampling_rate=1000, frequency=1)
      peaks_true = nk.signal_findpeaks(signal)["Peaks"]
      peaks = np.delete(peaks_true, [1])  # create gaps due to missing peaks

      signal = nk.signal_simulate(duration=20, sampling_rate=1000, frequency=1)
      peaks_true = nk.signal_findpeaks(signal)["Peaks"]
      peaks = np.delete(peaks_true, [5, 15])  # create gaps
      peaks = np.sort(np.append(peaks, [1350, 11350, 18350]))  # add artifacts

      # Identify and Correct Peaks using 'NeuroKit' Method
      info, peaks_corrected = nk.signal_fixpeaks(
          peaks=peaks, interval_min=0.5, interval_max=1.5, method="neurokit"
      )

      # Plot and shift original peaks to the right to see the difference.
      @savefig p_signal_fixpeaks3.png scale=100%
      nk.events_plot([peaks + 50, peaks_corrected], signal)
      @suppress
      plt.close()


    References
    ----------
    * Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate variability time
      series artefact correction using novel beat classification. Journal of medical engineering &
      technology, 43(3), 173-181. 10.1080/03091902.2019.1640306

    """
    # Format input
    # peaks = (peaks)

    info, peaks_clean = _signal_fixpeaks_kubios(
        env_diff, sampling_rate=sampling_rate, iterative=iterative, show=show, **kwargs
    )

    return info, peaks_clean


# =============================================================================
# Methods
# =============================================================================

def _signal_fixpeaks_kubios(
    env_diff, sampling_rate=1000, iterative=True, show=False, **kwargs
):
    """kubios method."""

    # Get corrected peaks and normal-to-normal intervals.
    artifacts, subspaces = _find_artifacts(env_diff, sampling_rate=sampling_rate, **kwargs)
    peaks_clean = _correct_artifacts(artifacts, env_diff)

    if iterative:
        # Iteratively apply the artifact correction until the number
        # of artifacts stops decreasing.
        n_artifacts_current = sum([len(i) for i in artifacts.values()])

        while True:
            new_artifacts, new_subspaces = _find_artifacts(
                peaks_clean, sampling_rate=sampling_rate, **kwargs
            )

            n_artifacts_previous = n_artifacts_current
            n_artifacts_current = sum([len(i) for i in new_artifacts.values()])
            if n_artifacts_current >= n_artifacts_previous:
                break
            artifacts = new_artifacts
            subspaces = new_subspaces
            peaks_clean = _correct_artifacts(artifacts, peaks_clean)

    artifacts["method"] = "kubios"
    artifacts.update(subspaces)

    return artifacts, peaks_clean


# =============================================================================
# Kubios: Lipponen & Tarvainen (2019).
# =============================================================================
def _find_artifacts(
    env_diff,
    c1=0.13,
    c2=0.17,
    alpha=5.2,
    window_width=91,
    medfilt_order=11,
    sampling_rate=1000,
):
    # Compute period series (make sure it has same numer of elements as peaks);
    # peaks are in samples, convert to seconds.
    # rr = np.ediff1d(peaks, to_begin=0) / sampling_rate
    # For subsequent analysis it is important that the first element has
    # a value in a realistic range (e.g., for median filtering).
    # rr[0] = np.mean(rr[1:])
    rr = env_diff
    # Artifact identification #################################################
    ###########################################################################

    # Compute dRRs: time series of differences of consecutive periods (dRRs).
    drrs = np.ediff1d(rr, to_begin=0)
    drrs[0] = np.mean(drrs[1:])
    # Normalize by threshold.
    th1 = _compute_threshold(drrs, alpha, window_width)
    # ignore division by 0 warning
    old_setting = np.seterr(divide="ignore", invalid="ignore")
    drrs /= th1
    # return old setting
    np.seterr(**old_setting)

    # Cast dRRs to subspace s12.
    # Pad drrs with one element.
    padding = 2
    drrs_pad = np.pad(drrs, padding, "reflect")

    s12 = np.zeros(drrs.size)
    for d in np.arange(padding, padding + drrs.size):
        if drrs_pad[d] > 0:
            s12[d - padding] = np.max([drrs_pad[d - 1], drrs_pad[d + 1]])
        elif drrs_pad[d] < 0:
            s12[d - padding] = np.min([drrs_pad[d - 1], drrs_pad[d + 1]])
    # Cast dRRs to subspace s22.
    s22 = np.zeros(drrs.size)
    for d in np.arange(padding, padding + drrs.size):
        if drrs_pad[d] >= 0:
            s22[d - padding] = np.min([drrs_pad[d + 1], drrs_pad[d + 2]])
        elif drrs_pad[d] < 0:
            s22[d - padding] = np.max([drrs_pad[d + 1], drrs_pad[d + 2]])
    # Compute mRRs: time series of deviation of RRs from median.
    df = pd.DataFrame({"signal": rr})
    medrr = df.rolling(medfilt_order, center=True, min_periods=1).median().signal.values
    mrrs = rr - medrr
    mrrs[mrrs < 0] = mrrs[mrrs < 0] * 2
    # Normalize by threshold.
    th2 = _compute_threshold(mrrs, alpha, window_width)
    mrrs /= th2

    # Artifact classification #################################################
    ###########################################################################

    # Artifact classes.
    extra_idcs = []
    missed_idcs = []
    ectopic_idcs = []
    longshort_idcs = []

    i = 0
    while i < rr.size - 2:  # The flow control is implemented based on Figure 1
        if np.abs(drrs[i]) <= 1:  # Figure 1
            i += 1
            continue
        eq1 = np.logical_and(
            drrs[i] > 1, s12[i] < (-c1 * drrs[i] - c2)
        )  # pylint: disable=E1111
        eq2 = np.logical_and(
            drrs[i] < -1, s12[i] > (-c1 * drrs[i] + c2)
        )  # pylint: disable=E1111

        if np.any([eq1, eq2]):
            # If any of the two equations is true.
            ectopic_idcs.append(i)
            i += 1
            continue
        # If none of the two equations is true.
        if ~np.any([np.abs(drrs[i]) > 1, np.abs(mrrs[i]) > 3]):  # Figure 1
            i += 1
            continue
        longshort_candidates = [i]
        # Check if the following beat also needs to be evaluated.
        if np.abs(drrs[i + 1]) < np.abs(drrs[i + 2]):
            longshort_candidates.append(i + 1)
        for j in longshort_candidates:
            # Long beat.
            eq3 = np.logical_and(drrs[j] > 1, s22[j] < -1)  # pylint: disable=E1111
            # Long or short.
            eq4 = np.abs(mrrs[j]) > 3  # Figure 1
            # Short beat.
            eq5 = np.logical_and(drrs[j] < -1, s22[j] > 1)  # pylint: disable=E1111

            if ~np.any([eq3, eq4, eq5]):
                # If none of the three equations is true: normal beat.
                i += 1
                continue
            # If any of the three equations is true: check for missing or extra
            # peaks.

            # Missing.
            eq6 = np.abs(rr[j] / 2 - medrr[j]) < th2[j]  # Figure 1
            # Extra.
            eq7 = np.abs(rr[j] + rr[j + 1] - medrr[j]) < th2[j]  # Figure 1

            # Check if extra.
            if np.all([eq5, eq7]):
                extra_idcs.append(j)
                i += 1
                continue
            # Check if missing.
            if np.all([eq3, eq6]):
                missed_idcs.append(j)
                i += 1
                continue
            # If neither classified as extra or missing, classify as "long or
            # short".
            longshort_idcs.append(j)
            i += 1
    # Prepare output
    artifacts = {
        "ectopic": ectopic_idcs,
        "missed": missed_idcs,
        "extra": extra_idcs,
        "longshort": longshort_idcs,
    }

    subspaces = {
        "rr": rr,
        "drrs": drrs,
        "mrrs": mrrs,
        "s12": s12,
        "s22": s22,
        "c1": c1,
        "c2": c2,
    }

    return artifacts, subspaces


def _compute_threshold(signal, alpha, window_width):
    df = pd.DataFrame({"signal": np.abs(signal)})
    q1 = (
        df.rolling(window_width, center=True, min_periods=1)
        .quantile(0.25)
        .signal.values
    )
    q3 = (
        df.rolling(window_width, center=True, min_periods=1)
        .quantile(0.75)
        .signal.values
    )
    th = alpha * ((q3 - q1) / 2)

    return th


def _correct_artifacts(artifacts, env_diff):
    # Artifact correction
    #####################
    # The integrity of indices must be maintained if peaks are inserted or
    # deleted: for each deleted beat, decrease indices following that beat in
    # all other index lists by 1. Likewise, for each added beat, increment the
    # indices following that beat in all other lists by 1.
    extra_idcs = artifacts["extra"]
    missed_idcs = artifacts["missed"]
    ectopic_idcs = artifacts["ectopic"]
    longshort_idcs = artifacts["longshort"]

    # Delete extra peaks.
    if extra_idcs:
        peaks = _correct_extra(extra_idcs, env_diff)
        # Update remaining indices.
        missed_idcs = _update_indices(extra_idcs, missed_idcs, -1)
        ectopic_idcs = _update_indices(extra_idcs, ectopic_idcs, -1)
        longshort_idcs = _update_indices(extra_idcs, longshort_idcs, -1)
    # Add missing peaks.
    if missed_idcs:
        env_diff = _correct_missed(missed_idcs, env_diff)
        # Update remaining indices.
        ectopic_idcs = _update_indices(missed_idcs, ectopic_idcs, 1)
        longshort_idcs = _update_indices(missed_idcs, longshort_idcs, 1)
    if ectopic_idcs:
        env_diff = _correct_misaligned(ectopic_idcs, env_diff)
    if longshort_idcs:
        env_diff = _correct_misaligned(longshort_idcs, env_diff)
    return env_diff


def _correct_extra(extra_idcs, peaks):
    corrected_peaks = peaks.copy()
    corrected_peaks = np.delete(corrected_peaks, extra_idcs)

    return corrected_peaks


def _correct_missed(missed_idcs, peaks):
    corrected_peaks = peaks.copy()
    missed_idcs = np.array(missed_idcs)
    # Calculate the position(s) of new beat(s). Make sure to not generate
    # negative indices. prev_peaks and next_peaks must have the same
    # number of elements.
    valid_idcs = np.logical_and(
        missed_idcs > 1, missed_idcs < len(corrected_peaks)
    )  # pylint: disable=E1111
    missed_idcs = missed_idcs[valid_idcs]
    prev_peaks = corrected_peaks[[i - 1 for i in missed_idcs]]
    next_peaks = corrected_peaks[missed_idcs]
    added_peaks = prev_peaks + (next_peaks - prev_peaks) / 2
    # Add the new peaks before the missed indices (see numpy docs).
    corrected_peaks = np.insert(corrected_peaks, missed_idcs, added_peaks)

    return corrected_peaks


def _correct_misaligned(misaligned_idcs, peaks):
    corrected_peaks = peaks.copy()
    misaligned_idcs = np.array(misaligned_idcs)
    # Make sure to not generate negative indices, or indices that exceed
    # the total number of peaks. prev_peaks and next_peaks must have the
    # same number of elements.
    valid_idcs = np.logical_and(
        misaligned_idcs > 1,
        misaligned_idcs < len(corrected_peaks) - 1,  # pylint: disable=E1111
    )
    misaligned_idcs = misaligned_idcs[valid_idcs]
    prev_peaks = corrected_peaks[[i - 1 for i in misaligned_idcs]]
    next_peaks = corrected_peaks[[i + 1 for i in misaligned_idcs]]

    half_ibi = (next_peaks - prev_peaks) / 2
    peaks_interp = prev_peaks + half_ibi
    # Shift the R-peaks from the old to the new position.
    corrected_peaks = np.delete(corrected_peaks, misaligned_idcs)
    corrected_peaks = np.concatenate((corrected_peaks, peaks_interp)).astype(int)
    corrected_peaks.sort(kind="mergesort")

    return corrected_peaks


def _update_indices(source_idcs, update_idcs, update):
    """For every element s in source_idcs, change every element u in update_idcs according to update, if u is larger
    than s."""
    if not update_idcs:
        return update_idcs
    for s in source_idcs:
        update_idcs = [u + update if u > s else u for u in update_idcs]
    return list(np.unique(update_idcs))
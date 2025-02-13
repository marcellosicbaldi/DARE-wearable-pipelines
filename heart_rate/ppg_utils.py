from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend


def MSPTDfast(
    signal,
    sampling_rate=1000,
    show=False,
):
    """Implementation of Charlton et al (2024) MSPTDfast: An Efficient Photoplethysmography
    Beat Detection Algorithm. 2024 Computing in Cardiology (CinC), Karlsruhe, Germany,
    doi:10.1101/2024.07.18.24310627.
    ** Updated and faster version of Aboy++ **
    """

    # Inner functions

    def find_m_max(x, N, max_scale, m_max):
        """Find local maxima scalogram for peaks
        """

        for k in range(1, max_scale + 1):  # scalogram scales
            for i in range(k + 2, N - k + 2):
                if x[i - 2] > x[i - k - 2] and x[i - 2] > x[i + k - 2]:
                    m_max[k - 1, i - 2] = True

        return m_max

    def find_m_min(x, N, max_scale, m_min):
        """Find local minima scalogram for onsets
        """

        for k in range(1, max_scale + 1):  # scalogram scales
            for i in range(k + 2, N - k + 2):
                if x[i - 2] < x[i - k - 2] and x[i - 2] < x[i + k - 2]:
                    m_min[k - 1, i - 2] = True

        return m_min

    def find_lms_using_msptd_approach(max_scale, x, options):
        """Find local maxima (or minima) scalogram(s) using the
        MSPTD approach
        """

        # Setup
        N = len(x)

        # Find local maxima scalogram (if required)
        if options["find_pks"]:
            m_max = np.full((max_scale, N), False)  # matrix for maxima
            m_max = find_m_max(x, N, max_scale, m_max)
        else:
            m_max = None

        # Find local minima scalogram (if required)
        if options["find_trs"]:
            m_min = np.full((max_scale, N), False)  # matrix for minima
            m_min = find_m_min(x, N, max_scale, m_min)
        else:
            m_min = None

        return m_max, m_min

    def downsample(win_sig, ds_factor):
        """Downsamples signal by picking out every nth sample, where n is
        specified by ds_factor
        """

        return win_sig[::ds_factor]

    def detect_peaks_and_onsets_using_msptd(signal, fs, options):
        """Detect peaks and onsets in a PPG signal using a modified MSPTD approach
        (where the modifications are those specified in Charlton et al. 2024)
        """

        # Setup
        N = len(signal)
        L = int(np.ceil(N / 2) - 1)

        # Step 0: Don't calculate scales outside the range of plausible HRs

        plaus_hr_hz = np.array(options['plaus_hr_bpm']) / 60  # in Hz
        init_scales = np.arange(1, L + 1)
        durn_signal = len(signal) / fs
        init_scales_fs = (L / init_scales) / durn_signal
        if options['use_reduced_lms_scales']:
            init_scales_inc_log = init_scales_fs >= plaus_hr_hz[0]
        else:
            init_scales_inc_log = np.ones_like(init_scales_fs, dtype=bool)  # DIDN"T FULLY UNDERSTAND

        max_scale_index = np.where(init_scales_inc_log)[0]  # DIDN"T FULLY UNDERSTAND THIS AND NEXT FEW LINES
        if max_scale_index.size > 0:
            max_scale = max_scale_index[-1] + 1  # Add 1 to convert from 0-based to 1-based index
        else:
            max_scale = None  # Or handle the case where no scales are valid

        # Step 1: calculate local maxima and local minima scalograms

        # - detrend
        x = detrend(signal, type="linear")

        # - populate LMS matrices
        [m_max, m_min] = find_lms_using_msptd_approach(max_scale, x, options)

        # Step 2: find the scale with the most local maxima (or local minima)

        # - row-wise summation (i.e. sum each row)
        if options["find_pks"]:
            gamma_max = np.sum(m_max, axis=1)  # the "axis=1" option makes it row-wise
        if options["find_trs"]:
            gamma_min = np.sum(m_min, axis=1)
        # - find scale with the most local maxima (or local minima)
        if options["find_pks"]:
            lambda_max = np.argmax(gamma_max)
        if options["find_trs"]:
            lambda_min = np.argmax(gamma_min)

        # Step 3: Use lambda to remove all elements of m for which k>lambda
        first_scale_to_include = np.argmax(init_scales_inc_log)
        if options["find_pks"]:
            m_max = m_max[first_scale_to_include:lambda_max + 1, :]
        if options["find_trs"]:
            m_min = m_min[first_scale_to_include:lambda_min + 1, :]

        # Step 4: Find peaks (and onsets)
        # - column-wise summation
        if options["find_pks"]:
            m_max_sum = np.sum(m_max == False, axis=0)
            peaks = np.where(m_max_sum == 0)[0].astype(int)
        else:
            peaks = []

        if options["find_trs"]:
            m_min_sum = np.sum(m_min == False, axis=0)
            onsets = np.where(m_min_sum == 0)[0].astype(int)
        else:
            onsets = []

        return peaks, onsets

    # ~~~ Main function ~~~

    # Specify settings
    # - version: optimal selection (CinC 2024)
    options = {
        'find_trs': True,  # whether or not to find onsets
        'find_pks': True,  # whether or not to find peaks
        'do_ds': True,  # whether or not to do downsampling
        'ds_freq': 20,  # the target downsampling frequency
        'use_reduced_lms_scales': True,  # whether or not to reduce the number of scales (default 30 bpm)
        'win_len': 8,  # duration of individual windows for analysis
        'win_overlap': 0.2,  # proportion of window overlap
        'plaus_hr_bpm': [30, 200]  # range of plausible HRs (only the lower bound is used)
    }

    # Split into overlapping windows
    no_samps_in_win = options["win_len"] * sampling_rate
    if len(signal) <= no_samps_in_win:
        win_starts = 0
        win_ends = len(signal) - 1
    else:
        win_offset = int(round(no_samps_in_win * (1 - options["win_overlap"])))
        win_starts = list(range(0, len(signal) - no_samps_in_win + 1, win_offset))
        win_ends = [start + 1 + no_samps_in_win for start in win_starts]
        if win_ends[-1] < len(signal):
            win_starts.append(len(signal) - 1 - no_samps_in_win)
            win_ends.append(len(signal))
        # this ensures that the windows include the entire signal duration

    # Set up downsampling if the sampling frequency is particularly high
    if options["do_ds"]:
        min_fs = options["ds_freq"]
        if sampling_rate > min_fs:
            ds_factor = int(np.floor(sampling_rate / min_fs))
            ds_fs = sampling_rate / np.floor(sampling_rate / min_fs)
        else:
            options["do_ds"] = False

    # detect peaks and onsets in each window
    peaks = []
    onsets = []

    # cycle through each window
    for win_no in range(len(win_starts)):
        # Extract this window's data
        win_sig = signal[win_starts[win_no]:win_ends[win_no]]

        # Downsample signal
        if options['do_ds']:
            rel_sig = downsample(win_sig, ds_factor)
            rel_fs = ds_fs
        else:
            rel_sig = win_sig
            rel_fs = sampling_rate

        # Detect peaks and onsets
        p, t = detect_peaks_and_onsets_using_msptd(rel_sig, rel_fs, options)

        # Resample peaks
        if options['do_ds']:
            p = [peak * ds_factor for peak in p]
            t = [onset * ds_factor for onset in t]

        # Correct peak indices by finding highest point within tolerance either side of detected peaks
        tol_durn = 0.05
        if rel_fs < 10:
            tol_durn = 0.2
        elif rel_fs < 20:
            tol_durn = 0.1
        tol = int(np.ceil(rel_fs * tol_durn))

        for pk_no in range(len(p)):
            segment = win_sig[(p[pk_no] - tol):(p[pk_no] + tol + 1)]
            temp = np.argmax(segment)
            p[pk_no] = p[pk_no] - tol + temp

        # Correct onset indices by finding highest point within tolerance either side of detected onsets
        for onset_no in range(len(t)):
            segment = win_sig[(t[onset_no] - tol):(t[onset_no] + tol + 1)]
            temp = np.argmin(segment)
            t[onset_no] = t[onset_no] - tol + temp

        # Store peaks and onsets
        win_peaks = [peak + win_starts[win_no] for peak in p]
        peaks.extend(win_peaks)
        win_onsets = [onset + win_starts[win_no] for onset in t]
        onsets.extend(win_onsets)

    # Tidy up detected peaks and onsets (by ordering them and only retaining unique ones)
    peaks = sorted(set(peaks))
    onsets = sorted(set(onsets))

    # Plot results (optional)
    if show:
        _, ax0 = plt.subplots(nrows=1, ncols=1, sharex=True)
        ax0.plot(signal, label="signal")
        ax0.scatter(peaks, signal[peaks], c="r")
        ax0.scatter(onsets, signal[onsets], c="b")
        ax0.set_title("PPG Onsets (Method by Charlton et al., 2024)")

    return peaks, onsets

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

    if show:
        _plot_artifacts_lipponen2019(artifacts)

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


def _plot_artifacts_lipponen2019(info):
    # Covnenience function to extract relevant stuff.
    def _get_which_endswith(info, string):
        return [s for key, s in info.items() if key.endswith(string)][0]

    # Extract parameters
    longshort_idcs = _get_which_endswith(info, "longshort")
    ectopic_idcs = _get_which_endswith(info, "ectopic")
    extra_idcs = _get_which_endswith(info, "extra")
    missed_idcs = _get_which_endswith(info, "missed")

    # Extract subspace info
    rr = _get_which_endswith(info, "rr")
    drrs = _get_which_endswith(info, "drrs")
    mrrs = _get_which_endswith(info, "mrrs")
    s12 = _get_which_endswith(info, "s12")
    s22 = _get_which_endswith(info, "s22")
    c1 = _get_which_endswith(info, "c1")
    c2 = _get_which_endswith(info, "c2")

    # Visualize artifact type indices.

    # Set grids
    gs = matplotlib.gridspec.GridSpec(ncols=2, nrows=6)
    fig = plt.figure(constrained_layout=False, figsize=(17, 12))
    fig.suptitle("Peak Correction", fontweight="bold")
    ax0 = fig.add_subplot(gs[0:2, 0])
    ax1 = fig.add_subplot(gs[2:4, 0])
    ax2 = fig.add_subplot(gs[4:6, 0])
    ax3 = fig.add_subplot(gs[0:3:, 1])
    ax4 = fig.add_subplot(gs[3:6, 1])

    ax0.set_title("Artifact types")
    ax0.plot(rr, label="heart period")
    ax0.scatter(
        longshort_idcs,
        rr[longshort_idcs],
        marker="x",
        c="m",
        s=100,
        zorder=3,
        label="long/short",
    )
    ax0.scatter(
        ectopic_idcs,
        rr[ectopic_idcs],
        marker="x",
        c="g",
        s=100,
        zorder=3,
        label="ectopic",
    )
    ax0.scatter(
        extra_idcs,
        rr[extra_idcs],
        marker="x",
        c="y",
        s=100,
        zorder=3,
        label="false positive",
    )
    ax0.scatter(
        missed_idcs,
        rr[missed_idcs],
        marker="x",
        c="r",
        s=100,
        zorder=3,
        label="false negative",
    )
    ax0.legend(loc="upper right")

    # Visualize first threshold.
    ax1.set_title("Consecutive-difference criterion")
    ax1.plot(np.abs(drrs), label="normalized difference consecutive heart periods")
    ax1.axhline(1, c="r", label="artifact threshold")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 5)

    # Visualize second threshold.
    ax2.set_title("Difference-from-median criterion")
    ax2.plot(np.abs(mrrs), label="difference from median over 11 periods")
    ax2.axhline(3, c="r", label="artifact threshold")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0, 5)

    # Visualize subspaces.
    ax4.set_title("Subspace 1")
    ax4.set_xlabel("S11")
    ax4.set_ylabel("S12")
    ax4.scatter(drrs, s12, marker="x", label="heart periods")
    ax4.set_ylim(-5, 5)
    ax4.set_xlim(-10, 10)
    verts0 = [(-10, 5), (-10, -c1 * -10 + c2), (-1, -c1 * -1 + c2), (-1, 5)]

    poly0 = matplotlib.patches.Polygon(
        verts0, alpha=0.3, facecolor="r", edgecolor=None, label="ectopic periods"
    )
    ax4.add_patch(poly0)
    verts1 = [(1, -c1 * 1 - c2), (1, -5), (10, -5), (10, -c1 * 10 - c2)]
    poly1 = matplotlib.patches.Polygon(verts1, alpha=0.3, facecolor="r", edgecolor=None)
    ax4.add_patch(poly1)
    ax4.legend(loc="upper right")

    ax3.set_title("Subspace 2")
    ax3.set_xlabel("S21")
    ax3.set_ylabel("S22")
    ax3.scatter(drrs, s22, marker="x", label="heart periods")
    ax3.set_xlim(-10, 10)
    ax3.set_ylim(-10, 10)
    verts2 = [(-10, 10), (-10, 1), (-1, 1), (-1, 10)]
    poly2 = matplotlib.patches.Polygon(
        verts2, alpha=0.3, facecolor="r", edgecolor=None, label="short periods"
    )
    ax3.add_patch(poly2)
    verts3 = [(1, -1), (1, -10), (10, -10), (10, -1)]
    poly3 = matplotlib.patches.Polygon(
        verts3, alpha=0.3, facecolor="y", edgecolor=None, label="long periods"
    )
    ax3.add_patch(poly3)
    ax3.legend(loc="upper right")
    plt.tight_layout()
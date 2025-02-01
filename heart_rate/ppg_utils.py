from warnings import warn

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dotmap import DotMap
from scipy.signal import kaiserord, firwin, filtfilt, detrend, periodogram, lfilter, find_peaks, firls, resample
from scipy import signal
import copy

def get_peak_onset(ppg, fs, peak_detector='PPGdet'):
        '''PPGdet detects beats in a photoplethysmogram (PPG) signal
        using the improved 'Automatic Beat Detection' of Aboy M et al.

        '''

        # inputs
        x = copy.deepcopy(ppg)                    #signal
        fso=fs
        # fs = 64
        x = resample(x, int(len(ppg)*(fs/fso)))
        up = set_beat_detection()                 #settings
        win_sec=10
        w = fs * win_sec                                    #window length(number of samples)
        win_starts = np.array(list(range(0,len(x),round(0.8*w))))
        win_starts = win_starts[0:min(np.where([win_starts >= len(x) - w])[1])]
        win_starts = np.insert(win_starts,len(win_starts), len(x) + 1 - w)

        # before pre-processing
        hr_win=0  #the estimated systolic peak-to-peak distance, initially it is 0
        hr_win_v=[]
        px = detect_maxima(x, 0, hr_win, peak_detector) # detect all maxima
        if len(px)==0:
            peaks = []
            onsets = []
            return peaks, onsets

        # detect peaks in windows
        all_p4 = []
        all_hr = np.empty(len(win_starts)-1)
        all_hr [:] = np.nan
        hr_past = 0 # the actual heart rate
        hrvi = 0    # heart rate variability index

        for win_no in range(0,len(win_starts) - 1):
            curr_els = range(win_starts[win_no],win_starts[win_no] + w)
            curr_x = x[curr_els]

            y1 = def_bandpass(sig = curr_x, fs = fs, lower_cutoff = 0.9 * 30/60, upper_cutoff = 3 * 200/60)   # Filter no.1
            hr = estimate_HR(y1, fs, up, hr_past)               # Estimate HR from weakly filtered signal
            hr_past=hr
            all_hr[win_no] = hr

            if (peak_detector=='PPGdet') and (hr>40):
                if win_no==0:
                    p1 = detect_maxima(y1, 0, hr_win, peak_detector)
                    tr = np.percentile(np.diff(p1), 50)
                    pks_diff = np.diff(p1)
                    pks_diff = pks_diff[pks_diff>=tr]
                    hrvi = np.std(pks_diff) / np.mean(pks_diff) * 5

                hr_win = fs / ((1 + hrvi) * 3)
                hr_win_v.append(hr_win)
            else:
                hr_win=0

            y2 = def_bandpass(curr_x, fs, 0.9 * up.fl_hz, 2.5 * hr / 60)           # Filter no. 2
            y2_deriv = estimate_deriv(y2)                                          # Estimate derivative from highly filtered signal
            p2 = detect_maxima(y2_deriv, up.deriv_threshold,hr_win, peak_detector) # Detect maxima in derivative
            y3 = def_bandpass(curr_x, fs, 0.9 * up.fl_hz, 10 * hr / 60)
            p3 = detect_maxima(y3, 50, hr_win, peak_detector)                      # Detect maxima in moderately filtered signal
            p4 = find_pulse_peaks(p2, p3)
            p4 = np.unique(p4)

            if peak_detector=='PPGdet':
                if len(p4)>round(win_sec/2):
                    pks_diff = np.diff(p4)
                    tr = np.percentile(pks_diff, 30)
                    pks_diff = pks_diff[pks_diff >= tr]

                    med_hr=np.median(all_hr[np.where(all_hr>0)])
                    if ((med_hr*0.5<np.mean(pks_diff)) and (med_hr*1.5<np.mean(pks_diff))):
                        hrvi = np.std(pks_diff) / np.mean(pks_diff)*10

            all_p4 = np.concatenate((all_p4, win_starts[win_no] + p4), axis=None)

        all_p4=all_p4.astype(int)
        all_p4 = np.unique(all_p4)

        peaks = (all_p4/fs*fso).astype(int)
        onsets, peaks = find_onsets(ppg, fso, up, peaks,60/np.median(all_hr)*fs)
        return peaks, onsets




def detect_maxima(sig: np.array, percentile: int ,hr_win: int, peak_detector: str):
        #Table VI pseudocode
        """
        Detect Maxima function detects all peaks in the raw and also in the filtered signal to find.

        :param sig: array of signal with shape (N,) where N is the length of the signal
        :type sig: 1-d array
        :param percentile: in each signal partition, a rank filter detects the peaks above a given percentile
        :type percentile: int
        :param hr_win: window for adaptive the heart rate estimate
        :type hr_win: int
        :param peak_detector: type of peak detector
        :type peak_detector: str

        :return: maximum peaks of signal, 1-d array.

        """

        tr = np.percentile(sig, percentile)

        if peak_detector=='ABD':

            s1,s2,s3 = sig[2:], sig[1:-1],sig[0:-2]
            m = 1 + np.array(np.where((s1 < s2) & (s3 < s2)))
            max_pks = m[sig[m] > tr]

        if peak_detector=='PPGdet':
            s1,s2,s3 = sig[2:], sig[1:-1],sig[0:-2]

            max_loc = []
            min_loc = []
            max_pks=[]
            intensity_v = []
            if hr_win == 0:
                m = 1 + np.array(np.where((s1 < s2) & (s3 < s2)))
                max_pks = m[sig[m] > tr]
            else:
                max_loc = find_peaks(sig, distance=hr_win)[0]
                min_loc = find_peaks(-sig, distance=hr_win)[0]

                for i in range(0,len(max_loc)):
                    values = abs(max_loc[i] - min_loc)
                    min_v = min(values)
                    min_i = np.where(min_v==values)[0][0]
                    intensity_v.append(sig[max_loc[i]] - sig[min_loc[min_i]])

                # possible improvements:
                #   - using adaptive thresholding for the maximum
                #   - estimate probability density of the maximum

                tr2 = np.mean(intensity_v)*0.25
                max_pks = find_peaks(sig+min(sig),prominence=tr2,distance=hr_win)[0]

        return max_pks

def find_pulse_peaks(p2: np.array, p3: np.array):
        """
        Pulse detection function detect the pulse peaks according to the peaks of 1st and 2nd derivatives
        General least-squares smoothing and differentiation by the convolution (Savitzky Golay) method

        :param p2: peaks of the 1st derivatives
        :type p2: 1-d array
        :param p3: peaks of the 2nd derivatives
        :type p2: 1-d array

        :return: pulse peaks, 1-d array.

        """

        p4 = np.empty(len(p2))
        p4[:] = np.nan
        for k in range(0,len(p2)):
            rel_el = np.where(p3>p2[k])
            if np.any(rel_el) and ~np.isnan(rel_el[0][0]):
                p4[k] = p3[rel_el[0][0]]

        p4 = p4[np.where(~np.isnan(p4))]
        p4 = p4.astype(int)
        return p4

def set_beat_detection():
        """
        This function setups the filter parameters of the algorithm

        :return: filter parameters of the algorithm, DotMap

        """
        # plausible HR limits
        up=DotMap()
        up.fl = 30               #lower bound for HR
        up.fh = 200              #upper bound for HR
        up.fl_hz = up.fl/60
        up.fh_hz = up.fh/60

        # Thresholds
        up.deriv_threshold = 75          #originally 90
        up.upper_hr_thresh_prop = 2.25   #originally 1.75
        up.lower_hr_thresh_prop = 0.5    #originally 0.75

        # Other parameters
        up.win_size = 10    #in secs

        return up

def def_bandpass(sig: np.array, fs: int, lower_cutoff: float, upper_cutoff: float):
        """
        def_bandpass filter function detects all peaks in the raw and also in the filtered signal to find.

        :param sig: array of signal with shape (N,) where N is the length of the signal
        :type sig: 1-d array
        :param fs: sampling frequency
        :type fs: int
        :param lower_cutoff: lower cutoff frequency
        :type lower_cutoff: float
        :param upper_cutoff: upper cutoff frequency
        :type upper_cutoff: float

        :return: bandpass filtered signal, 1-d array.

        """

        # Filter characteristics: Eliminate VLFs (below resp freqs): For 4bpm cutoff
        up = DotMap()
        up.paramSet.elim_vlf.Fpass = 1.3*lower_cutoff   #in Hz
        up.paramSet.elim_vlf.Fstop = 0.8*lower_cutoff   #in Hz
        up.paramSet.elim_vlf.Dpass = 0.05
        up.paramSet.elim_vlf.Dstop = 0.01

        # Filter characteristics: Eliminate VHFs (above frequency content of signals)
        up.paramSet.elim_vhf.Fpass = 1.2*upper_cutoff   #in Hz
        up.paramSet.elim_vhf.Fstop = 0.8*upper_cutoff   #in Hz
        up.paramSet.elim_vhf.Dpass = 0.05
        up.paramSet.elim_vhf.Dstop = 0.03

        # perform BPF
        s = DotMap()
        s.v = sig
        s.fs = fs

        b, a = signal.iirfilter(5, [2 * np.pi * lower_cutoff, 2 * np.pi * upper_cutoff], rs=60,
                                btype='band', analog=True, ftype='cheby2')

        bpf_sig = filtfilt(b, 1, s.v)

        return bpf_sig
def estimate_HR(sig: np.array, fs: int, up: DotMap, hr_past: int):
        """
        Heart Rate Estimation function estimate the heart rate according to the previous heart rate in given time window

        :param sig: array of signal with shape (N,) where N is the length of the signal
        :type sig: 1-d array
        :type fs: int
        :param up: setup up parameters of the algorithm
        :type up: DotMap
        :param hr_past: the average heart rate in the past in given time window
        :type hr_past: int

        :return: estimated heart rate, 1-d array.

        """

        # Estimate PSD
        blackman_window = np.blackman(len(sig))
        f, pxx = periodogram(sig,fs, blackman_window)
        ph = pxx
        fh = f

        # Extract HR
        if (hr_past / 60 < up.fl_hz) | (hr_past / 60 > up.fh_hz):
            rel_els = np.where((fh >= up.fl_hz) & (fh <= up.fh_hz))
        else:
            rel_els = np.where((fh >= hr_past / 60 * 0.5) & (fh <= hr_past / 60 * 1.4))

        rel_p = ph[rel_els]
        rel_f = fh[rel_els]
        max_el = np.where(rel_p==max(rel_p))
        hr = rel_f[max_el]*60
        hr = int(hr[0])

        return hr    

def estimate_deriv(sig: np.array):
        """
        Derivative Estimation function estimate derivative from highly filtered signal based on the
        General least-squares smoothing and differentiation by the convolution (Savitzky Golay) method

        :param sig: array of signal with shape (N,) where N is the length of the signal
        :type sig: 1-d array

        :return: derivative, 1-d array.

        """

        #Savitzky Golay
        deriv_no = 1
        win_size = 5
        deriv = savitzky_golay(sig, deriv_no, win_size)

        return deriv

def find_onsets(sig: np.array, fs: int, up: DotMap, peaks: np.array, med_hr: float):
        """
        This function finds the onsets of PPG sigal

        :param sig: array of signal with shape (N,) where N is the length of the signal
        :type sig: 1-d array
        :param fs: sampling frequency
        :type fs: int
        :param up: setup up parameters of the algorithm
        :type up: DotMap
        :param peaks: peaks of the signal
        :type peaks: 1-d array
        :param med_hr: median heart rate
        :type med_hr: float

        :return: onsets, 1-d array

        """

        Y1=def_bandpass(sig, fs, 0.9*up.fl_hz, 3*up.fh_hz)
        temp_oi0=find_peaks(-Y1,distance=med_hr*0.3)[0]

        null_indexes = np.where(temp_oi0<peaks[0])
        if len(null_indexes[0])!=0:
            if len(null_indexes[0])==1:
                onsets = [null_indexes[0][0]]
            else:
                onsets = [null_indexes[0][-1]]
        else:
            onsets = [peaks[0]-round(fs/50)]

        i=1
        while i < len(peaks):
            min_SUT=fs*0.12     # minimum Systolic Upslope Time 120 ms
            min_DT=fs*0.3       # minimum Diastolic Time 300 ms

            before_peak=temp_oi0 <peaks[i]
            after_last_onset=temp_oi0 > onsets[i - 1]
            SUT_time=peaks[i]-temp_oi0>min_SUT
            DT_time = temp_oi0-peaks[i-1]  > min_DT
            temp_oi1 = temp_oi0[np.where(before_peak * after_last_onset*SUT_time*DT_time)]
            if len(temp_oi1)>0:
                if len(temp_oi1) == 1:
                    onsets.append(temp_oi1[0])
                else:
                    onsets.append(temp_oi1[-1])
                i=i+1
            else:
                peaks = np.delete(peaks, i)

        return onsets,peaks

def savitzky_golay(sig: np.array, deriv_no: int, win_size: int):
        """
        This function estimate the Savitzky Golay derivative from highly filtered signal

        :param sig: array of signal with shape (N,) where N is the length of the signal
        :type sig: 1-d array
        :param deriv_no: number of derivative
        :type deriv_no: int
        :param win_size: size of window
        :type win_size: int

        :return: Savitzky Golay derivative, 1-d array.

        """

        ##assign coefficients
        # From: https: // en.wikipedia.org / wiki / Savitzky % E2 % 80 % 93 Golay_filter  # Tables_of_selected_convolution_coefficients
        # which are calculated from: A., Gorry(1990). "General least-squares smoothing and differentiation by the convolution (Savitzky?Golay) method".Analytical Chemistry. 62(6): 570?3. doi: 10.1021 / ac00205a007.

        if deriv_no==0:
            #smoothing
            if win_size == 5:
                coeffs = [-3, 12, 17, 12, -3]
                norm_factor = 35
            elif win_size == 7:
                coeffs = [-2, 3, 6, 7, 6, 3, -2]
                norm_factor = 21
            elif win_size == 9:
                coeffs = [-21, 14, 39, 54, 59, 54, 39, 14, -21]
                norm_factor = 231
            else:
                print('Can''t do this window size')
        elif deriv_no==1:
            # first derivative
            if win_size == 5:
                coeffs = range(-2,3)
                norm_factor = 10
            elif win_size == 7:
                coeffs = range(-3,4)
                norm_factor = 28
            elif win_size == 9:
                coeffs = range(-4,5)
                norm_factor = 60
            else:
                print('Can''t do this window size')
        elif deriv_no == 2:
            # second derivative
            if win_size == 5:
                coeffs = [2, -1, -2, -1, 2]
                norm_factor = 7
            elif win_size == 7:
                coeffs = [5, 0, -3, -4, -3, 0, 5]
                norm_factor = 42
            elif win_size == 9:
                coeffs = [28, 7, -8, -17, -20, -17, -8, 7, 28]
                norm_factor = 462
            else:
                print('Can''t do this window size')
        elif deriv_no == 3:
            # third derivative
            if win_size == 5:
                coeffs = [-1, 2, 0, -2, 1]
                norm_factor = 2
            elif win_size == 7:
                coeffs = [-1, 1, 1, 0, -1, -1, 1]
                norm_factor = 6
            elif win_size == 9:
                coeffs = [-14, 7, 13, 9, 0, -9, -13, -7, 14]
                norm_factor = 198
            else:
                print('Can''t do this window size')
        elif deriv_no == 4:
            # fourth derivative
            if win_size == 7:
                coeffs = [3, -7, 1, 6, 1, -7, 3]
                norm_factor = 11
            elif win_size == 9:
                coeffs = [14, -21, -11, 9, 18, 9, -11, -21, 14]
                norm_factor = 143
            else:
                print('Can''t do this window size')
        else:
            print('Can''t do this order of derivative')


        if deriv_no % 2 == 1:
            coeffs = -np.array(coeffs)

        A = [1, 0]
        filtered_sig = lfilter(coeffs, A, sig)
        # filtered_sig = filtfilt(coeffs, A, sig)
        s = len(sig)
        half_win_size = np.floor(win_size * 0.5)
        zero_pad=filtered_sig[win_size] * np.ones(int(half_win_size))
        sig_in=filtered_sig[win_size-1:s]
        sig_end=filtered_sig[s-1] * np.ones(int(half_win_size))
        deriv = [*zero_pad,*sig_in,*sig_end]
        deriv = deriv / np.array(norm_factor)

        return deriv

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
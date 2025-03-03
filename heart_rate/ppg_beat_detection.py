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


## ABOY ++

from dotmap import DotMap
from scipy.signal import kaiserord, firwin, filtfilt, detrend, periodogram, lfilter, find_peaks, firls, resample
from scipy import signal
import copy

def aboy_plusplus(ppg, fs, peak_detector='PPGdet'):
        '''PPGdet detects beats in a photoplethysmogram (PPG) signal
        using the improved 'Automatic Beat Detection' of Aboy M et al.

        '''

        # inputs
        x = copy.deepcopy(ppg)                    #signal
        fso=fs
        fs = 75
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

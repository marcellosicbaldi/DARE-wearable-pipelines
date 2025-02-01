import neurokit2 as nk
import pandas as pd
import numpy as np

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Compute high and low envelopes of a signal s
    Parameters
    ----------
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases

    Returns
    -------
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

def compute_envelope(acc, resample = True):
    """
    Compute the envelope of the acceleration signal

    Parameters
    ----------
    acc : pd.Series
        Band-pass filtered accelerometer signal magnitude vector
    resample : bool, optional
        If True, resample the envelope to the original size of the signal

    Returns
    -------
    env_diff : pd.Series
        Envelope difference of the acceleration signal
    """

    lmin, lmax = hl_envelopes_idx(acc.values, dmin = 10, dmax = 10)

    # adjust shapes
    if len(lmin) > len(lmax):
        lmin = lmin[:-1]
    if len(lmax) > len(lmin):
        lmax = lmax[1:]
        
    upper_envelope = acc.values[lmax]
    lower_envelope = acc.values[lmin]
                                
    if resample:
        upper_envelope_res = np.interp(np.arange(len(acc)), lmax, upper_envelope)
        lower_envelope_res = np.interp(np.arange(len(acc)), lmin, lower_envelope)
        env_diff = pd.Series(upper_envelope_res - lower_envelope_res, index = acc.index)
    else:
        env_diff = pd.Series(upper_envelope - lower_envelope, index = acc.index[lmax])

    return env_diff

def detect_bursts(acc, sampling_rate, envelope = True, resample_envelope = True, alfa = None):
    """
    Detect bursts in acceleration signal

    Parameters
    ----------
    acc : pd.Series
        Band-pass filtered accelerometer signal magnitude vector
    envelope : bool, optional
        If True, detect bursts based on the envelope of the signal
        If False, detect bursts based on the std of the signal
    resample_envelope : bool, optional
        If True, resample the envelope to the original size of the signal
    alfa : float, optional
        Threshold for detecting bursts

    Returns
    -------
    bursts : pd.Series
        pd.DataFrame with burst start times, end times, duration, peak-to-peak amplitude, and AUC
    """

    # band-pass filter the signal
    acc = pd.Series(nk.signal_filter(acc.values, sampling_rate = sampling_rate, lowcut=0.1, highcut=10, method='butterworth', order=8), index = acc.index)

    if envelope:
        env_diff = compute_envelope(acc, resample = resample_envelope)
        th = alfa
    else:
        std_acc = acc.resample("1 s").std()
        std_acc.index.round("1 s")
        th = np.percentile(std_acc, 10) * alfa
        env_diff = std_acc

    bursts1 = (env_diff > th).astype(int)
    start_burst = bursts1.where(bursts1.diff()==1).dropna()
    end_burst = bursts1.where(bursts1.diff()==-1).dropna()
    if bursts1.iloc[0] == 1:
            start_burst = pd.concat([pd.Series(0, index = [bursts1.index[0]]), start_burst])
    if bursts1.iloc[-1] == 1:
        end_burst = pd.concat([end_burst, pd.Series(0, index = [bursts1.index[-1]])])
    bursts_df = pd.DataFrame({"duration": pd.to_datetime(end_burst.index) - pd.to_datetime(start_burst.index)}, index = start_burst.index)

    start = bursts_df.index
    end = pd.to_datetime((bursts_df.index + bursts_df["duration"]).values)

    end = end.to_series().reset_index(drop = True)
    start = start.to_series().reset_index(drop = True)

    duration_between_bursts = (start.iloc[1:].values - end.iloc[:-1].values)

    # If two bursts are too close to each other (5s), consider them as one burst
    for i in range(len(start)-1):
        if duration_between_bursts[i] < pd.Timedelta("5 s"):
            end[i] = np.nan
            start[i+1] = np.nan
    end.dropna(inplace = True)
    start.dropna(inplace = True)

    # extract amplitude of the bursts
    bursts = pd.DataFrame({"start": start.reset_index(drop = True), "end": end.reset_index(drop = True)})
    p2p = []
    auc = []
    for i in range(len(bursts)):
        # peak-to-peak amplitude of bp acceleration
        p2p.append(acc.loc[bursts["start"].iloc[i]:bursts["end"].iloc[i]].max() - acc.loc[bursts["start"].iloc[i]:bursts["end"].iloc[i]].min())
        # AUC of env_diff
        auc.append(np.trapz(env_diff.loc[bursts["start"].iloc[i]:bursts["end"].iloc[i]]))
    bursts["duration"] = bursts["end"] - bursts["start"]
    bursts["peak-to-peak"] = p2p
    bursts["AUC"] = auc
    return bursts
import numpy as np
import pandas as pd

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    
    if isinstance(datax, np.ndarray):
        datax = pd.Series(datax)
    if isinstance(datay, np.ndarray):
        datay = pd.Series(datay)

    return datay.corr(datax.shift(lag))
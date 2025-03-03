# Author: Marcello Sicbaldi

# All of this is implemented based on the seminal work of:
# Till Ronnenmberg et. al., (2015) Human Activity and Rest In Situ, Chapter 3.5: Applying cosine fits
# They propose a very smart and efficient way to fit a cosine curve to time-series data using the projection method.
# The cosine_fit function is a Python implementation of their method.

import numpy as np
from scipy.stats import pearsonr


def cosine_fit(data, time_index, period):
    """
    Fit a static cosine curve to time-series data using the projection method.

    Parameters:
        data (np.array): The time-series data (e.g., LIDS.raw values).
        time_index (np.array): The corresponding time points in minutes.
        period (float): The period of the cosine curve.

    Returns:
        fitted_values (np.array): The fitted values of the cosine curve.
        amplitude (float): The amplitude of the cosine curve.
        phase (float): The phase of the cosine curve.
    """
    # Normalize the time points to the cycle
    t_normalized = time_index * (2 * np.pi / period)
    
    # Compute a_t and b_t
    a_t = data * np.cos(t_normalized)
    b_t = data * np.sin(t_normalized)
    
    # Calculate a and b as the means of a_t and b_t, multiplied by 2
    a = 2 * np.mean(a_t)
    b = 2 * np.mean(b_t)
    
    # Generate the fitted values
    fitted_values = a * np.cos(t_normalized) + b * np.sin(t_normalized)

    # Calculate amplitude and phase
    amplitude = np.sqrt(a**2 + b**2)
    phase = np.arctan2(b, a)

    return fitted_values, amplitude, phase

def lids_mri(lids, fitted_cosine, amplitude):
        '''Munich Rhythmicity Index

        The Munich Rhythmicity Index (MRI) is defined as
        :math:`MRI = A \times r` with :math:`A`, the cosine fit amplitude and
        :math:`r`, the bivariate correlation coefficient (a.k.a. Pearson'r).

        Parameters
        ----------


        Returns
        -------
        mri: numpy.float64
            Munich Rhythmicity Index
        '''

        # Pearson's r
        pearson_r = lids_pearson_r(lids, fitted_cosine)[0]

        # Oscillation range 2*amplitude
        oscillation_range = 2*amplitude

        # MRI
        mri = pearson_r*oscillation_range

        return mri

def lids_pearson_r(lids, fitted_cosine):
        '''Pearson correlation factor

        Pearson correlation factor between LIDS data and its fit function

        Parameters
        ----------
        lids: pandas.Series
            Output data from LIDS transformation.
        params: lmfit.Parameters, optional
            Parameters for the fit function.
            If None, self.lids_fit_params is used instead.
            Default is None.

        Returns
        -------
        r: numpy.float64
            Pearson’s correlation coefficient
        p: numpy.float64
            2-tailed p-value
        '''
        
        return pearsonr(lids, fitted_cosine)

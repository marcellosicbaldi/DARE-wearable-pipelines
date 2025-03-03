import numpy as np
import pandas as pd

def compute_acc_SMV(acc_df):
    return np.sqrt(acc_df.iloc[:,0]**2 + acc_df.iloc[:,1]**2 + acc_df.iloc[:,2]**2)

def compute_anglez(acc_df):
    """
    Compute the angle in the xy plane (z-axis) of the accelerometer data (vanhees2015)

    """

    z_angle = np.arctan(acc_df['z'].rolling('5 s').mean() /
                     acc_df['x'].rolling('5 s').mean()**2 + acc_df['y'].rolling('5 s').mean()**2) * 180 / np.pi
    return z_angle.resample('5 s').mean()

def compute_enmo(acc_df):
    """
    Compute the euclidean norm minus 1. Works best when the accelerometer data has been calibrated
    so that devices at rest measure acceleration norms of 1g.

    Returns
    -------
    """
    enmo1 = np.maximum(compute_acc_SMV(acc_df) - 1, 0) # subtract 1g and if negative, set to 0
    enmo_df = pd.Series(enmo1, index=acc_df.index)
    return enmo_df.resample('5 s').mean()


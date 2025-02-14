import numpy as np

def compute_acc_SMV(acc_df):
    return np.sqrt(acc_df["x"]**2 + acc_df["y"]**2 + acc_df["z"]**2)
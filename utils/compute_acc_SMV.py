import numpy as np

def compute_acc_SMV(acc_df):
    return np.sqrt(acc_df.iloc[:,0]**2 + acc_df.iloc[:,1]**2 + acc_df.iloc[:,2]**2)
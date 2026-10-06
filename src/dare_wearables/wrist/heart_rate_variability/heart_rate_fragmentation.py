
import numpy as np

def compute_HRF(ppi):
    """"Indices of **Heart Rate Fragmentation** (Costa, 2017) include:

    * **PIP**: Percentage of inflection points of the RR intervals series.
    * **IALS**: Inverse of the average length of the acceleration/deceleration segments.
    * **PSS**: Percentage of short segments.
    * **PAS**: Percentage of NN intervals in alternation segments."""

    n_inflection = np.zeros(len(ppi))

    diff_ppi = np.diff(ppi)

    eps = 1e-10

    for i in range(1, len(diff_ppi)):
        if diff_ppi[i] * diff_ppi[i-1] <= eps:
            n_inflection[i] = 1

    # PIP is the number of inflection points divided by the total number of NN intervals
    PIP = np.sum(n_inflection) / len(ppi)

    return PIP

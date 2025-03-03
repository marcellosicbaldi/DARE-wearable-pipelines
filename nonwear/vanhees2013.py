import numpy as np
import pandas as pd

def vanhees_nonwear(x_values, y_values, z_values, non_wear_window=60.0, window_step_size=15, std_thresh_mg=13.0,
                    value_range_thresh_mg=50.0, num_axes_required=2, freq=75.0, quiet=False):
    """
    Calculated non-wear predictions based on the GGIR algorithm created by Vanhees
    https://cran.r-project.org/web/packages/GGIR/vignettes/GGIR.html#non-wear-detection
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0061691

    Args:
        x_values: numpy array of the accelerometer x values
        y_values: numpy array of the accelerometer y values
        z_values: numpy array of the accelerometer z values
        non_wear_window: window size in minutes
        window_step_size: the distance in minutes that the window will step between loops
        std_thresh_mg: the value which the std of an axis in the window must be below
        value_range_thresh_mg: the value which the value range of an axis in the window must be below
        num_axes_required: the number of axes that must be below the std threshold to be considered NW
        freq: frequency of accelerometer in hz
        quiet: Whether or not to quiet print statements

    Returns:
        A numpy array with the length of the accelerometer data marked as either wear time (0) or non-wear time (1)

    """
    if not quiet:
        print("Starting Vanhees Calculation...")

    # Change from minutes in input to data points
    non_wear_window = int(non_wear_window * freq * 60)
    window_step_size = int(window_step_size * freq * 60)

    # Make thresholds from mg to g
    std_thresh_g = std_thresh_mg / 1000
    value_range_thresh_g = value_range_thresh_mg / 1000

    # Create array with all the raw data in it with their respective timestamps
    data = np.array([x_values, y_values, z_values])

    # Initially assuming all wear time, create a vector of wear (1) and input non-wear (0) later
    non_wear_vector = np.zeros(data.shape[1], dtype=bool)

    # Loop over data
    for n in range(0, data.shape[1], window_step_size):
        # Define Start and End points of window
        start = n
        end = start + non_wear_window

        # Grab data in window
        windowed_vector = data[:, start:end]

        # Remove final window of collection to maintain uniform window sizes
        if windowed_vector.shape[1] < non_wear_window:
            break

        # Calculate std
        window_std = windowed_vector.astype(float).std(axis=1)

        # Check how many axes are below std threshold
        std_axes_count = (window_std < std_thresh_g).sum()

        # Calculate value range
        window_value_range = np.ptp(windowed_vector, axis=1)

        # Check how many axes are below value range threshold
        value_range_axes_count = (window_value_range < value_range_thresh_g).sum()

        if (value_range_axes_count >= num_axes_required) or (std_axes_count >= num_axes_required):
            non_wear_vector[start:end] = True

    # Border Criteria
    df = pd.DataFrame({'NW Vector': non_wear_vector,
                       'idx': np.arange(len(non_wear_vector)),
                       "Unique Period Key": (pd.Series(non_wear_vector).diff(1) != 0).astype('int').cumsum(),
                       'Duration': np.ones(len(non_wear_vector))})
    period_durations = df.groupby('Unique Period Key').sum() / (freq * 60)
    period_durations['Wear'] = [False if val == 0 else True for val in period_durations['NW Vector']]
    period_durations['Adjacent Sum'] = period_durations['Duration'].shift(1, fill_value=0) + period_durations[
        'Duration'].shift(-1, fill_value=0)
    period_durations['Period Start'] = df.groupby('Unique Period Key').min()['idx']
    period_durations['Period End'] = df.groupby('Unique Period Key').max()['idx']+1
    for index, row in period_durations.iterrows():
        if not row['Wear']:
            if row['Duration'] <= 180:
                if row['Duration'] / row['Adjacent Sum'] < 0.8:
                    non_wear_vector[row['Period Start']:row['Period End']] = True
            elif row['Duration'] <= 360:
                if row['Duration'] / row['Adjacent Sum'] < 0.3:
                    non_wear_vector[row['Period Start']:row['Period End']] = True
    if not quiet:
        print("Finished Vanhees Calculation.")
    return non_wear_vector
import numpy as np
import pandas as pd
from avro.datafile import DataFileReader
from avro.io import DatumReader

def read_avro(avro_file):
    """
    Read a single avro file (only ACC, PPG and SysPeaks for now)
    TODO: add steps data.
    Parameters
    ----------
    avro_file : str
        Path to the avro file.
    Returns
    ----------
    ppg : numpy array
        PPG signal.
    t_ppg_datetime : pandas datetime
        Timestamps for the PPG signal.
    acc : numpy array
        Accelerometer data.
    t_acc_datetime : pandas datetime
        Timestamps for the accelerometer data.
    t_sys_peaks_datetime : pandas datetime
        Timestamps of the systolic peaks
    temp : numpy array
        Temperature data.
    t_temp_datetime : pandas datetime
        Timestamps for the temperature data.
    eda : numpy array
        EDA data.
    t_eda_datetime : pandas datetime
        Timestamps for the EDA data.
    """

    reader = DataFileReader(open(avro_file, "rb"), DatumReader())
    data = next(reader)

    timezone_offset = data["timezone"]

    # ######## PPG data ########
    ppg_raw = data["rawData"]["bvp"]
    fs_ppg = ppg_raw["samplingFrequency"]
    ppg = ppg_raw["values"]
    t_ppg_unix = [round(ppg_raw["timestampStart"] + timezone_offset*1e6 +
            i * (1e6 / fs_ppg)) for i in range(len(ppg_raw["values"]))] # Create a list of timestamps: start + i * (1/fs) for i in range(len(ppg))
    t_ppg_datetime =  pd.to_datetime(t_ppg_unix, unit= 'us') # us == microseconds
    
    # ######## ACC data ########  
    acc_data = data["rawData"]["accelerometer"]
    fs_acc = acc_data["samplingFrequency"]
    start_acc_unix = acc_data['timestampStart']
    # # Convert ADC counts in g
    delta_physical = acc_data["imuParams"]["physicalMax"] - acc_data["imuParams"]["physicalMin"]
    delta_digital = acc_data["imuParams"]["digitalMax"] - acc_data["imuParams"]["digitalMin"]
    acc_x = [val * delta_physical / delta_digital for val in acc_data["x"]]
    acc_y = [val * delta_physical / delta_digital for val in acc_data["y"]]
    acc_z = [val * delta_physical / delta_digital for val in acc_data["z"]]
    acc = np.vstack((acc_x, acc_y, acc_z)).T # accelerations on the columns
    t_acc_unix = [round(start_acc_unix + timezone_offset*1e6 + 
                i * (1e6 / fs_acc)) for i in range(len(acc_data["x"]))]
    t_acc_datetime = pd.to_datetime(t_acc_unix, unit='us')

    # ######## Systolic peaks ########
    sys_peaks_data = data["rawData"]["systolicPeaks"]["peaksTimeNanos"]
    sys_peaks_unix = np.array(sys_peaks_data) / 10**3  + timezone_offset*1e6 # from ns to us
    t_sys_peaks_datetime = pd.to_datetime(sys_peaks_unix, unit="us")

    ######## Temperature data ########
    temp_data = data["rawData"]["temperature"]
    fs_temp = temp_data["samplingFrequency"]
    temp = temp_data["values"]
    t_temp_unix = [round(temp_data["timestampStart"] + timezone_offset*1e6 +
            i * (1e6 / fs_temp)) for i in range(len(temp_data["values"]))] 
    t_temp_datetime = pd.to_datetime(t_temp_unix, unit= 'us')

    ######## EDA data ########
    eda_data = data["rawData"]["eda"]
    fs_eda = eda_data["samplingFrequency"]
    eda = eda_data["values"]
    t_eda_unix = [round(eda_data["timestampStart"] + timezone_offset*1e6 +
            i * (1e6 / fs_eda)) for i in range(len(eda_data["values"]))]
    t_eda_datetime = pd.to_datetime(t_eda_unix, unit= 'us')

    return ppg, t_ppg_datetime, acc, t_acc_datetime, t_sys_peaks_datetime, temp, t_temp_datetime, eda, t_eda_datetime

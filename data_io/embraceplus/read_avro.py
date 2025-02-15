import numpy as np
import pandas as pd
from avro.datafile import DataFileReader
from avro.io import DatumReader

from warnings import warn

from avro.datafile import DataFileReader
from avro.io import DatumReader
from numpy import (
    round,
    arange,
    vstack,
    ascontiguousarray,
    isclose,
    full,
    argmin,
    abs,
    nan,
    float64,
)

from utils.resample_signal import apply_resample_empatica

class ReadEmpaticaAvro():
    """
    Read Empatica data from an avro file.

    Parameters
    ----------
    trim_keys : {None, tuple}, optional
        Trim keys provided in the `predict` method. Default (None) will not do any trimming.
        Trimming of either start or end can be accomplished by providing None in the place
        of the key you do not want to trim. If provided, the tuple should be of the form
        (start_key, end_key). When provided, trim datetimes will be assumed to be in the
        same timezone as the data (ie naive if naive, or in the timezone provided).
    resample_to_accel : bool, optional
        Resample any additional data streams to match the accelerometer data stream.
        Default is True.
    """

    _file = "file"
    _time = "time"
    _acc = "acc"
    _gyro = "gyro"
    _mag = "magnet"
    _temp = "temperature"
    _days = "day_ends"

    def __init__(self, trim_keys=None, resample_to_bvp=True):
        
        self.trim_keys = trim_keys
        self.resample_to_bvp = resample_to_bvp

    def get_accel(self, raw_accel_dict, results_dict, key):
        """
        Get the raw acceleration data from the avro file record.

        Parameters
        ----------
        raw_accel_dict : dict
            The record from the avro file for a raw data stream.
        results_dict : dict
            Dictionary where the results will go.
        key : str
            Name for the results in `results_dict`.
        """
        # sampling frequency
        fs = raw_accel_dict["samplingFrequency"]

        # timestamp start
        ts_start = raw_accel_dict["timestampStart"] / 1e6  # convert to seconds

        # imu parameters for scaling to actual values
        phys_min = raw_accel_dict["imuParams"]["physicalMin"]
        phys_max = raw_accel_dict["imuParams"]["physicalMax"]
        dig_min = raw_accel_dict["imuParams"]["digitalMin"]
        dig_max = raw_accel_dict["imuParams"]["digitalMax"]

        # raw acceleration data
        accel = ascontiguousarray(
            vstack((raw_accel_dict["x"], raw_accel_dict["y"], raw_accel_dict["z"])).T
        )

        # scale the raw acceleration data to actual values
        accel = (accel - dig_min) / (dig_max - dig_min) * (
            phys_max - phys_min
        ) + phys_min

        # create the timestamp array using ts_start, fs, and the number of samples
        time = arange(ts_start, ts_start + accel.shape[0] / fs, 1 / fs)[
            : accel.shape[0]
        ]

        if time.size != accel.shape[0]:
            raise ValueError("Time does not have enough samples for accel array")

        # use special names here so we can just update dictionary later for returning
        results_dict[key] = {self._time: time, "fs": fs, "values": accel}
    
    def get_bvp(self, raw_bvp_dict, results_dict, key):
        """
        Get the raw blood volume pulse data from the avro file record.

        Parameters
        ----------
        raw_bvp_dict : dict
            The record from the avro file for a raw data stream.
        results_dict : dict
            Dictionary where the results will go.
        key : str
            Name for the results in `results_dict`.
        """
        # sampling frequency
        fs = round(raw_bvp_dict["samplingFrequency"], decimals=3)
        # timestamp start
        ts_start = raw_bvp_dict["timestampStart"] / 1e6 # convert to seconds

        # raw bvp data
        bvp = ascontiguousarray(raw_bvp_dict["values"])

        # timestamp array
        time = arange(ts_start, ts_start + bvp.size / fs, 1 / fs)[: bvp.shape[0]]

        if time.size != bvp.shape[0]:
            raise ValueError("Time does not have enough samples for bvp array")
        
        results_dict[key] = {self._time: time, "fs": fs, "bvp": bvp}

    def get_eda(self, raw_dict, results_dict, key):
        """
        Get the raw electrodermal activity data from the avro file record.

        Parameters
        ----------
        raw_dict : dict
            The record from the avro file for a raw data stream.
        results_dict : dict
            Dictionary where the results will go.
        key : str
            Name for the results in `results_dict`.
        """
        if not raw_dict["values"]:
            return

        # sampling frequency
        fs = round(raw_dict["samplingFrequency"], decimals=3)
        # timestamp start
        ts_start = raw_dict["timestampStart"] / 1e6

        # raw eda data
        eda = ascontiguousarray(raw_dict["values"])

        # timestamp array
        time = arange(ts_start, ts_start + eda.size / fs, 1 / fs)[: eda.shape[0]]

        if time.size != eda.shape[0]:
            raise ValueError("Time does not have enough samples for eda array")
        
        results_dict[key] = {"time_eda": time, "fs_eda": fs, "eda": eda}

    def get_temp(self, raw_dict, results_dict, key):
        """
        Get the raw temperature data from the avro file record.

        Parameters
        ----------
        raw_dict : dict
            The record from the avro file for a raw data stream.
        results_dict : dict
            Dictionary where the results will go.
        key : str
            Name for the results in `results_dict`.
        """
        if not raw_dict["values"]:
            return

        # sampling frequency
        fs = round(raw_dict["samplingFrequency"], decimals=3)
        # timestamp start
        ts_start = raw_dict["timestampStart"] / 1e6

        # raw temperature data
        temp = ascontiguousarray(raw_dict["values"])

        # timestamp array
        time = arange(ts_start, ts_start + temp.size / fs, 1 / fs)[: temp.shape[0]]

        if time.size != temp.shape[0]:
            raise ValueError("Time does not have enough samples for temp array")
        
        results_dict[key] = {"time_temp": time, "fs_temp": fs, "temp": temp}

    def get_values_1d(self, raw_dict, results_dict, key):
        """
        Get the raw 1-dimensional values data from the avro file record.
        i.e, PPG, EDA, and temperature

        Parameters
        ----------
        raw_dict : dict
            The record from the avro file for a raw 1-dimensional values data stream.
        results_dict : dict
            Dictionary where the results will go.
        key : str
            Name for the results in `results_dict`.
        """
        if not raw_dict["values"]:
            return

        # sampling frequency
        fs = round(raw_dict["samplingFrequency"], decimals=3)
        # timestamp start
        ts_start = raw_dict["timestampStart"] / 1e6  # convert to seconds

        # raw values data
        values = ascontiguousarray(raw_dict["values"])

        # timestamp array
        time = arange(ts_start, ts_start + values.size / fs, 1 / fs)[: values.shape[0]]

        if time.size != values.shape[0]:
            raise ValueError(f"Time does not have enough samples for {key} array")

        results_dict[key] = {self._time: time, "fs": fs, "values": values}

    @staticmethod
    def get_systolic_peaks(raw_dict, results_dict, key):
        """
        Get the systolic peaks data from the avro file record.

        Parameters
        ----------
        raw_dict : dict
            The record from the avro file for systolic peaks data.
        results_dict : dict
            Dictionary where the results will go.
        key : str
            Name for the results in `results_dict`.
        """
        if not raw_dict["peaksTimeNanos"]:
            return

        peaks = (
            ascontiguousarray(raw_dict["peaksTimeNanos"]) / 1e9
        )  # convert to seconds

        results_dict[key] = {"values": peaks}

    def get_steps(self, raw_dict, results_dict, key):
        """
        Get the raw steps data from the avro file record.

        Parameters
        ----------
        raw_dict : dict
            The record from the avro file for raw steps data.
        results_dict : dict
            Dictionary where the results will go.
        key : str
            Name for the results in `results_dict`.
        """
        if not raw_dict["values"]:
            return

        # sampling frequency
        fs = round(raw_dict["samplingFrequency"], decimals=3)

        # timestamp start
        ts_start = raw_dict["timestampStart"] / 1e6  # convert to seconds

        # raw steps data
        steps = ascontiguousarray(raw_dict["values"])

        # timestamp array
        time = arange(ts_start, ts_start + steps.size / fs, 1 / fs)[: steps.size]

        if time.size != steps.size:
            raise ValueError("Time does not have enough samples for steps array")

        results_dict[key] = {self._time: time, "fs": fs, "values": steps}

    def handle_resampling(self, streams):
        """
        Handle resampling of data streams. Data will be resampled to match the
        BVP (Blood Volume Pulse) data stream.
        """
        if "bvp" not in streams:
            raise ValueError("BVP data stream is missing, cannot resample.")
        
        # Remove BVP data stream
        bvp_dict = streams.pop("bvp")
        # Remove Temp data stream
        temp_dict = streams.pop("temperature")
        # Remove EDA data stream
        eda_dict = streams.pop("eda")

        # Remove keys that cannot be resampled
        rs_streams = {d: streams.pop(d) for d in ["systolic_peaks", "steps"] if d in streams}
        
        for name, stream in streams.items():
            if stream["values"] is None:
                continue
            
            # Check that the stream doesn't start significantly later than BVP
            # if (dt := (stream["time"][0] - bvp_dict["time"][0])) > 1:
            #     warn(
            #         f"Data stream {name} starts more than 1 second ({dt}s) after "
            #         f"the BVP stream. Data will be filled with the first (and "
            #         f"last) value as needed."
            #     )
            
            # Check if resampling is needed
            if isclose(stream["fs"], bvp_dict["fs"], atol=1e-3):
                new_shape = list(stream["values"].shape)
                new_shape[0] = bvp_dict["bvp"].shape[0]
                rs_streams[name] = full(new_shape, nan, dtype=float64)
                i1 = argmin(abs(bvp_dict["time"] - stream["time"][0]))
                i2 = i1 + stream["time"].size
                rs_streams[name][i1:i2] = stream["values"][: stream["values"].shape[0] - (i2 - bvp_dict["time"].size)]
                rs_streams[name][:i1] = stream["values"][0]
                rs_streams[name][i2:] = stream["values"][-1]
                continue
            
            # Resample the stream to match BVP
            _, (stream_rs,) = apply_resample_empatica(
                time=stream["time"],
                time_rs=bvp_dict["time"],
                data=(stream["values"],),
                aa_filter=True,
                fs=stream["fs"],
            )
            rs_streams[name] = stream_rs
        
        rs_streams.update(bvp_dict)
        rs_streams.update(temp_dict)
        rs_streams.update(eda_dict)
        return rs_streams

    def get_datastreams(self, raw_record):
        """
        Extract the various data streams from the raw avro file record.
        """
        fn_map = {
            "accelerometer": ("acc", self.get_accel),
            "eda": ("eda", self.get_eda),
            "temperature": ("temperature", self.get_temp),
            "bvp": ("bvp", self.get_bvp),
            "systolicPeaks": ("systolic_peaks", self.get_systolic_peaks),
            "steps": ("steps", self.get_steps),
        }

        raw_data_streams = {}
        for full_name, (stream_name, fn) in fn_map.items():
            fn(raw_record[full_name], raw_data_streams, stream_name)
        
        if self.resample_to_bvp:
            data_streams = self.handle_resampling(raw_data_streams)
        else:
            data_streams = raw_data_streams.pop("bvp")
            data_streams.update(raw_data_streams)
        
        return data_streams

    def read(self, *, file, tz_name=None, **kwargs):
        """
        Read the input .avro file.

        Parameters
        ----------
        file : {path-like, str}
            The path to the input file.
        tz_name : {None, optional}
            IANA time-zone name for the recording location. If not provided, timestamps
            will represent local time naively. This means they will not account for
            any time changes due to Daylight Saving Time.

        Returns
        -------
        results : dict
            Dictionary containing the data streams from the file. See Notes
            for different output options.

        Notes
        -----
        There are two output formats, based on if `resample_to_accel` is True or False.
        If True, all available data streams except for `systolic_peaks` and `steps`
        are resampled to match the accelerometer data stream, which results in their
        values being present in the top level of the `results` dictionary, ie
        `results['gyro']`, etc.

        If False, everything except accelerometer will be present in dictionaries
        containing the keys `time`, `fs`, and `values`, and the top level will be these
        dictionaries plus the accelerometer data (keys `time`, `fs`, and `accel`).

        `systolic_peaks` will always be a dictionary of the form `{'systolic_peaks': array}`.
        """

        reader = DataFileReader(open(file, "rb"), DatumReader())
        records = []
        for record in reader:
            records.append(record)
        reader.close()

        # get the timezone offset
        tz_offset = records[0]["timezone"]  # in seconds

        # as needed, deviceSn, deviceModel

        # get the data streams
        results = self.get_datastreams(records[0]["rawData"])

        # update the timestamps to be local. Do this as we don't have an actual
        # timezone from the data.
        if tz_name is None:
            results["time"] += tz_offset
            results["time_temp"] += tz_offset
            results["time_eda"] += tz_offset

            for k in results:
                if k == "time":
                    continue
                if (
                    isinstance(results[k], dict)
                    and "time" in results[k]
                    and results[k]["time"] is not None
                ):
                    results[k]["time"] += tz_offset
        # do nothing if we have the time-zone name, the timestamps are already
        # UTC

        # adjust systolic_peaks
        if "systolic_peaks" in results:
            results["systolic_peaks"]["values"] += tz_offset

        return results


def read_avro_old(avro_file):
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

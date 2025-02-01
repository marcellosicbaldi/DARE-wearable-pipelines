import numpy as np
import pandas as pd
import os

save_data_path = "/Users/marcellosicbaldi/Library/CloudStorage/OneDrive-AlmaMaterStudiorumUniversitàdiBologna/tesi_Sara/Empatica/data/parquet/"
#### Change the subject ID and device ID below to the subject and device you want to process ####
sub_ID = "00007"
device_ID = "3YK3J151VJ"

days = sorted(os.listdir(save_data_path))
days = [day for day in days if day[0] != "."] # remove hidden files (needed for MacOS users)

for i, day in enumerate(days[:1]):
    print(f"Processing day {day}")
    ppg = pd.read_parquet(save_data_path + day + "/ppg.parquet")
    acc = pd.read_parquet(save_data_path + day + "/acc.parquet")
    sys_peaks = pd.read_parquet(save_data_path + day + "/sys_peaks.parquet")

    # Check if timestamps are monotonically increasing
    assert ppg.index.is_monotonic_increasing
    assert acc.index.is_monotonic_increasing
    assert sys_peaks["SysPeakTime"].is_monotonic_increasing
    
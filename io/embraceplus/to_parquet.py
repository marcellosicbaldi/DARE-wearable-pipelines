# This script reads the avro files for a given subject and device and stores the data in parquet format, one file per day.
# Marcello Sicbaldi, 2025 - marcello.sicbaldi2@unibo.it

import os
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from read_avro import read_avro
from tqdm import tqdm

#### Change the paths below to the location of the data on your machine ####
data_path = "/Users/marcellosicbaldi/Library/CloudStorage/OneDrive-AlmaMaterStudiorumUniversitàdiBologna/tesi_Sara/Empatica/data/participant_data/"
save_data_path = "/Users/marcellosicbaldi/Library/CloudStorage/OneDrive-AlmaMaterStudiorumUniversitàdiBologna/tesi_Sara/Empatica/data/parquet/"
#### Change the subject ID and device ID below to the subject and device you want to process ####
sub_ID = "00007"
device_ID = "3YK3J151VJ"

days = sorted(os.listdir(data_path))
days = [day for day in days if day[0] != "."] # remove hidden files (needed for MacOS users)

with ThreadPoolExecutor() as executor: # Use ThreadPoolExecutor to parallelize the reading of the avro files

  for i, day in enumerate(days[5:]):

    # Initialize lists
    ppg_day = []
    t_ppg_day = []
    acc_day = []
    t_acc_day = []
    sys_peaks_day = []

    os.makedirs(save_data_path + day, exist_ok=True)

    print(f"Processing day {day}")
    
    folder_day = data_path + day + "/" + sub_ID + "-" + device_ID + "/raw_data/v6"

    avro_files = sorted(glob.glob(folder_day + "/*.avro"))

    results = list(tqdm(executor.map(read_avro, avro_files), total=len(avro_files))) # Read avro files in parallel

    print("Concatenating data...")

    for ppg, t_ppg, acc, t_acc, t_sys_peaks in results:
      
      # Concatenate data for the current day
      ppg_day.extend(ppg)
      acc_day.extend(acc)
      t_ppg_day.extend(t_ppg)
      t_acc_day.extend(t_acc)
      sys_peaks_day.extend(t_sys_peaks)

    print("Storing data in parquet format...")
    # Store data for the current day in parquet format
    pd.DataFrame(ppg_day, index=t_ppg_day, columns = ["ppg"]).to_parquet(save_data_path + day + "/ppg.parquet")
    pd.DataFrame(acc_day, index=t_acc_day, columns=["x", "y", "z"]).to_parquet(save_data_path + day + "/acc.parquet")
    pd.DataFrame(sys_peaks_day, columns=["SysPeakTime"]).to_parquet(save_data_path + day + "/sys_peaks.parquet")

    del ppg_day, t_ppg_day, acc_day, t_acc_day, sys_peaks_day # free memory
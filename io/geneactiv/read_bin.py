"""
Read .bin files coming from GENEActiv devices and save them as a pd.DataFrame in parquet format.

This module is based on the code from the scikit-digital-health python package (https://github.com/pfizer-opensource/scikit-digital-health)

IMP: scikit-digital-health create conflicts with versions needed for other packages. To solve this:
1. I tried to just copy the code can be copied and modified it to work as a standalone module -- however, scikit-digital-health 
    uses a lot of other modules (also implemented in .c) and it is not clear how to separate them without breaking the code.
2. Solution for now: he scikit-digital-health package is installed in a separate conda environment and the code is run from there.

"""

import numpy as np
import pandas as pd
import os
from skdh.io import ReadBin

def bin2parquet(file_path):
    """
    Read a .bin file from a GENEActiv device.
    """
    reader = ReadBin()
    data = reader.predict(file = file_path)
    acc_df = pd.DataFrame(data["accel"], 
                          columns = ["x", "y", "z"],
                          index = pd.to_datetime(data["time"], unit = "s"))
    # save as parquet
    acc_df.to_parquet(file_path.replace(".bin", ".parquet"))

if __name__ == "__main__":
    data_path = "/Users/augenpro/Documents/Age-IT" # path to the folder containing the subjects
    participants = sorted([p for p in os.listdir(data_path) if not p.startswith(".")]) # list of the participants
    visit = "T0 (baseline)" # T0 (baseline), T1 (follow-up @ 6 months), T2 (follow-up @ 12 months)

    sensors = ["GeneActivPolso", "GeneActivCaviglia"]

    for participant in participants:
        print(participant)
        for sensor in sensors:
            path = os.path.join(data_path, participant, visit, sensor)
            # if the path contains nothing, go to the next participant
            files = os.listdir(os.path.join(data_path, participant, visit, sensor))
            for f in files:
                if f.endswith(".bin"):
                    bin2parquet(os.path.join(path, f))
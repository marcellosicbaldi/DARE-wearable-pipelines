import os
from skdh.preprocessing import CalibrateAccelerometer

def calibrate_acc(file_path):
    """
    Calibrate the accelerometer data.
    """
    calibrator = CalibrateAccelerometer()
    acc_cal = calibrator.predict(file = file_path)
    acc_cal.to_parquet(file_path.replace(".parquet", "_calibrated.parquet"))

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
                if f.endswith(".parquet"):
                    calibrate_acc(os.path.join(path, f))
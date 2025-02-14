import numpy as np
import pandas as pd

def compute_hr_stats(hr_series):
    return {
        "median": hr_series.median(),
        "q1": hr_series.quantile(0.25),
        "q3": hr_series.quantile(0.75)
    }

def extract_summary_metrics(hr_belief, nights):
    source_iqr_data = {
        "time": [],
        "median": [],
        "q1": [],
        "q3": []
    }
    time_source = []
    end_last_day = hr_belief.index[-1]
    for i in range(len(nights)):
        # Night period (during sleep)
        night_start = nights.iloc[i]["start"]
        night_end = nights.iloc[i]["end"]
        mid_night = night_start + (night_end - night_start) / 2

        # Extract HR data for the night
        hr_segment = hr_belief.loc[night_start:night_end]
        if not hr_segment.empty:
            hr_stats = compute_hr_stats(hr_segment["HR"])

            # Store results for night
            source_iqr_data["time"].append(mid_night)
            source_iqr_data["median"].append(hr_stats["median"])
            source_iqr_data["q1"].append(hr_stats["q1"])
            source_iqr_data["q3"].append(hr_stats["q3"])

        # Daytime period (after night)
        if i < len(nights) - 1:
            next_night_start = nights.iloc[i + 1]["start"]
        else:
            next_night_start = end_last_day  # Last period extends to the last timestamp

        mid_day = night_end + (next_night_start - night_end) / 2

        # Extract HR data for the day
        hr_segment = hr_belief.loc[night_end:next_night_start]
        if not hr_segment.empty:
            hr_stats = compute_hr_stats(hr_segment["HR"])

            # Store results for day
            source_iqr_data["time"].append(mid_day)
            source_iqr_data["median"].append(hr_stats["median"])
            source_iqr_data["q1"].append(hr_stats["q1"])
            source_iqr_data["q3"].append(hr_stats["q3"])

        time_source.append(mid_night)
        time_source.append(mid_day)

    return source_iqr_data, time_source
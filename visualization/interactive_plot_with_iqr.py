import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, Select, DatetimeTickFormatter, Whisker, VBar, Circle, BoxAnnotation #, HoverTool
from bokeh.layouts import column
from bokeh.io import curdoc

# Load the HR data
hr_file_path = "/Users/augenpro/Documents/Empatica/data_sara/data/GGIR_input/"

# Load the HR data
file_path = hr_file_path + "hr_belief.pkl" # Update with your actual file path
hr_belief = pd.read_pickle(file_path)#.loc[:pd.Timestamp("2024-05-21 23:45:00")]

# Start and end of each SPT
start_end_sleep = np.load("/Users/augenpro/Documents/Empatica/data_sara/data/GGIR_input/SPT_window_GGIR.npy", allow_pickle=True)
nights = pd.DataFrame(start_end_sleep, columns=["start", "end"])
days = pd.DataFrame(columns=["start", "end"])
# days["start"] = nights["end"].iloc[:-1].reset_index(drop=True)
# days["end"] = nights["start"].iloc[1:].reset_index(drop=True)
start_first_day = hr_belief.index[0]
end_last_day =  hr_belief.index[-1]

hr_belief_day_vs_night = []
t_hr_belief_day_vs_night = []
hr_belief_day_vs_night.append(hr_belief.loc[:nights.iloc[0]["start"]].agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]))
t_hr_belief_day_vs_night.append(start_first_day + (nights.iloc[0]["start"] - start_first_day)/2)
for i in range(nights.shape[0]-1):
    hr_belief_day_vs_night.append(hr_belief.loc[nights.iloc[i]["end"]:nights.iloc[i+1]["start"]].agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]))
    t_hr_belief_day_vs_night.append(nights.iloc[i]["end"] + (nights.iloc[i+1]["start"] - nights.iloc[i]["end"])/2)
hr_belief_day_vs_night.append(hr_belief.loc[nights.iloc[-1]["end"]:].agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]))
t_hr_belief_day_vs_night.append(nights.iloc[-1]["end"] + (end_last_day - nights.iloc[-1]["end"])/2)

hr_belief = hr_belief.to_frame(name="HR")
hr_belief.index = pd.to_datetime(hr_belief.index)
x_start = hr_belief.index[0] - pd.Timedelta("30min")
x_end = hr_belief.index[-1] + pd.Timedelta("30min")

# Initial ColumnDataSource
source = ColumnDataSource(data={"time": hr_belief.index, "hr": hr_belief.values})
source_iqr = ColumnDataSource(data={"time": [], "q1": [], "q3": [], "median": []})

# Create figure
p = figure(x_axis_type="datetime", title="Mean Heart Rate", x_range=(x_start, x_end), width=1400, height=720)
line = p.line("time", "hr", source=source, line_width=2, color="blue")
p.vbar(x="time", top="q3", bottom="q1", width=800000, source=source_iqr, fill_color="blue", fill_alpha=0.3, legend_label="IQR")
p.scatter(x="time", y="median", source=source_iqr, size=8, color="black", legend_label="Median")

# BoxAnnotation for sleep period
for i in range(nights.shape[0]):
    box = BoxAnnotation(left=nights["start"].iloc[i], right=nights["end"].iloc[i], fill_color="gray", fill_alpha=0.2, line_color="black")#, line_width=2)
    p.add_layout(box)
# box = BoxAnnotation(left=night1_GGIR[0], right=night1_GGIR[1], fill_color="gray", fill_alpha=0.2, line_color="black")#, line_width=2)
p.add_layout(box)

# Helper function to compute HR statistics
def compute_hr_stats(hr_series):
    return {
        "median": hr_series.median(),
        "q1": hr_series.quantile(0.25),
        "q3": hr_series.quantile(0.75),
    }

# Convert results into a DataFrame
# iqr_summary = pd.DataFrame(source_iqr_data)

# Hover tool
# hover = HoverTool(tooltips=[("Time", "@time{%d %B %H:%M:%S}"), ("HR", "@hr{0.0}")], formatters={"@time": "datetime"}, mode="vline")
# p.add_tools(hover)

# Resampling options
resample_select = Select(title="Resample Interval", value="2 Sec", options=["2 s", "1 min", "5 min", "30 min", "1 h", "Day vs Night"], styles={"font-size": "16pt"})

# Callback function
def update_resample(attr, old, new):
    interval = resample_select.value

    if interval in ["Day vs Night"]:
        # Reinitialize the data storage
        source_iqr_data = {"time": [], "q1": [], "q3": [], "median": []}

        # Add the first day
        hr_segment = hr_belief.loc[start_first_day:nights.iloc[0]["start"]]
        if not hr_segment.empty:
            hr_stats = compute_hr_stats(hr_segment["HR"])
            source_iqr_data["time"].append(start_first_day + (nights.iloc[0]["start"] - start_first_day) / 2)
            source_iqr_data["median"].append(hr_stats["median"])
            source_iqr_data["q1"].append(hr_stats["q1"])
            source_iqr_data["q3"].append(hr_stats["q3"])
        time_source = [start_first_day + (nights.iloc[0]["start"] - start_first_day) / 2]
        # Iterate over all sleep periods
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

        source_iqr.data = source_iqr_data
        source.data = {"time": time_source, "hr": source_iqr_data["median"]}
        
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = "16pt"
        
    elif interval in ["30 min", "1 h"]:
        processed_df = hr_belief.resample(interval).median().dropna()
        source.data = {"time": processed_df.index, "hr": processed_df["HR"]}
        iqr_df = hr_belief.resample(interval).agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]).dropna()
        iqr_df.columns = ["median", "q1", "q3"]
        source_iqr.data = {"time": iqr_df.index, "q1": iqr_df["q1"], "q3": iqr_df["q3"], "median": iqr_df["median"]}
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = "16pt"
    else:
        processed_df = hr_belief.resample(interval).median().dropna()
        source.data = {"time": processed_df.index, "hr": processed_df["HR"]}
        source_iqr.data = {"time": [], "q1": [], "q3": [], "median": []}
        # Remove legend

# Link widgets to callback
resample_select.on_change("value", update_resample)

# Formatting
p.xaxis.axis_label = "Time"
p.yaxis.axis_label = "HR (bpm)"
p.title.text_font_size = "20pt"
p.xaxis.axis_label_text_font_size = "16pt"
p.yaxis.axis_label_text_font_size = "16pt"
p.xaxis.major_label_text_font_size = "16pt"
p.yaxis.major_label_text_font_size = "16pt"
p.xaxis.formatter = DatetimeTickFormatter(
    hours="%H:%M", days="%d %B", months="%d %B", years="%d %B"
)

p.xaxis.major_label_orientation = np.pi/4

# Layout
layout = column(p, resample_select)
curdoc().add_root(layout)
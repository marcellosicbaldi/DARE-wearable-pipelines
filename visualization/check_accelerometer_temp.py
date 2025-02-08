import numpy as np
import pandas as pd
import os

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Select, DatetimeTickFormatter
from bokeh.layouts import column
from bokeh.io import curdoc

def plot_acc_temp(parquet_folder):
    """
    Plot the accelerometer and temperature signal, toghether with nonwear. The user can select the day to visualize using a dropdown menu.
    Parameters
    ----------
    parquet_folder : str
        Path to the folder containing Parquet files.
    """
    days = sorted(os.listdir(parquet_folder))
    days = [day for day in days if not day.startswith(".")]  # Remove hidden files

    if not days:
        raise ValueError("No valid data days found in the directory.")

    # Dropdown menu
    day_select = Select(title="Day", options=days, value=days[0], styles={"font-size": "16pt"})

    # Read initial data
    day = day_select.value
    acc = pd.read_parquet(os.path.join(parquet_folder, day, "acc.parquet")).resample("1s").mean()
    temp = pd.read_parquet(os.path.join(parquet_folder, day, "temp.parquet"))

    # Convert to ColumnDataSource for dynamic updates
    acc_source = ColumnDataSource(data={"time": acc.index, "x": acc["x"], "y": acc["y"], "z": acc["z"]})
    temp_source = ColumnDataSource(data={"time": temp.index, "temp": temp["temp"]})

    # Figure for Accelerometer
    p1 = figure(title="Accelerometer Signal", x_axis_label="Time", y_axis_label="Acceleration (g)",
                x_axis_type='datetime', height=600, width=1250)
    p1.line("time", "x", source=acc_source, line_width=2, color="red", legend_label="Acc x")
    p1.line("time", "y", source=acc_source, line_width=2, color="green", legend_label="Acc y")
    p1.line("time", "z", source=acc_source, line_width=2, color="orange", legend_label="Acc z")

    # Figure for Temperature
    p2 = figure(title="Temperature Signal", x_axis_label="Time", y_axis_label="Temperature (°C)",
                x_axis_type='datetime', height=600, width=1250, x_range=p1.x_range)
    p2.line("time", "temp", source=temp_source, line_width=2, color="blue", legend_label="Temperature")

    # Formatting
    for p in [p1, p2]:
        p.title.text_font_size = "16pt"
        p.xaxis.axis_label_text_font_size = "16pt"
        p.yaxis.axis_label_text_font_size = "16pt"
        p.xaxis.major_label_text_font_size = "16pt"
        p.yaxis.major_label_text_font_size = "16pt"
        p.xaxis.formatter = DatetimeTickFormatter(
            hours="%H:%M",
            days="%d %B",
            months="%d %B",
            years="%d %B",
        )
        p.xaxis.major_label_orientation = np.pi / 4

    # Update function
    def update_plot(attrname, old, new):
        day = day_select.value
        acc = pd.read_parquet(os.path.join(parquet_folder, day, "acc.parquet")).resample("1s").mean()
        temp = pd.read_parquet(os.path.join(parquet_folder, day, "temp.parquet"))

        acc_source.data = {"time": acc.index, "x": acc["x"], "y": acc["y"], "z": acc["z"]}
        temp_source.data = {"time": temp.index, "temp": temp["temp"]}

    # Link dropdown to update function
    day_select.on_change("value", update_plot)

    # Layout
    layout = column(day_select, p1, p2)
    curdoc().add_root(layout)

# Run Bokeh app
parquet_path = "/Users/augenpro/Documents/Empatica/data_sara/data/parquet/"
plot_acc_temp(parquet_path)

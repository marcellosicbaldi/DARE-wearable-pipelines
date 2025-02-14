import numpy as np
import pandas as pd
import os

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Select, DatetimeTickFormatter
from bokeh.layouts import column
from bokeh.io import curdoc

def plot_acc_temp(data_folder, accSMV=False):  
    """
    Plot the accelerometer and temperature signal.
    Parameters
    ----------
    parquet_folder : str
        Path to the folder containing Parquet files.
    """
    acc = pd.read_parquet(os.path.join(data_folder,"acc.parquet")).resample("1s").mean() # Resample for better visualization
    temp = pd.read_parquet(os.path.join(data_folder, "temp.parquet")).resample("1s").mean()

    # Convert to ColumnDataSource for dynamic updates
    temp_source = ColumnDataSource(data={"time": temp.index, "temp": temp["temp"]})

    # Figure for Accelerometer
    if accSMV:
        acc["smv"] = np.sqrt(acc["x"]**2 + acc["y"]**2 + acc["z"]**2)
        acc_source = ColumnDataSource(data={"time": acc.index, "smv": acc["smv"]})
        p1 = figure(title="Accelerometer Signal", x_axis_label="Time", y_axis_label="Acceleration (g)",
                    x_axis_type='datetime', height=600, width=1400)
        p1.line("time", "smv", source=acc_source, line_width=2, color="black", legend_label="Acc SMV")

    else:
        acc_source = ColumnDataSource(data={"time": acc.index, "x": acc["x"], "y": acc["y"], "z": acc["z"]})
        p1 = figure(title="Accelerometer Signal", x_axis_label="Time", y_axis_label="Acceleration (g)",
                    x_axis_type='datetime', height=500, width=1400)
        p1.line("time", "x", source=acc_source, line_width=2, color="red", legend_label="Acc x")
        p1.line("time", "y", source=acc_source, line_width=2, color="green", legend_label="Acc y")
        p1.line("time", "z", source=acc_source, line_width=2, color="orange", legend_label="Acc z")

    # Figure for Temperature
    p2 = figure(title="Temperature Signal", x_axis_label="Time", y_axis_label="Temperature (°C)",
               x_axis_type='datetime', height=500, width=1400, x_range=p1.x_range)    
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

    # Layout
    layout = column(p1, p2)
    show(layout)


def plot_acc_temp_daily(parquet_folder):
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
if __name__ == "__main__":

    parquet_path = "/Users/augenpro/Documents/Empatica/data_sara/data/parquet/"
    plot_acc_temp(parquet_path)

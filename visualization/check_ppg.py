import numpy as np
import pandas as pd
import os

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Select, DatetimeTickFormatter
from bokeh.layouts import column
from bokeh.io import curdoc

def plot_ppg(parquet_folder):
    """
    Plot the ppg signal. The user can select the day to visualize using a dropdown menu.
    Parameters
    ----------
    parquet_folder : str
        Path to the folder containing Parquet files.
    """

    days = sorted(os.listdir(parquet_folder))
    days = [day for day in days if day[0] != "."]  # Remove hidden files

    if not days:
        raise ValueError("No valid data days found in the directory.")

    # Dropdown menu
    day_select = Select(title="Day", options=days, value=days[0], styles={"font-size": "16pt"})

    # Read initial data
    day = day_select.value
    ppg = pd.read_parquet(os.path.join(parquet_folder, day, "ppg.parquet")).resample("0.05s").mean()

    # Convert to ColumnDataSource for dynamic updates
    ppg_source = ColumnDataSource(data={"time": ppg.index, "ppg": ppg["ppg"]})

    # Figure for PPG
    p1 = figure(title="PPG Signal", x_axis_label="Time", y_axis_label="PPG",
                x_axis_type='datetime', height=600, width=1250)
    p1.line("time", "ppg", source=ppg_source, line_width=2, color="blue", legend_label="PPG")

    # Formatting
    for p in [p1]:
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
        ppg = pd.read_parquet(os.path.join(parquet_folder, day, "ppg.parquet")).resample("0.05s").mean()
        ppg_source.data = {"time": ppg.index, "ppg": ppg["ppg"]}

    day_select.on_change("value", update_plot)

    layout = column(day_select, p1)
    curdoc().add_root(layout)

# Run the visualization
path = "/Users/marcellosicbaldi/Library/CloudStorage/OneDrive-AlmaMaterStudiorumUniversitàdiBologna/tesi_Sara/Empatica/data/parquet/"
plot_ppg(path)
# Run the Bokeh server with the following command:
# bokeh serve --show visualization/check_ppg.py
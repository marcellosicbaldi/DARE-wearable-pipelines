import numpy as np
import pandas as pd
import os

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Select, DatetimeTickFormatter
from bokeh.layouts import column
from bokeh.io import curdoc

def plot_eda(parquet_folder):
    """
    Plot the EDA signal. The user can select the day to visualize using a dropdown menu.
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
    eda = pd.read_parquet(os.path.join(parquet_folder, day, "eda.parquet"))
    eda.index = pd.to_datetime(eda.index, unit="s")

    # Convert to ColumnDataSource for dynamic updates
    eda_source = ColumnDataSource(data={"time": eda.index, "eda": eda["eda"]})

    # Figure for EDA
    p1 = figure(title="EDA Signal", x_axis_label="Time", y_axis_label="EDA (µS)",
                x_axis_type='datetime', height=600, width=1250)
    p1.line("time", "eda", source=eda_source, line_width=2, color="purple", legend_label="EDA")

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
        eda = pd.read_parquet(os.path.join(parquet_folder, day, "eda.parquet"))
        eda.index = pd.to_datetime(eda.index, unit="s")
        eda_source.data = {"time": eda.index, "eda": eda["eda"]}

    # Set up callbacks
    day_select.on_change("value", update_plot)

    # Set up layout
    layout = column(day_select, p1)
    curdoc().add_root(layout)

# Run Bokeh app
parquet_path = "/Users/marcellosicbaldi/Library/CloudStorage/OneDrive-AlmaMaterStudiorumUniversitàdiBologna/tesi_Sara/Empatica/data/parquet/"
plot_eda(parquet_path)
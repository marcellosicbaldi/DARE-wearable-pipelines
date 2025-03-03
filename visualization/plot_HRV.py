import numpy as np
import pandas as pd
import os

from bokeh.plotting import figure, show
from bokeh.models import DatetimeTickFormatter
from bokeh.layouts import column, gridplot

def plot_HRV(ibi, hrv, hrv_metric = "rmssd"):
    """
    Plot the IBI and HRV signal.
    Parameters
    ----------
    ibi : pd.Series
        Night nter-beat interval signal (possibly already filtered for accelerometer bursts).
    hrv : pd.DataFrame
        Night heart rate variability signal.
    hrv_metric : str, optional
        HRV metric to plot.
    """

    p1 = figure(x_axis_type="datetime", title="IBI", width=1200, height=400)
    p1.line(ibi.index, ibi.values.flatten(), line_width=2)
    p1.title.text_font_size = "16pt"
    p1.yaxis.axis_label = "IBI (s)"
    p1.xaxis.axis_label_text_font_size = "16pt"
    p1.yaxis.axis_label_text_font_size = "16pt"
    p1.xaxis.major_label_text_font_size = "16pt"
    p1.yaxis.major_label_text_font_size = "16pt"
    p1.xaxis.formatter=DatetimeTickFormatter(
            hours="%H:%M",
            days="%d %B",
            months="%d %B",
            years="%d %B",
        )
    p1.xaxis.major_label_orientation = np.pi/4

    hrv.index = pd.to_datetime(hrv["time"])
    # HRV_df.drop(columns=["time"], inplace=True)
    HRV_df_plot = hrv.resample("1min").mean()

    p2 = figure(x_axis_type="datetime", title=hrv_metric, width=1200, height=400, x_range=p1.x_range)
    p2.line(HRV_df_plot["time"], HRV_df_plot[hrv_metric], legend_label=hrv_metric, line_width=2, color="blue")
    p2.scatter(HRV_df_plot["time"], HRV_df_plot[hrv_metric], legend_label=hrv_metric, color="blue", size = 11)
    # Horizontal line at the median, dashed
    p2.line([HRV_df_plot["time"].iloc[0], HRV_df_plot["time"].iloc[-1]], [np.median(HRV_df_plot[hrv_metric]), np.median(HRV_df_plot[hrv_metric])], 
            line_width=2, color="black", line_dash="dashed", legend_label="Median")
    p2.title.text_font_size = "16pt"
    p2.legend.location = "top_left"
    p2.legend.click_policy="hide"
    p2.yaxis.axis_label = "rmssd (ms)"
    p2.xaxis.axis_label_text_font_size = "16pt"
    p2.yaxis.axis_label_text_font_size = "16pt"
    p2.xaxis.major_label_text_font_size = "16pt"
    p2.yaxis.major_label_text_font_size = "16pt"
    # p2.y_range = Range1d(20, 110)
    p2.xaxis.formatter=DatetimeTickFormatter(
            hours="%H:%M",
            days="%d %B",
            months="%d %B",
            years="%d %B",
        )
    p2.xaxis.major_label_orientation = np.pi/4


    # Show the plot
    show(gridplot([[p1], [p2]]))
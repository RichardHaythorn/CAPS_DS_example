from datetime import time
from dataclasses import dataclass

import streamlit as st
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

st.set_page_config(layout="wide")


@dataclass
class Flyby:
    """Class for holding information about a flyby"""

    filepath: str
    start_time: time
    end_time: time


flyby_info = {
    "T55": Flyby("ELS_data_T55.csv", time(21, 22, 0), time(21, 35, 0)),
    "T56": Flyby("ELS_data_T56_a3.csv", time(19, 57, 0), time(20, 7, 0)),
}


def get_els_figure(df):
    x = df.index.to_numpy()
    y = range(63)
    z = df.to_numpy().transpose()

    fig, ax = plt.subplots(figsize=(20, 10))
    minmax = (1, 6e5)
    ax.pcolormesh(
        x, y, z, shading="nearest", norm=LogNorm(vmin=minmax[0], vmax=minmax[1])
    )
    ax.set_xlabel("Date/Time")
    ax.set_ylabel("Energy Level")
    fig.autofmt_xdate()
    return fig


st.title("Data exploration")


selected_flyby = st.sidebar.selectbox("Choose training flybys", flyby_info.keys())
start_time = st.sidebar.time_input(
    "Choose start time", value=flyby_info[selected_flyby].start_time, help="Only 6 hours loaded at a time"
)
end_time = st.sidebar.time_input(
    "Choose end time", value=flyby_info[selected_flyby].end_time, help="Only 6 hours loaded at a time"
)

df = pd.read_csv(flyby_info[selected_flyby].filepath).drop("Unnamed: 0", axis=1)
df = (
    df.astype({"Time": "datetime64[ms]"})
    .set_index("Time")
    .between_time(start_time, end_time)
)  # "21:22","21:35")
st.pyplot(get_els_figure(df))

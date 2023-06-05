import streamlit as st
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from processing.processing import get_els_figure

st.set_page_config(layout="wide")


st.title("Data exploration")

flyby_info = st.session_state["flyby_info"]

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
)
fig, ax = get_els_figure(df)
st.pyplot(fig)

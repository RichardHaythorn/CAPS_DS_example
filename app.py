import streamlit as st
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def get_els_figure(df):
    x = df.index.to_numpy()
    y = range(63)
    z = df.to_numpy().transpose()

    fig, ax  =plt.subplots(figsize=(10,10))
    minmax = (1,6e4)
    ax.pcolormesh(x,y,z,shading="nearest",norm=LogNorm(vmin=minmax[0], vmax=minmax[1]))
    fig.autofmt_xdate()
    return fig

st.title("Cassini Plasma Spectrometer")

df = pd.read_csv("ELS_data_T55.csv").drop("Unnamed: 0",axis=1)
df = df.astype({"Time":"datetime64[ms]"}).set_index("Time").between_time("21:22","21:35")
st.pyplot(get_els_figure(df))

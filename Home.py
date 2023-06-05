import streamlit as st
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


st.set_page_config(layout="wide")

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

flybys = ["T55","T56"]
side_chk = st.sidebar.multiselect("Choose training flybys",flybys)
text = st.sidebar.write(side_chk)

from datetime import datetime
from dataclasses import dataclass

import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, ConfusionMatrixDisplay

pl.Config.set_fmt_str_lengths(100)

from processing.processing import (
    RamModel,
    get_df,
    get_els_figure,
    make_X_y,
    incorrect_preds,
)

st.set_page_config(layout="wide")


@dataclass
class Flyby:
    """Class for holding information about a flyby"""

    filepath: list[str]
    start_time: datetime
    end_time: datetime
    start_ram_time: dict
    end_ram_time: dict


flyby_info = {
    "T55": Flyby(
        ["data/ELS_data_T55_a3.csv","data/ELS_data_T55_a4.csv"],
        datetime(2009, 5, 21, 21, 22, 0),
        datetime(2009, 5, 21, 21, 35, 0),
        {"3":datetime(2009, 5, 21, 21, 25, 0),"4":datetime(2009, 5, 21, 21, 25, 0)}, 
        {"3":datetime(2009, 5, 21, 21, 30, 0),"4":datetime(2009, 5, 21, 21, 30, 0)}, 
    ),
    "T56": Flyby(
        ["data/ELS_data_T56_a3.csv","data/ELS_data_T56_a4.csv"],
        datetime(2009, 6, 6, 19, 57, 0),
        datetime(2009, 6, 6, 20, 7, 0),
        datetime(2009, 6, 6, 19, 59, 0),
        datetime(2009, 6, 6, 20, 3, 0),
    ),
}

st.session_state["flyby_info"] = flyby_info

st.title("Cassini Plasma Spectrometer")

train_flybys = st.sidebar.multiselect(
    "Choose model training flybys", flyby_info.keys(), default="T55"
)

val_flyby = st.sidebar.selectbox("Choose validation flyby", flyby_info.keys())
val_anode = st.sidebar.selectbox("Choose validation anode", ["3","4"])
anode_map = {"3":0,"4":1}
val_filepath = [flyby_info[val_flyby].filepath[anode_map[val_anode]]]

val_df = get_df(
    val_filepath,
    flyby_info[val_flyby].start_ram_time,
    flyby_info[val_flyby].end_ram_time,
    scale=False,
)
X_val, y_val = make_X_y(val_df)

if list(set(train_flybys) & set([val_flyby])):
    st.sidebar.warning("Warning: Flyby chosen for validation in training set")

model = RamModel(train_flybys)
model.load_train_data(flyby_info)
model.fit()
y_pred = model.predict()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Model Performance")
    st.write("Flybys:", str(train_flybys))
    st.write(f"F1 Score: {model.f1_score(y_pred):.3f}")    
    st.write(f"Precision Score: {model.precision_score(y_pred):.3f}")
    conf_matrix = ConfusionMatrixDisplay.from_predictions(model.y_test, y_pred)
    st.pyplot(conf_matrix.figure_)

with col2:
    st.subheader("Validation")

    y_val_pred = model.predict(X_val)
    wrong_preds = incorrect_preds(X_val, y_val_pred, y_val)
    start_time = min(wrong_preds["Time"])
    end_time = max(wrong_preds["Time"])

    val_df = val_df.set_index("Time").drop(columns="Rammed")[start_time:end_time]

    fig, ax = get_els_figure(val_df)
    ax.vlines(wrong_preds["Time"], 0, 62, color="k")
    st.pyplot(fig)

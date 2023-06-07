"""Main entrypoint for the app"""
import warnings

import pandas as pd
import polars as pl
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score, precision_score

from processing.processing import RamModel, get_df, get_els_figure, make_x_y, join_y
from processing.data import flyby_info

if "flyby_info" not in st.session_state:
    st.session_state["flyby_info"] = flyby_info

pl.Config.set_fmt_str_lengths(100)

st.set_page_config(layout="wide")


st.title("Cassini Plasma Spectrometer")
st.divider()

train_flybys = st.sidebar.multiselect(
    "Choose model training flybys", flyby_info.keys(), default="T55"
)
max_iter = st.sidebar.slider("Max Iterations", min_value=100, max_value=1000)
st.sidebar.divider()


model = RamModel(train_flybys, max_iter)
model.load_train_data(st.session_state["flyby_info"])
with warnings.catch_warnings(record=True) as w:  # Alert user to when convergence fails
    model.fit()
y_pred = model.predict()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Model Scores")
    st.write(f"F1 Score: {f1_score(model.y_test, y_pred):.3f}")
    st.write(f"Precision Score: {precision_score(model.y_test,y_pred):.3f}")
    if w:
        st.warning(w[0].message)

with col2:
    conf_matrix = ConfusionMatrixDisplay.from_predictions(model.y_test, y_pred)
    st.pyplot(conf_matrix.figure_)


st.divider()
# -----------------Performance section of Page----------------
st.subheader("Model Performance")
val_flyby_options = [
    flyby
    for flyby in st.session_state["flyby_info"].keys()
    if flyby not in train_flybys
]
if val_flyby_options:
    val_flyby = st.sidebar.selectbox(
        "Choose flyby to check performance", val_flyby_options
    )
    val_anode = st.sidebar.selectbox("Choose flyby anode", ["3", "4"])
    val_filepath = st.session_state["flyby_info"][val_flyby].anodes[val_anode].filepath

    val_df = get_df(
        val_filepath,
        st.session_state["flyby_info"][val_flyby].anodes[val_anode].start_ram_time,
        st.session_state["flyby_info"][val_flyby].anodes[val_anode].end_ram_time,
        scale=False,
    )
    X_val, y_val = make_x_y(val_df)

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        shade_ram = st.checkbox("Shade Predicted Ram", value=True)

    y_val_pred = model.predict(X_val)
    joint_y = join_y(X_val, y_val_pred, y_val)

    start_time = joint_y.query("predicted == 1")["Time"].iloc[0]
    end_time = joint_y.query("predicted == 1")["Time"].iloc[-1]

    val_df = val_df.set_index("Time").drop(columns="Rammed")[
        start_time - pd.Timedelta(1, "min") : end_time + pd.Timedelta(1, "min")
    ]

    fig, ax = get_els_figure(val_df)

    if shade_ram:
        ax.fill_between(
            [
                start_time,
                end_time,
            ],
            0,
            63,
            alpha=0.35,
            color="r",
        )
    ax.set_ylim(0, 62)
    st.pyplot(fig)
else:
    st.write("No available flybys")

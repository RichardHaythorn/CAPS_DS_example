import streamlit as st
import polars as pl
from sklearn.metrics import ConfusionMatrixDisplay

from processing.processing import (
    RamModel,
    get_df,
    get_els_figure,
    make_X_y,
    incorrect_preds,
)
from processing.data import flyby_info

pl.Config.set_fmt_str_lengths(100)

st.set_page_config(layout="wide")


st.session_state["flyby_info"] = flyby_info

st.title("Cassini Plasma Spectrometer")

train_flybys = st.sidebar.multiselect(
    "Choose model training flybys", flyby_info.keys(), default="T55"
)

val_flyby = st.sidebar.selectbox("Choose validation flyby", flyby_info.keys())
val_anode = st.sidebar.selectbox("Choose validation anode", ["3", "4"])
val_filepath = flyby_info[val_flyby].anodes[val_anode].filepath

val_df = get_df(
    val_filepath,
    flyby_info[val_flyby].anodes[val_anode].start_ram_time,
    flyby_info[val_flyby].anodes[val_anode].end_ram_time,
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

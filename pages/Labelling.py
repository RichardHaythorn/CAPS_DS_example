"""Page for labelling the data"""
from datetime import datetime
import pandas as pd
import streamlit as st

from processing.processing import get_els_figure

st.set_page_config(layout="wide")


def get_df(flyby_info, selected_flyby, selected_anode, start_time, end_time):
    """Get dataframe for a given flyby & anode"""
    dataframe = pd.read_parquet(
        flyby_info[selected_flyby].anodes[selected_anode].filepath
    ).drop("Unnamed: 0", axis=1)
    dataframe = (
        dataframe.astype({"Time": "datetime64[ms]"})
        .set_index("Time")
        .between_time(start_time, end_time)
    )

    return dataframe


def update_flyby_attr(selected_flyby, name):
    """Update a given flyby attribute in the session state"""
    set_time = getattr(st.session_state, name)
    set_date = getattr(st.session_state["flyby_info"][selected_flyby], name).date()
    new_datetime = datetime.combine(set_date, set_time)
    setattr(
        st.session_state["flyby_info"][selected_flyby],
        name,
        new_datetime,
    )


def update_anode_attr(selected_flyby, selected_anode, name):
    """Update a given flyby attribute in the session state"""
    set_time = getattr(st.session_state, name)
    set_date = getattr(
        st.session_state["flyby_info"][selected_flyby].anodes[selected_anode], name
    ).date()
    new_datetime = datetime.combine(set_date, set_time)
    setattr(
        st.session_state["flyby_info"][selected_flyby].anodes[selected_anode],
        name,
        new_datetime,
    )


st.title("Data exploration")

flyby_info = st.session_state["flyby_info"]

selected_flyby = st.sidebar.selectbox("Choose flyby", flyby_info.keys())
selected_anode = st.sidebar.selectbox("Choose anode", ["3", "4"])
start_time = st.sidebar.time_input(
    "Plot start time",
    value=flyby_info[selected_flyby].start_time,
    help="Only 6 hours loaded at a time",
    step=300,
    on_change=update_flyby_attr,
    args=(selected_flyby, "start_time"),
    key="start_time",
)
end_time = st.sidebar.time_input(
    "Plot end time",
    value=flyby_info[selected_flyby].end_time,
    help="Only 6 hours loaded at a time",
    step=300,
    on_change=update_flyby_attr,
    args=(selected_flyby, "end_time"),
    key="end_time",
)
start_ram_time = st.sidebar.time_input(
    "Set start ram time",
    value=flyby_info[selected_flyby].anodes[selected_anode].start_ram_time,
    step=60,
    on_change=update_anode_attr,
    args=(selected_flyby, selected_anode, "start_ram_time"),
    key="start_ram_time",
)
end_ram_time = st.sidebar.time_input(
    "Set end ram time",
    value=flyby_info[selected_flyby].anodes[selected_anode].end_ram_time,
    step=60,
    on_change=update_anode_attr,
    args=(selected_flyby, selected_anode, "end_ram_time"),
    key="end_ram_time",
)

dataframe = get_df(flyby_info, selected_flyby, selected_anode, start_time, end_time)

plot_col1, plot_col2 = st.columns(2)
with plot_col1:
    shade_ram = st.checkbox("Shade Labelled Ram", value=True)
fig, ax = get_els_figure(dataframe)
if shade_ram:
    ax.fill_between(
        [
            flyby_info[selected_flyby].anodes[selected_anode].start_ram_time,
            flyby_info[selected_flyby].anodes[selected_anode].end_ram_time,
        ],
        0,
        63,
        alpha=0.35,
        color="r",
    )
ax.set_ylim(0, 62)
st.pyplot(fig)

with st.expander("Session State"):
    st.write(st.session_state)

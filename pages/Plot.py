import streamlit as st
import pandas as pd


from processing.processing import get_els_figure

st.set_page_config(layout="wide")


st.title("Data exploration")

flyby_info = st.session_state["flyby_info"]

selected_flyby = st.sidebar.selectbox("Choose flyby", flyby_info.keys())
selected_anode = st.sidebar.selectbox("Choose anode", ["3", "4"])
start_time = st.sidebar.time_input(
    "Choose start time",
    value=flyby_info[selected_flyby].start_time,
    help="Only 6 hours loaded at a time",
)
end_time = st.sidebar.time_input(
    "Choose end time",
    value=flyby_info[selected_flyby].end_time,
    help="Only 6 hours loaded at a time",
)

df = pd.read_csv(flyby_info[selected_flyby].anodes[selected_anode].filepath).drop(
    "Unnamed: 0", axis=1
)
df = (
    df.astype({"Time": "datetime64[ms]"})
    .set_index("Time")
    .between_time(start_time, end_time)
)
fig, ax = get_els_figure(df)
st.pyplot(fig)

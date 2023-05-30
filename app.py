import streamlit as st
import pandas as pd


st.title("Cassini Plasma Spectrometer")

df = pd.read_csv("ELS_data_1.csv")
st.dataframe(df)
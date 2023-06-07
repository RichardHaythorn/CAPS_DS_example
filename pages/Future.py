"""Page for labelling the data"""
import streamlit as st

st.set_page_config(layout="wide")

st.title("Possible things to add")

st.markdown(
    """
- Clean data to remove anomalies
- Cross-validation of data
- More flybys
- Investigate different classifiers
- Hyperparameter tuning
"""
)

# shl_catalog_viewer.py
import pandas as pd
import streamlit as st

# Load scraped CSV
df = pd.read_csv("shl_data.csv")

st.title("SHL Assessment Catalog")
st.markdown("View scraped assessments below:")

st.dataframe(df)


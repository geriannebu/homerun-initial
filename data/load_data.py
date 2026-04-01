import pandas as pd
import streamlit as st

@st.cache_data
def load_all_data():
    listings = pd.read_csv("data/final.csv")

    # -------------------------------
    # CONVERT UNITS
    # -------------------------------
    listings["floor_area_sqm"] = listings["floor_area_sqft"] * 0.092903
    listings["floor_area_sqm"] = listings["floor_area_sqm"].round().astype(int)


    return listings, None
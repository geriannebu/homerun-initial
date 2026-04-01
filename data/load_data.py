import pandas as pd
import streamlit as st

@st.cache_data
def load_all_data():
    """
    Load and clean the listings dataset.
    Returns:
        df: cleaned DataFrame
        None: placeholder (for compatibility)
    """
    df = pd.read_csv("data/final.csv")
    df.columns = df.columns.str.strip()

    # ── Numeric conversions ───────────────────────
    df["floor_area_sqm"] = pd.to_numeric(df["floor_area_sqm"], errors="coerce").fillna(0).round().astype(int)
    df["asking_price"] = pd.to_numeric(df["asking_price"], errors="coerce").fillna(0)
    df["predicted_price"] = pd.to_numeric(df["predicted_price"], errors="coerce").fillna(0)
    df["valuation_pct"] = pd.to_numeric(df.get("valuation_pct", 0), errors="coerce").fillna(0)

    # ── IDs & fallback columns ───────────────────
    df["listing_id"] = df.index.astype(str)
    df["storey_range"] = df["storey_midpoint"].fillna(0).astype(int).astype(str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["median_6m_similar"] = pd.to_numeric(df.get("median_6m_similar", 0), errors="coerce").fillna(0)

    return df, None
import numpy as np
import pandas as pd

from backend.utils.constants import TOWNS
from backend.utils.scoring import classify_listing
from backend.schemas.inputs import UserInputs

from data.load_data import load_all_data

def get_active_listings(inputs: UserInputs) -> pd.DataFrame:
    df, _ = load_all_data()

    # --- FILTERING ---
    if inputs.town:
        df = df[df["town"] == inputs.town]

    df = df[df["flat_type"] == inputs.flat_type]

    # Optional: area filter
    df = df[
        (df["floor_area_sqm"] >= inputs.floor_area_sqm - 10) &
        (df["floor_area_sqm"] <= inputs.floor_area_sqm + 10)
    ]


      # ── REQUIRED FIELDS ───────────────────────
    df["listing_id"] = df.index.astype(str)
    df["listing_url"] = "#"

    # already exists in your dataset
    df["recent_median_transacted"] = df["median_6m_similar"]

    # valuation already computed
    df["asking_vs_predicted_pct"] = df["valuation_pct"]

    return df.copy()
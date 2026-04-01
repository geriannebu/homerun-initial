# backend/services/recommendation_service.py

import pandas as pd
import numpy as np

from data.load_data import load_all_data
from backend.utils.constants import TOWNS, AMENITY_KEYS
from backend.schemas.inputs import UserInputs


def _distance_score(dist: float) -> float:
    """Convert distance in meters to a score out of 100."""
    if pd.isna(dist):
        return 40
    if dist <= 300:
        return 90
    elif dist <= 600:
        return 75
    elif dist <= 1000:
        return 60
    else:
        return 40


def compute_amenity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute amenity scores per listing."""
    mapping = {
        "mrt": "train_1_dist_m",
        "bus": "bus_1_dist_m",
        "schools": "school_1_dist_m",
        "hawker": "hawker_1_dist_m",
        "retail": "mall_1_dist_m",
        "healthcare": "polyclinic_1_dist_m",
    }
    for amen, col in mapping.items():
        df[f"{amen}_score"] = df[col].apply(_distance_score)
    df["amenity_avg"] = df[[f"{amen}_score" for amen in AMENITY_KEYS]].mean(axis=1)
    return df


def recommend_towns_real(inputs: UserInputs, top_n: int = 5) -> pd.DataFrame:
    """
    Returns top town recommendations based on predicted price, budget, and amenity scores.
    """

    listings_df, _ = load_all_data()

    # Filter by flat type & approximate floor area if provided
    if inputs.flat_type:
        listings_df = listings_df[listings_df["flat_type"] == inputs.flat_type]
    if inputs.floor_area_sqm:
        listings_df = listings_df[
            (listings_df["floor_area_sqm"] >= inputs.floor_area_sqm - 10) &
            (listings_df["floor_area_sqm"] <= inputs.floor_area_sqm + 10)
        ]

    if listings_df.empty:
        # Fallback if no listings match
        return pd.DataFrame(columns=[
            "town", "estimated_price", "within_budget", "match_score", "why_it_matches"
        ])

    # Compute amenity scores
    listings_df = compute_amenity_scores(listings_df)

    # Compute predicted price per listing (use 'predicted_price' if exists)
    listings_df["predicted_price"] = listings_df.get("predicted_price", listings_df["asking_price"])

    # Aggregate per town
    town_group = listings_df.groupby("town").agg(
        median_predicted=("predicted_price", "median"),
        median_amenity=("amenity_avg", "median"),
        count_listings=("listing_id", "count")
    ).reset_index()

    # Budget filter
    if inputs.budget:
        town_group["within_budget"] = town_group["median_predicted"] <= inputs.budget
    else:
        town_group["within_budget"] = True

    # Compute overall match score (amenity + budget + median price favorability)
    town_group["match_score"] = (
        0.6 * (town_group["median_predicted"].max() - town_group["median_predicted"]) /
        max(town_group["median_predicted"].max(), 1) * 100
        + 0.4 * town_group["median_amenity"]
    ).clip(0, 100).round(1)

    # Sort by match score descending
    town_group = town_group.sort_values("match_score", ascending=False).head(top_n)

    # Add why_it_matches blurb
    def _why(row):
        reasons = []
        if row["within_budget"]:
            reasons.append("affordable within budget")
        if row["median_amenity"] >= 70:
            reasons.append("good amenities nearby")
        if row["count_listings"] >= 5:
            reasons.append("lots of available flats")
        return "Strong overall fit: " + " · ".join(reasons) if reasons else "Good fit overall"

    town_group["why_it_matches"] = town_group.apply(_why, axis=1)
    town_group.rename(columns={"median_predicted": "estimated_price"}, inplace=True)

    return town_group[["town", "estimated_price", "within_budget", "match_score", "why_it_matches"]]
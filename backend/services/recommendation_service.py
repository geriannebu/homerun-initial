# backend/services/recommendation_service.py
import pandas as pd
from data.load_data import load_all_data
from backend.schemas.inputs import UserInputs
from backend.services.recommender import run_recommender
from backend.utils.constants import AMENITY_KEYS

# ── Amenity scoring helper ─────────────────────────────────────────────────────
def compute_amenity_scores(df: pd.DataFrame) -> pd.DataFrame:
    for amen in AMENITY_KEYS:
        col = f"walk_{amen}_avg_mins"
        if col in df.columns:
            df[f"{amen}_score"] = df[col].apply(
                lambda x: 100 * np.exp(-x / 8) if pd.notna(x) else 40
            )
        else:
            df[f"{amen}_score"] = 40

    df["amenity_avg"] = df[[f"{a}_score" for a in AMENITY_KEYS]].mean(axis=1)
    return df

# ── Town-level recommendations ────────────────────────────────────────────────
def recommend_towns_real(
    inputs: UserInputs,
    df: pd.DataFrame,
    amenity_ranking: list[str] = None,
    amenity_weights: dict[str, float] = None,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Returns top town recommendations using the real cleaned dataset.
    Relies on run_recommender for listing-level scoring.
    """
    results = []

    # Optional filters from user input
    if getattr(inputs, "flat_type", None):
        df = df[df["flat_type"] == inputs.flat_type]
    if getattr(inputs, "floor_area_sqm", None):
        df = df[
            (df["floor_area_sqm"] >= inputs.floor_area_sqm - 10) &
            (df["floor_area_sqm"] <= inputs.floor_area_sqm + 10)
        ]
    
    towns = df["town"].unique()
    for town in towns:
        town_listings = df[df["town"] == town]
        if town_listings.empty:
            continue

        rec = run_recommender(
            listings_df=town_listings,
            amenity_ranking=amenity_ranking or list(amenity_weights.keys()) if amenity_weights else [],
            amenity_weights=amenity_weights or {a: 1 for a in (amenity_ranking or [])},
            alpha=0.5,
            budget=getattr(inputs, "budget", None) or 1_000_000,
            rooms=[],
            preferred_towns=[],
            min_sqft=0,
            top_n=len(town_listings)
        )

        if rec["top"].empty:
            continue

        median_price = rec["top"]["predicted_price"].median()
        median_score = rec["top"]["final_score"].median()
        count_listings = len(rec["top"])
        results.append({
            "town": town,
            "estimated_price": median_price,
            "match_score": median_score * 100,  # scale 0..1 → 0..100
            "count_listings": count_listings
        })

    town_df = pd.DataFrame(results)
    if town_df.empty:
        return pd.DataFrame(columns=[
            "town", "estimated_price", "match_score", "why_it_matches"
        ])

    # Sort and keep top_n
    town_df = town_df.sort_values("match_score", ascending=False).head(top_n)

    # Add descriptive blurbs
    def _why(row):
        reasons = []
        if getattr(inputs, "budget", None) and row["estimated_price"] <= inputs.budget:
            reasons.append("affordable within budget")
        if row["match_score"] >= 70:
            reasons.append("good amenities nearby")
        if row["count_listings"] >= 5:
            reasons.append("lots of available flats")
        return "Strong overall fit: " + " · ".join(reasons) if reasons else "Good fit overall"

    town_df["why_it_matches"] = town_df.apply(_why, axis=1)

    return town_df[["town", "estimated_price", "match_score", "why_it_matches"]]

# ── Example usage ─────────────────────────────────────────────────────────────
def get_top_towns(inputs: UserInputs, top_n: int = 5):
    listings_df, _ = load_all_data()
    return recommend_towns_real(inputs, df=listings_df, top_n=top_n)
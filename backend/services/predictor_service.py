import pandas as pd
from data.load_data import load_all_data
from backend.schemas.inputs import UserInputs
from backend.services.recommendation_service import recommend_towns_real
from backend.utils.scoring import compute_listing_scores

# Default amenity weights
DEFAULT_AMENITY_WEIGHTS = {
    "mrt": 1,
    "bus": 1,
    "schools": 1,
    "hawker": 1,
    "retail": 1,
    "healthcare": 1,
}

def get_prediction_bundle(inputs: UserInputs, ranking_profile: str = "balanced") -> dict:
    """
    Returns a bundle of:
    - filtered listings dataframe with computed scores
    - town recommendations if no town specified
    - predicted price, median, confidence intervals
    """
    # ── Load data ───────────────────────────────
    listings_df, _ = load_all_data()

    # ── Filter by user inputs ───────────────────
    if inputs.town:
        listings_df = listings_df[listings_df["town"] == inputs.town]
    if inputs.flat_type:
        listings_df = listings_df[listings_df["flat_type"] == inputs.flat_type]
    if inputs.floor_area_sqm:
        listings_df = listings_df[
            (listings_df["floor_area_sqm"] >= inputs.floor_area_sqm - 10) &
            (listings_df["floor_area_sqm"] <= inputs.floor_area_sqm + 10)
        ]

    # Ensure listing_id exists
    if "listing_id" not in listings_df.columns:
        listings_df["listing_id"] = listings_df.index.astype(str)

    # ── Compute scores ──────────────────────────
    amenity_weights = getattr(inputs, "amenity_weights", DEFAULT_AMENITY_WEIGHTS)
    listings_df = compute_listing_scores(
        listings_df=listings_df,
        budget=getattr(inputs, "budget", None),
        amenity_weights=amenity_weights,
        ranking_profile=ranking_profile
    )

    # ── Summary stats ──────────────────────────
    viable_count = len(listings_df)
    predicted_price = int(listings_df["predicted_price"].median()) if viable_count else 0
    median_asking = int(listings_df["asking_price"].median()) if viable_count else 0
    recent_median_transacted = int(listings_df["median_6m_similar"].median()) if viable_count else 0
    confidence_low = round(predicted_price * 0.96)
    confidence_high = round(predicted_price * 1.04)

    # ── Recommendations if no town selected ───
    recommendations_df = recommend_towns_real(inputs) if not inputs.town else None

    # ── Filter report for frontend ─────────────
    filter_report = {
        "viable_listing_count": viable_count,
        "budget_filter": getattr(inputs, "budget", None),
        "flat_type": inputs.flat_type,
        "floor_area_sqm": inputs.floor_area_sqm,
    }

    mode = "town" if inputs.town else "recommendation"
    mode_label = f"Town mode: {inputs.town}" if inputs.town else "Recommendation mode"

    return {
        "predicted_price": predicted_price,
        "recent_median_transacted": recent_median_transacted,
        "confidence_low": confidence_low,
        "confidence_high": confidence_high,
        "recent_period": "last 6 months",
        "listings_df": listings_df,
        "recommendations_df": recommendations_df,
        "viable_listing_count": viable_count,
        "median_asking_active": median_asking,
        "mode": mode,
        "mode_label": mode_label,
        "ranking_profile": ranking_profile,
        "filter_report": filter_report,
    }
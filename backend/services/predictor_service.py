# backend/services/predictor_service.py
import pandas as pd
from data.load_data import load_all_data
from backend.schemas.inputs import UserInputs
from backend.services.recommender import run_recommender, load_listings
from backend.services.recommendation_service import recommend_towns_real

# Default amenity weights (fallback)
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
    # ── Load real dataset ─────────────────────────────
    listings_df, _ = load_all_data()

    # ── Filter by user inputs ─────────────────────────
    if getattr(inputs, "town", None):
        listings_df = listings_df[listings_df["planning_area"] == inputs.town]
    if getattr(inputs, "flat_type", None):
        listings_df = listings_df[listings_df["flat_type"] == inputs.flat_type]
    if getattr(inputs, "floor_area_sqm", None):
        listings_df = listings_df[
            (listings_df["floor_area_sqm"] >= inputs.floor_area_sqm - 10) &
            (listings_df["floor_area_sqm"] <= inputs.floor_area_sqm + 10)
        ]

    # Ensure listing_id exists
    if "listing_id" not in listings_df.columns:
        listings_df["listing_id"] = listings_df.index.astype(str)

    # ── Use ranking & weights already collected during onboarding ─────────
    amenity_ranking = getattr(inputs, "amenity_rank", None) or list(DEFAULT_AMENITY_WEIGHTS.keys())
    amenity_weights = getattr(inputs, "amenity_weights", None) or DEFAULT_AMENITY_WEIGHTS

    # ── Compute listing-level scores using recommender.py ─────────────────
    rec_results = run_recommender(
        listings_df=listings_df,
        amenity_ranking=amenity_ranking,
        amenity_weights=amenity_weights,
        alpha=0.5,  # can also be dynamic based on user slider
        budget=getattr(inputs, "budget", None) or 1_000_000,
        rooms=[],  # can expand later
        preferred_towns=[inputs.town] if getattr(inputs, "town", None) else [],
        min_sqft=0,
        top_n=15,
    )
    scored_listings = rec_results["filtered"]

    # ── Summary stats ────────────────────────────────
    viable_count = len(scored_listings)
    predicted_price = int(scored_listings["predicted_price"].median()) if viable_count else 0
    median_asking = int(scored_listings["asking_price"].median()) if viable_count else 0
    recent_median_transacted = int(scored_listings.get("median_6m_similar", 0).median()) if viable_count else 0
    confidence_low = round(predicted_price * 0.96)
    confidence_high = round(predicted_price * 1.04)

    # ── Town recommendations if no specific town selected ────────────────
    recommendations_df = None
    if not getattr(inputs, "town", None):
        recommendations_df = recommend_towns_real(
            inputs=inputs,
            df=listings_df,
            amenity_ranking=amenity_ranking,
            amenity_weights=amenity_weights,
            top_n=5,
        )

    # ── Filter report for frontend ─────────────────────
    filter_report = {
        "viable_listing_count": viable_count,
        "budget_filter": getattr(inputs, "budget", None),
        "flat_type": getattr(inputs, "flat_type", None),
        "floor_area_sqm": getattr(inputs, "floor_area_sqm", None),
    }

    mode = "town" if getattr(inputs, "town", None) else "recommendation"
    mode_label = f"Town mode: {inputs.town}" if getattr(inputs, "town", None) else "Recommendation mode"

    return {
        "predicted_price": predicted_price,
        "recent_median_transacted": recent_median_transacted,
        "confidence_low": confidence_low,
        "confidence_high": confidence_high,
        "recent_period": "last 6 months",
        "listings_df": scored_listings,
        "recommendations_df": recommendations_df,
        "viable_listing_count": viable_count,
        "median_asking_active": median_asking,
        "mode": mode,
        "mode_label": mode_label,
        "ranking_profile": ranking_profile,
        "filter_report": filter_report,
    }
from typing import Any, Dict
import streamlit as st

from backend.schemas.inputs import UserInputs
from backend.utils.formatters import fmt_sgd
from backend.utils.constants import AMENITY_LABELS

from frontend.components.listing_detail import show_listing_detail

# DELETED PRICE STORY AT THE TOP

# ---------------------------------------------------------------------------
# HomeRun Pick Card
# ---------------------------------------------------------------------------
def render_homerun_pick(inputs: UserInputs, bundle: Dict[str,Any]):
    if bundle["listings_df"].empty:
        st.info("No listings available to pick from.")
        return

    # Ensure amenity weights exist
    amenity_weights = getattr(inputs, "amenity_weights", None) or {
        "mrt":1, "bus":1, "schools":1, "hawker":1, "retail":1, "healthcare":1
    }

    if bundle["top"].empty:
        st.info("No listings available to pick from.")
        return

    top = bundle["top"].iloc[0]  # Already scored and ranked in recommender.py

    st.markdown("### 🏆 HomeRun Pick Right Now")
    st.write(f"**Listing ID:** {top['listing_id']}")
    st.write(f"**Town:** {top['town']}")
    st.write(f"**Flat Type:** {top.get('flat_type', 'N/A')}")
    st.write(f"**Floor Area:** {top.get('floor_area_sqm', 0)} sqm")
    st.write(f"**Asking Price:** {fmt_sgd(top['asking_price'])}")
    st.write(f"**Predicted Price:** {fmt_sgd(top['predicted_price'])}")
    st.write(f"**Valuation:** {top['valuation_label']}")
    st.write(f"**Overall Score:** {top['overall_value_score']:.1f}/100")

    # Button to open listing detail dialog
    payload = top.to_dict()
    st.button("View Details", on_click=show_listing_detail, args=(payload,))

    st.write("**Amenities Nearby:**")
    amenity_keys = ["train", "bus", "primary_school", "hawker", "mall", "polyclinic"]

    for amen in amenity_keys:
        score_col = f"walk_acc_{amen}"      # from stage3_score in recommender.py
        dist_col  = f"walk_{amen}_avg_mins" # display avg walking time

        distance = top.get(dist_col, None)
        score    = top.get(score_col, None)
        if distance is not None and score is not None:
            if score >= 0.8:
                closeness = "Very close"
            elif score >= 0.6:
                closeness = "Close"
            elif score >= 0.4:
                closeness = "Moderate"
            else:
                closeness = "Far"

            st.write(f"• {AMENITY_LABELS[amen]}: {closeness} ({distance:.0f} min walk, score: {score:.2f})")

        
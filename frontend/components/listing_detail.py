"""
frontend/components/listing_detail.py

Fully data-driven listing detail dialog.
Uses real dataset columns from final.csv.
"""

import streamlit as st
import streamlit.components.v1 as components

from backend.utils.formatters import fmt_sgd
from frontend.state.session import record_swipe


# ── Map (uses real lat/lon) ─────────────────────────────────────────────
def _map_iframe(lat: float, lon: float, height: int = 220) -> str:
    if lat is None or lon is None:
        return ""

    src = (
        f"https://www.openstreetmap.org/export/embed.html"
        f"?bbox={lon-0.01},{lat-0.006},{lon+0.01},{lat+0.006}"
        f"&layer=mapnik&marker={lat},{lon}"
    )
    return f"""
    <iframe src="{src}" width="100%" height="{height}"
    style="border:none;border-radius:12px;"></iframe>
    """


# ── Simple distance → score conversion ──────────────────────────────────
def _distance_score(dist):
    if dist is None:
        return 40
    if dist <= 300:
        return 90
    elif dist <= 600:
        return 75
    elif dist <= 1000:
        return 60
    else:
        return 40


# ── Valuation styling ───────────────────────────────────────────────────
def _val_style(diff):
    if diff <= -5:
        return "Great Deal", "#059E87"
    elif diff <= 3:
        return "Fair Value", "#2563eb"
    elif diff <= 10:
        return "Slightly High", "#d97706"
    else:
        return "Overpriced", "#dc2626"


@st.dialog("Listing Details", width="large")
def show_listing_detail(listing_id: str):

    # ── Find listing ─────────────────────────────────────────────────────
    row = None
    session_data = None

    for s in st.session_state.get("search_sessions", []):
        df = s["bundle"]["listings_df"]
        match = df[df["listing_id"] == listing_id]
        if not match.empty:
            row = match.iloc[0]
            session_data = s
            break

    if row is None:
        st.error("Listing not found.")
        return

    # ── Extract core data ────────────────────────────────────────────────
    asking = int(row["asking_price"])
    predicted = int(row["predicted_price"])
    diff = float(row["asking_vs_predicted_pct"])

    town = row.get("town", "")
    flat_type = row.get("flat_type", "")
    area = round(row.get("floor_area_sqm", 0))
    storey = row.get("storey_midpoint", "")

    lat = row.get("lat")
    lon = row.get("lon")

    # ── Valuation ────────────────────────────────────────────────────────
    val_label, val_color = _val_style(diff)
    sign = "+" if diff >= 0 else ""

    # ── Real Amenity Scores ─────────────────────────────────────────────
    mrt_score = _distance_score(row.get("train_1_dist_m"))
    bus_score = _distance_score(row.get("bus_1_dist_m"))
    school_score = _distance_score(row.get("school_1_dist_m"))
    hawker_score = _distance_score(row.get("hawker_1_dist_m"))
    retail_score = _distance_score(row.get("mall_1_dist_m"))
    health_score = _distance_score(row.get("polyclinic_1_dist_m"))

    amenity_avg = (
        mrt_score + bus_score + school_score +
        hawker_score + retail_score + health_score
    ) / 6

    # ── Value score ─────────────────────────────────────────────────────
    value_score = max(0, min(100, 100 - abs(diff)))

    overall_score = round((value_score * 0.6 + amenity_avg * 0.4), 1)

    # ── HEADER ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;">
        <div>
            <h2>{town}</h2>
            <p>{flat_type} · {area} sqm · Storey {storey}</p>
        </div>
        <div style="text-align:right;">
            <h3>{fmt_sgd(asking)}</h3>
            <p style="color:#9ca3af;">est. {fmt_sgd(predicted)}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Price analysis ───────────────────────────────────────────────────
    st.markdown("### 💰 Price Analysis")
    st.markdown(f"""
    - Asking: **{fmt_sgd(asking)}**
    - Estimated: **{fmt_sgd(predicted)}**
    - Difference: **{sign}{diff:.1f}%**
    - Valuation: <span style="color:{val_color};font-weight:700;">{val_label}</span>
    """, unsafe_allow_html=True)

    # ── Scores ──────────────────────────────────────────────────────────
    st.markdown("### 🎯 Match Scores")

    st.progress(overall_score / 100)
    st.write(f"Overall Match: **{overall_score:.0f}%**")

    st.write(f"💰 Value Score: {value_score:.0f}")
    st.write(f"📍 Amenity Score: {amenity_avg:.0f}")

    # ── Amenity Breakdown ───────────────────────────────────────────────
    st.markdown("### 📍 Amenities Nearby")

    def row_display(label, score, dist):
        return f"{label}: {score:.0f} ({dist:.0f}m)"

    st.write(row_display("🚇 MRT", mrt_score, row.get("train_1_dist_m", 0)))
    st.write(row_display("🚌 Bus", bus_score, row.get("bus_1_dist_m", 0)))
    st.write(row_display("🏫 School", school_score, row.get("school_1_dist_m", 0)))
    st.write(row_display("🍜 Hawker", hawker_score, row.get("hawker_1_dist_m", 0)))
    st.write(row_display("🛍️ Mall", retail_score, row.get("mall_1_dist_m", 0)))
    st.write(row_display("🏥 Polyclinic", health_score, row.get("polyclinic_1_dist_m", 0)))

    # ── Market context ──────────────────────────────────────────────────
    st.markdown("### 📊 Market Context")

    st.write(f"Recent median: {fmt_sgd(row.get('median_6m_similar', 0))}")

    # ── Map ─────────────────────────────────────────────────────────────
    st.markdown("### 🗺️ Location")
    st.markdown(_map_iframe(lat, lon), unsafe_allow_html=True)

    # ── Actions ─────────────────────────────────────────────────────────
    st.markdown("---")

    is_saved = session_data and listing_id in session_data.get("liked_ids", [])
    is_super = session_data and listing_id in session_data.get("super_ids", [])

    col1, col2 = st.columns(2)

    with col1:
        if not is_saved:
            if st.button("♥ Save", use_container_width=True):
                record_swipe(session_data["session_id"], listing_id, "right")
                st.rerun()
        else:
            st.success("Saved")

    with col2:
        if not is_super:
            if st.button("⭐ Super Save", use_container_width=True):
                record_swipe(session_data["session_id"], listing_id, "up")
                st.rerun()
        else:
            st.warning("Super Saved")
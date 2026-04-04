import math
from typing import Any, Dict
import streamlit as st

from backend.utils.formatters import fmt_sgd
from frontend.state.session import record_swipe


# ── Map helper ──────────────────────────────────────────────────────────────
def _map_iframe(lat, lon, height: int = 220) -> str:
    if lat is None or lon is None:
        return ""
    src = (
        f"https://www.openstreetmap.org/export/embed.html"
        f"?bbox={lon-0.01},{lat-0.006},{lon+0.01},{lat+0.006}"
        f"&layer=mapnik&marker={lat},{lon}"
    )
    return f'<iframe src="{src}" width="100%" height="{height}" style="border:none;border-radius:12px;"></iframe>'


# ── Distance scoring helpers ───────────────────────────────────────────────
def _distance_score(dist):
    if dist is None:
        return 40
    try:
        if math.isnan(float(dist)):
            return 40
    except Exception:
        pass

    if dist <= 300:
        return 90
    if dist <= 600:
        return 75
    if dist <= 1000:
        return 60
    return 40


def _proximity_label(dist):
    if dist is None:
        return "Very far"
    try:
        if math.isnan(float(dist)) or float(dist) > 1500:
            return "Very far"
    except Exception:
        return "Very far"

    if dist > 1000:
        return "Far"
    if dist > 600:
        return "Moderate"
    if dist > 300:
        return "Close"
    return "Very close"


def _val_style(diff):
    if diff <= -5:
        return "Great Deal", "#059E87"
    if diff <= 3:
        return "Fair Value", "#2563eb"
    if diff <= 10:
        return "Slightly High", "#d97706"
    return "Overpriced", "#dc2626"


def _find_listing_row(listing_id):
    target_id = str(listing_id)

    for s in st.session_state.get("search_sessions", []):
        df = s["bundle"]["listings_df"].copy()
        if "listing_id" not in df.columns:
            continue

        match = df[df["listing_id"].astype(str) == target_id]
        if not match.empty:
            return match.iloc[0], s

    return None, None


# ── Main dialog ──────────────────────────────────────────────────────────────
@st.dialog("Listing Details", width="large")
def show_listing_detail(payload: Dict[str, Any] | str | int, show_actions: bool = True):
    import json

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            pass

    if isinstance(payload, dict):
        listing_id = payload.get("listing_id") or payload.get("id")
        row = payload
    elif isinstance(payload, (int, str)):
        listing_id = payload
        row = None
    else:
        st.error("Invalid payload type")
        return

    if listing_id is None:
        st.error("Listing ID missing")
        return

    db_row, session_data = _find_listing_row(listing_id)

    # Prefer the full dataframe row from the recommender dataset
    if db_row is not None:
        row = db_row

    if row is None:
        st.error("Listing not found.")
        return

    asking = int(row.get("asking_price", 0))
    predicted = int(row.get("predicted_price", 0))

    # Use the same field as the card deck
    diff = float(row.get("asking_vs_predicted_pct", row.get("valuation_pct", 0)))

    ci_low = row.get("predicted_price_lower")
    ci_high = row.get("predicted_price_upper")
    try:
        ci_low = int(ci_low) if ci_low is not None and not math.isnan(float(ci_low)) else None
        ci_high = int(ci_high) if ci_high is not None and not math.isnan(float(ci_high)) else None
    except (TypeError, ValueError):
        ci_low = ci_high = None

    town = str(row.get("town", ""))
    flat_type = str(row.get("flat_type", ""))
    area = round(float(row.get("floor_area_sqm", 0)))
    storey = row.get("storey_range", row.get("storey_midpoint", ""))
    address = str(row.get("address", row.get("full_address", "")))
    lat = row.get("lat")
    lon = row.get("lon")
    remaining = row.get("remaining_lease")

    mrt_dist = row.get("train_1_dist_m")
    bus_dist = row.get("bus_1_dist_m")
    school_dist = row.get("school_1_dist_m")
    hawker_dist = row.get("hawker_1_dist_m")
    retail_dist = row.get("mall_1_dist_m")
    health_dist = row.get("polyclinic_1_dist_m")

    mrt_score = _distance_score(mrt_dist)
    bus_score = _distance_score(bus_dist)
    school_score = _distance_score(school_dist)
    hawker_score = _distance_score(hawker_dist)
    retail_score = _distance_score(retail_dist)
    health_score = _distance_score(health_dist)

    amenity_avg = (mrt_score + bus_score + school_score + hawker_score + retail_score + health_score) / 6
    value_score = max(0, min(100, 100 - abs(diff)))
    overall_score = round(value_score * 0.6 + amenity_avg * 0.4, 1)

    val_label, val_color = _val_style(diff)
    sign = "+" if diff >= 0 else ""

    # ── Actions ─────────────────────────────────────────────────────────────
    if show_actions:
        liked_ids = [str(x) for x in session_data.get("liked_ids", [])] if session_data else []
        passed_ids = [str(x) for x in session_data.get("passed_ids", [])] if session_data else []
        listing_id_str = str(listing_id)

        is_saved = listing_id_str in liked_ids
        is_passed = listing_id_str in passed_ids

        col1, col2 = st.columns(2)
        with col1:
            if not is_passed:
                if st.button("✕ Pass", use_container_width=True, key=f"detail_pass_{listing_id_str}"):
                    if session_data:
                        record_swipe(session_data["session_id"], listing_id_str, "left")
                    st.rerun()
            else:
                st.info("Passed")

        with col2:
            if not is_saved:
                if st.button("♥ Save", use_container_width=True, type="primary", key=f"detail_save_{listing_id_str}"):
                    if session_data:
                        record_swipe(session_data["session_id"], listing_id_str, "right")
                    st.rerun()
            else:
                st.success("Saved ♥")

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(f"## {town} · {flat_type}")
    st.caption(address or "Address unavailable")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Asking price", fmt_sgd(asking))
    with c2:
        st.metric("Predicted fair value", fmt_sgd(predicted))
    with c3:
        st.metric("Value gap", f"{sign}{diff:.1f}%")

    st.markdown(
        f"""
        <div style="
            margin: 10px 0 18px 0;
            display:inline-block;
            padding:6px 12px;
            border-radius:999px;
            background:{val_color};
            color:white;
            font-weight:700;
            font-size:0.82rem;">
            {val_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    c4, c5, c6 = st.columns(3)
    with c4:
        st.metric("Floor area", f"{area} sqm")
    with c5:
        st.metric("Storey", str(storey) if storey else "-")
    with c6:
        st.metric("Overall score", f"{overall_score:.1f}/100")

    if remaining:
        st.write(f"**Remaining lease:** {remaining}")

    if ci_low is not None and ci_high is not None:
        st.write(f"**Confidence band:** {fmt_sgd(ci_low)} – {fmt_sgd(ci_high)}")

    # ── Amenities ───────────────────────────────────────────────────────────
    st.markdown("### Nearby amenities")

    a1, a2, a3 = st.columns(3)
    a4, a5, a6 = st.columns(3)

    with a1:
        st.metric("MRT", _proximity_label(mrt_dist))
    with a2:
        st.metric("Bus", _proximity_label(bus_dist))
    with a3:
        st.metric("Schools", _proximity_label(school_dist))
    with a4:
        st.metric("Hawker", _proximity_label(hawker_dist))
    with a5:
        st.metric("Retail", _proximity_label(retail_dist))
    with a6:
        st.metric("Healthcare", _proximity_label(health_dist))

    # ── Map ─────────────────────────────────────────────────────────────────
    if lat is not None and lon is not None:
        st.markdown("### Map")
        st.markdown(_map_iframe(lat, lon), unsafe_allow_html=True)
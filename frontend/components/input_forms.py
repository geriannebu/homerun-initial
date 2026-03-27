import streamlit as st

from backend.schemas.inputs import UserInputs
from backend.utils.constants import FLAT_TYPES, TOWNS, SCHOOL_OPTIONS, AMENITY_LABELS
from backend.utils.scoring import RANKING_LABELS
from frontend.components.sections import render_section


def build_user_inputs() -> UserInputs:
    render_section(
        "1",
        "Tell us what you're looking for",
        "We'll generate price insights and recommendations.",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        budget = st.number_input(
            "Budget (S$)",
            min_value=150_000,
            max_value=2_000_000,
            value=650_000,
            step=10_000,
        )
        flat_type = st.selectbox("Flat type", FLAT_TYPES, index=2)

    with c2:
        floor_area_sqm = st.number_input(
            "Min floor area (sqm)",
            min_value=35.0,
            max_value=160.0,
            value=95.0,
            step=1.0,
            help="Listings below this area will be filtered out.",
        )
        lease_commence_year = st.number_input(
            "Lease commence year",
            min_value=1966,
            max_value=2025,
            value=2005,
            step=1,
        )

    with c3:
        town_choice = st.selectbox(
            "Preferred town",
            ["Recommendation mode"] + TOWNS,
        )

    # ── Ranking preference ──
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    render_section(
        "2",
        "Ranking preference",
        "How should listings be ordered when scores are close?",
    )

    ranking_options = list(RANKING_LABELS.keys())
    ranking_display = list(RANKING_LABELS.values())
    default_idx = ranking_options.index("balanced")

    ranking_col, _ = st.columns([2, 3])
    with ranking_col:
        ranking_choice_label = st.selectbox(
            "Ranking profile",
            options=ranking_display,
            index=default_idx,
            label_visibility="collapsed",
        )
    ranking_profile = ranking_options[ranking_display.index(ranking_choice_label)]
    st.session_state["ranking_profile"] = ranking_profile

    # ── Amenity priorities ──
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    render_section(
        "3a",
        "Amenity priorities",
        "Rate how important each amenity is to you (1 = low, 5 = high).",
    )

    a1, a2, a3 = st.columns(3)

    amenity_keys = list(AMENITY_LABELS.keys())
    default_weights = {
        "mrt":        5,
        "bus":        3,
        "healthcare": 2,
        "schools":    3,
        "hawker":     4,
        "retail":     3,
    }

    amenity_weights = {}
    school_scope = SCHOOL_OPTIONS[0]
    for idx, key in enumerate(amenity_keys):
        col = [a1, a2, a3][idx % 3]
        with col:
            amenity_weights[key] = st.slider(
                AMENITY_LABELS[key],
                min_value=1,
                max_value=5,
                value=default_weights.get(key, 3),
            )
            if key == "schools":
                school_scope = st.selectbox(
                    "School preference",
                    SCHOOL_OPTIONS,
                    index=0,
                )

    # ── Anchor locations ──
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    render_section(
        "3b",
        "Anchor locations",
        "Optional: add up to 2 postal codes for places you want to stay close to.",
    )

    p1, p2 = st.columns(2)
    with p1:
        postal_1 = st.text_input("Postal code 1", value="")
    with p2:
        postal_2 = st.text_input("Postal code 2", value="")

    landmark_postals = [p.strip() for p in [postal_1, postal_2] if p.strip()]

    return UserInputs(
        budget=int(budget),
        flat_type=flat_type,
        floor_area_sqm=float(floor_area_sqm),
        lease_commence_year=int(lease_commence_year),
        town=None if town_choice == "Recommendation mode" else town_choice,
        school_scope=school_scope,
        amenity_weights=amenity_weights,
        landmark_postals=landmark_postals,
    )


def render_user_profile(inputs: UserInputs):
    town_text    = inputs.town if inputs.town else "Recommendation mode"
    anchors_text = ", ".join(inputs.landmark_postals) if inputs.landmark_postals else "None"
    rank_profile = st.session_state.get("ranking_profile", "balanced")

    from backend.utils.scoring import RANKING_LABELS
    rank_label = RANKING_LABELS.get(rank_profile, rank_profile.capitalize())

    st.markdown("### Your search profile")
    st.markdown(
        f"""
        <div style="background:#ffffff;border:1px solid #e4e7ed;border-radius:18px;padding:1rem 1.1rem;box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(0,0,0,0.06);margin-bottom:0.85rem;"><div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px 20px;"><div><span style="font-family:'DM Sans',-apple-system,sans-serif;color:#9ca3af;font-size:0.78rem;">Budget</span><br><strong style="font-family:'DM Sans',-apple-system,sans-serif;color:#0f172a;">S${inputs.budget:,.0f}</strong></div><div><span style="font-family:'DM Sans',-apple-system,sans-serif;color:#9ca3af;font-size:0.78rem;">Flat type</span><br><strong style="font-family:'DM Sans',-apple-system,sans-serif;color:#0f172a;">{inputs.flat_type}</strong></div><div><span style="font-family:'DM Sans',-apple-system,sans-serif;color:#9ca3af;font-size:0.78rem;">Min floor area</span><br><strong style="font-family:'DM Sans',-apple-system,sans-serif;color:#0f172a;">{inputs.floor_area_sqm:.0f} sqm</strong></div><div><span style="font-family:'DM Sans',-apple-system,sans-serif;color:#9ca3af;font-size:0.78rem;">Lease year</span><br><strong style="font-family:'DM Sans',-apple-system,sans-serif;color:#0f172a;">{inputs.lease_commence_year}</strong></div><div><span style="font-family:'DM Sans',-apple-system,sans-serif;color:#9ca3af;font-size:0.78rem;">Town</span><br><strong style="font-family:'DM Sans',-apple-system,sans-serif;color:#0f172a;">{town_text}</strong></div><div><span style="font-family:'DM Sans',-apple-system,sans-serif;color:#9ca3af;font-size:0.78rem;">Ranking</span><br><strong style="font-family:'DM Sans',-apple-system,sans-serif;color:#0f172a;">{rank_label.split(' (')[0]}</strong></div><div><span style="font-family:'DM Sans',-apple-system,sans-serif;color:#9ca3af;font-size:0.78rem;">School scope</span><br><strong style="font-family:'DM Sans',-apple-system,sans-serif;color:#0f172a;">{inputs.school_scope}</strong></div><div><span style="font-family:'DM Sans',-apple-system,sans-serif;color:#9ca3af;font-size:0.78rem;">Anchors</span><br><strong style="font-family:'DM Sans',-apple-system,sans-serif;color:#0f172a;">{anchors_text}</strong></div></div></div>
        """,
        unsafe_allow_html=True,
    )

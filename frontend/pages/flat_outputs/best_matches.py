"""
frontend/pages/flat_outputs/best_matches.py

Tinder-style swipe deck:
  Right / ♥  → Like (shortlist)
  Left  / ✕  → Pass (skip)
  Up    / ⭐  → Super-save (highlight + shortlist)
  Click card → Detail overlay
"""

import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from backend.utils.formatters import fmt_sgd, valuation_tag_html
from backend.utils.constants import TOWN_COORDS
from frontend.state.session import get_active_session, record_swipe
from frontend.components.listing_detail import show_listing_detail

DEFAULT_COORD = (1.3521, 103.8198)

AMENITY_ICONS = {
    "mrt": "🚇",
    "bus": "🚌",
    "healthcare": "🏥",
    "schools": "🏫",
    "hawker": "🍜",
    "retail": "🛍️",
}

AMENITY_LABELS = {
    "mrt": "MRT",
    "bus": "Bus",
    "healthcare": "Health",
    "schools": "Schools",
    "hawker": "Hawker",
    "retail": "Shops",
}


def _map_url(town: str) -> str:
    lat, lon = TOWN_COORDS.get(town, DEFAULT_COORD)
    return (
        f"https://www.openstreetmap.org/export/embed.html"
        f"?bbox={lon-0.012},{lat-0.008},{lon+0.012},{lat+0.008}"
        f"&layer=mapnik&marker={lat},{lon}"
    )


def _val_color(label: str) -> str:
    if "Steal" in label:
        return "#059E87"
    if "Fair" in label:
        return "#2563eb"
    if "Slight" in label:
        return "#d97706"
    return "#dc2626"


def _why_match(row, inputs) -> str:
    """Generate a short 'why this matches you' blurb."""
    rank = getattr(inputs, "amenity_rank", [])
    top_amenities = rank[:2] if rank else []

    reasons = []

    diff = float(row.get("asking_vs_predicted_pct", 0))
    if diff <= -5:
        reasons.append("priced below model estimate")
    elif diff <= 3:
        reasons.append("fairly priced")

    for amen in top_amenities:
        score = float(row.get(f"{amen}_score", 0))
        if score >= 50:
            icon = AMENITY_ICONS.get(amen, "")
            label = {
                "mrt": "MRT access",
                "bus": "bus connectivity",
                "healthcare": "healthcare nearby",
                "schools": "schools nearby",
                "hawker": "hawker food nearby",
                "retail": "shopping nearby",
            }.get(amen, amen)
            reasons.append(f"{icon} good {label}")

    return "Matches on " + " · ".join(reasons) if reasons else "Good overall fit"


def _build_card_data(df: pd.DataFrame, inputs, unseen_ids: list) -> list:
    """Return card dicts, sorted by overall_value_score descending."""
    df_unseen = df[df["listing_id"].isin(unseen_ids)].copy()
    if "overall_value_score" in df_unseen.columns:
        df_unseen = df_unseen.sort_values("overall_value_score", ascending=False)

    cards = []
    for _, row in df_unseen.iterrows():
        diff = float(row.get("asking_vs_predicted_pct", 0))
        label = str(row.get("valuation_label", ""))
        town = str(row.get("town", ""))
        card = {
            "id": str(row.get("listing_id", "")),
            "town": town,
            "flat_type": str(row.get("flat_type", "")),
            "area": float(row.get("floor_area_sqm", 0)),
            "storey": str(row.get("storey_range", "")),
            "asking": int(row.get("asking_price", 0)),
            "predicted": int(row.get("predicted_price", 0)),
            "diff_pct": round(diff, 1),
            "label": label,
            "label_color": _val_color(label),
            "map_url": _map_url(town),
            "why": _why_match(row, inputs),
            "overall_value_score": float(row.get("overall_value_score", 0)),
        }
        # Add amenity scores for badge display
        for amen in AMENITY_ICONS:
            card[amen] = float(row.get(f"{amen}_score", 0))
        cards.append(card)
    return cards


def render_listing_tab(listings_df: pd.DataFrame):
    if listings_df is None or listings_df.empty:
        st.info("No listings available. Run a search first.")
        return

    session = get_active_session()
    if session is None:
        st.info("No active search session found.")
        return

    inputs = session["inputs"]

    # Ensure unseen_ids are populated after quiz
    if not session["unseen_ids"]:
        session["unseen_ids"] = list(listings_df["listing_id"])
        session["liked_ids"] = []
        session["passed_ids"] = []
        session["super_ids"] = []

    unseen_ids = session["unseen_ids"]
    liked_ids = session["liked_ids"]
    passed_ids = session["passed_ids"]

    # ── Deck exhausted ─────────────────────────────────────────────
    if not unseen_ids:
        _render_deck_done(session, listings_df)
        return

    # ── Build cards ────────────────────────────────────────────────
    cards = _build_card_data(listings_df, inputs, unseen_ids)
    cards_json = json.dumps(cards)
    session_id = session["session_id"]

    # ── Swipe deck iframe ─────────────────────────────────────────
    html = _build_swipe_html(cards_json, session["session_id"])
    components.html(html, height=720, scrolling=False)
    

    # ── Top card info ─────────────────────────────────────────────
    top_card = cards[0] if cards else None
    if top_card:
        # Match score bar
        score = top_card["overall_value_score"]
        color = "#059E87" if score >= 75 else "#d97706" if score >= 50 else "#FF4458"
        st.markdown(
            f"""
            <div style='margin:8px 0 10px;'>
                <div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;
                     letter-spacing:0.08em;color:#94a3b8;margin-bottom:5px;'>🎯 Match Score</div>
                <div style='display:flex;align-items:center;gap:10px;'>
                    <div style='flex:1;height:8px;border-radius:4px;background:#f1f5f9;overflow:hidden;'>
                        <div style='width:{score}%;height:100%;background:{color};border-radius:4px;
                             transition:width 0.4s;'></div>
                    </div>
                    <span style='font-weight:800;color:{color};font-size:0.88rem;min-width:2.8rem;'>{score:.0f}%</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Amenity badges
        badges = ""
        for amen, icon in AMENITY_ICONS.items():
            val = top_card.get(amen, 0)
            good = val >= 60
            border = "#059E87" if good else "#d97706"
            label = AMENITY_LABELS.get(amen, amen)
            badges += (
                f'<span style="display:inline-flex;align-items:center;gap:4px;'
                f'padding:4px 9px;border-radius:999px;border:1.5px solid {border};'
                f'font-size:0.72rem;font-weight:700;color:#334155;white-space:nowrap;">'
                f'{icon} {label}</span>'
            )
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px;">{badges}</div>',
            unsafe_allow_html=True,
        )

    # ── Swipe buttons ──────────────────────────────────────────────
    _render_swipe_controls(session_id, top_card["id"] if top_card else None)

    # View details button
    if top_card:
        _, detail_col, _ = st.columns([2, 1.5, 2])
        with detail_col:
            if st.button("View details", key=f"deck_detail_{top_card['id']}", use_container_width=True):
                show_listing_detail(top_card["id"])

    # ── Saved / passed counts ──────────────────────────────────────
    if liked_ids or passed_ids:
        total = len(listings_df)
        seen = len(liked_ids) + len(passed_ids)
        st.markdown(
            f"<p style='text-align:center;font-size:0.78rem;color:#9ca3af;margin-top:0.4rem;'>"
            f"{seen} of {total} seen · {len(liked_ids)} saved · {len(passed_ids)} passed</p>",
            unsafe_allow_html=True,
        )


# ───────────────────────── Swipe Controls ────────────────────────────
def _render_swipe_controls(session_id: str, top_id: str | None):
    if not top_id:
        return
    col1, col2, col3, col4, col5 = st.columns([1, 1, 0.6, 1, 1])
    with col2:
        if st.button("✕  Pass", key=f"pass_{top_id}", use_container_width=True):
            record_swipe(session_id, top_id, "left")
            st.rerun()
    with col3:
        st.markdown("<div style='height:38px'></div>", unsafe_allow_html=True)
    with col4:
        if st.button("♥  Save", key=f"save_{top_id}", type="primary", use_container_width=True):
            record_swipe(session_id, top_id, "right")
            st.rerun()
    _, mid, _ = st.columns([2, 1, 2])
    with mid:
        if st.button("⭐ Super", key=f"super_{top_id}", use_container_width=True):
            record_swipe(session_id, top_id, "up")
            st.rerun()


# ───────────────────────── Deck Done / Saved Preview ─────────────────────────
def _render_deck_done(session: dict, listings_df: pd.DataFrame):
    liked = session["liked_ids"]
    supers = session["super_ids"]
    passed = session["passed_ids"]

    st.markdown(
        f"""
        <div style="text-align:center;padding:2rem 1rem;">
            <div style="font-size:3rem;margin-bottom:0.8rem;">🎉</div>
            <h2 style="font-size:1.6rem;font-weight:800;color:#0f172a;margin-bottom:0.4rem;">
            You've seen them all</h2>
            <p style="font-size:0.88rem;color:#9ca3af;margin-bottom:1.6rem;">
            Head to the <strong>Saved</strong> tab to review your picks.</p>
            <div style="display:flex;gap:24px;justify-content:center;margin-bottom:1.8rem;">
                <div style="text-align:center;">
                    <div style="font-size:2rem;font-weight:800;color:#059E87;">{len(liked)}</div>
                    <div style="font-size:0.72rem;color:#9ca3af;font-weight:600;
                         text-transform:uppercase;letter-spacing:0.06em;">Saved</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:2rem;font-weight:800;color:#d97706;">{len(supers)}</div>
                    <div style="font-size:0.72rem;color:#9ca3af;font-weight:600;
                         text-transform:uppercase;letter-spacing:0.06em;">Super-saved</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:2rem;font-weight:800;color:#9ca3af;">{len(passed)}</div>
                    <div style="font-size:0.72rem;color:#9ca3af;font-weight:600;
                         text-transform:uppercase;letter-spacing:0.06em;">Passed</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Review saved →", type="primary", use_container_width=True):
            st.session_state.active_page = "Saved"
            st.rerun()
    with c2:
        if st.button("Restart deck", use_container_width=True):
            for s in st.session_state.search_sessions:
                if s["session_id"] == session["session_id"]:
                    s["unseen_ids"] = list(listings_df["listing_id"].values)
                    s["liked_ids"] = []
                    s["super_ids"] = []
                    s["passed_ids"] = []
            st.rerun()


# ───────────────────────── Swipe Deck HTML ────────────────────────────
def _build_swipe_html(cards_json: str, session_id: str) -> str:
    """
    Returns HTML string for a swipeable card deck.
    All JS template literals are preserved; Python does not try to evaluate offsetX/offsetY.
    """
    return """
    <html>
    <head>
    <style>
    body { font-family: 'Segoe UI', sans-serif; margin:0; padding:0; background:#f1f5f9;
           display:flex; justify-content:center; align-items:center; height:100vh; }
    #deck { width:360px; height:650px; position:relative; perspective:1000px; }
    .card { width:340px; height:460px; border-radius:16px; background:#fff;
            box-shadow:0 8px 20px rgba(0,0,0,0.25); position:absolute; top:20px; left:10px;
            display:flex; flex-direction:column; justify-content:space-between; padding:16px;
            overflow-wrap: break-word; transition: transform 0.3s ease, opacity 0.3s ease; cursor:grab; }
    .card h4 { margin:0 0 4px 0; font-size:1.1rem; font-weight:700; color:#1e293b; }
    .card p { margin:2px 0; font-size:0.8rem; color:#334155; }
    .val-label { font-weight:700; padding:2px 6px; border-radius:6px; font-size:0.75rem;
                 color:#fff; display:inline-block; margin-top:4px; }
    </style>
    </head>
    <body>
    <div id="deck"></div>
    <script>
    const cardsData = """ + cards_json + """;
    const deck = document.getElementById('deck');
    let topIndex = 0;

    function createCard(c) {
        const card = document.createElement('div');
        card.className = 'card';
        card.style.zIndex = cardsData.length - topIndex;
        const valColor = c.label_color || '#2563eb';
        card.innerHTML = `
            <h4>${c.town} · ${c.flat_type}</h4>
            <p style="font-weight:600; font-size:0.9rem;">${c.address || 'No address'}</p>
            <p>Area: ${c.area} sqm · Storey: ${c.storey}</p>
            <p>Asking: ${c.asking.toLocaleString()} SGD · Predicted: ${c.predicted.toLocaleString()} SGD</p>
            <p class="val-label" style="background:${valColor}">${c.label}</p>
            <p>${c.diff_pct >=0? '+' : ''}${c.diff_pct}% vs model</p>
            <p style="margin-top:8px; font-size:0.78rem; color:#475569;">${c.why}</p>
        `;
        return card;
    }

    function renderDeck() {
        deck.innerHTML = '';
        cardsData.slice(topIndex).forEach(c => {
            const card = createCard(c);
            deck.appendChild(card);
            makeSwipeable(card, c.id);
        });
    }

    function makeSwipeable(card, cardId) {
        let offsetX = 0;
        let offsetY = 0;
        let startX = 0;
        let startY = 0;
        let isDragging = false;

        card.addEventListener('pointerdown', e => {
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            card.setPointerCapture(e.pointerId);
        });

        card.addEventListener('pointermove', e => {
            if (!isDragging) return;
            offsetX = e.clientX - startX;
            offsetY = e.clientY - startY;
            card.style.transform = `translate(${offsetX}px, ${offsetY}px) rotate(${offsetX/15}deg)`;
        });

        card.addEventListener('pointerup', e => {
            isDragging = false;
            let direction = null;
            if (Math.abs(offsetX) > 120 || Math.abs(offsetY) > 150) {
                if (Math.abs(offsetY) > Math.abs(offsetX) && offsetY < -150) {
                    direction = 'up';       // super-swipe
                } else if (offsetX > 120) {
                    direction = 'right';    // save
                } else {
                    direction = 'left';     // pass
                }

                card.style.transition = 'transform 0.5s ease, opacity 0.5s ease';
                card.style.opacity = 0;
                card.style.transform = `translate(${offsetX*3}px, ${offsetY*3}px) rotate(${offsetX*10}deg)`;

                if (direction) {
                    const swipeEvent = {
                        session_id: '""" + session_id + """',
                        card_id: cardId,
                        direction: direction
                    };
                    window.parent.postMessage({type:'swipe', data: swipeEvent}, "*");
                }
                topIndex += 1;
                setTimeout(renderDeck, 300);
            } else {
                card.style.transition = 'transform 0.3s ease';
                card.style.transform = 'translate(0,0) rotate(0)';
            }
        });
    }

    renderDeck();
    </script>
    </body>
    </html>
    """
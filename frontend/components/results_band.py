"""results_band.py
Renders the unified results summary band that replaces the old scattered
results_band.py
Renders the unified results summary band.
All styles are INLINE — no dependency on external CSS class rules.
This is the correct pattern for Streamlit HTML components.
"""

from typing import Any, Dict

import streamlit as st

from backend.schemas.inputs import UserInputs
from backend.utils.formatters import fmt_sgd
from backend.utils.scoring import RANKING_LABELS, RANKING_ALPHA


# ── shared tokens ──────────────────────────────────────────────────────
_MINT        = "#059E87"
_MINT_DARK   = "#037a68"
_MINT_LIGHT  = "#e6f7f4"
_MINT_BORDER = "#a7e8dc"
_AMBER_SOFT  = "#fffaeb"
_AMBER_BDR   = "#fedf89"
_AMBER_TEXT  = "#92400e"
_RED_TEXT    = "#991b1b"
_TEXT1       = "#0f172a"
_TEXT2       = "#4b5563"
_TEXT3       = "#9ca3af"
_BG_SOFT     = "#f7f8fa"
_BG_WHITE    = "#ffffff"
_BORDER      = "#e4e7ed"
_FONT        = "'DM Sans', -apple-system, sans-serif"
_RADIUS      = "14px"
_RADIUS_LG   = "18px"
_SHADOW_SM   = "0 1px 2px rgba(0,0,0,0.05)"
_SHADOW_CARD = "0 2px 8px rgba(0,0,0,0.07)"


def _metric(label: str, value: str, sub: str, *, value_color: str = "#0f172a") -> str:
    return (
        f'<div style="background:{_BG_SOFT};border:1px solid {_BORDER};'
        f'border-radius:{_RADIUS};padding:10px 13px;">'
        f'<div style="font-family:{_FONT};font-size:0.67rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.06em;color:{_TEXT3};margin-bottom:4px;">{label}</div>'
        f'<div style="font-family:{_FONT};font-size:1.05rem;font-weight:800;'
        f'letter-spacing:-0.025em;color:{value_color};line-height:1.15;">{value}</div>'
        f'<div style="font-family:{_FONT};font-size:0.67rem;color:{_TEXT3};margin-top:2px;">{sub}</div>'
        f'</div>'
    )


def _badge(text: str, bg: str, color: str, border: str) -> str:
    return (
        f'<span style="display:inline-block;font-family:{_FONT};font-size:0.71rem;'
        f'font-weight:700;padding:3px 10px;border-radius:999px;background:{bg};'
        f'color:{color};border:1px solid {border};">{text}</span>'
    )


def render_results_band(inputs: UserInputs, bundle: Dict[str, Any]) -> None:
    pred          = bundle["predicted_price"]
    trans         = bundle["recent_median_transacted"]
    budget        = inputs.budget
    count         = bundle.get("viable_listing_count", 0)
    median        = bundle.get("median_asking_active", 0)
    mode_label    = bundle.get("mode_label", "Search mode")
    rank_profile  = bundle.get("ranking_profile", "balanced")
    rank_label    = RANKING_LABELS.get(rank_profile, rank_profile.capitalize())
    alpha         = RANKING_ALPHA.get(rank_profile, 0.5)
    amenity_pct   = round(alpha * 100)
    value_pct     = 100 - amenity_pct
    recent_period = bundle.get("recent_period", "last 6 months")

    reference     = median if median else pred
    headroom      = budget - reference
    headroom_text = (
        f"+{fmt_sgd(headroom)} headroom"
        if headroom >= 0
        else f"{fmt_sgd(abs(headroom))} over median"
    )
    budget_color  = _MINT_DARK if headroom >= 0 else _RED_TEXT

    filter_desc = (
        f"{inputs.flat_type} flats in {inputs.town}"
        if inputs.town
        else f"{inputs.flat_type} flats across recommended towns"
    )
    rank_short = rank_label.split(" (")[0].strip()
    why_text = (
        f"Filtered to {filter_desc} within your budget of {fmt_sgd(budget)}. "
        f"Ranked by {rank_short} ({amenity_pct}% amenity &middot; {value_pct}% value). "
        f"Top 5 listings shown below."
    )

    m_count  = _metric("Viable listings",    str(count),                  "passed all filters")
    m_pred   = _metric("Predicted fair value", fmt_sgd(pred),             "hedonic model estimate", value_color=_MINT_DARK)
    m_trans  = _metric("Recent benchmark",   fmt_sgd(trans),              f"median, {recent_period}")
    m_median = _metric("Median asking",      fmt_sgd(median) if median else "&mdash;", "active comparable listings")
    m_budget = _metric("Your budget",        fmt_sgd(budget),             headroom_text, value_color=budget_color)

    badge_mode = _badge(mode_label, _MINT_LIGHT, _MINT_DARK, _MINT_BORDER)
    badge_rank = _badge(rank_label, _AMBER_SOFT, _AMBER_TEXT, _AMBER_BDR)

    html = (
        f'<div style="background:{_BG_WHITE};border:1px solid {_BORDER};'
        f'border-radius:{_RADIUS_LG};padding:1.1rem 1.3rem 1rem;'
        f'box-shadow:{_SHADOW_CARD};margin-bottom:1rem;">'

        # header row
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'flex-wrap:wrap;gap:6px;margin-bottom:0.85rem;">'
        f'<span style="font-family:{_FONT};font-size:0.68rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.07em;color:{_TEXT3};">Results summary</span>'
        f'<div style="display:flex;gap:6px;flex-wrap:wrap;">{badge_mode}{badge_rank}</div>'
        f'</div>'

        # metrics grid
        f'<div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));'
        f'gap:8px;margin-bottom:0.8rem;">'
        f'{m_count}{m_pred}{m_trans}{m_median}{m_budget}'
        f'</div>'

        # why box
        f'<div style="background:{_MINT_LIGHT};border-left:3px solid {_MINT_BORDER};'
        f'border-radius:0 {_RADIUS} {_RADIUS} 0;padding:0.55rem 0.9rem;'
        f'font-family:{_FONT};font-size:0.8rem;color:{_MINT_DARK};line-height:1.55;">'
        f'{why_text}'
        f'</div>'

        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)

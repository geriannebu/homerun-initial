"""towns.py — area-level town insights, fully inline styles."""

from typing import Optional

import pandas as pd
import streamlit as st

from backend.schemas.inputs import UserInputs
from backend.utils.formatters import fmt_sgd

_MINT        = "#059E87"
_MINT_DARK   = "#037a68"
_MINT_LIGHT  = "#e6f7f4"
_MINT_BORDER = "#a7e8dc"
_GREEN_SOFT  = "#ecfdf3"
_GREEN_BDR   = "#abefc6"
_RED_SOFT    = "#fef3f2"
_RED_BDR     = "#fda29b"
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


def render_towns_tab(inputs: UserInputs, reco_df: Optional[pd.DataFrame]) -> None:
    if inputs.town:
        st.info(
            "Town insights are only available in **Recommendation mode**. "
            "Clear the town field in your search to see town-level guidance."
        )
        return

    if reco_df is None or reco_df.empty:
        st.info("No town recommendations available for this search.")
        return

    st.markdown(
        f'<div style="margin-bottom:0.9rem;">'
        f'<span style="font-family:{_FONT};font-size:1rem;font-weight:800;color:{_TEXT1};">'
        f'Recommended towns</span><br>'
        f'<span style="font-family:{_FONT};font-size:0.82rem;color:{_TEXT3};">'
        f'Ranked by match score across affordability, amenities, and anchor proximity. '
        f'Select a town below to focus your search there.</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    medals = ["🥇", "🥈", "🥉", "4th", "5th"]

    for rank, (_, row) in enumerate(reco_df.iterrows()):
        medal          = medals[rank] if rank < len(medals) else str(rank + 1)
        within_budget  = bool(row.get("within_budget", False))
        match_score    = float(row.get("match_score", 0))
        est_price      = row.get("estimated_price", 0)
        why_text       = row.get("why_it_matches", "—")
        bar_pct        = min(100, max(0, match_score))

        budget_bg     = _GREEN_SOFT if within_budget else _RED_SOFT
        budget_color  = "#166534"   if within_budget else _RED_TEXT
        budget_border = _GREEN_BDR  if within_budget else _RED_BDR
        budget_label  = "✓ Within budget" if within_budget else "↑ Above budget"

        budget_tag = (
            f'<span style="display:inline-block;font-family:{_FONT};font-size:0.72rem;'
            f'font-weight:700;padding:2px 8px;border-radius:999px;'
            f'background:{budget_bg};border:1px solid {budget_border};'
            f'color:{budget_color};">{budget_label}</span>'
        )

        html = (
            f'<div style="background:{_BG_WHITE};border:1px solid {_BORDER};'
            f'border-radius:{_RADIUS_LG};padding:1rem 1.1rem;'
            f'margin-bottom:0.7rem;box-shadow:{_SHADOW_SM};">'

            # header
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:flex-start;gap:1rem;">'
            f'<div>'
            f'<div style="font-family:{_FONT};font-size:0.97rem;font-weight:700;color:{_TEXT1};">'
            f'{medal} &nbsp; {row["town"]}</div>'
            f'<div style="font-family:{_FONT};font-size:0.8rem;color:{_TEXT3};'
            f'margin-top:3px;line-height:1.4;">{why_text}</div>'
            f'</div>'
            f'<div style="text-align:right;min-width:70px;flex-shrink:0;">'
            f'<div style="font-family:{_FONT};font-size:1.25rem;font-weight:800;'
            f'color:{_MINT_DARK};letter-spacing:-0.02em;">{match_score:.1f}</div>'
            f'<div style="font-family:{_FONT};font-size:0.68rem;color:{_TEXT3};">/ 100</div>'
            f'</div>'
            f'</div>'

            # score bar
            f'<div style="margin:0.5rem 0 0.4rem;">'
            f'<div style="height:3px;background:{_BORDER};border-radius:2px;overflow:hidden;">'
            f'<div style="height:3px;width:{bar_pct:.0f}%;background:{_MINT};border-radius:2px;"></div>'
            f'</div>'
            f'</div>'

            # footer
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'flex-wrap:wrap;gap:6px;">'
            f'<span style="font-family:{_FONT};font-size:0.9rem;font-weight:800;color:{_TEXT1};">'
            f'Est. {fmt_sgd(est_price)}</span>'
            f'{budget_tag}'
            f'</div>'

            f'</div>'
        )

        st.markdown(html, unsafe_allow_html=True)

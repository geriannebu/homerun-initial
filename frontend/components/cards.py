from typing import Any, Dict

import streamlit as st

from backend.schemas.inputs import UserInputs
from backend.utils.formatters import fmt_sgd, valuation_tag_html
from backend.utils.scoring import compute_listing_scores


def render_value_cards(bundle: Dict[str, Any], budget: int):
    pred = bundle["predicted_price"]
    trans = bundle["recent_median_transacted"]
    low = bundle.get("confidence_low", round(pred * 0.96))
    high = bundle.get("confidence_high", round(pred * 1.04))
    gap_pct = ((budget - pred) / pred) * 100 if pred else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Predicted fair value", fmt_sgd(pred))
    with c2:
        st.metric("Recent transacted median", fmt_sgd(trans))
    with c3:
        st.metric("Confidence band", f"{fmt_sgd(low)} – {fmt_sgd(high)}")
    with c4:
        st.metric("Budget vs fair value", f"{gap_pct:+.1f}%")


def render_budget_banner(bundle: Dict[str,Any], budget: int):
    pred = bundle["predicted_price"]
    gap  = (budget-pred)/pred
    if gap >= 0.05:
        css, icon = "nw-budget-ok",   "✓"
        msg = f"{icon} Your budget is {gap*100:.1f}% above the predicted fair value — you have good room to negotiate."
    elif gap >= -0.05:
        css, icon = "nw-budget-warn", "△"
        msg = f"{icon} Your budget is close to the predicted fair value ({gap*100:+.1f}%). Look for steals."
    else:
        css, icon = "nw-budget-over", "↓"
        msg = f"{icon} Your budget is {abs(gap)*100:.1f}% below the predicted fair value. Recommendation mode may surface better-value options."
    st.markdown(f'<div class="{css}">{msg}</div>', unsafe_allow_html=True)


def render_nestwise_pick(inputs: UserInputs, bundle: Dict[str,Any]):
    if bundle["listings_df"].empty: return
    ranked = compute_listing_scores(bundle["listings_df"], inputs.budget, inputs.amenity_weights)
    top = ranked.sort_values("overall_value_score", ascending=False).iloc[0]
    tag = valuation_tag_html(top["valuation_label"])
    st.markdown(f"""
    <div class="nw-pick">
        <div class="nw-pick-icon">🏆</div>
        <div style="flex:1">
            <div class="nw-pick-label">NestWise pick right now</div>
            <div class="nw-pick-value">{top['listing_id']} &nbsp;·&nbsp; {top['town']}</div>
            <div class="nw-pick-sub">
                Asking {fmt_sgd(top['asking_price'])} &nbsp;·&nbsp; {tag} &nbsp;·&nbsp;
                Overall score <strong>{top['overall_value_score']:.1f}</strong>/100
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)



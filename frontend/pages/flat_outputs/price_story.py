from typing import Any, Dict

import pandas as pd
import streamlit as st

from backend.utils.formatters import fmt_sgd


def render_price_story_tab(bundle: Dict[str,Any]):
    pred   = bundle["predicted_price"]
    trans  = bundle["recent_median_transacted"]
    c_low  = bundle["confidence_low"]
    c_high = bundle["confidence_high"]

    st.markdown("#### Price story")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fair value",    fmt_sgd(pred),   f"Model estimate")
    col2.metric("Transacted",    fmt_sgd(trans),  f"Median {bundle['recent_period']}")
    col3.metric("Band low",      fmt_sgd(c_low),  "−4% confidence")
    col4.metric("Band high",     fmt_sgd(c_high), "+4% confidence")

    compare_df = pd.DataFrame({
        "Metric": ["Predicted fair value","Recent transacted","Conf. low","Conf. high"],
        "Value":  [pred, trans, c_low, c_high],
    })
    st.bar_chart(compare_df.set_index("Metric"), color="#059E87")

    gap = (trans-pred)/pred*100
    direction = "above" if gap > 0 else "below"
    st.markdown(f"""
    <div class="nw-method">
        The model estimates this flat type at <strong>{fmt_sgd(pred)}</strong>.
        Recent transactions show a median of <strong>{fmt_sgd(trans)}</strong> —
        <strong>{abs(gap):.1f}% {direction}</strong> the predicted value.
        The confidence band spans <strong>{fmt_sgd(c_low)}</strong> to <strong>{fmt_sgd(c_high)}</strong>.
    </div>
    """, unsafe_allow_html=True)
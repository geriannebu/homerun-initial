import streamlit as st

_FONT  = "'DM Sans', -apple-system, sans-serif"
_TEXT1 = "#0f172a"
_TEXT2 = "#4b5563"
_TEXT3 = "#9ca3af"
_MINT  = "#059E87"


def render_methodology():
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    with st.expander("How NestWise works"):
        st.markdown(
            f'<div style="font-family:{_FONT};font-size:0.87rem;color:{_TEXT2};line-height:1.75;">'

            f'<div style="font-size:0.67rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.07em;color:{_TEXT3};margin-bottom:6px;">Three pricing anchors</div>'
            f'<b style="color:{_TEXT1};">1. Predicted fair value</b> — hedonic model estimate based on flat type, size, lease, location, and amenities.<br>'
            f'<b style="color:{_TEXT1};">2. Recent transacted median</b> — what similar flats have actually sold for in the same town and period.<br>'
            f'<b style="color:{_TEXT1};">3. Current asking prices</b> — live listings from the backend, compared against the model estimate.<br><br>'

            f'<div style="font-size:0.67rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.07em;color:{_TEXT3};margin-bottom:6px;">Valuation labels</div>'
            f'🔥 <b style="color:{_TEXT1};">Steal</b> &nbsp;·&nbsp; −5% or below &emsp;'
            f'✅ <b style="color:{_TEXT1};">Fair</b> &nbsp;·&nbsp; within ±3%<br>'
            f'⚠️ <b style="color:{_TEXT1};">Slightly overpriced</b> &nbsp;·&nbsp; +3–10% &emsp;'
            f'🚩 <b style="color:{_TEXT1};">Overpriced</b> &nbsp;·&nbsp; above +10%<br><br>'

            f'<div style="font-size:0.67rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.07em;color:{_TEXT3};margin-bottom:6px;">Town recommendation</div>'
            f'Leave town blank and the engine scores every town on affordability, amenity fit, and proximity to your anchor locations — then surfaces the top 5.'

            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

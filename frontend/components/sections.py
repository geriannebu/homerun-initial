import streamlit as st


def render_section(step: str, title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="nw-section">
            <div class="nw-section-step">Step {step}</div>
            <div class="nw-section-title">{title}</div>
            <div class="nw-section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
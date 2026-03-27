import streamlit as st

from frontend.pages.flat_outputs.best_matches import render_listing_tab
from frontend.pages.flat_outputs.towns import render_towns_tab
from frontend.pages.flat_outputs.map_view import render_map_tab
from frontend.pages.flat_outputs.scenario_testing import render_scenario_testing_output


def render_flat_finder_page(inputs, bundle, map_bundle):
    if inputs.town:
        subtab1, subtab2, subtab3 = st.tabs([
            "🏘️ Best matches",
            "🔭 What-if explorer",
            "📍 Map",
        ])

        with subtab1:
            render_listing_tab(inputs, bundle)

        with subtab2:
            render_scenario_testing_output(inputs)

        with subtab3:
            render_map_tab(inputs, map_bundle)

    else:
        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "🏘️ Best matches",
            "🏙️ Town insights",
            "🔭 What-if explorer",
            "📍 Map",
        ])

        with subtab1:
            render_listing_tab(inputs, bundle)

        with subtab2:
            render_towns_tab(inputs, bundle["recommendations_df"])

        with subtab3:
            render_scenario_testing_output(inputs)

        with subtab4:
            render_map_tab(inputs, map_bundle)

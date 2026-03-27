import streamlit as st
from backend.services.predictor_service import get_prediction_bundle
from backend.schemas.inputs import UserInputs

def render_scenario_testing_output(user_inputs: UserInputs, bundle):
    st.markdown("#### Scenario Testing")
    st.caption("Test a what-if flat profile separately from the real listing results to see how the estimated fair value changes.")

    # --- Original prediction for reference ---
    original_bundle = get_prediction_bundle(user_inputs)
    original_price = original_bundle["predicted_price"]

    # --- Flat type ---
    flat_type_options = ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"]
    adjusted_flat_type = st.selectbox(
        "Test a different flat type",
        options=flat_type_options,
        index=flat_type_options.index(user_inputs.flat_type)
    )

    # --- Floor area using + / - stepper ---
    adjusted_area = st.number_input(
        "Floor area for testing (sqm)",
        min_value=10.0,
        max_value=200.0,
        value=user_inputs.floor_area_sqm,
        step=1.0,
        format="%.1f"
    )

    # --- Lease commence year using + / - stepper ---
    adjusted_lease_year = st.number_input(
        "Lease commence year for testing",
        min_value=1950,
        max_value=2026,
        value=user_inputs.lease_commence_year,
        step=1
    )

    # --- Amenity scoring ---
    adjusted_amenities = {}
    st.markdown("**Adjust amenity priorities (1-5)**")
    amenity_names = list(user_inputs.amenity_weights.keys())
    num_cols = 3  # sliders per row
    for i in range(0, len(amenity_names), num_cols):
        cols = st.columns(num_cols)
        for j, amenity in enumerate(amenity_names[i:i+num_cols]):
            adjusted_score = cols[j].slider(
                f"{amenity} priority",
                min_value=1,
                max_value=5,
                value=int(user_inputs.amenity_weights[amenity]),
                step=1
            )
            adjusted_amenities[amenity] = adjusted_score

   

    # --- Create scenario input object ---
    scenario_inputs = UserInputs(
        budget=user_inputs.budget,
        flat_type=adjusted_flat_type,
        floor_area_sqm=adjusted_area,
        lease_commence_year=adjusted_lease_year,
        town=user_inputs.town,
        school_scope=user_inputs.school_scope,
        amenity_weights=adjusted_amenities,
        landmark_postals=user_inputs.landmark_postals
    )


    # --- Compute scenario predicted price ---
    bundle = get_prediction_bundle(scenario_inputs)
    scenario_price = bundle["predicted_price"]

    price_diff = scenario_price - original_price
    percent_diff = (price_diff / original_price) * 100 if original_price != 0 else 0

    # --- Display comparison cards ---
    st.markdown("### 💰 Price Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Original Estimated Price",
            f"${original_price:,.0f}"
        )

    with col2:
        st.metric(
            "Scenario Estimated Price",
            f"${scenario_price:,.0f}",
            delta=f"{price_diff:,.0f} ({percent_diff:.1f}%)"
        )

    with col3:
        # --- Budget comparison ---
        budget_diff = scenario_price - user_inputs.budget

        if budget_diff > 0:
            # ❌ Over budget → RED
            col3.metric(
                "Vs Your Budget",
                f"${scenario_price:,.0f}",
                delta=f"+${budget_diff:,.0f} over budget",
                delta_color="inverse"   # makes positive = red
            )
        else:
            # ✅ Under budget → GREEN
            col3.metric(
                "Vs Your Budget",
                f"${scenario_price:,.0f}",
                delta=f"-${abs(budget_diff):,.0f} under budget",
                delta_color="normal"   # makes negative = green
            )

    # ----------------------------
    # Insights
    # ----------------------------
    st.markdown("### 🔄 Scenario Changes")

    changes = []

    # Track changes
    if adjusted_flat_type != user_inputs.flat_type:
        changes.append(f"Flat type: {user_inputs.flat_type} → {adjusted_flat_type}")

    if adjusted_area != user_inputs.floor_area_sqm:
        changes.append(f"Floor area: {user_inputs.floor_area_sqm} → {adjusted_area} sqm")

    if adjusted_lease_year != user_inputs.lease_commence_year:
        changes.append(f"Lease year: {user_inputs.lease_commence_year} → {adjusted_lease_year}")

    for amenity, new_score in adjusted_amenities.items():
        old_score = user_inputs.amenity_weights[amenity]
        if new_score != old_score:
            changes.append(f"{amenity}: priority {old_score} → {new_score}")

    # Display changes
    if changes:
        for change in changes:
            st.write(f"• {change}")
    else:
        st.write("No changes made.")

    # ----------------------------
    # Price impact summary
    # ----------------------------
    st.markdown("### 💡 Impact on Price")

    if price_diff > 0:
        st.warning(f"Price increased by **${price_diff:,.0f} ({percent_diff:.1f}%)**.")
    else:
        st.success(f"Price decreased by **${abs(price_diff):,.0f} ({abs(percent_diff):.1f}%)**.")

    # Budget insight
    if scenario_price > user_inputs.budget:
        st.error("⚠️ This scenario exceeds your budget.")
    else:
        st.success("✅ This scenario is within your budget.")
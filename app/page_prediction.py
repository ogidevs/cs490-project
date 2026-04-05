import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import load_and_clean_data
from src.predict import predict_value


def render_prediction_page(active_dataset_filename, active_data_path):
    """Renders a clean, user-friendly prediction dashboard with market comparison visualizations."""
    st.header("Property Valuation Dashboard")
    st.write(
        "Input property details to generate a valuation and see how it compares to the local market."
    )

    if not active_dataset_filename:
        st.warning("Please select a dataset from the sidebar to make predictions.")
        return

    # Load dataset to populate accurate city/municipality dropdowns
    df = load_and_clean_data(active_data_path)

    # --- UI: Inputs ---
    col1, col2 = st.columns(2)

    with col1:
        cities_list = sorted(df["City"].dropna().unique().tolist())
        input_city = st.selectbox("City", cities_list)

        # Cascade logic: Only show municipalities that exist in the chosen city
        muni_filtered = (
            df[df["City"] == input_city]["Municipality"].dropna().unique().tolist()
        )
        muni_list = sorted(muni_filtered) if muni_filtered else ["Unknown"]
        input_muni = st.selectbox("Municipality", muni_list)

    with col2:
        input_area = st.number_input(
            "Property Area (m²)", min_value=10.0, max_value=1000.0, value=50.0, step=1.0
        )

        # Clean selectbox instead of an awkward slider for rooms
        room_options = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
        input_rooms = st.selectbox(
            "Number of Rooms", room_options, index=3
        )  # Defaults to 2.0

    st.write("---")

    # User selects the model they trained
    prediction_type = st.radio(
        "Model Target Trained On:",
        ["Total Price", "Price per Square Meter"],
        horizontal=True,
    )
    target_suffix = "price" if prediction_type == "Total Price" else "price_per_m2"

    if st.button("Calculate Valuation", type="primary"):
        # We only pass the clean features we defined in config.py
        input_data = {
            "Area": input_area,
            "Rooms_Numeric": input_rooms,
            "City": input_city,
            "Municipality": input_muni,
        }

        result = predict_value(input_data, active_dataset_filename, target_suffix)

        if isinstance(result, str):
            st.error(f"Prediction Failed: {result}")
            st.info(
                "Did you remember to retrain your model in the 'Model Training' tab after updating features?"
            )
            return

        # Standardize outputs so we can show BOTH total and per m2
        if target_suffix == "price":
            total_price = result
            price_per_m2 = result / input_area
        else:
            price_per_m2 = result
            total_price = result * input_area

        # --- UI: Result Metrics ---
        st.subheader("Valuation Results")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Estimated Total Price", f"€ {total_price:,.0f}")
        res_col2.metric("Estimated Price per m²", f"€ {price_per_m2:,.0f} /m²")

        # Calculate market median for context
        df_local = df[(df["City"] == input_city) & (df["Municipality"] == input_muni)]
        market_median = (
            df_local["Price_per_Unit_EUR"].median() if not df_local.empty else 0
        )

        if market_median > 0:
            diff = ((price_per_m2 / market_median) - 1) * 100
            res_col3.metric(
                "Market Median (m²)",
                f"€ {market_median:,.0f} /m²",
                f"{diff:+.1f}% vs Median",
                delta_color="inverse",
            )

        # --- UI: Market Context Graph ---
        st.write("---")
        st.subheader(f"Market Distribution in {input_muni}")

        if len(df_local) > 5:
            # We use Matplotlib/Seaborn to plot the local market and drop a line exactly where our prediction sits
            fig, ax = plt.subplots(figsize=(10, 5))

            sns.histplot(
                df_local["Price_per_Unit_EUR"],
                kde=True,
                color="lightgray",
                ax=ax,
                edgecolor="white",
            )

            # Plot Prediction Line
            ax.axvline(
                price_per_m2,
                color="blue",
                linestyle="-",
                linewidth=2.5,
                label=f"Your Property: € {price_per_m2:,.0f} /m²",
            )

            # Plot Market Average Line
            ax.axvline(
                market_median,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Market Median: € {market_median:,.0f} /m²",
            )

            ax.set_xlabel("Price per m² (€)")
            ax.set_ylabel("Number of Properties")
            ax.legend()

            st.pyplot(fig)
        else:
            st.info(
                f"There are not enough properties currently listed in {input_muni} to generate a reliable market distribution graph."
            )

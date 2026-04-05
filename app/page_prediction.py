import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
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

    dataset_base = active_dataset_filename.replace(".csv", "")
    dataset_property_type = "flat"
    meta_path = os.path.join("data", "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        dataset_property_type = metadata.get(dataset_base, {}).get("property_type", "flat")

    subtype_defaults = {
        "flat": "Stan",
        "house": "Kuća",
        "land": "Zemljište",
        "garage": "Garaža",
    }

    # Load dataset to populate accurate city/municipality dropdowns
    df = load_and_clean_data(active_data_path)

    def _scope_slice(dataframe, city, municipality, neighborhood):
        scoped = dataframe[dataframe["City"] == city]
        if municipality != "All Municipalities":
            scoped = scoped[scoped["Municipality"] == municipality]
        if neighborhood != "All Neighborhoods":
            scoped = scoped[scoped["Neighborhood"] == neighborhood]
        return scoped

    # --- UI: Inputs ---
    col1, col2 = st.columns(2)

    with col1:
        cities_list = sorted(df["City"].dropna().unique().tolist())
        input_city = st.selectbox("City", cities_list if cities_list else ["Unknown"])

        # Cascade logic: allow city-wide predictions or narrower location scope.
        muni_filtered = df[df["City"] == input_city]["Municipality"].dropna().unique().tolist()
        muni_list = ["All Municipalities"] + sorted(muni_filtered) if muni_filtered else ["All Municipalities"]
        input_muni = st.selectbox("Municipality", muni_list)

        if input_muni == "All Municipalities":
            neighborhood_filtered = df[df["City"] == input_city]["Neighborhood"].dropna().value_counts().index.tolist()
        else:
            neighborhood_filtered = df[
                (df["City"] == input_city) & (df["Municipality"] == input_muni)
            ]["Neighborhood"].dropna().value_counts().index.tolist()

        neighborhood_list = ["All Neighborhoods"] + neighborhood_filtered[:40] if neighborhood_filtered else ["All Neighborhoods"]
        input_neighborhood = st.selectbox("Neighborhood", neighborhood_list)

        advertiser_options = sorted(df["Advertiser_Type"].dropna().unique().tolist())
        input_advertiser = st.selectbox(
            "Advertiser Type", advertiser_options if advertiser_options else ["Unknown"]
        )

    with col2:
        input_area = st.number_input(
            "Property Area (m²)", min_value=1.0, max_value=50000.0, value=50.0, step=1.0
        )

        if dataset_property_type in ["flat", "house"]:
            input_rooms = st.number_input(
                "Number of Rooms",
                min_value=0.5,
                max_value=20.0,
                value=2.0,
                step=0.5,
            )
        else:
            input_rooms = None

        if dataset_property_type == "flat":
            input_current_floor = st.number_input(
                "Current Floor",
                min_value=-2,
                max_value=100,
                value=2,
                step=1,
            )

            input_total_floors = st.number_input(
                "Total Floors",
                min_value=1,
                max_value=100,
                value=5,
                step=1,
            )
        else:
            input_current_floor = None
            input_total_floors = None

        input_photo_count = st.number_input(
            "Photo Count",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
        )

    scope_label = input_muni if input_muni != "All Municipalities" else input_city
    if input_neighborhood != "All Neighborhoods":
        scope_label = f"{scope_label} / {input_neighborhood}"

    df_local = _scope_slice(df, input_city, input_muni, input_neighborhood)
    st.info(f"Selected scope: {scope_label} | Total properties: {len(df_local):,}")

    st.write("---")

    # User selects the model they trained
    prediction_type = st.radio(
        "Model Target Trained On:",
        ["Total Price", "Price per Square Meter"],
        horizontal=True,
    )
    target_suffix = "price" if prediction_type == "Total Price" else "price_per_m2"

    if st.button("Calculate Valuation", type="primary"):
        # We pass raw listing signals plus the engineered-friendly numeric fields.
        input_data = {
            "City": input_city,
            "Municipality": input_muni,
            "Neighborhood": input_neighborhood,
            "Property_Type": dataset_property_type,
            "Property_Subtype": subtype_defaults.get(dataset_property_type, "Unknown"),
            "Area": input_area,
            "Photo_Count": input_photo_count,
            "Advertiser_Type": input_advertiser,
        }

        if input_rooms is not None:
            input_data["Rooms_Numeric"] = input_rooms
        if input_current_floor is not None:
            input_data["Current_Floor_Num"] = input_current_floor
        if input_total_floors is not None:
            input_data["Total_Floors_Num"] = input_total_floors

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
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.metric("Estimated Total Price", f"€ {total_price:,.0f}")
        res_col2.metric("Estimated Price per m²", f"€ {price_per_m2:,.0f} /m²")
        res_col3.metric("Properties in Scope", f"{len(df_local):,}")

        # Calculate market median for context
        market_median = (
            df_local["Price_per_Unit_EUR"].median() if not df_local.empty else 0
        )

        if market_median > 0:
            diff = ((price_per_m2 / market_median) - 1) * 100
            res_col4.metric(
                "Market Median (m²)",
                f"€ {market_median:,.0f} /m²",
                f"{diff:+.1f}% vs Median",
                delta_color="inverse",
            )
        else:
            res_col4.metric("Market Median (m²)", "N/A")

        # --- UI: Market Context Graph ---
        st.write("---")
        st.subheader(f"Market Distribution in {scope_label}")

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
                f"There are not enough properties currently listed in {scope_label} to generate a reliable market distribution graph."
            )

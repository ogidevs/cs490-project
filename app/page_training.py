import streamlit as st
import pandas as pd
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import MODEL_CLASSES
from src.train_model import train_and_evaluate
from app.ui_utils import render_dynamic_hyperparameters


def render_training_page(active_dataset_filename):
    """Renders the dynamic hyperparameter UI and triggers the backend training engine."""
    st.header("Train Machine Learning Models")

    if active_dataset_filename:
        # Display dataset metadata info (property type, cities, etc.)
        dataset_base = active_dataset_filename.replace(".csv", "")
        meta_path = os.path.join("data", "metadata.json")

        dataset_property_type = "flat"
        dataset_cities = []
        dataset_records = 0

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            dataset_info = metadata.get(dataset_base, {})
            dataset_property_type = dataset_info.get("property_type", "flat")
            dataset_cities = dataset_info.get("cities", [])
            dataset_records = dataset_info.get("records", 0)

        # Display info banner
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Property Type", dataset_property_type.capitalize())
        with col_info2:
            st.metric("Cities", len(dataset_cities))
        with col_info3:
            st.metric("Records", f"{dataset_records:,}")

        col1, col2 = st.columns(2)
        with col1:
            target_var = st.selectbox(
                "Target Variable to Predict:", ["Total_Price_EUR", "Price_per_Unit_EUR"]
            )
            selected_models = st.multiselect(
                "Select Models to Train:",
                list(MODEL_CLASSES.keys()),
                default=["Random Forest", "XGBoost", "Gradient Boosting"],
            )

        st.write("---")
        st.subheader("⚙️ Configure Hyperparameters")

        model_params = {}
        if selected_models:
            tabs = st.tabs(selected_models)
            for i, model_name in enumerate(selected_models):
                with tabs[i]:
                    st.write(f"Settings for **{model_name}**")
                    model_params[model_name] = render_dynamic_hyperparameters(
                        model_name
                    )

            st.write("---")
            if st.button("Train Selected Models"):
                with st.spinner("Training models..."):
                    try:
                        results, best_score = train_and_evaluate(
                            active_dataset_filename,
                            selected_models,
                            target_var,
                            model_params,
                        )
                        st.success(
                            f"Training Complete! Best R-Squared Score: {best_score:.4f}"
                        )

                        st.write("### Model Performance")

                        # 1. Transpose the results dictionary into a Pandas DataFrame
                        res_df = pd.DataFrame(results).T.reset_index()
                        res_df.columns = ["Model", "R2", "MAE"]

                        # 2. Sort the values so the best model is on top
                        res_df = res_df.sort_values(by="R2", ascending=False)

                        # 3. Format the numbers manually to look clean
                        res_df["R2"] = res_df["R2"].astype(float).map("{:.4f}".format)
                        res_df["MAE"] = (
                            res_df["MAE"].astype(float).map("{:,.2f}".format)
                        )

                        html_table = res_df.to_html(index=False)
                        st.markdown(html_table, unsafe_allow_html=True)

                        # Show property type used for training
                        st.info(
                            f"✓ Models trained using **{dataset_property_type}**-specific features from {len(dataset_cities)} cities"
                        )

                    except Exception as e:
                        st.error(f"Training error: {e}")
        else:
            st.info("Select at least one model to train.")
    else:
        st.error("No dataset available. Please scrape data first.")

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model_config import MODEL_CLASSES
from src.train_model import train_and_evaluate
from app.ui_utils import render_dynamic_hyperparameters


def render_training_page(active_dataset_filename):
    """Renders the dynamic hyperparameter UI and triggers the backend training engine."""
    st.header("Train Machine Learning Models")

    if active_dataset_filename:
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

                    except Exception as e:
                        st.error(f"Training error: {e}")
        else:
            st.info("Select at least one model to train.")
    else:
        st.error("No dataset available. Please scrape data first.")

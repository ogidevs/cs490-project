import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import load_and_clean_data
from src.plotting import (
    plot_distributions,
    plot_relationships,
    plot_correlation_matrix,
    plot_advanced_features,
)


def render_eda_page(active_dataset_filename, active_data_path):
    """Renders all visualizations and data distribution profiles based on the active dataset."""
    st.header("Exploratory Data Analysis")
    if active_dataset_filename and os.path.exists(active_data_path):
        df_clean = load_and_clean_data(active_data_path)

        st.subheader("Feature Distributions")
        st.pyplot(plot_distributions(df_clean))
        st.subheader("Price vs. Area")
        st.pyplot(plot_relationships(df_clean))
        st.subheader("Rooms & Municipalities")
        st.pyplot(plot_advanced_features(df_clean))
        st.subheader("Feature Correlation")
        st.pyplot(plot_correlation_matrix(df_clean))
    else:
        st.warning("Please generate and select a dataset first.")

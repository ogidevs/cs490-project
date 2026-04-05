import streamlit as st
import os
import sys
import json

# Ensure global imports route perfectly from parent folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ui_utils import get_available_datasets
from page_scraping import render_scraping_page
from page_eda import render_eda_page
from page_training import render_training_page
from page_prediction import render_prediction_page

# Set top-level page configs globally
st.set_page_config(page_title="Real Estate ML System", layout="wide")
st.title("Real Estate Price Prediction System")

# Master Sidebar Navigation State
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:",["1. Data & Scraping", "2. EDA (Analysis)", "3. Model Training", "4. Prediction"])

# Manage the Shared Dataset State App-wide
dataset_map = get_available_datasets()
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Selection")

if dataset_map:
    selected_display = st.sidebar.selectbox("Active Dataset:", list(dataset_map.keys()))
    active_dataset_filename = dataset_map[selected_display]
    active_data_path = os.path.join('data', active_dataset_filename)
    
    # --- NEW: DISPLAY DATASET AND MODEL INFO IN SIDEBAR ---
    file_id = active_dataset_filename.replace('.csv', '')
    meta_path = os.path.join('data', 'metadata.json')
    
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        dataset_info = meta.get(file_id, {})
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Dataset Details")
        st.sidebar.markdown(f"**Type:** {dataset_info.get('property_type', 'N/A').capitalize()}")
        st.sidebar.markdown(f"**Cities:** {', '.join(dataset_info.get('cities',[]))}")
        st.sidebar.markdown(f"**Records:** {dataset_info.get('records', 0)}")
        st.sidebar.markdown(f"**Date:** {dataset_info.get('scraped_at', 'N/A').split()[0]}")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Trained Models")
        models_info = dataset_info.get('models', {})
        
        if models_info:
            for target, info in models_info.items():
                target_name = "Total Price" if target == "price" else "Price / m²"
                st.sidebar.success(f"**Target:** {target_name}\n\n**Best Model:** {info['best_model']}\n\n**R² Score:** {info['r2']:.4f}")
        else:
            st.sidebar.warning("No models trained on this dataset yet.")
            
else:
    active_dataset_filename = None
    active_data_path = None
    st.sidebar.info("No datasets available. Scrape data to begin.")

# Component Router Mechanism
if page == "1. Data & Scraping":
    render_scraping_page(active_dataset_filename, active_data_path)
    
elif page == "2. EDA (Analysis)":
    render_eda_page(active_dataset_filename, active_data_path)
    
elif page == "3. Model Training":
    render_training_page(active_dataset_filename)
    
elif page == "4. Prediction":
    render_prediction_page(active_dataset_filename, active_data_path)
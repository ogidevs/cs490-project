import os
import json
import streamlit as st
import sys

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import HYPERPARAM_CONFIG


def get_available_datasets():
    """Reads the JSON catalog and constructs beautiful human-readable UI dropdown options."""
    meta_path = os.path.join("data", "metadata.json")

    # If registry doesn't exist, return empty
    if not os.path.exists(meta_path):
        return {}

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Sort datasets chronologically (newest first)
    sorted_meta = sorted(
        metadata.items(), key=lambda x: x[1]["scraped_at"], reverse=True
    )

    dataset_map = {}
    for file_id, info in sorted_meta:
        # Check if the CSV actually exists so we don't crash if a user manually deleted a file
        filename = f"{file_id}.csv"
        if not os.path.exists(os.path.join("data", filename)):
            continue

        prop_type = info.get("property_type", "Unknown").capitalize()
        cities = ", ".join(info.get("cities", []))
        date_str = info.get("scraped_at", "Unknown Date")
        records = info.get("records", 0)

        # Build the beautiful UI display string
        display_name = f"{prop_type} | {cities} | {records} properties | {date_str}"
        dataset_map[display_name] = filename

    return dataset_map


def render_dynamic_hyperparameters(model_name):
    """Dynamically generates Streamlit input widgets for a specific machine learning model."""
    params = {}
    config_list = HYPERPARAM_CONFIG.get(model_name, [])
    for param in config_list:
        p_name, p_label, p_type = param["name"], param["label"], param["type"]
        key = f"{model_name}_{p_name}"

        if p_type == "slider":
            params[p_name] = st.slider(
                p_label,
                min_value=param.get("min"),
                max_value=param.get("max"),
                value=param.get("default"),
                step=param.get("step"),
                key=key,
            )
        elif p_type == "number":
            params[p_name] = st.number_input(
                p_label,
                min_value=param.get("min"),
                max_value=param.get("max"),
                value=param.get("default"),
                step=param.get("step"),
                format=param.get("format"),
                key=key,
            )
        elif p_type == "selectbox":
            opts = param.get("options")
            default_idx = (
                opts.index(param.get("default")) if param.get("default") in opts else 0
            )
            fmt = param.get("format_func", lambda x: str(x))
            params[p_name] = st.selectbox(
                p_label, options=opts, index=default_idx, format_func=fmt, key=key
            )
        elif p_type == "checkbox":
            params[p_name] = st.checkbox(p_label, value=param.get("default"), key=key)
    return params

import os
import streamlit as st
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model_config import HYPERPARAM_CONFIG


def get_available_datasets():
    """Identifies and indexes successfully extracted scraping artifacts in the data folder."""
    if not os.path.exists("data"):
        return {}
    files = [
        f for f in os.listdir("data") if f.startswith("dataset_") and f.endswith(".csv")
    ]
    files.sort(key=lambda x: os.path.getmtime(os.path.join("data", x)), reverse=True)

    dataset_map = {}
    for f in files:
        base = f.replace(".csv", "")
        parts = base.split("_")
        if len(parts) >= 5:
            prop_type = parts[1].capitalize()
            cities = parts[2].replace("-", ", ")
            date_str = parts[3]
            time_str = parts[4].replace("-", ":")
            display_name = f"{prop_type} | {cities} | {date_str} {time_str}"
        else:
            display_name = f
        dataset_map[display_name] = f

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

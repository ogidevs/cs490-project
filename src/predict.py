import joblib
import pandas as pd
import os


def predict_value(input_data_dict, dataset_filename, target_suffix):
    """Loads the compiled ML Pipeline and predicts the price directly from a raw dictionary."""
    dataset_base = dataset_filename.replace(".csv", "")
    pipeline_path = os.path.join(
        "data", f"best_pipeline_{dataset_base}_{target_suffix}.pkl"
    )

    if not os.path.exists(pipeline_path):
        return "Model Pipeline not found. Please train a model for this dataset and target first."

    try:
        # The pipeline contains the imputer, scaler, one-hot encoder, AND the model
        pipeline = joblib.load(pipeline_path)
    except Exception as e:
        return f"File Reading Asset Corrupted: {e}"

    # Convert input dict to dataframe
    df_input = pd.DataFrame([input_data_dict])

    # The pipeline automatically handles missing columns, encoding, and scaling
    prediction = pipeline.predict(df_input)
    return prediction[0]

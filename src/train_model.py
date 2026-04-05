import os
import json
import joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from src.preprocessing import load_and_clean_data, prepare_data_for_training, get_preprocessor
from src.config import MODEL_CLASSES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def train_and_evaluate(dataset_filename, selected_models, target_variable='Total_Price_EUR', model_params=None):
    """
    Compiles ML Pipelines, fits them against data, benchmarks performance, and saves the best pipeline.
    """
    if model_params is None: 
        model_params = {}

    filepath = os.path.join('data', dataset_filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset {dataset_filename} not found.")

    # 1. Load data and split into features (X) and target (y)
    df = load_and_clean_data(filepath)
    X, y = prepare_data_for_training(df, target_col=target_variable)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Get the master preprocessor tailored EXACTLY to the columns we have
    preprocessor = get_preprocessor(X_train)
    
    results = {}
    best_pipeline = None
    best_score = -float('inf')
    best_model_name = "None"
    
    for name in selected_models:
        if name in MODEL_CLASSES:
            params = model_params.get(name, {})
            if name in["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"]:
                params['random_state'] = 42
                
            model = MODEL_CLASSES[name](**params)
            
            pipeline = Pipeline(steps=[
                ('preprocessor', clone(preprocessor)), 
                ('model', model)
            ])
            
            pipeline.fit(X_train, y_train)
            
            preds = pipeline.predict(X_test)
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            
            results[name] = {"R2": round(r2, 4), "MAE": round(mae, 2)}
            
            if r2 > best_score:
                best_score = r2
                best_pipeline = pipeline
                best_model_name = name
            
    # 3. Save the FULL pipeline to disk securely
    os.makedirs('data', exist_ok=True)
    dataset_base = dataset_filename.replace('.csv', '')
    target_suffix = "price" if "Total" in target_variable else "price_per_m2"
    
    joblib.dump(best_pipeline, os.path.join('data', f'best_pipeline_{dataset_base}_{target_suffix}.pkl'))
    
    # 4. UPDATE METADATA JSON WITH MODEL INFO
    meta_path = os.path.join('data', 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        if dataset_base in meta:
            if 'models' not in meta[dataset_base]:
                meta[dataset_base]['models'] = {}
                
            meta[dataset_base]['models'][target_suffix] = {
                "best_model": best_model_name,
                "r2": round(best_score, 4),
                "numerical_features": NUMERICAL_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "target_column": target_variable
            }
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=4, ensure_ascii=False)
    
    return results, best_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# --- SCRAPER CONFIGURATION ---
URL_MAPPING = {
    "flat": "prodaja-stanova",
    "house": "prodaja-kuca",
    "land": "prodaja-zemljista",
    "garage": "prodaja-garaza"
}

CITIES = {
    "Beograd": "beograd", "Novi Sad": "novi-sad", "Niš": "nis",
    "Kragujevac": "kragujevac", "Subotica": "subotica", "Zrenjanin": "zrenjanin",
    "Pančevo": "pancevo", "Čačak": "cacak", "Kruševac": "krusevac", "Sombor": "sombor"
}

# --- FEATURE ENGINEERING CONFIGURATION ---
# The model uses raw listing fields plus engineered signals that can be
# reconstructed at prediction time from the same inputs.
NUMERICAL_FEATURES = [
    'Area',
    'Rooms_Numeric',
    'Current_Floor_Num',
    'Total_Floors_Num',
    'Floor_Ratio',
    'Area_per_Room',
    'Rooms_per_100sqm',
    'Area_Log',
    'Photo_Count',
    'Photo_Count_Log',
    'Is_Ground_Floor',
    'Is_Top_Floor',
    'Is_Agency',
]
CATEGORICAL_FEATURES = ['City', 'Municipality', 'Neighborhood', 'Advertiser_Type', 'Property_Type', 'Property_Subtype', 'Floor_Category']

# --- MODEL REGISTRY ---
MODEL_CLASSES = {
    "Linear Regression": LinearRegression,
    "Ridge Regression": Ridge,
    "Lasso Regression": Lasso,
    "Decision Tree": DecisionTreeRegressor,
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "XGBoost": XGBRegressor,
    "LightGBM": LGBMRegressor,
    "K-Neighbors": KNeighborsRegressor
}

# --- UI HYPERPARAMETER CONFIGURATION ---
HYPERPARAM_CONFIG = {
    "Linear Regression":[{"name": "fit_intercept", "label": "Fit Intercept", "type": "checkbox", "default": True}],
    "Ridge Regression":[{"name": "alpha", "label": "Alpha", "type": "number", "min": 0.01, "max": 200.0, "default": 1.0, "step": 0.5}],
    "Lasso Regression":[
        {"name": "alpha", "label": "Alpha", "type": "number", "min": 0.01, "max": 200.0, "default": 1.0, "step": 0.5},
        {"name": "max_iter", "label": "Max Iterations", "type": "number", "min": 1000, "max": 10000, "default": 5000, "step": 500}
    ],
    "Decision Tree":[
        {"name": "max_depth", "label": "Max Depth", "type": "selectbox", "options":[None, 5, 10, 20, 50], "default": None, "format_func": lambda x: "Unlimited" if x is None else str(x)},
        {"name": "min_samples_split", "label": "Min Samples Split", "type": "slider", "min": 2, "max": 20, "default": 2},
        {"name": "min_samples_leaf", "label": "Min Samples Leaf", "type": "slider", "min": 1, "max": 10, "default": 1},
        {"name": "max_features", "label": "Max Features", "type": "selectbox", "options": [None, 'sqrt', 'log2'], "default": None, "format_func": lambda x: 'Auto' if x is None else str(x)}
    ],
    "Random Forest":[
        {"name": "n_estimators", "label": "Number of Trees", "type": "slider", "min": 100, "max": 1000, "default": 400, "step": 50},
        {"name": "max_depth", "label": "Max Depth", "type": "selectbox", "options": [None, 10, 20, 30, 50], "default": 20, "format_func": lambda x: "Unlimited" if x is None else str(x)},
        {"name": "min_samples_split", "label": "Min Samples Split", "type": "slider", "min": 2, "max": 20, "default": 4},
        {"name": "min_samples_leaf", "label": "Min Samples Leaf", "type": "slider", "min": 1, "max": 10, "default": 2},
        {"name": "max_features", "label": "Max Features", "type": "selectbox", "options": [None, 'sqrt', 'log2'], "default": 'sqrt', "format_func": lambda x: 'Auto' if x is None else str(x)}
    ],
    "Gradient Boosting":[
        {"name": "n_estimators", "label": "Number of Boosting Stages", "type": "slider", "min": 100, "max": 1500, "default": 400, "step": 50},
        {"name": "learning_rate", "label": "Learning Rate", "type": "number", "min": 0.001, "max": 1.0, "default": 0.05, "step": 0.01, "format": "%.3f"},
        {"name": "max_depth", "label": "Max Depth", "type": "slider", "min": 1, "max": 10, "default": 3},
        {"name": "subsample", "label": "Subsample", "type": "number", "min": 0.5, "max": 1.0, "default": 0.8, "step": 0.05, "format": "%.2f"},
        {"name": "min_samples_leaf", "label": "Min Samples Leaf", "type": "slider", "min": 1, "max": 10, "default": 2},
        {"name": "max_features", "label": "Max Features", "type": "selectbox", "options": [None, 'sqrt', 'log2'], "default": None, "format_func": lambda x: 'Auto' if x is None else str(x)}
    ],
    "XGBoost":[
        {"name": "n_estimators", "label": "Number of Stages", "type": "slider", "min": 100, "max": 1500, "default": 500, "step": 50},
        {"name": "learning_rate", "label": "Learning Rate", "type": "number", "min": 0.001, "max": 1.0, "default": 0.05, "step": 0.01, "format": "%.3f"},
        {"name": "max_depth", "label": "Max Depth", "type": "slider", "min": 2, "max": 12, "default": 6},
        {"name": "subsample", "label": "Subsample", "type": "number", "min": 0.5, "max": 1.0, "default": 0.85, "step": 0.05, "format": "%.2f"},
        {"name": "colsample_bytree", "label": "Column Sample", "type": "number", "min": 0.5, "max": 1.0, "default": 0.8, "step": 0.05, "format": "%.2f"},
        {"name": "reg_alpha", "label": "L1 Regularization", "type": "number", "min": 0.0, "max": 10.0, "default": 0.0, "step": 0.1, "format": "%.2f"},
        {"name": "reg_lambda", "label": "L2 Regularization", "type": "number", "min": 0.1, "max": 20.0, "default": 1.0, "step": 0.1, "format": "%.2f"}
    ],
    "LightGBM":[
        {"name": "n_estimators", "label": "Number of Stages", "type": "slider", "min": 100, "max": 1500, "default": 500, "step": 50},
        {"name": "learning_rate", "label": "Learning Rate", "type": "number", "min": 0.001, "max": 1.0, "default": 0.05, "step": 0.01, "format": "%.3f"},
        {"name": "num_leaves", "label": "Number of Leaves", "type": "slider", "min": 16, "max": 256, "default": 64, "step": 1},
        {"name": "max_depth", "label": "Max Depth", "type": "selectbox", "options": [None, -1, 10, 20, 30], "default": -1, "format_func": lambda x: 'Unlimited' if x in [None, -1] else str(x)},
        {"name": "subsample", "label": "Subsample", "type": "number", "min": 0.5, "max": 1.0, "default": 0.85, "step": 0.05, "format": "%.2f"},
        {"name": "colsample_bytree", "label": "Column Sample", "type": "number", "min": 0.5, "max": 1.0, "default": 0.8, "step": 0.05, "format": "%.2f"},
        {"name": "min_child_samples", "label": "Min Child Samples", "type": "slider", "min": 5, "max": 100, "default": 20, "step": 1}
    ],
    "K-Neighbors":[
        {"name": "n_neighbors", "label": "Neighbors (k)", "type": "slider", "min": 1, "max": 30, "default": 5},
        {"name": "weights", "label": "Weight Function", "type": "selectbox", "options": ['uniform', 'distance'], "default": 'uniform'}
    ]
}
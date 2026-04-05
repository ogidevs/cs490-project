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
# Stripped out "junk" features (photos, floors, advertiser type). 
# The ML Pipeline will now ONLY train on real property metrics.
NUMERICAL_FEATURES =['Area', 'Rooms_Numeric']
CATEGORICAL_FEATURES = ['City', 'Municipality']

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
        {"name": "min_samples_split", "label": "Min Samples Split", "type": "slider", "min": 2, "max": 20, "default": 2}
    ],
    "Random Forest":[
        {"name": "n_estimators", "label": "Number of Trees", "type": "slider", "min": 10, "max": 500, "default": 100, "step": 10},
        {"name": "max_depth", "label": "Max Depth", "type": "selectbox", "options": [None, 10, 20, 30, 50], "default": None, "format_func": lambda x: "Unlimited" if x is None else str(x)}
    ],
    "Gradient Boosting":[
        {"name": "n_estimators", "label": "Number of Boosting Stages", "type": "slider", "min": 50, "max": 1000, "default": 100, "step": 50},
        {"name": "learning_rate", "label": "Learning Rate", "type": "number", "min": 0.001, "max": 1.0, "default": 0.1, "step": 0.01, "format": "%.3f"},
        {"name": "max_depth", "label": "Max Depth", "type": "slider", "min": 1, "max": 15, "default": 3}
    ],
    "XGBoost":[
        {"name": "n_estimators", "label": "Number of Stages", "type": "slider", "min": 50, "max": 1000, "default": 100, "step": 50},
        {"name": "learning_rate", "label": "Learning Rate", "type": "number", "min": 0.001, "max": 1.0, "default": 0.1, "step": 0.01, "format": "%.3f"},
        {"name": "max_depth", "label": "Max Depth", "type": "slider", "min": 1, "max": 15, "default": 6}
    ],
    "LightGBM":[
        {"name": "n_estimators", "label": "Number of Stages", "type": "slider", "min": 50, "max": 1000, "default": 100, "step": 50},
        {"name": "learning_rate", "label": "Learning Rate", "type": "number", "min": 0.001, "max": 1.0, "default": 0.1, "step": 0.01, "format": "%.3f"},
        {"name": "num_leaves", "label": "Number of Leaves", "type": "slider", "min": 10, "max": 200, "default": 31, "step": 1}
    ],
    "K-Neighbors":[
        {"name": "n_neighbors", "label": "Neighbors (k)", "type": "slider", "min": 1, "max": 30, "default": 5},
        {"name": "weights", "label": "Weight Function", "type": "selectbox", "options": ['uniform', 'distance'], "default": 'uniform'}
    ]
}
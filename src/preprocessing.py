import pandas as pd
import numpy as np
import re
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def parse_floor_numeric(val):
    """
    Converts textual floor descriptions to numeric values.
    Examples: 'V 5 Spratnost' -> 5.0, 'I Spratnost' -> 1.0, 'Prizemlje' -> 0.0
    """
    if pd.isna(val) or str(val).strip() == "":
        return np.nan
    
    val_str = str(val).lower().strip()

    special_terms = {
        'prizemlje': 0.0,
        'pr': 0.0,
        'visoko prizemlje': 0.5,
        'vpr': 0.5,
        'suteren': -1.0,
        'sut': -1.0,
        'potkrovlje': 99.0,
        'ptk': 99.0,
        'blagi suteren': -0.5
    }
    
    if val_str in special_terms:
        return special_terms[val_str]

    roman_map = {
        'ix': 9, 'viii': 8, 'vii': 7, 'vi': 6, 'iv': 4, 
        'v': 5, 'iii': 3, 'ii': 2, 'i': 1, 'x': 10
    }

    val_str = val_str.replace('spratnost', '').replace('sprat', '').strip()

    numbers = re.findall(r'\d+', val_str)
    if numbers:
        return float(numbers[0])

    for roman, num in roman_map.items():
        if roman in val_str.split():
            return float(num)
            
    return np.nan

def load_and_clean_data(filepath):
    """Loads CSV source files and applies statistical IQR outlier clipping."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['Total_Price_EUR', 'Area'])
    
    df['Price_per_Unit_EUR'] = df['Total_Price_EUR'] / df['Area']

    if 'Rooms' in df.columns and not df['Rooms'].isnull().all():
        df['Rooms_Numeric'] = df['Rooms'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

    if 'Current_Floor' in df.columns:
        df['Current_Floor_Num'] = df['Current_Floor'].apply(parse_floor_numeric)
    
    if 'Total_Floors' in df.columns:
        df['Total_Floors_Num'] = df['Total_Floors'].apply(parse_floor_numeric)

    if df.empty:
        return df

    for col in ['Total_Price_EUR', 'Area']:
        q1 = df[col].quantile(0.10)
        q3 = df[col].quantile(0.90)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
    return df

def prepare_data_for_training(df, target_col='Total_Price_EUR'):
    """Isolates the target variable and drops completely empty columns to prevent imputer crashes."""
    df = df.dropna(subset=[target_col]).copy()

    df = df.dropna(axis=1, how='all')
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def get_preprocessor(X):
    """Builds a robust scikit-learn preprocessing pipeline dynamically based on available columns."""

    num_features = [col for col in NUMERICAL_FEATURES if col in X.columns]
    cat_features = [col for col in CATEGORICAL_FEATURES if col in X.columns]

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ], remainder='drop') 
    
    return preprocessor
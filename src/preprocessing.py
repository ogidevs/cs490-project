import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def load_and_clean_data(filepath):
    """Loads CSV source files and applies statistical IQR outlier clipping."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['Total_Price_EUR', 'Area'])
    
    df['Price_per_Unit_EUR'] = df['Total_Price_EUR'] / df['Area']
    
    # Safely try to extract rooms, otherwise it's skipped
    if 'Rooms' in df.columns and not df['Rooms'].isnull().all():
        df['Rooms_Numeric'] = df['Rooms'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
        
    if df.empty:
        return df
        
    # Standardize IQR outlier thresholding on Pricing and Sizing
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
    
    # Drop any columns that are 100% empty (e.g., if we scraped Land and there are no Rooms)
    df = df.dropna(axis=1, how='all')
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def get_preprocessor(X):
    """Builds a robust scikit-learn preprocessing pipeline dynamically based on available columns."""
    
    # ONLY use features that actually exist in the incoming dataframe X
    num_features =[col for col in NUMERICAL_FEATURES if col in X.columns]
    cat_features =[col for col in CATEGORICAL_FEATURES if col in X.columns]

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
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def _normalize_text(val):
    if pd.isna(val):
        return "Unknown"
    text = re.sub(r"\s+", " ", str(val).strip())
    return text if text else "Unknown"


def parse_room_numeric(val):
    """Converts room descriptions like 'dvoiposoban' or '3.5' into a numeric value."""
    if pd.isna(val) or str(val).strip() == "":
        return np.nan

    val_str = _normalize_text(val).lower()
    room_terms = {
        "garsonjera": 0.5,
        "studio": 0.5,
        "jednoiposoban": 1.5,
        "1.5": 1.5,
        "jednosoban": 1.0,
        "1 soban": 1.0,
        "dvosoban": 2.0,
        "dvoiposoban": 2.5,
        "trosoban": 3.0,
        "troiposoban": 3.5,
        "cetvorosoban": 4.0,
        "četvorosoban": 4.0,
        "petosoban": 5.0,
        "vise soban": 5.0,
        "višesoban": 5.0,
        "visesoban": 5.0,
    }

    for term, value in room_terms.items():
        if term in val_str:
            return float(value)

    match = re.search(r"(\d+(?:[\.,]\d+)?)", val_str)
    if match:
        return float(match.group(1).replace(",", "."))

    return np.nan


def parse_floor_numeric(val):
    """Converts floor labels into numeric values for both raw and cleaned inputs."""
    if pd.isna(val) or str(val).strip() == "":
        return np.nan

    val_str = _normalize_text(val).lower()

    special_terms = {
        "prizemlje": 0.0,
        "pr": 0.0,
        "visoko prizemlje": 0.5,
        "vpr": 0.5,
        "suteren": -1.0,
        "sut": -1.0,
        "blagi suteren": -0.5,
        "potkrovlje": 99.0,
        "ptk": 99.0,
    }

    if val_str in special_terms:
        return special_terms[val_str]

    roman_map = {
        "ix": 9,
        "viii": 8,
        "vii": 7,
        "vi": 6,
        "v": 5,
        "iv": 4,
        "iii": 3,
        "ii": 2,
        "i": 1,
        "x": 10,
    }

    cleaned = (
        val_str.replace("spratnost", "")
        .replace("sprat", "")
        .replace("/", " ")
        .strip()
    )

    numbers = re.findall(r"\d+(?:[\.,]\d+)?", cleaned)
    if numbers:
        return float(numbers[0].replace(",", "."))

    for token in cleaned.split():
        if token in roman_map:
            return float(roman_map[token])

    return np.nan


def engineer_features(df):
    """Builds reusable features that can be reconstructed during prediction."""
    df = df.copy()

    for column in ["City", "Municipality", "Neighborhood", "Advertiser_Type", "Property_Type", "Property_Subtype"]:
        if column not in df.columns:
            df[column] = "Unknown"
        df[column] = df[column].apply(_normalize_text)

    if "Area" not in df.columns:
        df["Area"] = np.nan
    df["Area"] = pd.to_numeric(df["Area"], errors="coerce")

    if "Photo_Count" not in df.columns:
        df["Photo_Count"] = np.nan
    df["Photo_Count"] = pd.to_numeric(df["Photo_Count"], errors="coerce")

    if "Rooms_Numeric" not in df.columns:
        if "Rooms" in df.columns:
            df["Rooms_Numeric"] = df["Rooms"].apply(parse_room_numeric)
        else:
            df["Rooms_Numeric"] = np.nan
    else:
        df["Rooms_Numeric"] = pd.to_numeric(df["Rooms_Numeric"], errors="coerce")

    if "Current_Floor_Num" not in df.columns:
        if "Current_Floor" in df.columns:
            df["Current_Floor_Num"] = df["Current_Floor"].apply(parse_floor_numeric)
        else:
            df["Current_Floor_Num"] = np.nan
    else:
        df["Current_Floor_Num"] = pd.to_numeric(df["Current_Floor_Num"], errors="coerce")

    if "Total_Floors_Num" not in df.columns:
        if "Total_Floors" in df.columns:
            df["Total_Floors_Num"] = df["Total_Floors"].apply(parse_floor_numeric)
        else:
            df["Total_Floors_Num"] = np.nan
    else:
        df["Total_Floors_Num"] = pd.to_numeric(df["Total_Floors_Num"], errors="coerce")

    current_floor = df["Current_Floor_Num"].replace([np.inf, -np.inf], np.nan)
    total_floors = df["Total_Floors_Num"].replace([np.inf, -np.inf], np.nan)
    area = df["Area"].replace([np.inf, -np.inf], np.nan)
    rooms = df["Rooms_Numeric"].replace([np.inf, -np.inf], np.nan)
    photos = df["Photo_Count"].replace([np.inf, -np.inf], np.nan)

    df["Floor_Ratio"] = current_floor / total_floors.replace(0, np.nan)
    df["Floor_Ratio"] = df["Floor_Ratio"].replace([np.inf, -np.inf], np.nan)

    df["Area_per_Room"] = area / rooms.replace(0, np.nan)
    df["Rooms_per_100sqm"] = (rooms / area.replace(0, np.nan)) * 100
    df["Area_Log"] = np.log1p(area)
    df["Photo_Count_Log"] = np.log1p(photos)
    df["Is_Ground_Floor"] = current_floor.le(0).fillna(False).astype(int)
    df["Is_Top_Floor"] = (
        current_floor.notna() & total_floors.notna() & (current_floor >= total_floors)
    ).astype(int)
    df["Is_Agency"] = df["Advertiser_Type"].str.contains(
        r"agenc|agency|posrednik", case=False, na=False
    ).astype(int)

    def _floor_bucket(row):
        floor = row["Current_Floor_Num"]
        total = row["Total_Floors_Num"]
        if pd.isna(floor):
            return "Unknown"
        if floor <= 0:
            return "Ground"
        if pd.notna(total) and floor >= total:
            return "Top"
        if floor <= 3:
            return "Low"
        if floor <= 7:
            return "Mid"
        return "High"

    df["Floor_Category"] = df.apply(_floor_bucket, axis=1)
    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return engineer_features(X)


def load_and_clean_data(filepath):
    """Loads CSV source files, removes extreme price-area outliers, and engineers features."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["Total_Price_EUR", "Area"])
    df = df[df["Area"] > 0]
    df = df[df["Total_Price_EUR"] > 0]

    df = engineer_features(df)

    if "Price_per_Unit_EUR" not in df.columns:
        df["Price_per_Unit_EUR"] = np.nan
    df["Price_per_Unit_EUR"] = df["Total_Price_EUR"] / df["Area"].replace(0, np.nan)

    if df.empty:
        return df

    for col in ["Total_Price_EUR", "Area"]:
        q1 = df[col].quantile(0.10)
        q3 = df[col].quantile(0.90)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df


def prepare_data_for_training(df, target_col='Total_Price_EUR'):
    """Isolates the target variable and removes leakage columns derived from it."""
    df = df.dropna(subset=[target_col]).copy()
    df = df.dropna(axis=1, how='all')

    leakage_columns = {
        'Total_Price_EUR': 'Price_per_Unit_EUR',
        'Price_per_Unit_EUR': 'Total_Price_EUR',
    }
    leakage_col = leakage_columns.get(target_col)
    if leakage_col in df.columns:
        df = df.drop(columns=[leakage_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def get_preprocessor(X=None):
    """Builds a preprocessing pipeline for the engineered feature schema."""
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=10))
    ])

    return Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('column_transformer', ColumnTransformer(transformers=[
            ('num', num_transformer, NUMERICAL_FEATURES),
            ('cat', cat_transformer, CATEGORICAL_FEATURES),
        ], remainder='drop')),
    ])
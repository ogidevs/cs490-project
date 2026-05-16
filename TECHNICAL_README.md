# Serbian Real Estate Prediction System - Technical Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Technology Stack](#architecture--technology-stack)
3. [Data Pipeline](#data-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Machine Learning Models](#machine-learning-models)
6. [User Interface & Workflows](#user-interface--workflows)
7. [Visualization & Analysis](#visualization--analysis)
8. [API & Backend](#api--backend)
9. [Testing & Validation](#testing--validation)
10. [Configuration & Metadata](#configuration--metadata)

---

## Project Overview

### Purpose

A machine learning system that predicts real estate property prices in Serbia based on property characteristics, location, and market signals. Supports multiple property types (apartments/flats, houses, land, garages) with property-type-aware feature engineering.

### Key Features

- **Dynamic Web Scraping**: Collects live real estate listings from HaloOglasi
- **Property-Type Aware**: Different feature sets for flats, houses, land, and garages
- **Multi-Model Training**: Supports 8+ ML algorithms with automatic best model selection
- **Interactive UI**: Streamlit-based dashboard for data exploration, training, and predictions
- **Reproducible Pipelines**: Scikit-Learn pipelines for consistent preprocessing and prediction
- **Comprehensive Testing**: 74-test suite with 47% code coverage

### Supported Property Types

| Type       | Features                     | Use Case                            |
| ---------- | ---------------------------- | ----------------------------------- |
| **Flat**   | 13 numerical + 7 categorical | Apartment buildings with floor info |
| **House**  | 8 numerical + 6 categorical  | Single-family homes, villas         |
| **Land**   | 5 numerical + 5 categorical  | Raw land, plots                     |
| **Garage** | 5 numerical + 5 categorical  | Parking spaces, garages             |

---

## Architecture & Technology Stack

### Technology Summary

```
Frontend:           Streamlit (Python web framework)
Backend:            Python 3.13.12
Package Manager:    uv (ultra-fast Python package manager)
ML/Data:            scikit-learn, XGBoost, LightGBM, pandas, numpy
Testing:            pytest, pytest-cov
Web Scraping:       requests, BeautifulSoup (custom selenium alternative)
Visualization:      matplotlib, seaborn
Storage:            CSV files (data/), JSON (metadata), pickle (models)
```

### Directory Structure

```
project/
├── app/                          # Streamlit UI application
│   ├── streamlit_app.py         # Main app entry point
│   ├── page_scraping.py         # Web scraping UI
│   ├── page_eda.py              # Exploratory Data Analysis UI
│   ├── page_training.py         # Model training UI
│   ├── page_prediction.py       # Price prediction UI
│   └── ui_utils.py              # Shared UI utilities
│
├── src/                          # Core business logic
│   ├── config.py                # Configuration, features, models
│   ├── scraper.py               # Web scraping logic
│   ├── preprocessing.py         # Data cleaning & feature engineering
│   ├── train_model.py           # Model training pipeline
│   ├── predict.py               # Prediction engine
│   └── plotting.py              # Visualization functions
│
├── tests/                        # Comprehensive test suite (74 tests)
│   ├── conftest.py              # pytest fixtures
│   ├── test_preprocessing.py    # Feature engineering tests (34)
│   ├── test_training.py         # Training tests (7)
│   ├── test_prediction.py       # Prediction tests (6)
│   ├── test_data_validation.py  # Data integrity tests (10)
│   └── test_property_types.py   # Property type tests (18)
│
├── data/                         # Data storage
│   ├── dataset_*.csv            # Raw scraped datasets
│   ├── metadata.json            # Dataset & model metadata
│   └── best_pipeline_*.pkl      # Trained model pipelines
│
├── notebooks/                    # Analysis notebooks
│   └── analysis.ipynb           # Exploratory analysis
│
├── pyproject.toml               # Project configuration and dependencies
├── tox.ini                      # Testing automation
└── README.md                    # User-facing documentation
```

---

## Data Pipeline

### Data Collection (Scraping)

**Module**: `src/scraper.py`

**Workflow**:

1. **URL Building**: Constructs HaloOglasi URLs based on property type and city
2. **Page Iteration**: Crawls multiple pages per city (configurable)
3. **HTML Parsing**: Extracts property details from listing pages
4. **Data Standardization**: Converts scraped values to consistent units
5. **Deduplication**: Filters out duplicate listings
6. **CSV Export**: Saves to timestamped CSV file

**Supported Property Types**:

```python
URL_MAPPING = {
    "flat": "prodaja-stanova",
    "house": "prodaja-kuca",
    "land": "prodaja-zemljista",
    "garage": "prodaja-garaza",
}
```

**Supported Cities** (10 major Serbian cities):

- Beograd, Novi Sad, Niš, Kragujevac, Subotica
- Zrenjanin, Pančevo, Čačak, Kruševac, Sombor

**Data Collected**:

```
- Total_Price_EUR: Property price in euros
- Area: Property size in square meters (standardized)
- Rooms: Room count (parsed from text descriptions)
- Current_Floor: Floor number (1-100 scale)
- Total_Floors: Building height
- City, Municipality, Neighborhood: Location hierarchy
- Property_Type, Property_Subtype: Classification
- Advertiser_Type: Agency vs. private seller
- Photo_Count: Number of listing photos
- Scrape_Date, Scrape_Time: Metadata
```

**Output Format**:

```
data/dataset_{property_type}_{cities}_{date}_{time}.csv
Example: dataset_flat_beograd-novi-sad_2026-05-16_14-30-45.csv
```

### Data Cleaning & Validation

**Module**: `src/preprocessing.py` → `load_and_clean_data()`

**Cleaning Steps** (in order):

1. **Drop Missing Essentials**: Remove rows missing `Total_Price_EUR` or `Area`
2. **Remove Invalid Values**: Drop rows where `Area <= 0` or `Price <= 0`
3. **Feature Engineering**: Create derived features (see next section)
4. **Price Per Unit**: Calculate `Price_per_Unit_EUR = Total_Price_EUR / Area`
5. **Outlier Removal**: Remove extreme 10th-90th percentile outliers using IQR method
   - Formula: `lower = Q1 - 1.5*IQR`, `upper = Q3 + 1.5*IQR`
   - Applied to both `Total_Price_EUR` and `Area`

**Data Quality Metrics**:

- Input: ~500-2000 raw listings per city
- After cleaning: ~300-1500 valid records (60-75% retention)
- Typical dataset: 5,000-15,000 properties across multiple cities

---

## Feature Engineering

### Feature Creation Strategy

**Principle**: Create features that can be reconstructed at prediction time from raw inputs.

**Module**: `src/preprocessing.py` → `engineer_features()`

### Original Features (Raw Input)

```
Numerical: Area, Photo_Count, Rooms_Numeric, Current_Floor_Num, Total_Floors_Num
Categorical: City, Municipality, Neighborhood, Advertiser_Type,
            Property_Type, Property_Subtype
```

### Engineered Features

#### 1. **Location-Based Features**

- All property types benefit from location data
- Aggregated via One-Hot Encoding in preprocessing pipeline
- Captures neighborhood price premiums

#### 2. **Floor-Based Features** (Flats Only - _Not created for other types_)

| Feature           | Calculation                     | Purpose                                             | Type        |
| ----------------- | ------------------------------- | --------------------------------------------------- | ----------- |
| `Floor_Ratio`     | `Current_Floor / Total_Floors`  | Relative floor position (0-1)                       | float       |
| `Is_Ground_Floor` | `Current_Floor <= 0`            | Binary: true for ground floor                       | int         |
| `Is_Top_Floor`    | `Current_Floor >= Total_Floors` | Binary: true for top floor                          | int         |
| `Floor_Category`  | Bucketing logic                 | Discrete: Ground, Low(1-3), Mid(4-7), High(8+), Top | categorical |

**Bucketing Logic**:

```
if floor <= 0:      "Ground"
elif floor >= total: "Top"
elif floor <= 3:    "Low"
elif floor <= 7:    "Mid"
else:               "High"
```

_Why?_ Different floors have different market prices in apartments.

#### 3. **Room-Based Features** (Flats & Houses Only - _Not created for land/garages_)

| Feature            | Calculation            | Purpose        | Type  |
| ------------------ | ---------------------- | -------------- | ----- |
| `Area_per_Room`    | `Area / Rooms`         | Room size (m²) | float |
| `Rooms_per_100sqm` | `(Rooms / Area) * 100` | Room density   | float |

_Why?_ Market values spacious rooms differently than cramped units.

#### 4. **Area-Based Features** (All Property Types)

| Feature    | Calculation   | Purpose                              | Type  |
| ---------- | ------------- | ------------------------------------ | ----- |
| `Area_Log` | `log1p(Area)` | Log-transformed area (handles range) | float |

_Why?_ Log transformation linearizes price-area relationship and handles outliers.

#### 5. **Photo-Based Features** (All Property Types)

| Feature           | Calculation          | Purpose                     | Type  |
| ----------------- | -------------------- | --------------------------- | ----- |
| `Photo_Count_Log` | `log1p(Photo_Count)` | Log-transformed photo count | float |

_Why?_ More photos = better listed property = potential quality signal.

#### 6. **Advertiser Signal** (All Property Types)

| Feature     | Calculation                                        | Purpose                    | Type |
| ----------- | -------------------------------------------------- | -------------------------- | ---- |
| `Is_Agency` | Does name contain "agency"/"agencija"/"posrednik"? | Binary: agency vs. private | int  |

_Why?_ Agencies vs. private sellers may have different negotiation patterns.

### Feature Set by Property Type

**Flats (13 numerical + 7 categorical)**:

- All floors features ✓
- All rooms features ✓
- All area features ✓
- Floor_Category categorical ✓

**Houses (8 numerical + 6 categorical)**:

- Floor features ✗ (excluded)
- Room features ✓
- Area features ✓
- No Floor_Category (only 6 categorical)

**Land (5 numerical + 5 categorical)**:

- Floor features ✗
- Room features ✗
- Area features ✓ (only area and photo-based)

**Garages (5 numerical + 5 categorical)**:

- Same as Land

### Property-Type Detection

Feature set is **automatically selected** based on `Property_Type` column:

```python
# In engineer_features():
if "Property_Type" not in df.columns:
    property_type = "flat"  # Default
else:
    property_type = df["Property_Type"].iloc[0]  # Detect from first row
```

_Important_: Process datasets with ONE property type per batch. Mixed types will use first row's type.

---

## Machine Learning Models

### Supported Models (9 total)

**Module**: `src/config.py` → `MODEL_CLASSES`

```python
MODEL_CLASSES = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
}
```

### Training Pipeline

**Module**: `src/train_model.py` → `train_and_evaluate()`

**Workflow**:

1. **Load & Clean Data**: `load_and_clean_data(dataset_file)`
2. **Split Data**: 80% training, 20% test (random_state=42)
3. **Get Preprocessor**: `get_preprocessor()` builds scikit-learn pipeline
4. **Train Models**: For each selected model:
   - Clone model class
   - Fit to training data with preprocessing
   - Evaluate on test data
   - Record metrics
5. **Select Best**: Winner by R² score (highest wins)
6. **Save Pipeline**: Pickle best*pipeline to `data/best_pipeline*{dataset}\_{target}.pkl`
7. **Update Metadata**: Log model info to `data/metadata.json`

### Preprocessing Pipeline (Scikit-Learn)

```python
Pipeline(
    steps=[
        ("feature_engineer", FeatureEngineer()),
        (
            "column_transformer",
            ColumnTransformer(
                transformers=[
                    ("num", StandardScaler + Imputation, NUMERICAL_FEATURES),
                    ("cat", OneHotEncoder + Imputation, CATEGORICAL_FEATURES),
                ],
                remainder="drop",
            ),
        ),
    ]
)
```

**Step-by-step**:

1. **Feature Engineering**: Create engineered features
2. **Numerical Transform**:
   - Imputation: Missing values filled with median
   - Scaling: StandardScaler (mean=0, std=1)
3. **Categorical Transform**:
   - Imputation: Missing values filled with "Unknown"
   - One-Hot Encoding: Handles unknown categories, min_frequency=10
4. **Remainder**: Drop unmapped columns

### Hyperparameter Configuration

**Module**: `src/config.py` → `HYPERPARAM_CONFIG`

Users can adjust per-model parameters via UI:

- `Random Forest`: n_estimators, max_depth, min_samples_split
- `XGBoost`: learning_rate, max_depth, subsample
- `LightGBM`: learning_rate, num_leaves, feature_fraction
- `Ridge/Lasso`: Regularization strength

### Evaluation Metrics

**Metrics Calculated** (on test set):

| Metric       | Formula                     | Interpretation                          |
| ------------ | --------------------------- | --------------------------------------- |
| **R² Score** | `1 - (SS_res / SS_tot)`     | Variance explained (0-1, higher better) |
| **MAE**      | `mean(\|y_true - y_pred\|)` | Average absolute error in euros         |

**Best Model Selection**: Highest R² score on test set

### Model Serialization

Trained pipelines are saved as pickle files:

```
data/best_pipeline_{dataset_id}_{target_suffix}.pkl

Examples:
- best_pipeline_dataset_1775390037_price.pkl (Total_Price_EUR)
- best_pipeline_dataset_1775390037_price_per_m2.pkl (Price_per_Unit_EUR)
```

---

## User Interface & Workflows

### Streamlit Application

**Entry Point**: `app/streamlit_app.py`

**Launch Command**:

```bash
uv run streamlit run app/streamlit_app.py
```

### Navigation Structure

```
Main Page: Real Estate Price Prediction System
├── Sidebar Navigation
│   └── 4 Main Pages + Dataset Selector
│
├── 1. Data & Scraping
│   ├── Property Type Selector (flat/house/land/garage)
│   ├── City Multi-Select
│   ├── Pages Per City Slider (1-500)
│   ├── "Start Scraping" Button
│   ├── Progress Bar & Stats
│   └── Dataset Preview (last 10 rows)
│
├── 2. EDA (Analysis)
│   ├── Feature Distributions (histogram)
│   ├── Price vs. Area (scatter plot)
│   ├── Rooms & Municipalities (bar charts)
│   └── Feature Correlation Matrix (heatmap)
│
├── 3. Model Training
│   ├── Dataset Info Cards (property type, cities, records)
│   ├── Target Variable Selector
│   ├── Model Multi-Select
│   ├── Hyperparameter Tabs (per-model settings)
│   ├── "Train Selected Models" Button
│   └── Performance Table (R², MAE, best model)
│
└── 4. Prediction
    ├── Property Details Input
    │   ├── City, Municipality, Neighborhood (cascading)
    │   ├── Area, Rooms, Floor, Total Floors (conditional)
    │   ├── Photo Count, Advertiser Type
    │   └── Property Type (auto-detected from dataset)
    ├── "Calculate Valuation" Button
    ├── Prediction Results (total price, price/m², market comparison)
    └── Market Distribution Graph (property overlaid on market)
```

### Sidebar Intelligence

**Dynamic Dataset Selection**:

- Lists available CSV files from `data/` folder
- Displays metadata:
  - Property Type
  - Cities included
  - Number of records
  - Scrape timestamp
  - Trained models (if any) with R² scores

### Page 1: Data & Scraping

**Purpose**: Collect new real estate data

**Inputs**:

- Property Type: flat, house, land, garage
- Cities: Multi-select from 10 supported cities
- Pages per City: 1-500 (5 is typical)

**Process**:

1. Constructs URLs for each city+type combination
2. Crawls HaloOglasi.rs listings
3. Extracts property data (100-300 listings per page)
4. Deduplicates results
5. Saves to timestamped CSV

**Output**: `data/dataset_[type]_[cities]_[date]_[time].csv`

**Time Taken**: 2-5 minutes for 5 pages × 2 cities (depends on network)

### Page 2: EDA (Analysis)

**Purpose**: Understand data distribution and relationships

**Visualizations**:

1. **Feature Distributions** (histogram)
   - Shows price, area, rooms distribution
   - Reveals skewness and outliers

2. **Price vs. Area** (scatter plot)
   - Reveals linear relationship strength
   - Shows price variance per area

3. **Rooms & Municipalities** (bar charts)
   - Room count frequency
   - Properties per municipality

4. **Feature Correlation** (heatmap)
   - Shows which features predict price
   - Identifies multicollinearity

**Usage**: Validate data quality before training

### Page 3: Model Training

**Purpose**: Build price prediction models

**Workflow**:

1. **Select Target**: Total_Price_EUR or Price_per_Unit_EUR
2. **Choose Models**: Multi-select from 9 options
3. **Configure Hyperparameters**: Adjust per-model settings in tabs
4. **Train**: Click button, wait 30-60 seconds
5. **Review Results**: Table shows all models ranked by R²

**Dataset Info Displayed**:

- Property type (determines which features used)
- Number of cities
- Total records
- Trained models (if previously trained)

**Output**: Best model saved as pickle file, metadata updated

### Page 4: Prediction

**Purpose**: Get price estimates for new properties

**Input Workflow**:

1. **Location Cascade**:
   - Select City (required)
   - Select Municipality (cascading, filtered by city)
   - Select Neighborhood (cascading, filtered by municipality)
   - Advertiser Type selector

2. **Property Details** (conditional on property type):
   - **All types**: Area (m²), Photo Count
   - **Flats & Houses**: Rooms
   - **Flats only**: Current Floor, Total Floors

3. **Model Selection**:
   - Choose: Total Price or Price/m²

4. **Predict**:
   - Click "Calculate Valuation"
   - Features auto-engineered
   - Pipeline applies preprocessing
   - Model predicts price

**Output Display**:

```
┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Estimated Total  │ Estimated Price  │ Properties in    │ Market Median    │
│ Price            │ per m²           │ Scope            │ (m²)             │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ €450,000         │ €4,500 /m²       │ 245              │ €4,200 /m² (+7%) │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘

+ Market Distribution Graph (property shown as blue line vs. market median as red dashed line)
```

**Pricing Variance Calculation**:

```
diff_percent = ((predicted_price_per_m2 / market_median_per_m2) - 1) * 100

Examples:
- +10% → Property priced above market
- -5% → Property priced below market
- ±0% → Market rate
```

---

## Visualization & Analysis

### Module: `src/plotting.py`

**Visualization Functions**:

#### 1. `plot_distributions(df)`

- Price distribution (histogram with KDE)
- Area distribution
- Rooms distribution
- Shows skewness and outliers

**Used in**: Page 2 (EDA) - Feature Distributions

#### 2. `plot_relationships(df)`

- Price vs. Area (scatter)
- Color-coded by municipality
- Shows linear vs. non-linear relationships

**Used in**: Page 2 (EDA) - Price vs. Area

#### 3. `plot_advanced_features(df)`

- Rooms frequency (bar)
- Municipality distribution (bar, top 15)

**Used in**: Page 2 (EDA) - Rooms & Municipalities

#### 4. `plot_correlation_matrix(df)`

- Heatmap of feature correlations
- Shows which features predict price
- Identifies multicollinearity

**Used in**: Page 2 (EDA) - Feature Correlation

#### 5. Market Distribution in Prediction

- Histogram of local market prices
- Blue line: predicted property
- Red dashed line: market median
- Shows fit in market context

**Used in**: Page 4 (Prediction) - Market Visualization

**Visualization Libraries**:

- `matplotlib`: Base plotting
- `seaborn`: Statistical graphics (KDE, heatmaps, styling)

---

## API & Backend

### Prediction Engine

**Module**: `src/predict.py` → `predict_value()`

**Function Signature**:

```python
def predict_value(
    input_data_dict: dict,     # Raw property data
    dataset_filename: str,      # e.g., "dataset_123.csv"
    target_suffix: str,         # "price" or "price_per_m2"
) -> float | str
```

**Workflow**:

1. Load saved pipeline from pickle file
2. Convert input dict to DataFrame
3. Apply preprocessing pipeline:
   - Feature engineering
   - Imputation
   - Scaling
   - One-hot encoding
4. Predict using trained model
5. Return price (float) or error message (str)

**Input Data Dict**:

```python
{
    "City": "Belgrade",
    "Municipality": "Voždovac",
    "Neighborhood": "Kumodraž",
    "Property_Type": "flat",
    "Property_Subtype": "Apartment",
    "Area": 75.0,
    "Rooms_Numeric": 2.0,
    "Current_Floor_Num": 3,
    "Total_Floors_Num": 5,
    "Photo_Count": 8,
    "Advertiser_Type": "Agency",
}
```

**Required vs. Optional**:

- Required: City, Municipality, Area, Property_Type
- Type-specific: Rooms_Numeric (flats/houses), Floor info (flats only)
- Always included: Photo_Count, Advertiser_Type

**Error Handling**:

- Missing pipeline: "Model Pipeline not found..."
- Corrupt file: "File Reading Asset Corrupted..."
- Prediction success: Returns float price

### Data Persistence

**Metadata Storage**: `data/metadata.json`

**Structure**:

```json
{
  "dataset_1775390037": {
    "property_type": "flat",
    "cities": ["Beograd", "Novi Sad"],
    "records": 12543,
    "scraped_at": "2026-05-16 14:30:45",
    "models": {
      "price": {
        "best_model": "XGBoost",
        "r2": 0.8234,
        "property_type": "flat",
        "numerical_features": [13 features list],
        "categorical_features": [7 features list],
        "target_column": "Total_Price_EUR"
      },
      "price_per_m2": {
        "best_model": "LightGBM",
        "r2": 0.7891,
        ...
      }
    }
  }
}
```

**Files Created**:

- CSV: Raw scraped data (never modified)
- Pickle: Trained pipelines (binary, reproducible)
- JSON: Metadata (human-readable)

---

## Testing & Validation

### Test Suite Statistics

**Total Tests**: 74 (all passing)

- Coverage: 47% overall, 95% for critical modules

**Test Categories**:

#### 1. Preprocessing Tests (34 tests)

- Room parsing (5 tests)
- Floor parsing (8 tests)
- Text normalization (4 tests)
- Feature engineering (5 tests)
- Data cleaning (5 tests)
- Data preparation (3 tests)
- Preprocessor pipeline (4 tests)

#### 2. Training Tests (7 tests)

- Single model training
- Multiple models training
- Pipeline saving
- Metadata updates
- File not found error handling
- Custom hyperparameters
- Metric retrieval

#### 3. Prediction Tests (6 tests)

- Numeric output
- Reasonable predictions
- Model not found errors
- Different input variations
- Missing column handling
- Prediction consistency

#### 4. Data Validation Tests (10 tests)

- Negative price removal
- Zero price removal
- Negative area removal
- Zero area removal
- Missing data removal
- Data preservation
- Feature integrity
- Row count preservation
- Categorical encoding

#### 5. Property Type Tests (18 tests)

- Property-type-specific features
- Feature exclusion per type
- Configuration consistency
- Preprocessor detection
- Feature counts validation

---

## Configuration & Metadata

### Configuration File: `src/config.py`

**Sections**:

#### 1. Scraper Configuration

```python
URL_MAPPING = {
    "flat": "prodaja-stanova",
    "house": "prodaja-kuca",
    ...
}

CITIES = {
    "Beograd": "beograd",
    ...
}
```

#### 2. Feature Configuration

```python
NUMERICAL_FEATURES_BY_TYPE = {
    "flat": [13 features],
    "house": [8 features],
    ...
}

CATEGORICAL_FEATURES_BY_TYPE = {
    "flat": [7 features],
    ...
}

# Backward compatibility defaults
NUMERICAL_FEATURES = NUMERICAL_FEATURES_BY_TYPE["flat"]
CATEGORICAL_FEATURES = CATEGORICAL_FEATURES_BY_TYPE["flat"]
```

#### 3. Model Configuration

```python
MODEL_CLASSES = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(...),
    ...
}
```

#### 4. Hyperparameter Configuration

```python
HYPERPARAM_CONFIG = {
    "Random Forest": [
        {
            "name": "n_estimators",
            "label": "Number of Trees",
            "type": "slider",
            "min": 10,
            "max": 500,
            "default": 100,
            "step": 10,
        },
        ...
    ],
    ...
}
```

### Metadata Storage: `data/metadata.json`

**Contents**:

- Dataset basic info (type, cities, count)
- Scrape timestamp
- Trained models (name, R² score)
- Feature lists (for audit trail)
- Target variable

**Auto-Updated**: When training completes

**Uses**:

- UI sidebar displays dataset stats
- Training page shows property type
- Prediction ensures correct features used

---

## Workflow Examples

### Example 1: Predicting an Apartment Price

**Scenario**: User wants to estimate price of a 2-room, 70m², 3rd floor apartment in Belgrade

**Steps**:

1. Open UI → Go to "4. Prediction"
2. City: Beograd
3. Municipality: Voždovac
4. Neighborhood: Kumodraž
5. Area: 70
6. Rooms: 2
7. Current Floor: 3
8. Total Floors: 5
9. Photo Count: 5
10. Advertiser Type: Agency
11. Click "Calculate Valuation"

**Behind the Scenes**:

1. App converts to input dict
2. Loads best_pipeline model
3. Pipeline creates features:
   - Floor_Ratio = 3/5 = 0.6
   - Area_per_Room = 70/2 = 35
   - Rooms_per_100sqm = (2/70)\*100 = 2.86
   - Area_Log = log(70) = 4.25
   - Is_Ground_Floor = 0
   - Is_Top_Floor = 0
   - Floor_Category = "Mid"
   - Is_Agency = 1
4. Imputes missing values, scales numericals, one-hot encodes categoricals
5. Model predicts: €320,000
6. Per m²: €320,000 / 70 = €4,571
7. Market median in Kumodraž: €4,200
8. Difference: +8.8% above median
9. Displays: Result metrics + market distribution graph

### Example 2: Finding Best Model for Land Prices

**Scenario**: User has land listings, wants to build model

**Steps**:

1. Go to "1. Data & Scraping" → Scrape land (prodaja-zemljista)
2. Select cities: Beograd, Novi Sad, Niš
3. Pages per city: 10
4. Click "Start Scraping" → Wait 5 minutes
5. Go to "3. Model Training"
6. Target Variable: Total_Price_EUR
7. Select Models: Random Forest, XGBoost, LightGBM
8. Adjust hyperparameters if desired
9. Click "Train Selected Models"

**Behind the Scenes**:

1. Loads land dataset
2. Cleans, engineers features (area-based only for land)
3. Splits 80/20 train-test
4. Trains 3 models in parallel
5. Evaluates on test set
6. Results:
   - XGBoost: R² = 0.82, MAE = 15,000€
   - LightGBM: R² = 0.81, MAE = 16,200€
   - Random Forest: R² = 0.79, MAE = 18,500€
7. Winner: XGBoost
8. Saves: best_pipeline_dataset_xxx_price.pkl
9. Updates metadata.json with model info
10. User ready to make predictions

---

## Performance Characteristics

### Data Processing

- **Scraping**: 100-300 listings/page, 2-5 min per session
- **Cleaning**: Removes 20-40% outliers (typical)
- **Feature Engineering**: < 1 second per 100k rows

### Model Training

- **Dataset Sizes**: Typical 5k-15k properties
- **Training Time**: 30-60 seconds for 9 models
- **Typical Performance**:
  - Flats (best): R² = 0.80-0.88
  - Houses: R² = 0.75-0.85
  - Land: R² = 0.70-0.80
  - Garages: R² = 0.60-0.75

### Prediction

- **Speed**: < 100ms per property
- **Accuracy**: ±15-25% typical MAPE (Mean Absolute Percentage Error)

### UI Responsiveness

- Page loads: < 2 seconds
- Scraping: Real-time progress updates
- EDA plots: < 3 seconds to render all
- Training: Progress spinner, updates every few seconds

---

## Future Enhancements

1. **Property-Type-Specific Models**: Train separate models per type
2. **Time Series Analysis**: Track price trends over months
3. **Spatial Analysis**: Map-based visualization
4. **API Server**: Expose prediction as REST API
5. **Model Explainability**: SHAP values for feature importance
6. **Real-Time Notifications**: Alert on price changes
7. **Mobile App**: React Native or Flutter adaptation
8. **Database**: Replace CSV/JSON with PostgreSQL
9. **Containerization**: Docker deployment for production
10. **A/B Testing**: Compare model versions

---

## Development Guide

### Adding a New Feature

1. **Feature Calculation**: Add to `engineer_features()` in `preprocessing.py`
2. **Add to Config**: Include in `NUMERICAL_FEATURES_BY_TYPE` or `CATEGORICAL_FEATURES_BY_TYPE`
3. **Write Tests**: Add test in `test_property_types.py` or `test_preprocessing.py`
4. **Update UI**: Modify input form if user-configurable
5. **Document**: Update docstring and this README

### Adding a New Model

1. **Import Model**: Add to `src/config.py`
2. **Add to MODEL_CLASSES**: Dict with model instance
3. **Hyperparameters**: Add to `HYPERPARAM_CONFIG` (optional)
4. **Test**: Verify in training pipeline
5. **Document**: Add to config comments

### Debugging

- **Scraping Issues**: Check network, verify URL structure
- **Data Cleaning**: Review outliers in `load_and_clean_data()`
- **Feature Engineering**: Check for NaN creation in `engineer_features()`
- **Model Training**: Review preprocessing pipeline, check feature alignment
- **Prediction Errors**: Verify input data types, check for missing columns

---

## Glossary

| Term                    | Definition                                              |
| ----------------------- | ------------------------------------------------------- |
| **Pipeline**            | Scikit-learn Pipeline combining preprocessing and model |
| **Feature Engineering** | Creation of new features from raw data                  |
| **Property Type**       | Classification (flat, house, land, garage)              |
| **R² Score**            | Coefficient of determination, variance explained (0-1)  |
| **MAE**                 | Mean Absolute Error, average prediction error           |
| **IQR**                 | Interquartile Range, used for outlier detection         |
| **OneHotEncoding**      | Conversion of categorical to binary features            |
| **Imputation**          | Filling missing values (median/constant)                |
| **Scaling**             | Standardization to mean=0, std=1                        |
| **Metadata**            | Data about data (dataset info, model info)              |

---

This technical documentation covers the complete system from data collection through prediction. For user-facing instructions, see [README.md](README.md).

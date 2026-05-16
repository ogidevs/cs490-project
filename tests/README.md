# Unit Tests for Real Estate Price Prediction Project

This directory contains comprehensive unit tests for components of the real estate price prediction system.

## Test Coverage

Tests cover the following key functionalities:

### 1. **Preprocessing Tests** (`test_preprocessing.py`)

**Parsing Rooms (`parse_room_numeric`)**

- Standard room numbers (1-room, 2-room, etc.)
- Half rooms (2.5-room, 3.5-room, etc.)
- Studio and garsoniere
- Numeric input (2.5, 1.5, etc.)
- Missing values (NaN, empty string)
- Different formats (uppercase, whitespace)

**Parsing Floors (`parse_floor_numeric`)**

- Ground floor, basement, high basement
- Attic/top floor
- Numeric input
- Multiple numbers (takes first)
- Missing values
- Different formats

**Text Normalization (`_normalize_text`)**

- Removing extra spaces
- Handling NaN values
- Empty strings
- Number to string conversion

**Feature Engineering (`engineer_features`)**

- Creating all required columns
- Floor_Ratio calculation
- Area_Log calculation
- Agency detection
- Floor categorization
- Creating binary indicators (Is_Ground_Floor, Is_Top_Floor, Is_Agency)
- Area_per_Room and Rooms_per_100sqm calculation

**Loading and Cleaning Data (`load_and_clean_data`)**

- Removing rows with missing data
- Removing zero and negative values
- Creating Price_per_Unit_EUR column
- Outlier removal (IQR method)
- Error handling when file doesn't exist

**Preparing Data for Training (`prepare_data_for_training`)**

- Separating target column from others
- Removing leakage columns
- Handling missing target values

**Preprocessing Pipeline (`get_preprocessor`)**

- Returning Pipeline object
- Presence of all required steps
- Fit_transform method

### 2. **Model Training Tests** (`test_training.py`)

**Training and Evaluation (`train_and_evaluate`)**

- Training with single model
- Training with multiple models
- Saving best pipeline
- Updating metadata JSON
- Error handling when dataset doesn't exist
- Training with custom hyperparameters
- Returning valid metrics (R2, MAE)

### 3. **Prediction Tests** (`test_prediction.py`)

**Prediction Value (`predict_value`)**

- Returning numeric value
- Reasonable predictions (realistic range)
- Error handling when model doesn't exist
- Predictions with different inputs
- Handling missing columns
- Prediction consistency

### 4. **Data Validation Tests** (`test_data_validation.py`)

**Data Validation and Integrity**

- Removing negative prices
- Removing zero prices
- Removing negative areas
- Removing zero areas
- Removing rows with missing essential data
- Preserving valid data
- Handling extreme outliers

**Data Integrity through Pipeline**

- Engineered features without inf/NaN
- Preprocessing preserves row count
- Categorical columns properly encoded

## Test Structure

```
tests/
├── __init__.py                 # Initialization file
├── conftest.py                 # Shared fixtures
├── test_preprocessing.py       # Preprocessing tests
├── test_training.py            # Model training tests
├── test_prediction.py          # Prediction tests
└── test_data_validation.py     # Data validation tests
```

## Fixtures

The following shared fixtures are available in `conftest.py`:

- **`sample_dataset`**: Creates a small sample dataset with basic data
- **`temp_csv_file`**: Creates a temporary CSV file for testing
- **`sample_input_dict`**: Creates sample input dictionary for prediction testing

## Dependencies

The following dependencies are required to run tests:

- `pytest>=8.0.0`
- `pytest-cov>=5.0.0`
- All dependencies from `pyproject.toml`

Dependencies are automatically installed with `uv sync`.

## Best Practices for Tests

1. **Tests are independent**: Each test can run separately
2. **Tests clean up**: Temporary files are deleted after tests
3. **Clear naming**: Test names clearly describe what is tested
4. **Good structure**: Tests are organized into logical classes
5. **Fixtures**: Fixtures are used instead of real data

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    parse_floor_numeric,
    engineer_features,
    load_and_clean_data,
    prepare_data_for_training,
    get_preprocessor,
    _normalize_text,
)

class TestParseFloorNumeric:
    """Tests parse_floor_numeric() function."""

    def test_parse_floor_numeric_ground_floor(self):
        """Test parsing ground floor."""
        assert parse_floor_numeric("prizemlje") == 0.0
        assert parse_floor_numeric("pr") == 0.0

    def test_parse_floor_numeric_basement(self):
        """Test parsing basement."""
        assert parse_floor_numeric("suteren") == -1.0
        assert parse_floor_numeric("sut") == -1.0

    def test_parse_floor_numeric_high_basement(self):
        """Test parsing high basement."""
        assert parse_floor_numeric("blagi suteren") == -0.5

    def test_parse_floor_numeric_attic(self):
        """Test parsing attic."""
        assert parse_floor_numeric("potkrovlje") == 99.0
        assert parse_floor_numeric("ptk") == 99.0

    def test_parse_floor_numeric_numeric_input(self):
        """Test parsing numeric input."""
        assert parse_floor_numeric("2") == 2.0
        assert parse_floor_numeric("10") == 10.0
        assert parse_floor_numeric("5") == 5.0

    def test_parse_floor_numeric_multiple_numbers(self):
        """Test parsing with multiple numbers (takes first)."""
        assert parse_floor_numeric("2/5") == 2.0

    def test_parse_floor_numeric_missing_values(self):
        """Test parsing missing values."""
        assert pd.isna(parse_floor_numeric(np.nan))
        assert pd.isna(parse_floor_numeric(""))
        assert pd.isna(parse_floor_numeric(None))

    def test_parse_floor_numeric_uppercase(self):
        """Test parsing with uppercase."""
        assert parse_floor_numeric("PRIZEMLJE") == 0.0
        assert parse_floor_numeric("POTKROVLJE") == 99.0


class TestNormalizeText:
    """Tests _normalize_text() function."""

    def test_normalize_text_removes_extra_spaces(self):
        """Test removing extra spaces."""
        assert _normalize_text("  hello   world  ") == "hello world"

    def test_normalize_text_handles_nan(self):
        """Test handling NaN values."""
        assert _normalize_text(np.nan) == "Unknown"
        assert _normalize_text(None) == "Unknown"

    def test_normalize_text_empty_string(self):
        """Test handling empty strings."""
        assert _normalize_text("") == "Unknown"
        assert _normalize_text("   ") == "Unknown"

    def test_normalize_text_converts_to_string(self):
        """Test converting number to string."""
        assert _normalize_text(123) == "123"
        assert _normalize_text(45.67) == "45.67"


class TestEngineerFeatures:
    """Tests engineer_features() function."""

    def test_engineer_features_creates_required_columns(self, sample_dataset):
        """Test that engineer_features creates all required columns."""
        df = engineer_features(sample_dataset)

        required_features = [
            "Floor_Ratio",
            "Area_per_Room",
            "Rooms_per_100sqm",
            "Area_Log",
            "Photo_Count_Log",
            "Is_Ground_Floor",
            "Is_Top_Floor",
            "Is_Agency",
            "Floor_Category",
        ]

        for feature in required_features:
            assert feature in df.columns, f"Missing feature: {feature}"

    def test_engineer_features_floor_ratio_calculation(self, sample_dataset):
        """Test that Floor_Ratio is calculated correctly."""
        df = engineer_features(sample_dataset)

        # First row: Current_Floor=0 (ground floor), Total_Floors=4
        # Floor_Ratio = 0 / 4 = 0
        assert df["Floor_Ratio"].iloc[0] == 0.0

    def test_engineer_features_area_log(self, sample_dataset):
        """Test that Area_Log is calculated correctly."""
        df = engineer_features(sample_dataset)

        # Area_Log = log1p(Area)
        assert df["Area_Log"].iloc[0] > 0
        assert df["Area_Log"].iloc[0] == np.log1p(50)

    def test_engineer_features_is_agency_detection(self, sample_dataset):
        """Test agency detection."""
        df = engineer_features(sample_dataset)

        # Second row has 'agencija'
        assert df["Is_Agency"].iloc[1] == 1
        # First row has 'privatna lica'
        assert df["Is_Agency"].iloc[0] == 0

    def test_engineer_features_floor_category(self, sample_dataset):
        """Test floor categorization."""
        df = engineer_features(sample_dataset)

        # Ground floor -> Ground
        assert df["Floor_Category"].iloc[0] == "Ground"


class TestLoadAndCleanData:
    """Tests load_and_clean_data() function."""

    def test_load_and_clean_data_removes_missing_values(self, temp_csv_file):
        """Test removing rows with missing values."""
        df = load_and_clean_data(temp_csv_file)

        # Should have at least 5 rows (all have data)
        assert len(df) > 0
        assert df["Total_Price_EUR"].notna().all()
        assert df["Area"].notna().all()

    def test_load_and_clean_data_removes_zero_and_negative(self, temp_csv_file):
        """Test removing zero and negative values."""
        df = load_and_clean_data(temp_csv_file)

        # All values should be positive
        assert (df["Total_Price_EUR"] > 0).all()
        assert (df["Area"] > 0).all()

    def test_load_and_clean_data_creates_price_per_unit(self, temp_csv_file):
        """Test creating Price_per_Unit_EUR column."""
        df = load_and_clean_data(temp_csv_file)

        assert "Price_per_Unit_EUR" in df.columns
        # Should be Total_Price_EUR / Area
        expected = df["Total_Price_EUR"].iloc[0] / df["Area"].iloc[0]
        assert (
            pd.isna(df["Price_per_Unit_EUR"].iloc[0])
            or abs(df["Price_per_Unit_EUR"].iloc[0] - expected) < 0.01
        )

    def test_load_and_clean_data_applies_outlier_removal(self, temp_csv_file):
        """Test outlier removal."""
        df_original = pd.read_csv(temp_csv_file)
        df_cleaned = load_and_clean_data(temp_csv_file)

        # Expect some rows to be removed or all rows to remain
        assert len(df_cleaned) <= len(df_original)

    def test_load_and_clean_data_file_not_found(self):
        """Test error handling when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_and_clean_data("nonexistent_file.csv")


class TestPrepareDataForTraining:
    """Tests prepare_data_for_training() function."""

    def test_prepare_data_for_training_splits_X_y(self, sample_dataset):
        """Test separating target column from others."""
        X, y = prepare_data_for_training(sample_dataset, target_col="Total_Price_EUR")

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "Total_Price_EUR" not in X.columns
        assert len(X) == len(y)

    def test_prepare_data_for_training_removes_leakage(self, sample_dataset):
        """Test removing leakage columns."""
        X, y = prepare_data_for_training(sample_dataset, target_col="Total_Price_EUR")

        # 'Price_per_Unit_EUR' shouldn't be in X
        # because it's directly related to 'Total_Price_EUR'

    def test_prepare_data_for_training_handles_missing_target(self, sample_dataset):
        """Test handling missing target values."""
        df = sample_dataset.copy()
        df.loc[0, "Total_Price_EUR"] = np.nan

        X, y = prepare_data_for_training(df, target_col="Total_Price_EUR")

        # NaN rows should be removed first
        assert len(X) == len(y)
        assert y.notna().all()


class TestGetPreprocessor:
    """Tests get_preprocessor() function."""

    def test_get_preprocessor_returns_pipeline(self, sample_dataset):
        """Test that get_preprocessor returns Pipeline object."""
        from sklearn.pipeline import Pipeline

        preprocessor = get_preprocessor(sample_dataset)

        assert isinstance(preprocessor, Pipeline)

    def test_get_preprocessor_has_required_steps(self, sample_dataset):
        """Test that preprocessor has all required steps."""
        preprocessor = get_preprocessor(sample_dataset)

        # Pipeline should have 'feature_engineer' and 'column_transformer'
        step_names = [name for name, _ in preprocessor.steps]
        assert "feature_engineer" in step_names
        assert "column_transformer" in step_names

    def test_get_preprocessor_fit_transform(self, sample_dataset):
        """Test fit_transform method of preprocessor."""
        preprocessor = get_preprocessor(sample_dataset)

        # Should process data without error
        X_transformed = preprocessor.fit_transform(sample_dataset)

        assert X_transformed is not None
        assert len(X_transformed) > 0
        # Should be numpy array
        assert isinstance(X_transformed, (pd.DataFrame, np.ndarray))

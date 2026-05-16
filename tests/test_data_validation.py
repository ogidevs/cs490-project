import pytest
import pandas as pd
import numpy as np
from src.preprocessing import load_and_clean_data


class TestDataValidation:
    """Tests data validation."""

    @pytest.fixture
    def create_test_data_with_issues(self, tmp_path):
        """Create dataset with various issues for testing."""
        data = {
            "Total_Price_EUR": [
                50000,
                75000,
                -100000,
                120000,
                0,
                95000,
                np.nan,
            ],  # Has negative and NaN
            "Area": [50, 60, 80, -100, 0, 75, np.nan],  # Has negative and NaN
            "Rooms": [
                "dvosoban",
                "trosoban",
                "troiposoban",
                "četvorosoban",
                "dvoiposoban",
                np.nan,
                "jednosoban",
            ],
            "Current_Floor": ["prizemlje", "2", "5", "10", "3", np.nan, "7"],
            "Total_Floors": ["4", "5", "8", "12", "6", "8", np.nan],
            "City": [
                "Beograd",
                "Beograd",
                "Novi Sad",
                "Beograd",
                "Novi Sad",
                "Niš",
                "Beograd",
            ],
            "Municipality": [
                "Vračar",
                "Kumudžija",
                "Grbavica",
                "Voždovac",
                "Mueška",
                np.nan,
                "Palilula",
            ],
            "Neighborhood": [
                "Pozitivac",
                "Čukurica",
                "Grbavica",
                "Autokomanda",
                "Pobeda",
                "Keramičar",
                np.nan,
            ],
            "Advertiser_Type": [
                "privatna lica",
                "agencija",
                "privatna lica",
                "posrednik",
                "privatna lica",
                "agencija",
                "privatna lica",
            ],
            "Property_Type": ["flat", "flat", "flat", "flat", "flat", "house", "flat"],
            "Property_Subtype": [
                "apartment",
                "apartment",
                "apartment",
                "apartment",
                "apartment",
                "house",
                "apartment",
            ],
            "Photo_Count": [5, 8, 12, 15, 10, np.nan, 20],
        }
        df = pd.DataFrame(data)

        csv_path = tmp_path / "test_data_with_issues.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_load_and_clean_removes_negative_prices(self, create_test_data_with_issues):
        """Test that negative prices are removed."""
        df = load_and_clean_data(create_test_data_with_issues)

        # All prices should be positive
        assert (df["Total_Price_EUR"] > 0).all()

    def test_load_and_clean_removes_zero_prices(self, create_test_data_with_issues):
        """Test that zero prices are removed."""
        df = load_and_clean_data(create_test_data_with_issues)

        # No zero prices
        assert (df["Total_Price_EUR"] != 0).all()

    def test_load_and_clean_removes_negative_areas(self, create_test_data_with_issues):
        """Test that negative areas are removed."""
        df = load_and_clean_data(create_test_data_with_issues)

        # All areas should be positive
        assert (df["Area"] > 0).all()

    def test_load_and_clean_removes_zero_areas(self, create_test_data_with_issues):
        """Test that zero areas are removed."""
        df = load_and_clean_data(create_test_data_with_issues)

        # No zero areas
        assert (df["Area"] != 0).all()

    def test_load_and_clean_removes_missing_essential_data(
        self, create_test_data_with_issues
    ):
        """Test that rows with missing essential data are removed."""
        df = load_and_clean_data(create_test_data_with_issues)

        # No NaN in Total_Price_EUR and Area
        assert df["Total_Price_EUR"].notna().all()
        assert df["Area"].notna().all()

    def test_load_and_clean_preserves_valid_data(self, sample_dataset):
        """Test that valid data is preserved."""
        # Create CSV from sample dataset
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            sample_dataset.to_csv(f, index=False, encoding="utf-8")
            temp_path = f.name

        try:
            df = load_and_clean_data(temp_path)

            # Should have more than 0 rows remaining
            assert len(df) > 0
        finally:
            import os

            os.unlink(temp_path)


class TestDataIntegrity:
    """Tests data integrity through pipeline."""

    def test_engineered_features_no_inf_or_nan(self, sample_dataset):
        """Test that engineered features don't contain inf or only NaN."""
        from src.preprocessing import engineer_features

        df = engineer_features(sample_dataset)

        # Proveravamo inf vrednosti
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Dozvoljavamo NaN, ali ne inf
            assert not (df[col].isin([np.inf, -np.inf]).any()), (
                f"Kolona {col} sadrži inf"
            )

    def test_preprocessor_maintains_row_count(self, sample_dataset):
        """Test da preprocessor čuva broj redova."""
        from src.preprocessing import get_preprocessor

        preprocessor = get_preprocessor(sample_dataset)
        X_transformed = preprocessor.fit_transform(sample_dataset)

        # Trebalo bi da bude isti broj redova
        if isinstance(X_transformed, np.ndarray):
            assert X_transformed.shape[0] == len(sample_dataset)
        else:
            assert len(X_transformed) == len(sample_dataset)

    def test_categorical_features_properly_encoded(self, sample_dataset):
        """Test da se kategorijske kolone pravilno kodiraju."""
        from src.preprocessing import get_preprocessor

        preprocessor = get_preprocessor(sample_dataset)
        X_transformed = preprocessor.fit_transform(sample_dataset)

        # Rešenje bi trebalo da bude numeričko
        assert isinstance(X_transformed, (np.ndarray, pd.DataFrame))

        # Sve vrednosti trebalo bi da budu numeričke
        if isinstance(X_transformed, pd.DataFrame):
            assert X_transformed.select_dtypes(include=[np.number]).shape[1] > 0

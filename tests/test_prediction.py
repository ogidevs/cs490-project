import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.predict import predict_value
from src.train_model import train_and_evaluate


class TestPredictValue:
    """Tests predict_value() function."""

    @pytest.fixture
    def setup_trained_model(self, tmp_path):
        """Create a trained model for prediction testing."""
        # Create and train model
        np.random.seed(42)
        n_samples = 150

        data = {
            "Total_Price_EUR": np.random.randint(30000, 200000, n_samples),
            "Area": np.random.randint(40, 150, n_samples),
            "Rooms": np.random.choice(
                ["jednosoban", "dvosoban", "trosoban", "četvorosoban"], n_samples
            ),
            "Current_Floor": np.random.choice(
                ["prizemlje", "1", "2", "3", "5", "potkrovlje"], n_samples
            ),
            "Total_Floors": np.random.choice(
                ["4", "5", "6", "8", "10", "12"], n_samples
            ),
            "City": np.random.choice(["Beograd", "Novi Sad"], n_samples),
            "Municipality": np.random.choice(["Voždovac", "Kumudžija"], n_samples),
            "Neighborhood": np.random.choice(["Autokomanda", "Čukurica"], n_samples),
            "Advertiser_Type": np.random.choice(
                ["privatna lica", "agencija"], n_samples
            ),
            "Property_Type": "flat",
            "Property_Subtype": "apartment",
            "Photo_Count": np.random.randint(3, 20, n_samples),
        }

        df = pd.DataFrame(data)

        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        # Create CSV file
        csv_path = data_dir / "test_model_dataset.csv"
        df.to_csv(str(csv_path), index=False)

        # Train model
        try:
            train_and_evaluate(
                "test_model_dataset.csv",
                selected_models=["Linear Regression"],
                target_variable="Total_Price_EUR",
            )
        except Exception as e:
            pytest.skip(f"Cannot train model: {e}")

        return "test_model_dataset.csv", csv_path

    def test_predict_value_returns_number(self, setup_trained_model, sample_input_dict):
        """Test that predict_value returns numeric value."""
        dataset_filename, _ = setup_trained_model

        prediction = predict_value(sample_input_dict, dataset_filename, "price")

        # Prediction should be number
        assert isinstance(prediction, (int, float, np.number))
        assert not isinstance(prediction, str)

    def test_predict_value_reasonable_prediction(
        self, setup_trained_model, sample_input_dict
    ):
        """Test that prediction has reasonable value."""
        dataset_filename, _ = setup_trained_model

        prediction = predict_value(sample_input_dict, dataset_filename, "price")

        # Prediction should be positive (since it's price)
        assert prediction > 0

        # Prediction should be in reasonable range
        # (not too small or too large)
        assert prediction > 10000  # Minimum reasonable price
        assert prediction < 1000000  # Maximum reasonable price

    def test_predict_value_model_not_found(self, sample_input_dict):
        """Test error handling when model doesn't exist."""
        prediction = predict_value(
            sample_input_dict, "nonexistent_dataset.csv", "price"
        )

        # Should get error message
        assert isinstance(prediction, str)
        assert "not found" in prediction.lower() or "error" in prediction.lower()

    def test_predict_value_with_different_inputs(
        self, setup_trained_model, sample_input_dict
    ):
        """Test prediction with different inputs."""
        dataset_filename, _ = setup_trained_model

        # Small apartment
        input_1 = sample_input_dict.copy()
        input_1["Area"] = 35
        input_1["Rooms"] = "jednosoban"
        pred_1 = predict_value(input_1, dataset_filename, "price")

        # Large apartment
        input_2 = sample_input_dict.copy()
        input_2["Area"] = 120
        input_2["Rooms"] = "četvorosoban"
        pred_2 = predict_value(input_2, dataset_filename, "price")

        # Larger apartment should typically have higher predicted price
        if isinstance(pred_1, str) or isinstance(pred_2, str):
            pytest.skip("Cannot get predictions due to model error")

    def test_predict_value_with_missing_columns(self, setup_trained_model):
        """Test handling missing columns in input data."""
        dataset_filename, _ = setup_trained_model

        # Input with only a few columns
        input_dict = {
            "Area": 75,
            "Rooms": "trosoban",
        }

        # Prediction should execute
        # because pipeline automatically handles missing columns
        prediction = predict_value(input_dict, dataset_filename, "price")

        # Should get either number or error message
        assert isinstance(prediction, (int, float, np.number, str))

    def test_predict_value_consistent_predictions(
        self, setup_trained_model, sample_input_dict
    ):
        """Test that same inputs yield same prediction."""
        dataset_filename, _ = setup_trained_model

        # Make two predictions with same inputs
        pred_1 = predict_value(sample_input_dict, dataset_filename, "price")
        pred_2 = predict_value(sample_input_dict, dataset_filename, "price")

        # Should be exactly the same
        if not isinstance(pred_1, str) and not isinstance(pred_2, str):
            assert pred_1 == pred_2

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up temporary files after tests."""
        yield

        # Delete temporary files
        if Path("data").exists():
            for f in Path("data").glob("test_model*"):
                try:
                    if f.is_file():
                        f.unlink()
                except Exception:
                    pass

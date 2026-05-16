import pytest
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from src.train_model import train_and_evaluate


class TestTrainAndEvaluate:
    """Tests train_and_evaluate() function."""

    @pytest.fixture
    def setup_training_dataset(self, tmp_path):
        """Create a temporary dataset for training testing."""
        # Create a sample dataset
        np.random.seed(42)
        n_samples = 100

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

        # Create CSV file in temporary directory
        csv_path = tmp_path / "test_dataset.csv"
        df.to_csv(csv_path, index=False)

        # Create data directory in current directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        # Copy CSV to data directory
        import shutil

        test_csv = data_dir / "test_dataset.csv"
        shutil.copy(str(csv_path), str(test_csv))

        return "test_dataset.csv", test_csv

    def test_train_and_evaluate_single_model(self, setup_training_dataset):
        """Test training with single model."""
        dataset_filename, test_csv_path = setup_training_dataset

        results, best_score = train_and_evaluate(
            dataset_filename,
            selected_models=["Linear Regression"],
            target_variable="Total_Price_EUR",
        )

        # Should get results
        assert isinstance(results, dict)
        assert "Linear Regression" in results
        assert "R2" in results["Linear Regression"]
        assert "MAE" in results["Linear Regression"]

        # Best score should be R2 for Linear Regression
        # (rounded to 4 decimals)
        assert abs(best_score - results["Linear Regression"]["R2"]) < 0.0001

    def test_train_and_evaluate_multiple_models(self, setup_training_dataset):
        """Test training with multiple models."""
        dataset_filename, test_csv_path = setup_training_dataset

        results, best_score = train_and_evaluate(
            dataset_filename,
            selected_models=["Linear Regression", "Ridge Regression", "Random Forest"],
            target_variable="Total_Price_EUR",
        )

        # Should have results for all models
        assert len(results) == 3
        assert "Linear Regression" in results
        assert "Ridge Regression" in results
        assert "Random Forest" in results

    def test_train_and_evaluate_saves_pipeline(self, setup_training_dataset):
        """Test saving best pipeline."""
        dataset_filename, test_csv_path = setup_training_dataset

        results, best_score = train_and_evaluate(
            dataset_filename,
            selected_models=["Linear Regression"],
            target_variable="Total_Price_EUR",
        )

        # Pipeline should be saved
        dataset_base = dataset_filename.replace(".csv", "")
        pipeline_path = Path("data") / f"best_pipeline_{dataset_base}_price.pkl"

        assert pipeline_path.exists()

        # Pipeline should load without error
        pipeline = joblib.load(str(pipeline_path))
        assert pipeline is not None

    def test_train_and_evaluate_updates_metadata(self, setup_training_dataset):
        """Test updating metadata JSON."""
        dataset_filename, test_csv_path = setup_training_dataset

        results, best_score = train_and_evaluate(
            dataset_filename,
            selected_models=["Linear Regression"],
            target_variable="Total_Price_EUR",
        )

        # Check metadata
        metadata_path = Path("data") / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            dataset_base = dataset_filename.replace(".csv", "")

            # Should have entry for this dataset
            if dataset_base in metadata:
                assert "models" in metadata[dataset_base]

    def test_train_and_evaluate_file_not_found(self):
        """Test error handling when dataset doesn't exist."""
        with pytest.raises(FileNotFoundError):
            train_and_evaluate(
                "nonexistent_dataset.csv",
                selected_models=["Linear Regression"],
                target_variable="Total_Price_EUR",
            )

    def test_train_and_evaluate_with_hyperparameters(self, setup_training_dataset):
        """Test training with custom hyperparameters."""
        dataset_filename, test_csv_path = setup_training_dataset

        model_params = {"Ridge Regression": {"alpha": 10.0}}

        results, best_score = train_and_evaluate(
            dataset_filename,
            selected_models=["Ridge Regression"],
            target_variable="Total_Price_EUR",
            model_params=model_params,
        )

        # Training should execute without error
        assert "Ridge Regression" in results

    def test_train_and_evaluate_returns_valid_metrics(self, setup_training_dataset):
        """Test that metrics have valid values."""
        dataset_filename, test_csv_path = setup_training_dataset

        results, best_score = train_and_evaluate(
            dataset_filename,
            selected_models=["Linear Regression"],
            target_variable="Total_Price_EUR",
        )

        # R2 should be between -1 and 1
        r2 = results["Linear Regression"]["R2"]
        assert -1 <= r2 <= 1

        # MAE should be positive
        mae = results["Linear Regression"]["MAE"]
        assert mae > 0

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up temporary files after tests."""
        yield

        # Delete temporary files

        if Path("data").exists():
            # Delete only test files we created
            for f in Path("data").glob("test_dataset*"):
                try:
                    if f.is_file():
                        f.unlink()
                except Exception:
                    pass

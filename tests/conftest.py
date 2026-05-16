import pytest
import pandas as pd


@pytest.fixture
def sample_dataset():
    """Create a small sample dataset for testing."""
    data = {
        "Total_Price_EUR": [50000, 75000, 100000, 120000, 95000],
        "Area": [50, 60, 80, 100, 75],
        "Rooms": ["dvosoban", "trosoban", "troiposoban", "četvorosoban", "dvoiposoban"],
        "Current_Floor": ["prizemlje", "2", "5", "10", "3"],
        "Total_Floors": ["4", "5", "8", "12", "6"],
        "City": ["Beograd", "Beograd", "Novi Sad", "Beograd", "Novi Sad"],
        "Municipality": ["Vračar", "Kumudžija", "Grbavica", "Voždovac", "Mueška"],
        "Neighborhood": ["Pozitivac", "Čukurica", "Grbavica", "Autokomanda", "Pobeda"],
        "Advertiser_Type": [
            "privatna lica",
            "agencija",
            "privatna lica",
            "posrednik",
            "privatna lica",
        ],
        "Property_Type": ["flat", "flat", "flat", "flat", "flat"],
        "Property_Subtype": [
            "apartment",
            "apartment",
            "apartment",
            "apartment",
            "apartment",
        ],
        "Photo_Count": [5, 8, 12, 15, 10],
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(sample_dataset, tmp_path):
    """Create a temporary CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    sample_dataset.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_input_dict():
    """Create sample input dictionary for prediction testing."""
    return {
        "Area": 75,
        "Rooms": "trosoban",
        "Current_Floor": "3",
        "Total_Floors": "6",
        "City": "Beograd",
        "Municipality": "Voždovac",
        "Neighborhood": "Autokomanda",
        "Advertiser_Type": "privatna lica",
        "Property_Type": "flat",
        "Property_Subtype": "apartment",
        "Photo_Count": 10,
    }

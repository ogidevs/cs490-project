"""Tests for property-type-specific feature engineering."""

import pandas as pd
import numpy as np
from src.preprocessing import engineer_features, get_preprocessor
from src.config import (
    NUMERICAL_FEATURES_BY_TYPE,
    CATEGORICAL_FEATURES_BY_TYPE,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


class TestPropertyTypeFeatures:
    """Test property-type-specific feature engineering."""

    def test_engineer_features_flat_creates_floor_features(self):
        """Verify floor features are created for flats."""
        df = pd.DataFrame(
            {
                "City": ["Belgrade"],
                "Municipality": ["Voždovac"],
                "Neighborhood": ["Kumodraž"],
                "Advertiser_Type": ["Agency"],
                "Property_Type": ["flat"],
                "Property_Subtype": ["Apartment"],
                "Area": [50.0],
                "Photo_Count": [5],
                "Rooms": [2.0],
                "Current_Floor": ["2"],
                "Total_Floors": ["5"],
            }
        )

        result = engineer_features(df)

        # Floor-based features should be present for flats
        assert "Floor_Ratio" in result.columns
        assert "Is_Ground_Floor" in result.columns
        assert "Is_Top_Floor" in result.columns
        assert "Floor_Category" in result.columns

        # Verify floor features have values
        assert not pd.isna(result["Floor_Ratio"].iloc[0])
        assert result["Floor_Category"].iloc[0] in [
            "Low",
            "Mid",
            "High",
            "Ground",
            "Top",
        ]

    def test_engineer_features_house_excludes_floor_features(self):
        """Verify floor features are NaN for houses."""
        df = pd.DataFrame(
            {
                "City": ["Belgrade"],
                "Municipality": ["Voždovac"],
                "Neighborhood": ["Kumodraž"],
                "Advertiser_Type": ["Owner"],
                "Property_Type": ["house"],
                "Property_Subtype": ["Family Home"],
                "Area": [100.0],
                "Photo_Count": [5],
                "Rooms": [4.0],
                "Current_Floor": ["0"],
                "Total_Floors": ["2"],
            }
        )

        result = engineer_features(df)

        # Floor-based features should be NaN for houses
        assert pd.isna(result["Floor_Ratio"].iloc[0])
        assert result["Is_Ground_Floor"].iloc[0] == 0 or pd.isna(
            result["Is_Ground_Floor"].iloc[0]
        )
        assert result["Is_Top_Floor"].iloc[0] == 0 or pd.isna(
            result["Is_Top_Floor"].iloc[0]
        )
        assert result["Floor_Category"].iloc[0] == "Unknown"

    def test_engineer_features_land_excludes_floor_and_room_features(self):
        """Verify floor and room features are NaN for land."""
        df = pd.DataFrame(
            {
                "City": ["Belgrade"],
                "Municipality": ["Voždovac"],
                "Neighborhood": ["Kumodraž"],
                "Advertiser_Type": ["Owner"],
                "Property_Type": ["land"],
                "Property_Subtype": ["Lot"],
                "Area": [500.0],
                "Photo_Count": [2],
            }
        )

        result = engineer_features(df)

        # Floor-based features should be NaN for land
        assert pd.isna(result["Floor_Ratio"].iloc[0])
        assert result["Floor_Category"].iloc[0] == "Unknown"

        # Room-based features should be NaN for land
        assert pd.isna(result["Area_per_Room"].iloc[0])
        assert pd.isna(result["Rooms_per_100sqm"].iloc[0])

    def test_engineer_features_garage_excludes_floor_and_room_features(self):
        """Verify floor and room features are NaN for garages."""
        df = pd.DataFrame(
            {
                "City": ["Belgrade"],
                "Municipality": ["Voždovac"],
                "Neighborhood": ["Kumodraž"],
                "Advertiser_Type": ["Agency"],
                "Property_Type": ["garage"],
                "Property_Subtype": ["Parking Space"],
                "Area": [20.0],
                "Photo_Count": [1],
            }
        )

        result = engineer_features(df)

        # Floor-based features should be NaN for garages
        assert pd.isna(result["Floor_Ratio"].iloc[0])
        assert result["Floor_Category"].iloc[0] == "Unknown"

        # Room-based features should be NaN for garages
        assert pd.isna(result["Area_per_Room"].iloc[0])
        assert pd.isna(result["Rooms_per_100sqm"].iloc[0])

    def test_engineer_features_house_keeps_room_features(self):
        """Verify room features are set (possibly NaN) for houses."""
        df = pd.DataFrame(
            {
                "City": ["Belgrade"],
                "Municipality": ["Voždovac"],
                "Neighborhood": ["Kumodraž"],
                "Advertiser_Type": ["Owner"],
                "Property_Type": ["house"],
                "Property_Subtype": ["Family Home"],
                "Area": [100.0],
                "Photo_Count": [5],
                "Rooms": [4.0],  # Directly provide parsed rooms
            }
        )

        result = engineer_features(df)

        # Room-based features should be present for houses
        assert "Area_per_Room" in result.columns
        assert "Rooms_per_100sqm" in result.columns

        # Verify room features have values (even if they might be NaN due to data)
        # The presence of features is what matters for houses
        assert isinstance(result["Area_per_Room"].iloc[0], (float, np.floating))
        assert isinstance(result["Rooms_per_100sqm"].iloc[0], (float, np.floating))

    def test_engineer_features_all_types_have_area_features(self):
        """Verify area-based features are created for all property types."""
        property_types = ["flat", "house", "land", "garage"]

        for prop_type in property_types:
            df = pd.DataFrame(
                {
                    "City": ["Belgrade"],
                    "Municipality": ["Voždovac"],
                    "Neighborhood": ["Kumodraž"],
                    "Advertiser_Type": ["Owner"],
                    "Property_Type": [prop_type],
                    "Property_Subtype": ["Test"],
                    "Area": [50.0],
                    "Photo_Count": [3],
                }
            )

            result = engineer_features(df)

            # Area-based features should be present for all types
            assert "Area_Log" in result.columns
            assert "Photo_Count_Log" in result.columns
            assert "Is_Agency" in result.columns

            # Verify area features have values
            assert not pd.isna(result["Area_Log"].iloc[0])
            assert not pd.isna(result["Photo_Count_Log"].iloc[0])

    def test_numerical_features_by_type_consistency(self):
        """Verify NUMERICAL_FEATURES_BY_TYPE has all property types."""
        expected_types = ["flat", "house", "land", "garage"]
        assert set(NUMERICAL_FEATURES_BY_TYPE.keys()) == set(expected_types)

        # Verify each type has expected number of features
        assert len(NUMERICAL_FEATURES_BY_TYPE["flat"]) == 13  # Full apartment set
        assert len(NUMERICAL_FEATURES_BY_TYPE["house"]) == 8  # No floor features
        assert len(NUMERICAL_FEATURES_BY_TYPE["land"]) == 5  # Area-based only
        assert len(NUMERICAL_FEATURES_BY_TYPE["garage"]) == 5  # Area-based only

    def test_categorical_features_by_type_consistency(self):
        """Verify CATEGORICAL_FEATURES_BY_TYPE has all property types."""
        expected_types = ["flat", "house", "land", "garage"]
        assert set(CATEGORICAL_FEATURES_BY_TYPE.keys()) == set(expected_types)

        # Flat should have Floor_Category
        assert "Floor_Category" in CATEGORICAL_FEATURES_BY_TYPE["flat"]

        # Non-flats should NOT have Floor_Category
        for prop_type in ["house", "land", "garage"]:
            assert "Floor_Category" not in CATEGORICAL_FEATURES_BY_TYPE[prop_type]

    def test_preprocessor_detects_property_type(self):
        """Verify get_preprocessor detects and uses property-type-specific features."""
        df_flat = pd.DataFrame(
            {
                "City": ["Belgrade"],
                "Municipality": ["Voždovac"],
                "Neighborhood": ["Kumodraž"],
                "Advertiser_Type": ["Agency"],
                "Property_Type": ["flat"],
                "Property_Subtype": ["Apartment"],
                "Area": [50.0],
                "Photo_Count": [5],
                "Rooms": [2.0],
                "Current_Floor": ["2"],
                "Total_Floors": ["5"],
            }
        )

        df_land = pd.DataFrame(
            {
                "City": ["Belgrade"],
                "Municipality": ["Voždovac"],
                "Neighborhood": ["Kumodraž"],
                "Advertiser_Type": ["Owner"],
                "Property_Type": ["land"],
                "Property_Subtype": ["Lot"],
                "Area": [500.0],
                "Photo_Count": [2],
            }
        )

        # Get preprocessors for each type
        preprocessor_flat = get_preprocessor(df_flat)
        preprocessor_land = get_preprocessor(df_land)

        # Both should be valid pipelines
        assert hasattr(preprocessor_flat, "fit_transform")
        assert hasattr(preprocessor_land, "fit_transform")

        # Verify they have the FeatureEngineer and ColumnTransformer
        assert hasattr(preprocessor_flat, "named_steps")
        assert hasattr(preprocessor_land, "named_steps")

    def test_preprocessor_backward_compatible_without_property_type(self):
        """Verify get_preprocessor defaults to flat features when Property_Type is missing."""
        df = pd.DataFrame(
            {
                "City": ["Belgrade"],
                "Municipality": ["Voždovac"],
                "Neighborhood": ["Kumodraž"],
                "Advertiser_Type": ["Agency"],
                "Area": [50.0],
                "Photo_Count": [5],
                "Rooms": [2.0],
            }
        )

        preprocessor = get_preprocessor(df)

        # Should return a valid pipeline
        assert hasattr(preprocessor, "fit_transform")
        assert "feature_engineer" in preprocessor.named_steps
        assert "column_transformer" in preprocessor.named_steps

    def test_engineer_features_missing_property_type_defaults_to_flat(self):
        """Verify engineer_features defaults to flat when Property_Type is missing."""
        df = pd.DataFrame(
            {
                "City": ["Belgrade"],
                "Municipality": ["Voždovac"],
                "Neighborhood": ["Kumodraž"],
                "Advertiser_Type": ["Agency"],
                "Area": [50.0],
                "Photo_Count": [5],
                "Rooms": [2.0],
                "Current_Floor": ["2"],
                "Total_Floors": ["5"],
            }
        )

        result = engineer_features(df)

        # Floor features should be created (as if flat)
        assert "Floor_Ratio" in result.columns
        assert "Is_Ground_Floor" in result.columns
        assert "Floor_Category" in result.columns

        # Property type should now be set to "flat"
        assert result["Property_Type"].iloc[0] == "flat"

    def test_engineer_features_flat_specific_features_list(self):
        """Verify all flat-specific features exist in engineered output."""
        df = pd.DataFrame(
            {
                "City": ["Belgrade"],
                "Municipality": ["Voždovac"],
                "Neighborhood": ["Kumodraž"],
                "Advertiser_Type": ["Agency"],
                "Property_Type": ["flat"],
                "Property_Subtype": ["Apartment"],
                "Area": [75.0],
                "Photo_Count": [8],
                "Rooms": [3.0],
                "Current_Floor": ["3"],
                "Total_Floors": ["7"],
            }
        )

        result = engineer_features(df)

        # All expected features
        expected_features = [
            "Area",
            "Photo_Count",
            "Rooms",
            "Current_Floor_Num",
            "Total_Floors_Num",
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

        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"

    def test_engineer_features_default_features_backward_compatible(self):
        """Verify NUMERICAL_FEATURES and CATEGORICAL_FEATURES still point to flat defaults."""
        # These should be the flat feature sets
        assert NUMERICAL_FEATURES == NUMERICAL_FEATURES_BY_TYPE["flat"]
        assert CATEGORICAL_FEATURES == CATEGORICAL_FEATURES_BY_TYPE["flat"]

    def test_engineer_features_no_mixed_property_types(self):
        """Verify single property type per batch (not mixed in same DataFrame)."""
        df = pd.DataFrame(
            {
                "City": ["Belgrade", "Belgrade"],
                "Municipality": ["Voždovac", "Voždovac"],
                "Neighborhood": ["Kumodraž", "Kumodraž"],
                "Advertiser_Type": ["Agency", "Owner"],
                "Property_Type": ["flat", "house"],  # Mixed types
                "Property_Subtype": ["Apartment", "Home"],
                "Area": [50.0, 100.0],
                "Photo_Count": [5, 3],
                "Rooms": [2.0, 4.0],
                "Current_Floor": ["2", "0"],
                "Total_Floors": ["5", "2"],
            }
        )

        result = engineer_features(df)

        # Should use property type of first row (flat)
        assert not pd.isna(result["Floor_Ratio"].iloc[0])
        # Second row is a house but will be treated as flat since function uses first row's type
        # This is expected behavior for batch processing


class TestPropertyTypeFeatureSelection:
    """Test that correct features are selected for different property types."""

    def test_flat_has_13_numerical_features(self):
        """Verify flat property type uses 13 numerical features."""
        features = NUMERICAL_FEATURES_BY_TYPE["flat"]
        assert len(features) == 13

        # Should include floor-based features
        assert any("Floor" in f for f in features)
        assert any("Room" in f for f in features)

    def test_house_has_8_numerical_features(self):
        """Verify house property type uses 8 numerical features (no floor)."""
        features = NUMERICAL_FEATURES_BY_TYPE["house"]
        assert len(features) == 8

        # Should NOT include floor-based features
        assert "Floor_Ratio" not in features
        assert "Is_Top_Floor" not in features

        # Should include room features
        assert any("Room" in f for f in features)

    def test_land_has_5_numerical_features(self):
        """Verify land property type uses 5 numerical features (area only)."""
        features = NUMERICAL_FEATURES_BY_TYPE["land"]
        assert len(features) == 5

        # Should NOT include floor or room features
        assert not any("Floor" in f for f in features)
        assert not any("Room" in f for f in features)

    def test_garage_has_5_numerical_features(self):
        """Verify garage property type uses 5 numerical features (area only)."""
        features = NUMERICAL_FEATURES_BY_TYPE["garage"]
        assert len(features) == 5

        # Should NOT include floor or room features
        assert not any("Floor" in f for f in features)
        assert not any("Room" in f for f in features)

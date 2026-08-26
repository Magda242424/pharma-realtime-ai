import pandas as pd

from src.features.feature_definitions import (
    SENSOR_FEATURES,
    LAG_STEPS,
    FORECASTING_FEATURES,
    ANOMALY_FEATURES,
    TARGET_COLUMN,
)


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

FEATURES_FILE = (
    ROOT
    / "data"
    / "processed"
    / "features.csv"
)


def test_sensor_features_contract():

    assert len(SENSOR_FEATURES) == 6

    expected = [
        "temperature",
        "ph",
        "dissolved_oxygen",
        "agitation_speed",
        "pressure",
        "flow_rate",
    ]

    assert SENSOR_FEATURES == expected


def test_lag_configuration():

    assert LAG_STEPS == [
        1,
        2,
        3,
        5,
        10,
        15,
        30,
    ]


def test_forecasting_features_contract():

    assert len(FORECASTING_FEATURES) == 7

    for lag in LAG_STEPS:

        expected_column = (
            f"dissolved_oxygen_lag_{lag}"
        )

        assert expected_column in FORECASTING_FEATURES


def test_anomaly_features_contract():

    assert len(ANOMALY_FEATURES) == 13

    for feature in SENSOR_FEATURES:
        assert feature in ANOMALY_FEATURES

    for feature in FORECASTING_FEATURES:
        assert feature in ANOMALY_FEATURES


def test_target_column():

    assert TARGET_COLUMN == "dissolved_oxygen"


def test_feature_dataset_contains_model_features():

    df = pd.read_csv(FEATURES_FILE)

    for feature in ANOMALY_FEATURES:

        assert feature in df.columns
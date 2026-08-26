from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "sensor_data.csv"
CLEAN_FILE = ROOT / "data" / "processed" / "sensor_data_clean.csv"
FEATURES_FILE = ROOT / "data" / "processed" / "features.csv"


def test_raw_data_exists():
    assert RAW_FILE.exists()


def test_clean_data_exists():
    assert CLEAN_FILE.exists()


def test_features_data_exists():
    assert FEATURES_FILE.exists()


def test_raw_data_is_not_empty():
    df = pd.read_csv(RAW_FILE)

    assert len(df) > 0


def test_clean_data_is_not_empty():
    df = pd.read_csv(CLEAN_FILE)

    assert len(df) > 0


def test_features_data_is_not_empty():
    df = pd.read_csv(FEATURES_FILE)

    assert len(df) > 0


def test_features_required_columns_exist():

    df = pd.read_csv(FEATURES_FILE)

    required_columns = [
        "equipment_id",
        "timestamp",
        "temperature",
        "ph",
        "dissolved_oxygen",
        "agitation_speed",
        "pressure",
        "flow_rate",
    ]

    for column in required_columns:
        assert column in df.columns
from pathlib import Path

import joblib
import pandas as pd

from src.features.feature_definitions import (
    ANOMALY_FEATURES,
    FORECASTING_FEATURES,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "features.csv"
)

ANOMALY_MODEL_FILE = (
    MODEL_DIR
    / "anomaly_model.joblib"
)

FORECASTING_MODEL_FILE = (
    MODEL_DIR
    / "forecasting_model.joblib"
)


def test_anomaly_model_exists():

    assert ANOMALY_MODEL_FILE.exists()


def test_forecasting_model_exists():

    assert FORECASTING_MODEL_FILE.exists()


def test_anomaly_model_feature_contract():

    model = joblib.load(
        ANOMALY_MODEL_FILE
    )

    assert list(
        model.feature_names_in_
    ) == ANOMALY_FEATURES


def test_forecasting_model_feature_contract():

    model = joblib.load(
        FORECASTING_MODEL_FILE
    )

    assert list(
        model.feature_names_in_
    ) == FORECASTING_FEATURES


def test_anomaly_model_prediction():

    df = pd.read_csv(DATA_FILE)

    row = (
        df[ANOMALY_FEATURES]
        .dropna()
        .head(1)
    )

    model = joblib.load(
        ANOMALY_MODEL_FILE
    )

    prediction = model.predict(row)

    assert len(prediction) == 1
    assert prediction[0] in [-1, 1]


def test_forecasting_model_prediction():

    df = pd.read_csv(DATA_FILE)

    row = (
        df[FORECASTING_FEATURES]
        .dropna()
        .head(1)
    )

    model = joblib.load(
        FORECASTING_MODEL_FILE
    )

    prediction = model.predict(row)

    assert len(prediction) == 1
    assert isinstance(
        float(prediction[0]),
        float,
    )
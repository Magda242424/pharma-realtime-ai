
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models"

ANOMALY_MODEL_FILE = MODEL_DIR / "anomaly_model.joblib"
FORECASTING_MODEL_FILE = MODEL_DIR / "forecasting_model.joblib"


# ============================================================
# FEATURE CONTRACT
# ============================================================

ANOMALY_FEATURES = [
    "temperature",
    "ph",
    "dissolved_oxygen",
    "agitation_speed",
    "pressure",
    "flow_rate",
    "dissolved_oxygen_lag_1",
    "dissolved_oxygen_lag_2",
    "dissolved_oxygen_lag_3",
    "dissolved_oxygen_lag_5",
    "dissolved_oxygen_lag_10",
    "dissolved_oxygen_lag_15",
    "dissolved_oxygen_lag_30",
]

FORECASTING_FEATURES = [
    "dissolved_oxygen_lag_1",
    "dissolved_oxygen_lag_2",
    "dissolved_oxygen_lag_3",
    "dissolved_oxygen_lag_5",
    "dissolved_oxygen_lag_10",
    "dissolved_oxygen_lag_15",
    "dissolved_oxygen_lag_30",
]

TARGET_COLUMN = "dissolved_oxygen"


# ============================================================
# LOAD MODELS
# ============================================================

if not ANOMALY_MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Anomaly model not found: {ANOMALY_MODEL_FILE}"
    )

if not FORECASTING_MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Forecasting model not found: {FORECASTING_MODEL_FILE}"
    )


anomaly_model = joblib.load(ANOMALY_MODEL_FILE)
forecasting_model = joblib.load(FORECASTING_MODEL_FILE)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="PHARMA Real-Time AI API",
    version="1.0.0",
    description="API for pharmaceutical sensor anomaly detection and dissolved oxygen forecasting.",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class SensorReading(BaseModel):
    temperature: float
    ph: float
    dissolved_oxygen: float
    agitation_speed: float
    pressure: float
    flow_rate: float

    dissolved_oxygen_lag_1: float
    dissolved_oxygen_lag_2: float
    dissolved_oxygen_lag_3: float
    dissolved_oxygen_lag_5: float
    dissolved_oxygen_lag_10: float
    dissolved_oxygen_lag_15: float
    dissolved_oxygen_lag_30: float


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "service": "PHARMA Real-Time AI API",
        "status": "ok",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "anomaly_model": ANOMALY_MODEL_FILE.exists(),
        "forecasting_model": FORECASTING_MODEL_FILE.exists(),
    }


# ============================================================
# FEATURE VALIDATION
# ============================================================

def build_dataframe(reading: SensorReading) -> pd.DataFrame:
    data = reading.model_dump()

    missing = [
        feature
        for feature in ANOMALY_FEATURES
        if feature not in data
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required features",
                "features": missing,
            },
        )

    return pd.DataFrame([data])


# ============================================================
# ANOMALY DETECTION
# ============================================================

@app.post("/predict/anomaly")
def predict_anomaly(reading: SensorReading):

    df = build_dataframe(reading)

    X = df[ANOMALY_FEATURES]

    prediction = int(anomaly_model.predict(X)[0])
    score = float(anomaly_model.decision_function(X)[0])

    return {
        "anomaly_prediction": prediction,
        "is_anomaly": prediction == -1,
        "anomaly_score": score,
    }


# ============================================================
# DISSOLVED OXYGEN FORECAST
# ============================================================

@app.post("/predict/forecast")
def predict_forecast(reading: SensorReading):

    df = build_dataframe(reading)

    X = df[FORECASTING_FEATURES]

    prediction = float(forecasting_model.predict(X)[0])

    return {
        "predicted_dissolved_oxygen": prediction,
    }


# ============================================================
# COMBINED PREDICTION
# ============================================================

@app.post("/predict")
def predict(reading: SensorReading):

    df = build_dataframe(reading)

    anomaly_X = df[ANOMALY_FEATURES]
    forecast_X = df[FORECASTING_FEATURES]

    anomaly_prediction = int(
        anomaly_model.predict(anomaly_X)[0]
    )

    anomaly_score = float(
        anomaly_model.decision_function(anomaly_X)[0]
    )

    forecast_prediction = float(
        forecasting_model.predict(forecast_X)[0]
    )

    return {
        "anomaly_prediction": anomaly_prediction,
        "is_anomaly": anomaly_prediction == -1,
        "anomaly_score": anomaly_score,
        "predicted_dissolved_oxygen": forecast_prediction,
    }

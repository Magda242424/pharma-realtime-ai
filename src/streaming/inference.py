from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

ANOMALY_MODEL_FILE = (
    ROOT
    / "models"
    / "anomaly_model.joblib"
)

FORECASTING_MODEL_FILE = (
    ROOT
    / "models"
    / "forecasting_model.joblib"
)


# ============================================================
# ANOMALY MODEL FEATURE CONTRACT
# ============================================================

ANOMALY_FEATURE_COLUMNS = [
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


# ============================================================
# FORECASTING MODEL FEATURE CONTRACT
# ============================================================

FORECASTING_FEATURE_COLUMNS = [
    "dissolved_oxygen_lag_1",
    "dissolved_oxygen_lag_2",
    "dissolved_oxygen_lag_3",
    "dissolved_oxygen_lag_5",
    "dissolved_oxygen_lag_10",
    "dissolved_oxygen_lag_15",
    "dissolved_oxygen_lag_30",
]


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    if not ANOMALY_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Anomaly model not found: "
            f"{ANOMALY_MODEL_FILE}"
        )

    if not FORECASTING_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Forecasting model not found: "
            f"{FORECASTING_MODEL_FILE}"
        )

    anomaly_model = joblib.load(
        ANOMALY_MODEL_FILE
    )

    forecasting_model = joblib.load(
        FORECASTING_MODEL_FILE
    )

    return (
        anomaly_model,
        forecasting_model,
    )


# ============================================================
# VALIDATE EVENT
# ============================================================

def validate_event(event):

    required_features = set(
        ANOMALY_FEATURE_COLUMNS
    )

    missing = [
        column
        for column in required_features
        if column not in event
    ]

    if missing:

        raise ValueError(
            "Missing required features: "
            + ", ".join(sorted(missing))
        )


# ============================================================
# PREDICT ONE STREAM EVENT
# ============================================================

def predict_event(event):

    validate_event(event)

    (
        anomaly_model,
        forecasting_model,
    ) = load_models()

    # --------------------------------------------------------
    # ANOMALY INPUT
    # --------------------------------------------------------

    anomaly_features = pd.DataFrame(
        [
            {
                column: float(event[column])
                for column in ANOMALY_FEATURE_COLUMNS
            }
        ]
    )

    # --------------------------------------------------------
    # ANOMALY PREDICTION
    # --------------------------------------------------------

    anomaly_prediction = int(
        anomaly_model.predict(
            anomaly_features
        )[0]
    )

    anomaly_score = float(
        anomaly_model.decision_function(
            anomaly_features
        )[0]
    )

    is_anomaly = (
        anomaly_prediction == -1
    )

    # --------------------------------------------------------
    # FORECASTING INPUT
    # --------------------------------------------------------

    forecasting_features = pd.DataFrame(
        [
            {
                column: float(event[column])
                for column in FORECASTING_FEATURE_COLUMNS
            }
        ]
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    forecast = forecasting_model.predict(
        forecasting_features
    )

    predicted_dissolved_oxygen = float(
        forecast[0]
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {
        "equipment_id": event.get(
            "equipment_id"
        ),
        "timestamp": event.get(
            "timestamp"
        ),
        "anomaly_prediction": (
            anomaly_prediction
        ),
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "predicted_dissolved_oxygen": (
            predicted_dissolved_oxygen
        ),
    }

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    from producer import create_stream

    print("=" * 60)
    print("PHARMA - STREAMING INFERENCE")
    print("=" * 60)

    print(
        f"Anomaly model: "
        f"{ANOMALY_MODEL_FILE}"
    )

    print(
        f"Forecasting model: "
        f"{FORECASTING_MODEL_FILE}"
    )

    print("-" * 60)

    # --------------------------------------------------------
    # LOAD FIRST STREAM EVENT
    # --------------------------------------------------------

    stream = create_stream()

    first_event = next(stream)

    print(
        f"Equipment: "
        f"{first_event.get('equipment_id')}"
    )

    print(
        f"Timestamp: "
        f"{first_event.get('timestamp')}"
    )

    print("-" * 60)

    # --------------------------------------------------------
    # RUN INFERENCE
    # --------------------------------------------------------

    result = predict_event(
        first_event
    )

    print("Prediction:")

    print(
        f"Anomaly prediction: "
        f"{result['anomaly_prediction']}"
    )

    print(
        f"Is anomaly: "
        f"{result['is_anomaly']}"
    )

    print(
        f"Anomaly score: "
        f"{result['anomaly_score']:.6f}"
    )

    print(
        f"Predicted dissolved oxygen: "
        f"{result['predicted_dissolved_oxygen']:.6f}"
    )

    print("=" * 60)
    print("PHARMA - STREAMING INFERENCE: OK")
    print("=" * 60)


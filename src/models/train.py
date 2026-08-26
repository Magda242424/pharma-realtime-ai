
from __future__ import annotations

import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.feature_definitions import (
    ANOMALY_FEATURES,
    FORECASTING_FEATURES,
    TARGET_COLUMN,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ANOMALY DETECTION
# ============================================================

def train_anomaly_model(df: pd.DataFrame):

    training_df = df[ANOMALY_FEATURES].dropna()

    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42,
    )

    model.fit(training_df)

    output = MODEL_DIR / "anomaly_model.joblib"

    joblib.dump(model, output)

    return model, output


# ============================================================
# FORECASTING
# ============================================================

def train_forecasting_model(df: pd.DataFrame):

    training_df = df[
        FORECASTING_FEATURES + [TARGET_COLUMN]
    ].dropna()

    X = training_df[FORECASTING_FEATURES]
    y = training_df[TARGET_COLUMN]

    # Chronological split
    split = int(len(X) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    output = MODEL_DIR / "forecasting_model.joblib"

    joblib.dump(model, output)

    return model, output, mae, rmse


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("PHARMA — MODEL TRAINING")
    print("=" * 60)

    print()
    print("Input file:")
    print(DATA_FILE)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print()
    print(f"Input rows: {len(df)}")

    # --------------------------------------------------------
    # MLflow configuration
    # --------------------------------------------------------

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(
        "pharma-realtime-ai"
    )

    # --------------------------------------------------------
    # MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="pharma-model-training"
    ):

        # ----------------------------------------------------
        # Anomaly model
        # ----------------------------------------------------

        anomaly_model, anomaly_path = train_anomaly_model(df)

        mlflow.log_param(
            "anomaly_model",
            "IsolationForest",
        )

        mlflow.log_param(
            "anomaly_features",
            len(ANOMALY_FEATURES),
        )

        mlflow.log_param(
            "anomaly_estimators",
            200,
        )

        # ----------------------------------------------------
        # Forecasting model
        # ----------------------------------------------------

        (
            forecasting_model,
            forecasting_path,
            mae,
            rmse,
        ) = train_forecasting_model(df)

        mlflow.log_param(
            "forecasting_model",
            "RandomForestRegressor",
        )

        mlflow.log_param(
            "forecasting_features",
            len(FORECASTING_FEATURES),
        )

        mlflow.log_param(
            "forecasting_estimators",
            200,
        )

        mlflow.log_metric(
            "mae",
            mae,
        )

        mlflow.log_metric(
            "rmse",
            rmse,
        )

        print()
        print("Anomaly model:")
        print(f"  Saved to: {anomaly_path}")

        print()
        print("Forecasting model:")
        print(f"  Saved to: {forecasting_path}")

        print()
        print("Forecast metrics:")
        print(f"  MAE  : {mae:.4f}")
        print(f"  RMSE : {rmse:.4f}")

    print()
    print("=" * 60)
    print("MODEL TRAINING: OK")
    print("=" * 60)


if __name__ == "__main__":
    main()

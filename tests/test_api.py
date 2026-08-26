from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app


ROOT = Path(__file__).resolve().parents[1]


client = TestClient(app)


def valid_payload():
    return {
        "temperature": 37.0,
        "ph": 7.1,
        "dissolved_oxygen": 8.0,
        "agitation_speed": 150.0,
        "pressure": 1.2,
        "flow_rate": 10.0,
        "dissolved_oxygen_lag_1": 8.0,
        "dissolved_oxygen_lag_2": 7.9,
        "dissolved_oxygen_lag_3": 7.8,
        "dissolved_oxygen_lag_5": 7.7,
        "dissolved_oxygen_lag_10": 7.6,
        "dissolved_oxygen_lag_15": 7.5,
        "dissolved_oxygen_lag_30": 7.4,
    }


def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["anomaly_model"] is True
    assert data["forecasting_model"] is True


def test_anomaly_prediction_endpoint():

    response = client.post(
        "/predict/anomaly",
        json=valid_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert "anomaly_prediction" in data
    assert "is_anomaly" in data
    assert "anomaly_score" in data

    assert data["anomaly_prediction"] in [-1, 1]
    assert isinstance(data["is_anomaly"], bool)
    assert isinstance(data["anomaly_score"], float)


def test_forecast_prediction_endpoint():

    response = client.post(
        "/predict/forecast",
        json=valid_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_dissolved_oxygen" in data

    assert isinstance(
        data["predicted_dissolved_oxygen"],
        float,
    )


def test_combined_prediction_endpoint():

    response = client.post(
        "/predict",
        json=valid_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert "anomaly_prediction" in data
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    assert "predicted_dissolved_oxygen" in data

    assert data["anomaly_prediction"] in [-1, 1]

    assert isinstance(
        data["is_anomaly"],
        bool,
    )

    assert isinstance(
        data["anomaly_score"],
        float,
    )

    assert isinstance(
        data["predicted_dissolved_oxygen"],
        float,
    )


def test_anomaly_endpoint_rejects_missing_features():

    payload = valid_payload()

    del payload["temperature"]

    response = client.post(
        "/predict/anomaly",
        json=payload,
    )

    assert response.status_code == 422
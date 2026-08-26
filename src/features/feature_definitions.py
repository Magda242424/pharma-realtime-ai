from __future__ import annotations


# ------------------------------------------------------------
# Raw sensor features
# ------------------------------------------------------------

SENSOR_FEATURES = [
    "temperature",
    "ph",
    "dissolved_oxygen",
    "agitation_speed",
    "pressure",
    "flow_rate",
]


# ------------------------------------------------------------
# Time-series lag configuration
# ------------------------------------------------------------

LAG_STEPS = [
    1,
    2,
    3,
    5,
    10,
    15,
    30,
]


# ------------------------------------------------------------
# Forecasting features
# ------------------------------------------------------------

FORECASTING_FEATURES = [
    f"dissolved_oxygen_lag_{lag}"
    for lag in LAG_STEPS
]


# ------------------------------------------------------------
# Anomaly detection features
# ------------------------------------------------------------

ANOMALY_FEATURES = (
    SENSOR_FEATURES
    + FORECASTING_FEATURES
)


# ------------------------------------------------------------
# Target
# ------------------------------------------------------------

TARGET_COLUMN = "dissolved_oxygen"


# ------------------------------------------------------------
# Contract validation
# ------------------------------------------------------------

assert len(SENSOR_FEATURES) == 6
assert len(FORECASTING_FEATURES) == 7
assert len(ANOMALY_FEATURES) == 13

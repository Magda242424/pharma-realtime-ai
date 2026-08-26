from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"


def generate_sensor_data(
    n_rows: int = 5000,
    n_equipment: int = 3,
    seed: int = 42,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start="2025-01-01",
        periods=n_rows,
        freq="min",
    )

    equipment_ids = [
        f"EQ-{i:03d}"
        for i in range(1, n_equipment + 1)
    ]

    equipment = rng.choice(
        equipment_ids,
        size=n_rows,
    )

    time_index = np.arange(n_rows)

    temperature = (
        25
        + 1.5 * np.sin(time_index / 120)
        + rng.normal(0, 0.3, n_rows)
    )

    ph = (
        7.0
        + 0.15 * np.sin(time_index / 90)
        + rng.normal(0, 0.03, n_rows)
    )

    dissolved_oxygen = (
        7.5
        - 0.015 * (temperature - 25)
        + 0.3 * np.sin(time_index / 80)
        + rng.normal(0, 0.12, n_rows)
    )

    agitation_speed = (
        150
        + 10 * np.sin(time_index / 100)
        + rng.normal(0, 3, n_rows)
    )

    pressure = (
        1.2
        + 0.05 * np.sin(time_index / 150)
        + rng.normal(0, 0.01, n_rows)
    )

    flow_rate = (
        100
        + 5 * np.sin(time_index / 110)
        + rng.normal(0, 1.5, n_rows)
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "batch_id": "BATCH-001",
            "equipment_id": equipment,
            "temperature": temperature,
            "ph": ph,
            "dissolved_oxygen": dissolved_oxygen,
            "agitation_speed": agitation_speed,
            "pressure": pressure,
            "flow_rate": flow_rate,
            "source_system": "iot_simulator",
        }
    )

    # --------------------------------------------------------
    # Synthetic anomalies
    # --------------------------------------------------------

    anomaly_indices = rng.choice(
        n_rows,
        size=max(1, n_rows // 100),
        replace=False,
    )

    df.loc[
        anomaly_indices,
        "dissolved_oxygen",
    ] += rng.normal(
        2.0,
        0.5,
        len(anomaly_indices),
    )

    return df


def save_raw_data(df: pd.DataFrame) -> Path:

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = RAW_DIR / "sensor_data.csv"

    df.to_csv(
        output,
        index=False,
    )

    return output


if __name__ == "__main__":

    df = generate_sensor_data()

    path = save_raw_data(df)

    print(f"Generated {len(df)} rows")
    print(f"Saved to: {path}")

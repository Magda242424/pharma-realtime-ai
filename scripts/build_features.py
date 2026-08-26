from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.feature_definitions import (
    LAG_STEPS,
    FORECASTING_FEATURES,
)


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "sensor_data_clean.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "features.csv"
)


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Sort observations by equipment and time
    # --------------------------------------------------------

    result = result.sort_values(
        [
            "equipment_id",
            "timestamp",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Create dissolved oxygen lag features
    # --------------------------------------------------------

    for lag in LAG_STEPS:

        feature_name = (
            f"dissolved_oxygen_lag_{lag}"
        )

        result[feature_name] = (
            result
            .groupby("equipment_id")[
                "dissolved_oxygen"
            ]
            .shift(lag)
        )

    # --------------------------------------------------------
    # Keep rows with complete forecasting history
    # --------------------------------------------------------

    result = result.dropna(
        subset=FORECASTING_FEATURES
    ).reset_index(drop=True)

    return result


def save_features(
    df: pd.DataFrame,
) -> Path:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return OUTPUT_FILE


def main():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Clean data not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    features = build_features(
        df
    )

    output = save_features(
        features
    )

    print(
        f"Input rows: {len(df)}"
    )

    print(
        f"Feature rows: {len(features)}"
    )

    print(
        f"Features created: "
        f"{len(FORECASTING_FEATURES)}"
    )

    print(
        f"Saved to: {output}"
    )


if __name__ == "__main__":

    main()

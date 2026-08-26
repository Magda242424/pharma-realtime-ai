from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    ROOT
    / "data"
    / "raw"
    / "sensor_data.csv"
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)


REQUIRED_COLUMNS = [
    "timestamp",
    "batch_id",
    "equipment_id",
    "temperature",
    "ph",
    "dissolved_oxygen",
    "agitation_speed",
    "pressure",
    "flow_rate",
    "source_system",
]


def validate_schema(df: pd.DataFrame) -> None:

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    validate_schema(df)

    result = df.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    numeric_columns = [
        "temperature",
        "ph",
        "dissolved_oxygen",
        "agitation_speed",
        "pressure",
        "flow_rate",
    ]

    for column in numeric_columns:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = result.dropna(
        subset=[
            "timestamp",
            "equipment_id",
            "dissolved_oxygen",
        ]
    )

    result = result.drop_duplicates(
        subset=[
            "timestamp",
            "equipment_id",
        ]
    )

    result = result.sort_values(
        [
            "equipment_id",
            "timestamp",
        ]
    )

    return result.reset_index(drop=True)


def save_processed_data(
    df: pd.DataFrame,
) -> Path:

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        PROCESSED_DIR
        / "sensor_data_clean.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    return output


if __name__ == "__main__":

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw data not found: {RAW_FILE}"
        )

    df = pd.read_csv(RAW_FILE)

    clean = clean_data(df)

    output = save_processed_data(clean)

    print(f"Clean dataset: {len(clean)} rows")
    print(f"Saved to: {output}")

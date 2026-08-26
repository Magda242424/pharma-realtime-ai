from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "exports" / "stream_predictions.jsonl"
OUTPUT_FILE = ROOT / "exports" / "powerbi_predictions.csv"


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "equipment_id",
    "timestamp",
    "anomaly_prediction",
    "is_anomaly",
    "anomaly_score",
    "predicted_dissolved_oxygen",
]


# ============================================================
# LOAD STREAMING PREDICTIONS
# ============================================================

def load_predictions() -> pd.DataFrame:
    """Load prediction events from the JSONL streaming output."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Streaming prediction file not found: {INPUT_FILE}"
        )

    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            records.append(record)

    if not records:
        raise ValueError(
            f"No prediction events found in {INPUT_FILE}"
        )

    return pd.DataFrame(records)


# ============================================================
# PREPARE POWER BI DATASET
# ============================================================

def prepare_powerbi_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare the streaming predictions for Power BI."""

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    output = df[REQUIRED_COLUMNS].copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    output["timestamp"] = pd.to_datetime(
        output["timestamp"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    output["anomaly_prediction"] = pd.to_numeric(
        output["anomaly_prediction"],
        errors="coerce",
    )

    output["anomaly_score"] = pd.to_numeric(
        output["anomaly_score"],
        errors="coerce",
    )

    output["predicted_dissolved_oxygen"] = pd.to_numeric(
        output["predicted_dissolved_oxygen"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Boolean anomaly flag
    # --------------------------------------------------------

    output["is_anomaly"] = output["is_anomaly"].astype(bool)

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    output = output.dropna(
        subset=[
            "equipment_id",
            "timestamp",
            "anomaly_score",
            "predicted_dissolved_oxygen",
        ]
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    output = output.sort_values(
        by="timestamp"
    ).reset_index(drop=True)

    return output


# ============================================================
# EXPORT
# ============================================================

def export_powerbi() -> Path:
    """Create the CSV dataset consumed by Power BI."""

    print("=" * 60)
    print("POWER BI EXPORT")
    print("=" * 60)

    print()
    print("Input:")
    print(INPUT_FILE)

    df = load_predictions()

    print()
    print(f"Input events: {len(df)}")

    powerbi_df = prepare_powerbi_dataframe(df)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    powerbi_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print()
    print(f"Exported events: {len(powerbi_df)}")

    print()
    print("Columns:")
    for column in powerbi_df.columns:
        print(f"  - {column}")

    print()
    print("=" * 60)
    print("POWER BI EXPORT: OK")
    print("=" * 60)

    return OUTPUT_FILE


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    export_powerbi()
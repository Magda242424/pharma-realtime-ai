# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import time


ROOT = Path(__file__).resolve().parents[2]

FEATURES_FILE = ROOT / "data" / "processed" / "features.csv"


def load_stream_data():
    """
    Load the existing processed feature dataset.

    This function does not modify the dataset.
    """
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Features file not found: {FEATURES_FILE}"
        )

    return pd.read_csv(FEATURES_FILE)


def stream_rows(df, delay_seconds=0.0):
    """
    Simulate a sensor stream by emitting rows sequentially.

    Existing feature data is used as the source.
    """
    for _, row in df.iterrows():

        event = row.to_dict()

        yield event

        if delay_seconds > 0:
            time.sleep(delay_seconds)


def create_stream(delay_seconds=0.0):
    """
    Create a local simulated stream from features.csv.
    """
    df = load_stream_data()

    return stream_rows(
        df,
        delay_seconds=delay_seconds,
    )


if __name__ == "__main__":

    df = load_stream_data()

    print("=" * 60)
    print("PHARMA - STREAMING PRODUCER")
    print("=" * 60)

    print(f"Features file: {FEATURES_FILE}")
    print(f"Rows available: {len(df)}")

    print("-" * 60)
    print("Streaming first 5 events")
    print("-" * 60)

    stream = create_stream()

    for index, event in enumerate(stream):

        print(
            f"Event {index + 1}: "
            f"equipment={event.get('equipment_id')} "
            f"timestamp={event.get('timestamp')}"
        )

        if index >= 4:
            break

    print("=" * 60)
    print("PHARMA - STREAMING PRODUCER: OK")
    print("=" * 60)

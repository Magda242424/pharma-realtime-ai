from pathlib import Path
import json

from src.streaming.producer import create_stream
from src.streaming.inference import predict_event
from src.streaming.consumer import consume_events


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = (
    ROOT
    / "exports"
    / "stream_predictions.jsonl"
)


def test_producer_generates_events():

    stream = create_stream()

    event = next(stream)

    assert isinstance(event, dict)

    assert "equipment_id" in event
    assert "timestamp" in event


def test_inference_returns_prediction():

    stream = create_stream()

    event = next(stream)

    result = predict_event(event)

    assert isinstance(result, dict)

    assert "equipment_id" in result
    assert "timestamp" in result
    assert "anomaly_prediction" in result
    assert "is_anomaly" in result
    assert "anomaly_score" in result
    assert "predicted_dissolved_oxygen" in result


def test_inference_anomaly_prediction():

    stream = create_stream()

    event = next(stream)

    result = predict_event(event)

    assert result[
        "anomaly_prediction"
    ] in [-1, 1]

    assert isinstance(
        result["is_anomaly"],
        bool,
    )


def test_consumer_writes_prediction():

    stream = create_stream()

    event = next(stream)

    result = predict_event(event)

    count = consume_events(
        [result]
    )

    assert count == 1

    assert OUTPUT_FILE.exists()

    lines = OUTPUT_FILE.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    assert len(lines) >= 1

    saved_event = json.loads(
        lines[-1]
    )

    assert (
        saved_event["equipment_id"]
        == result["equipment_id"]
    )


def test_streaming_end_to_end():

    stream = create_stream()

    event = next(stream)

    result = predict_event(event)

    count = consume_events(
        [result]
    )

    assert count == 1

    assert (
        "predicted_dissolved_oxygen"
        in result
    )
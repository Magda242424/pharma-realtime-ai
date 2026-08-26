from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = ROOT / "exports" / "stream_predictions.jsonl"


def ensure_output_directory():
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def consume_events(events):
    """
    Consume prediction events and persist them
    as JSON Lines.

    Each event is written independently.
    """
    ensure_output_directory()

    count = 0

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for event in events:

            file.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

            count += 1

    return count


if __name__ == "__main__":

    print("=" * 60)
    print("PHARMA - STREAMING CONSUMER")
    print("=" * 60)

    print(f"Output: {OUTPUT_FILE}")

    print("=" * 60)
    print("Consumer module ready.")
    print("=" * 60)

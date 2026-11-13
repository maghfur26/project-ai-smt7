"""Predict a single text with a saved model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.inference import predict_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one text.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--task",
        default="cyberbullying",
        choices=["cyberbullying", "buzzer"],
    )
    args = parser.parse_args()

    result = predict_text(args.text, args.model_dir, task=args.task)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Prepare processed datasets from public sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.prepare import prepare_task  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare a task dataset (cyberbullying or buzzer)."
    )
    parser.add_argument("--config", required=True, help="Path to a data YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = prepare_task(config)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()

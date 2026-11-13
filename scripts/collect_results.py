"""Collect experiment metrics, write tables, and generate figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.collect import write_aggregate  # noqa: E402
from src.evaluation.plots import plot_comparisons  # noqa: E402
from src.utils.config import resolve_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics and build figures.")
    parser.add_argument("--results-dir", default="results/models")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    metrics, aggregate = write_aggregate(args.results_dir, args.output_dir)
    figures = plot_comparisons(metrics, resolve_path(args.output_dir) / "figures")

    print(f"Runs collected: {len(metrics)}")
    if not aggregate.empty:
        print(aggregate.to_string(index=False))
    print(f"Figures written: {len(figures)}")


if __name__ == "__main__":
    main()

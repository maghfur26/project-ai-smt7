from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.collect import (
    aggregate_metrics,
    collect_metrics,
    write_aggregate,
)
from src.evaluation.plots import plot_comparisons
from src.utils.io import write_json


def _make_run(root: Path, name: str, model: str, f1: float, seed: int = 42) -> None:
    run_dir = root / name
    write_json(
        {
            "run_name": name,
            "task": "cyberbullying",
            "model": model,
            "seed": seed,
            "macro_f1": f1,
            "accuracy": f1 + 0.01,
        },
        run_dir / "metrics.json",
    )
    write_json(
        {"parameters": 110_000_000, "train_seconds": 100.0, "model_size_mb": 400.0},
        run_dir / "efficiency.json",
    )


def test_collect_and_aggregate(tmp_path):
    root = tmp_path / "models"
    _make_run(root, "run_a", "model/a", 0.8, seed=42)
    _make_run(root, "run_b", "model/a", 0.9, seed=123)
    _make_run(root, "run_c", "model/b", 0.7, seed=42)

    metrics = collect_metrics(root)
    assert len(metrics) == 3
    assert "eff_parameters" in metrics.columns

    aggregate = aggregate_metrics(metrics)
    row = aggregate[aggregate["model"] == "model/a"].iloc[0]
    assert row["macro_f1_mean"] == 0.85
    assert row["macro_f1_std"] > 0


def test_write_aggregate_and_plots(tmp_path):
    root = tmp_path / "models"
    _make_run(root, "run_a", "model/a", 0.8, seed=42)
    _make_run(root, "run_b", "model/a", 0.9, seed=123)
    _make_run(root, "run_c", "model/b", 0.7, seed=42)

    metrics, aggregate = write_aggregate(root, tmp_path / "results")
    assert (tmp_path / "results" / "metrics.csv").exists()
    assert (tmp_path / "results" / "tables" / "aggregate.csv").exists()
    assert not aggregate.empty

    figures = plot_comparisons(metrics, tmp_path / "results" / "figures")
    assert figures
    assert all(path.exists() for path in figures)


def test_collect_empty(tmp_path):
    metrics = collect_metrics(tmp_path / "missing")
    assert metrics.empty
    assert aggregate_metrics(metrics).empty

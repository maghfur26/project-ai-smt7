"""Collect and aggregate experiment metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import read_json

_META_COLUMNS = {"run_name", "task", "model", "seed"}


def collect_metrics(results_dir: str | Path) -> pd.DataFrame:
    """Read every ``*/metrics.json`` under ``results_dir`` into one frame."""
    root = Path(results_dir)
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        payload = read_json(metrics_path)
        run_dir = metrics_path.parent
        row: dict[str, Any] = {
            "run_name": payload.get("run_name", run_dir.name),
            "task": payload.get("task"),
            "model": payload.get("model"),
            "seed": payload.get("seed"),
        }
        for key, value in payload.items():
            if key in _META_COLUMNS:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[key] = float(value)

        efficiency_path = run_dir / "efficiency.json"
        if efficiency_path.exists():
            efficiency = read_json(efficiency_path)
            for key, value in efficiency.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[f"eff_{key}"] = float(value)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["run_name", "task", "model", "seed"])
    return pd.DataFrame(rows)


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per task/model as mean and standard deviation."""
    if df.empty:
        return pd.DataFrame()
    numeric = [
        column
        for column in df.columns
        if column not in _META_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]
    grouped = df.groupby(["task", "model"], dropna=False)[numeric]
    aggregated = grouped.agg(["mean", "std"]).round(4)
    aggregated.columns = [f"{metric}_{stat}" for metric, stat in aggregated.columns]
    return aggregated.reset_index()


def write_aggregate(
    results_dir: str | Path, output_dir: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write ``metrics.csv`` and ``aggregate.csv``; return both frames."""
    metrics = collect_metrics(results_dir)
    aggregate = aggregate_metrics(metrics)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False, encoding="utf-8")
    aggregate.to_csv(tables / "aggregate.csv", index=False, encoding="utf-8")
    return metrics, aggregate

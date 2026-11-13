"""Build processed, leakage-safe train/val/test splits from public sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.cleaning import clean_text
from src.data.labels import TASK_LABELS
from src.data.sources import load_source
from src.data.split import stratified_group_split
from src.data.weak_label import near_duplicate_clusters, weak_label_frame
from src.utils.config import resolve_path
from src.utils.io import read_json, write_json


def _source_specs(config: dict[str, Any], task: str) -> list[dict[str, Any]]:
    raw_dir = config.get("raw_dir", f"data/raw/{task}")
    specs: list[dict[str, Any]] = []
    for spec in config.get("sources", []):
        item = dict(spec)
        item.setdefault("raw_dir", raw_dir)
        item.setdefault(
            "keep_emoji", config.get("cleaning", {}).get("keep_emoji", True)
        )
        specs.append(item)
    if not specs:
        raise ValueError("Config must define at least one source under 'sources'")
    return specs


def _load_sources(config: dict[str, Any], task: str) -> tuple[pd.DataFrame, dict[str, int]]:
    frames: list[pd.DataFrame] = []
    dropped_unknown = 0
    for spec in _source_specs(config, task):
        frame = load_source(spec, task)
        if task != "buzzer":
            before = len(frame)
            frame = frame[frame["label"].notna()].copy()
            dropped_unknown += before - len(frame)
        frames.append(frame)
    if not frames:
        raise ValueError("No source data was loaded")
    combined = pd.concat(frames, ignore_index=True)
    combined["label"] = combined["label"].astype("Int64")
    return combined, {"dropped_unknown_labels": dropped_unknown}


def prepare_task(config: dict[str, Any]) -> dict[str, Path]:
    """Prepare splits for one task and return the output file paths."""
    task = config["task"]
    if task not in TASK_LABELS:
        raise ValueError(f"Unsupported task: {task!r}")
    seed = int(config.get("seed", 42))
    output_dir = resolve_path(config.get("output_dir", f"data/processed/{task}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    combined, stats = _load_sources(config, task)

    stats["raw_rows"] = int(len(combined))
    stats["empty_text_removed"] = int((combined["text"].str.strip() == "").sum())
    combined = combined[combined["text"].str.strip() != ""].copy()

    if bool(config.get("deduplicate", True)):
        before = len(combined)
        if task != "buzzer":
            label_counts = combined.groupby("text")["label"].nunique()
            ambiguous = label_counts[label_counts > 1].index
            stats["conflicting_duplicates_removed"] = int(
                combined["text"].isin(ambiguous).sum()
            )
            combined = combined[~combined["text"].isin(ambiguous)].copy()
        combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
        stats["exact_duplicates_removed"] = int(before - len(combined))

    if task == "buzzer":
        combined = weak_label_frame(combined, config.get("weak_label", {}))
    else:
        combined = combined[combined["label"].notna()].copy()
        combined["label"] = combined["label"].astype(int)
        if bool(config.get("group_near_duplicates", True)):
            clusters = near_duplicate_clusters(
                combined["text"].tolist(),
                threshold=float(
                    config.get("near_duplicate_threshold", 0.9)
                ),
                neighbors=int(config.get("near_duplicate_neighbors", 20)),
            )
            cluster_groups = [f"c{cluster}" if cluster >= 0 else f"u{index}" for index, cluster in enumerate(clusters)]
            combined["cluster_id"] = clusters
            provided = combined.get("group_id")
            if provided is None or combined["group_id"].isna().all():
                combined["group_id"] = cluster_groups
            else:
                combined["group_id"] = [
                    str(group) if pd.notna(group) else cluster
                    for group, cluster in zip(combined["group_id"], cluster_groups)
                ]
        else:
            combined["group_id"] = [f"r{index}" for index in range(len(combined))]

    ratios = config.get("split", {"train": 0.8, "val": 0.1, "test": 0.1})
    splits = stratified_group_split(
        combined,
        label_col="label",
        group_col="group_id",
        ratios=ratios,
        seed=seed,
    )

    output_paths: dict[str, Path] = {}
    for name, frame in splits.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        output_paths[name] = path

    stats["rows"] = {name: int(len(frame)) for name, frame in splits.items()}
    stats["label_distribution"] = {
        name: {str(key): int(value) for key, value in frame["label"].value_counts().sort_index().items()}
        for name, frame in splits.items()
    }
    stats["sources"] = {
        source: int(count)
        for source, count in combined["source"].value_counts().items()
    }
    stats["groups"] = {
        name: int(frame["group_id"].nunique()) for name, frame in splits.items()
    }
    stats["task"] = task
    stats["seed"] = seed
    stats["labels"] = TASK_LABELS[task]
    stats["note"] = (
        "Buzzer labels are weak-supervision proxies, not verified buzzer behaviour."
        if task == "buzzer"
        else "Labels are harmonized from public datasets; see data/DATASET_CARD.md."
    )

    stats_path = output_dir / "stats.json"
    write_json(stats, stats_path)
    output_paths["stats"] = stats_path
    return output_paths


def read_stats(output_dir: str | Path) -> dict[str, Any]:
    """Read the preparation statistics for a processed dataset."""
    return read_json(resolve_path(output_dir) / "stats.json")

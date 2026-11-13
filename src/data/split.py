"""Group-aware stratified splitting to prevent duplicate leakage."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _group_labels(df: pd.DataFrame, label_col: str, group_col: str) -> pd.Series:
    grouped = df.groupby(group_col, sort=False)[label_col]
    # A group takes the rounded mean of its row labels. This keeps a group that
    # mixes labels together (it must not be split) and gives stratification a
    # single value per group.
    return grouped.mean().round().astype(int)


def stratified_group_split(
    df: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "group_id",
    ratios: Mapping[str, float] | None = None,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split a frame so that every ``group_col`` value lands in one split only."""
    if label_col not in df.columns or group_col not in df.columns:
        raise KeyError(f"Need columns {label_col!r} and {group_col!r}")

    split_ratios = dict(ratios or {"train": 0.8, "val": 0.1, "test": 0.1})
    if set(split_ratios) != {"train", "val", "test"}:
        raise ValueError("ratios must define exactly train, val, and test")
    total = sum(split_ratios.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {total}")
    if split_ratios["train"] <= 0 or split_ratios["val"] <= 0 or split_ratios["test"] <= 0:
        raise ValueError("all split ratios must be positive")

    group_labels = _group_labels(df, label_col, group_col)
    groups = group_labels.index.to_numpy()
    labels = group_labels.to_numpy()
    if len(groups) < 3:
        raise ValueError("At least 3 distinct groups are required for a 3-way split")

    def safe_split(indices, test_size, stratify):
        # Stratification is impossible when a class has fewer than 2 members.
        counts = pd.Series(stratify).value_counts() if stratify is not None else pd.Series(dtype=int)
        if stratify is not None and (len(counts) < 2 or counts.min() < 2):
            stratify = None
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )

    indices = np.arange(len(groups))
    train_idx, temp_idx = safe_split(
        indices, split_ratios["val"] + split_ratios["test"], labels
    )
    relative_test = split_ratios["test"] / (split_ratios["val"] + split_ratios["test"])
    val_idx, test_idx = safe_split(temp_idx, relative_test, labels[temp_idx])

    group_to_split: dict[str, str] = {}
    for index in train_idx:
        group_to_split[str(groups[index])] = "train"
    for index in val_idx:
        group_to_split[str(groups[index])] = "val"
    for index in test_idx:
        group_to_split[str(groups[index])] = "test"

    split_names = df[group_col].astype(str).map(group_to_split)
    if split_names.isna().any():
        missing = df.loc[split_names.isna(), group_col].unique()[:5]
        raise RuntimeError(f"Some groups were not assigned to a split: {missing}")

    result = {
        name: df.loc[split_names == name].reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    for name, part in result.items():
        if len(part) == 0:
            raise ValueError(f"Split {name!r} is empty; provide more groups")
    return result

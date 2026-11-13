from __future__ import annotations

import pandas as pd
import pytest

from src.data.split import stratified_group_split


def _frame(groups: int = 60, rows_per_group: int = 2) -> pd.DataFrame:
    rows = []
    for group in range(groups):
        label = group % 2
        for row in range(rows_per_group):
            rows.append(
                {
                    "text": f"teks unik grup {group} baris {row}",
                    "label": label,
                    "group_id": f"g{group}",
                }
            )
    return pd.DataFrame(rows)


def test_split_has_no_group_leakage():
    frame = _frame()
    splits = stratified_group_split(frame, seed=42)
    groups = {name: set(part["group_id"]) for name, part in splits.items()}
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    assert sum(len(part) for part in splits.values()) == len(frame)


def test_split_is_stratified_and_covers_rows():
    frame = _frame()
    splits = stratified_group_split(frame, seed=123)
    for part in splits.values():
        assert set(part["label"].unique()) == {0, 1}
    train_ratio = len(splits["train"]) / len(frame)
    assert 0.7 <= train_ratio <= 0.9


def test_split_rejects_too_few_groups():
    frame = pd.DataFrame(
        {"text": ["a", "b"], "label": [0, 1], "group_id": ["g0", "g1"]}
    )
    with pytest.raises(ValueError):
        stratified_group_split(frame)


def test_split_rejects_bad_ratios():
    with pytest.raises(ValueError):
        stratified_group_split(_frame(), ratios={"train": 0.9, "val": 0.2, "test": 0.1})

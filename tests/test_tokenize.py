from __future__ import annotations

import pandas as pd

from src.data.tokenize import tokenize_splits


class DummyTokenizer:
    def __call__(self, texts, truncation=True, max_length=128, padding=False):
        return {
            "input_ids": [[1] + [len(text) % 50 + 2] for text in texts],
            "attention_mask": [[1, 1] for _ in texts],
        }


def test_tokenize_splits_keeps_rows_and_labels():
    splits = {
        "train": pd.DataFrame({"text": ["halo", "dunia"], "label": [0, 1]}),
        "val": pd.DataFrame({"text": ["tes"], "label": [1]}),
        "test": pd.DataFrame({"text": ["uji"], "label": [0]}),
    }
    result = tokenize_splits(splits, DummyTokenizer(), max_length=32)
    assert set(result.keys()) == {"train", "val", "test"}
    assert len(result["train"]) == 2
    assert set(result["train"].column_names) == {"input_ids", "attention_mask", "labels"}
    assert result["train"]["labels"] == [0, 1]


def test_tokenize_drops_empty_text():
    splits = {"train": pd.DataFrame({"text": ["halo", "   "], "label": [0, 1]})}
    result = tokenize_splits(splits, DummyTokenizer())
    assert len(result["train"]) == 1

"""Convert pandas splits into tokenized Hugging Face datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd


def tokenize_splits(
    splits: dict[str, pd.DataFrame],
    tokenizer: Any,
    max_length: int = 128,
    text_col: str = "text",
    label_col: str = "label",
) -> Any:
    """Tokenize each split and return a ``DatasetDict``.

    Only model inputs (``input_ids``, ``attention_mask``) and ``labels`` are
    kept. The tokenizer is expected to be a Hugging Face tokenizer or any
    callable with the same batched signature (useful in tests).
    """
    from datasets import Dataset, DatasetDict

    tokenized: dict[str, Dataset] = {}
    for name, frame in splits.items():
        if text_col not in frame.columns or label_col not in frame.columns:
            raise KeyError(
                f"Split {name!r} needs columns {text_col!r} and {label_col!r}"
            )
        subset = frame[[text_col, label_col]].reset_index(drop=True)
        subset = subset[subset[text_col].astype(str).str.strip() != ""]
        dataset = Dataset.from_pandas(subset, preserve_index=False)

        def encode(batch: dict[str, list[Any]]) -> dict[str, Any]:
            return tokenizer(
                list(batch[text_col]),
                truncation=True,
                max_length=max_length,
                padding=False,
            )

        dataset = dataset.map(encode, batched=True)
        dataset = dataset.rename_column(label_col, "labels")
        keep = {"input_ids", "attention_mask", "labels", "token_type_ids"}
        remove = [column for column in dataset.column_names if column not in keep]
        if remove:
            dataset = dataset.remove_columns(remove)
        tokenized[name] = dataset
    return DatasetDict(tokenized)

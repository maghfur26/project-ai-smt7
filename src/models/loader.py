"""Lazy wrappers around Hugging Face model loading."""

from __future__ import annotations

from typing import Any


def load_model_and_tokenizer(
    model_name: str, num_labels: int, cache_dir: str | None = None
) -> tuple[Any, Any]:
    """Load a sequence-classification model and its tokenizer."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=int(num_labels),
        cache_dir=cache_dir,
    )
    return model, tokenizer


def count_parameters(model: Any) -> int:
    """Return the total number of trainable parameters."""
    return int(sum(parameter.numel() for parameter in model.parameters()))


def model_disk_size_mb(model: Any) -> float:
    """Approximate the in-memory parameter footprint in megabytes."""
    total_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    return round(total_bytes / (1024 * 1024), 2)

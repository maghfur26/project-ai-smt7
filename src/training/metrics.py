"""Classification metrics shared by training and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def compute_metrics(eval_pred: tuple[Any, Any]) -> dict[str, float]:
    """Compute accuracy, macro/weighted precision-recall-F1, and ROC-AUC."""
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels).reshape(-1)
    if logits.ndim == 1:
        logits = np.stack([1.0 - logits, logits], axis=-1)
    predictions = np.argmax(logits, axis=-1)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, predictions)),
    }
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average=average,
            zero_division=0,
        )
        metrics[f"{average}_precision"] = float(precision)
        metrics[f"{average}_recall"] = float(recall)
        metrics[f"{average}_f1"] = float(f1)

    if logits.shape[-1] == 2 and len(np.unique(labels)) == 2:
        try:
            probabilities = _softmax(logits)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        except ValueError:  # pragma: no cover - defensive
            pass
    return metrics

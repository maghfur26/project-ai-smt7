from __future__ import annotations

import numpy as np

from src.training.metrics import compute_metrics


def test_binary_metrics_perfect_predictions():
    logits = np.array([[3.0, -1.0], [-1.0, 3.0]])
    labels = np.array([0, 1])
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_macro_metrics_with_errors():
    logits = np.array([[3.0, -1.0], [3.0, -1.0], [-1.0, 3.0], [-1.0, 3.0]])
    labels = np.array([0, 1, 1, 1])
    metrics = compute_metrics((logits, labels))
    assert 0.0 <= metrics["accuracy"] < 1.0
    assert 0.0 <= metrics["macro_f1"] < 1.0


def test_multiclass_metrics_have_no_roc_auc():
    logits = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    labels = np.array([0, 1, 2])
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 1.0
    assert "roc_auc" not in metrics

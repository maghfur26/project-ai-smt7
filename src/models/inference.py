"""Inference helpers shared by the CLI and the Streamlit demo."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from src.data.cleaning import clean_text
from src.data.labels import TASK_LABELS
from src.utils.io import read_json


@lru_cache(maxsize=6)
def _load_artifact(model_dir: str) -> tuple[Any, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def load_available_models(results_dir: str | Path = "results/models") -> list[dict[str, Any]]:
    """List run directories that contain both ``metrics.json`` and a model."""
    root = Path(results_dir)
    runs: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        model_dir = metrics_path.parent / "model"
        if not model_dir.exists():
            continue
        payload = read_json(metrics_path)
        runs.append(
            {
                "run_name": payload.get("run_name", metrics_path.parent.name),
                "task": payload.get("task"),
                "model": payload.get("model"),
                "seed": payload.get("seed"),
                "macro_f1": payload.get("macro_f1"),
                "model_dir": str(model_dir),
            }
        )
    runs.sort(key=lambda item: (str(item.get("task")), -(item.get("macro_f1") or 0)))
    return runs


def predict_text(
    text: str,
    model_dir: str | Path,
    task: str = "cyberbullying",
    max_length: int = 128,
) -> dict[str, Any]:
    """Predict the label of one text using a saved sequence-classification model."""
    import torch

    if task not in TASK_LABELS:
        raise ValueError(f"Unknown task: {task!r}")

    model, tokenizer = _load_artifact(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cleaned = clean_text(text)
    encoded = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
    probabilities = torch.softmax(logits, dim=-1)[0].tolist()
    label_id = int(torch.argmax(logits, dim=-1).item())
    label_names = TASK_LABELS[task]
    return {
        "text": cleaned,
        "label_id": label_id,
        "label": label_names[label_id],
        "confidence": float(probabilities[label_id]),
        "probabilities": {
            label_names[index]: float(value)
            for index, value in enumerate(probabilities)
            if index in label_names
        },
    }

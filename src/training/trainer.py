"""Run one fine-tuning experiment and persist its artefacts."""

from __future__ import annotations

import inspect
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.data.tokenize import tokenize_splits
from src.models.loader import (
    count_parameters,
    load_model_and_tokenizer,
    model_disk_size_mb,
)
from src.training.metrics import compute_metrics
from src.utils.config import resolve_path
from src.utils.io import write_json
from src.utils.seed import set_seed


def build_class_weights(labels: list[int] | np.ndarray, num_labels: int) -> Any:
    """Return inverse-frequency class weights as a float tensor."""
    import torch

    values = np.asarray(labels, dtype=int)
    counts = np.bincount(values, minlength=num_labels).astype(float)
    weights = np.ones(num_labels, dtype=float)
    present = counts > 0
    weights[present] = len(values) / (num_labels * counts[present])
    return torch.tensor(weights, dtype=torch.float)


def _training_arguments(
    config: dict[str, Any],
    output_dir: Path,
    fp16: bool,
    total_steps: int | None = None,
) -> Any:
    from transformers import TrainingArguments

    training = config.get("training", {})
    signature = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": float(training.get("epochs", 4)),
        "per_device_train_batch_size": int(training.get("batch_size", 16)),
        "per_device_eval_batch_size": int(training.get("eval_batch_size", 32)),
        "learning_rate": float(training.get("learning_rate", 2e-5)),
        "weight_decay": float(training.get("weight_decay", 0.01)),
        "gradient_accumulation_steps": int(
            training.get("gradient_accumulation_steps", 1)
        ),
        "logging_steps": int(training.get("logging_steps", 10)),
        "save_strategy": "epoch",
        strategy_key: "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": training.get("metric_for_best_model", "macro_f1"),
        "greater_is_better": True,
        "save_total_limit": 2,
        "save_only_model": True,
        "fp16": fp16,
        "report_to": [],
        "seed": int(config.get("seed", 42)),
        "data_seed": int(config.get("seed", 42)),
        "remove_unused_columns": True,
        "label_names": ["labels"],
        "disable_tqdm": bool(training.get("disable_tqdm", False)),
    }

    # transformers >= 5 replaced warmup_ratio with warmup_steps. Compute the
    # equivalent number of warmup steps when the ratio is configured.
    if "warmup_ratio" in signature:
        kwargs["warmup_ratio"] = float(training.get("warmup_ratio", 0.1))
    elif total_steps:
        warmup_ratio = float(training.get("warmup_ratio", 0.1))
        kwargs["warmup_steps"] = max(1, int(round(warmup_ratio * total_steps)))

    supported = set(signature)
    kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return TrainingArguments(**kwargs)


def _weighted_trainer_class() -> Any:
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def __init__(self, *args: Any, class_weights: Any = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(
            self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any
        ) -> Any:
            import torch

            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            if self.class_weights is not None:
                weight = self.class_weights.to(logits.device)
                loss_function = torch.nn.CrossEntropyLoss(weight=weight)
            else:
                loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def _build_trainer(**kwargs: Any) -> Any:
    from transformers import Trainer

    trainer_class = _weighted_trainer_class()
    try:
        return trainer_class(**kwargs)
    except TypeError:
        # Newer transformers renamed ``tokenizer`` to ``processing_class``.
        if "tokenizer" in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        return trainer_class(**kwargs)


def cleanup_checkpoints(output_dir: str | Path) -> None:
    """Remove trainer checkpoints; the final model is already saved separately."""
    checkpoints = Path(output_dir) / "checkpoints"
    if checkpoints.exists():
        shutil.rmtree(checkpoints, ignore_errors=True)


def _strip_prefix(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        key[len(prefix):] if key.startswith(prefix) else key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Fine-tune one model on one task and write metrics/efficiency artefacts."""
    import torch
    from transformers import EarlyStoppingCallback

    task = config["task"]
    seed = int(config.get("seed", 42))
    model_name = config["model"]["name"]
    run_name = config.get(
        "run_name",
        f"{task}_{model_name.split('/')[-1]}_{seed}",
    )
    output_dir = resolve_path(
        config.get("output_dir", f"results/models/{run_name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    data_config = config.get("data", {})
    processed_dir = resolve_path(
        data_config.get("processed_dir", f"data/processed/{task}")
    )
    text_col = data_config.get("text_col", "text")
    label_col = data_config.get("label_col", "label")
    num_labels = int(data_config.get("num_labels", 2))

    splits = {
        name: pd.read_csv(processed_dir / f"{name}.csv")
        for name in ("train", "val", "test")
    }

    max_length = int(config.get("model", {}).get("max_length", 128))
    cache_dir = config.get("model", {}).get("cache_dir")
    model, tokenizer = load_model_and_tokenizer(
        model_name, num_labels=num_labels, cache_dir=cache_dir
    )
    tokenized = tokenize_splits(
        splits,
        tokenizer,
        max_length=max_length,
        text_col=text_col,
        label_col=label_col,
    )

    class_weights = build_class_weights(
        splits["train"][label_col].astype(int).tolist(), num_labels
    )

    training = config.get("training", {})
    use_fp16 = bool(training.get("fp16", True)) and torch.cuda.is_available()
    batch_size = max(1, int(training.get("batch_size", 16)))
    accumulation = max(1, int(training.get("gradient_accumulation_steps", 1)))
    steps_per_epoch = max(
        1,
        (len(splits["train"]) + batch_size * accumulation - 1)
        // (batch_size * accumulation),
    )
    total_steps = steps_per_epoch * max(1, int(training.get("epochs", 4)))
    training_arguments = _training_arguments(
        config, output_dir, use_fp16, total_steps=total_steps
    )

    trainer = _build_trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=int(
                    training.get("early_stopping_patience", 2)
                )
            )
        ],
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    train_output = trainer.train()
    train_seconds = time.perf_counter() - started

    test_started = time.perf_counter()
    prediction = trainer.predict(tokenized["test"])
    test_seconds = time.perf_counter() - test_started

    logits = (
        prediction.predictions
        if hasattr(prediction, "predictions")
        else prediction["predictions"]
    )
    true_labels = (
        prediction.label_ids
        if hasattr(prediction, "label_ids")
        else prediction["label_ids"]
    )
    predicted = np.argmax(logits, axis=-1)
    true_labels = np.asarray(true_labels).reshape(-1)
    test_metrics = compute_metrics((logits, true_labels))
    labels_order = list(range(num_labels))
    report = classification_report(
        true_labels,
        predicted,
        labels=labels_order,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted, labels=labels_order)

    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))
    cleanup_checkpoints(output_dir)

    efficiency = {
        "parameters": count_parameters(model),
        "model_size_mb": model_disk_size_mb(model),
        "train_seconds": round(train_seconds, 3),
        "test_seconds": round(test_seconds, 3),
        "latency_ms_per_sample": round(
            (test_seconds * 1000) / max(1, len(true_labels)), 3
        ),
        "throughput_samples_per_second": round(
            len(true_labels) / max(1e-9, test_seconds), 2
        ),
        "peak_gpu_memory_mb": round(
            torch.cuda.max_memory_allocated() / (1024 * 1024), 2
        )
        if torch.cuda.is_available()
        else 0.0,
        "fp16": use_fp16,
    }

    metrics_payload = {
        "run_name": run_name,
        "task": task,
        "model": model_name,
        "seed": seed,
        **test_metrics,
    }
    write_json(metrics_payload, output_dir / "metrics.json")
    write_json(efficiency, output_dir / "efficiency.json")
    write_json(
        {
            "confusion_matrix": matrix.tolist(),
            "labels": labels_order,
            "classification_report": report,
        },
        output_dir / "test_report.json",
    )
    write_json(
        {
            "config": config,
            "train_metrics": _strip_prefix(
                getattr(train_output, "metrics", {}) or {}, "train_"
            ),
            "test_metrics": test_metrics,
            "efficiency": efficiency,
        },
        output_dir / "run_summary.json",
    )
    return {
        "run_name": run_name,
        "metrics": test_metrics,
        "efficiency": efficiency,
        "output_dir": str(output_dir),
    }

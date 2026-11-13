"""Generate comparison figures from collected metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _model_label(model: str) -> str:
    return str(model).split("/")[-1]


def plot_comparisons(metrics: pd.DataFrame, figures_dir: str | Path) -> list[Path]:
    """Create comparison figures and return the written paths."""
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if metrics.empty or "macro_f1" not in metrics.columns:
        return written

    frame = metrics.copy()
    frame["model_label"] = frame["model"].map(_model_label)

    for task, subset in frame.groupby("task", dropna=False):
        if subset.empty:
            continue
        summary = (
            subset.groupby("model_label")["macro_f1"]
            .agg(["mean", "std"])
            .sort_values("mean", ascending=False)
        )
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(
            summary.index,
            summary["mean"],
            yerr=summary["std"].fillna(0),
            capsize=4,
            color="#3b6ea5",
        )
        axis.set_title(f"Macro-F1 per model — {task}")
        axis.set_ylabel("Macro-F1")
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=20)
        figure.tight_layout()
        path = output_dir / f"macro_f1_{task}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)

        if {"eff_parameters", "macro_f1"}.issubset(subset.columns):
            figure, axis = plt.subplots(figsize=(7, 4.5))
            for label, group in subset.groupby("model_label"):
                axis.scatter(
                    group["eff_parameters"] / 1_000_000,
                    group["macro_f1"],
                    label=label,
                )
            axis.set_title(f"Efficiency vs Macro-F1 — {task}")
            axis.set_xlabel("Parameters (millions)")
            axis.set_ylabel("Macro-F1")
            axis.set_ylim(0, 1)
            axis.legend(fontsize=7)
            figure.tight_layout()
            path = output_dir / f"efficiency_{task}.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            written.append(path)

    if "seed" in frame.columns and frame["seed"].nunique() > 1:
        figure, axis = plt.subplots(figsize=(9, 4.5))
        frame.boxplot(column="macro_f1", by="model_label", ax=axis)
        axis.set_title("Macro-F1 across seeds")
        axis.set_xlabel("Model")
        axis.set_ylabel("Macro-F1")
        figure.suptitle("")
        axis.tick_params(axis="x", rotation=20)
        figure.tight_layout()
        path = output_dir / "macro_f1_by_seed.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)

    return written

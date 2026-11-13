"""Run every (model, seed) experiment for one task."""

from __future__ import annotations

import argparse
import copy
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.trainer import run_experiment  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _merge(base: dict, model: dict, seed: int, model_key: str) -> dict:
    config = copy.deepcopy(base)
    model_config = dict(model)
    model_config.pop("key", None)
    model_config.pop("optional", None)
    config["model"] = {**config.get("model", {}), **model_config}
    config["seed"] = seed
    config.pop("output_dir", None)
    config["run_name"] = f"{config['task'].replace('cyberbullying', 'cyber')}_{model_key}_s{seed}"
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all model/seed combinations for a task.")
    parser.add_argument("--task", required=True, choices=["cyberbullying", "buzzer"])
    parser.add_argument("--config", help="Base config; defaults to configs/<task>/base.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--models", help="Comma-separated model keys to run.")
    parser.add_argument("--seeds", help="Comma-separated seeds to override the registry.")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config or f"configs/{args.task}/base.yaml")
    registry = load_config(args.models_config)
    models = registry.get("models", [])
    if args.models:
        wanted = {item.strip() for item in args.models.split(",") if item.strip()}
        models = [model for model in models if model["key"] in wanted]
    elif not args.include_optional:
        models = [model for model in models if not model.get("optional", False)]

    seeds = (
        [int(item) for item in args.seeds.split(",")]
        if args.seeds
        else [int(seed) for seed in registry.get("seeds", [42])]
    )

    failures: list[tuple[str, str]] = []
    for model in models:
        for seed in seeds:
            config = _merge(base, model, seed, model["key"])
            print(f"\n=== {config['run_name']} ===")
            if args.dry_run:
                print(f"model={config['model']['name']} seed={seed}")
                continue
            try:
                result = run_experiment(config)
                print(f"macro_f1={result['metrics'].get('macro_f1')}")
            except Exception as error:  # pragma: no cover - CLI guard
                traceback.print_exc()
                failures.append((config["run_name"], str(error)))

    if failures:
        print("\nFailed runs:")
        for run_name, error in failures:
            print(f"- {run_name}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

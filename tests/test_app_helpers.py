from __future__ import annotations

from src.models.inference import load_available_models
from src.utils.io import write_json


def test_load_available_models_empty(tmp_path):
    assert load_available_models(tmp_path) == []


def test_load_available_models_returns_run(tmp_path):
    run_dir = tmp_path / "cyber_model_s42"
    (run_dir / "model").mkdir(parents=True)
    write_json(
        {
            "run_name": "cyber_model_s42",
            "task": "cyberbullying",
            "model": "indobenchmark/indobert-lite-base-p2",
            "seed": 42,
            "macro_f1": 0.81,
        },
        run_dir / "metrics.json",
    )
    runs = load_available_models(tmp_path)
    assert len(runs) == 1
    assert runs[0]["run_name"] == "cyber_model_s42"
    assert runs[0]["macro_f1"] == 0.81
    assert runs[0]["model_dir"].endswith("model")


def test_load_available_models_skips_missing_model_dir(tmp_path):
    run_dir = tmp_path / "incomplete"
    run_dir.mkdir()
    write_json({"run_name": "incomplete", "task": "buzzer"}, run_dir / "metrics.json")
    assert load_available_models(tmp_path) == []

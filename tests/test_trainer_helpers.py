from __future__ import annotations

from pathlib import Path

from src.training.trainer import cleanup_checkpoints
from src.utils.io import copy_json_artifacts


def test_cleanup_checkpoints_removes_directory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "optimizer.pt").write_text("large")

    cleanup_checkpoints(tmp_path)

    assert not (tmp_path / "checkpoints").exists()


def test_cleanup_checkpoints_missing_directory_is_noop(tmp_path: Path) -> None:
    cleanup_checkpoints(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_copy_json_artifacts_skips_model_weights(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text('{"macro_f1": 0.5}')
    (run_dir / "efficiency.json").write_text("{}")
    model_dir = run_dir / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_text("weights")

    copied = copy_json_artifacts(run_dir, tmp_path / "backup" / "run")

    assert (tmp_path / "backup" / "run" / "metrics.json").exists()
    assert (tmp_path / "backup" / "run" / "efficiency.json").exists()
    assert not (tmp_path / "backup" / "run" / "model").exists()
    assert len(copied) == 2

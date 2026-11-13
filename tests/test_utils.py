from __future__ import annotations

import random

import numpy as np
import pandas as pd
import yaml

from src.utils.config import load_config, project_root, resolve_path
from src.utils.io import read_json, read_table, write_json
from src.utils.seed import set_seed


def test_load_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"task": "cyberbullying", "seed": 42}))
    config = load_config(config_path)
    assert config["task"] == "cyberbullying"
    assert config["seed"] == 42


def test_read_table_latin1_semicolon(tmp_path):
    path = tmp_path / "data.csv"
    frame = pd.DataFrame({"text": ["halo", "dunia"], "label": [1, 0]})
    frame.to_csv(path, sep=";", index=False, encoding="latin-1")
    loaded = read_table(path)
    assert loaded.columns.tolist() == ["text", "label"]
    assert loaded.shape == (2, 2)


def test_read_table_tsv_utf8_bom(tmp_path):
    path = tmp_path / "data.tsv"
    pd.DataFrame({"text": ["a"], "label": [1]}).to_csv(
        path, sep="\t", index=False, encoding="utf-8-sig"
    )
    loaded = read_table(path)
    assert loaded.shape == (1, 2)


def test_json_roundtrip(tmp_path):
    path = tmp_path / "nested" / "payload.json"
    write_json({"a": 1, "b": "teks"}, path)
    assert read_json(path) == {"a": 1, "b": "teks"}


def test_resolve_path_relative_to_root():
    assert resolve_path("results/models") == project_root() / "results" / "models"
    absolute = resolve_path("/tmp/x")
    assert str(absolute) == "/tmp/x"


def test_set_seed_is_deterministic():
    set_seed(42)
    first = (random.random(), float(np.random.rand()))
    set_seed(42)
    second = (random.random(), float(np.random.rand()))
    assert first == second

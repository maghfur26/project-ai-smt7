from __future__ import annotations

import pandas as pd

from src.data.prepare import prepare_task


def _write_cyber_sources(tmp_path):
    first = tmp_path / "first.tsv"
    rows = []
    for index in range(40):
        rows.append(
            {
                "text": f"komentar pertama nomor {index} kata{index}",
                "label": "CB" if index % 2 else "Non_CB",
            }
        )
    pd.DataFrame(rows).to_csv(first, sep="\t", index=False, encoding="latin-1")

    second = tmp_path / "second.csv"
    rows = []
    for index in range(40):
        rows.append(
            {
                "Instagram Comment Text": f"komentar kedua nomor {index} unik{index}",
                "Sentiment": "negative" if index % 2 else "positive",
            }
        )
    pd.DataFrame(rows).to_csv(second, index=False, encoding="utf-8")

    unknown = tmp_path / "unknown.csv"
    pd.DataFrame(
        {"Instagram Comment Text": ["komentar aneh"], "Sentiment": ["tidak tahu"]}
    ).to_csv(unknown, index=False, encoding="utf-8")
    return first, second, unknown


def test_prepare_cyberbullying(tmp_path):
    first, second, unknown = _write_cyber_sources(tmp_path)
    config = {
        "task": "cyberbullying",
        "seed": 42,
        "output_dir": str(tmp_path / "out"),
        "deduplicate": True,
        "group_near_duplicates": False,
        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        "sources": [
            {
                "name": "first",
                "path": str(first),
                "text_col": "text",
                "label_col": "label",
                "label_map": {"CB": 1, "Non_CB": 0},
            },
            {
                "name": "second",
                "path": str(second),
                "text_col": "Instagram Comment Text",
                "label_col": "Sentiment",
                "label_map": {"negative": 1, "positive": 0},
            },
            {
                "name": "unknown",
                "path": str(unknown),
                "text_col": "Instagram Comment Text",
                "label_col": "Sentiment",
            },
        ],
    }
    paths = prepare_task(config)
    assert set(paths) == {"train", "val", "test", "stats"}
    for name in ("train", "val", "test"):
        frame = pd.read_csv(paths[name])
        assert {"text", "label", "source", "group_id"}.issubset(frame.columns)
        assert frame["label"].isin([0, 1]).all()
    stats = pd.read_json(paths["stats"], typ="series")
    assert stats["rows"]["train"] > 0
    assert stats["dropped_unknown_labels"] >= 1


def test_prepare_buzzer_weak_labels(tmp_path):
    rows = []
    spam = "ayo viralkan sebarkan ke semua grup dukung terus sampai trending"
    for index in range(8):
        rows.append({"Text Tweet": spam})
    for index in range(40):
        rows.append({"Text Tweet": f"diskusi politik biasa nomor {index} unik{index}"})
    source = tmp_path / "tweets.csv"
    pd.DataFrame(rows).to_csv(source, index=False, encoding="utf-8")

    config = {
        "task": "buzzer",
        "seed": 42,
        "output_dir": str(tmp_path / "out_buzzer"),
        "deduplicate": False,
        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        "weak_label": {
            "similarity_threshold": 0.85,
            "min_cluster_size": 3,
            "score_threshold": 0.5,
        },
        "sources": [
            {
                "name": "tweets",
                "path": str(source),
                "text_col": "Text Tweet",
                "label_col": None,
            }
        ],
    }
    paths = prepare_task(config)
    combined = pd.concat([pd.read_csv(paths[name]) for name in ("train", "val", "test")])
    assert len(combined) == len(rows)
    assert combined["label"].sum() >= 8
    assert "weak_score" in combined.columns

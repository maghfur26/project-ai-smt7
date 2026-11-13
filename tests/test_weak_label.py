from __future__ import annotations

import pandas as pd

from src.data.weak_label import near_duplicate_clusters, weak_label_frame


def test_near_duplicate_clusters_group_identical_texts():
    texts = ["sebarkan berita ini sekarang", "sebarkan berita ini sekarang", "makan siang enak"]
    clusters = near_duplicate_clusters(texts, threshold=0.85)
    assert clusters[0] == clusters[1]
    assert clusters[2] != clusters[0]


def test_near_duplicate_clusters_handle_empty():
    assert near_duplicate_clusters(["", "halo"])[0] == -1


def test_weak_label_marks_repeated_template_as_buzzer():
    text = "ayo viralkan sebarkan ke semua grup dukung terus sampai trending"
    frame = pd.DataFrame({"text": [text] * 5 + ["saya sedang membaca buku di rumah"]})
    result = weak_label_frame(
        frame,
        {"similarity_threshold": 0.85, "min_cluster_size": 3, "score_threshold": 0.5},
    )
    assert result.loc[0, "label"] == 1
    assert result.loc[5, "label"] == 0
    assert result.loc[0, "group_id"] == result.loc[4, "group_id"]
    assert "weak_score" in result.columns
    assert "cluster_id" in result.columns


def test_weak_label_empty_frame():
    frame = pd.DataFrame({"text": []})
    result = weak_label_frame(frame)
    assert result.empty
    assert {"label", "weak_score", "cluster_id", "group_id"}.issubset(result.columns)

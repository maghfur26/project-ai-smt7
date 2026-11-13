"""Weak supervision rules for buzzer-like Indonesian social-media text.

The rules intentionally operate on text and optional metadata (user id,
timestamp). They produce a *proxy* label, not verified buzzer behaviour.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

_HASHTAG_RE = re.compile(r"#\w+")
_MENTION_RE = re.compile(r"@\w+")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

DEFAULT_CONFIG = {
    "similarity_threshold": 0.85,
    "min_cluster_size": 3,
    "neighbors": 20,
    "score_threshold": 0.5,
    "weights": {
        "cluster": 0.5,
        "hashtag": 0.15,
        "link": 0.15,
        "lexicon": 0.2,
    },
    "lexicon": [
        "ayo",
        "sebarkan",
        "viralkan",
        "bagikan",
        "ikuti",
        "follow",
        "like",
        "komentar",
        "share",
        "klik",
        "promo",
        "diskon",
        "gratis",
        "dukung",
        "pilih",
        "menangkan",
        "golput",
        "buzzer",
        "buzzerbayaran",
        "sebarluaskan",
        "trending",
        "serbu",
        "balas",
        "tag",
        "viral",
        "kampanye",
        "provokasi",
    ],
}


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _dedupe_texts(texts: Sequence[str]) -> tuple[list[str], list[int]]:
    """Return unique non-empty texts and the original index for each."""
    seen: dict[str, int] = {}
    unique: list[str] = []
    origin: list[int] = []
    for index, value in enumerate(texts):
        text = "" if value is None else str(value).strip()
        if not text:
            continue
        if text not in seen:
            seen[text] = len(unique)
            unique.append(text)
        origin.append(seen[text])
    return unique, origin


def near_duplicate_clusters(
    texts: Sequence[str], threshold: float = 0.85, neighbors: int = 20
) -> list[int]:
    """Cluster near-duplicate texts and return a cluster id for every input.

    Empty inputs receive cluster id ``-1``. Non-empty texts are vectorized with
    character n-grams so that template variants with punctuation or small
    edits still land in the same cluster.
    """
    if not texts:
        return []
    unique, origin = _dedupe_texts(texts)
    if not unique:
        return [-1 for _ in texts]

    if len(unique) == 1:
        unique_clusters = [0]
    else:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        matrix = vectorizer.fit_transform(unique)
        k = min(max(1, neighbors), len(unique))
        model = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        model.fit(matrix)
        distances, indices = model.kneighbors(matrix)

        union_find = _UnionFind(len(unique))
        for row in range(len(unique)):
            for distance, column in zip(distances[row], indices[row]):
                if row == column:
                    continue
                similarity = 1.0 - float(distance)
                if similarity >= threshold:
                    union_find.union(row, int(column))
        unique_clusters = [union_find.find(index) for index in range(len(unique))]

    result: list[int] = []
    text_to_cluster: dict[str, int] = {}
    for text, cluster in zip(unique, unique_clusters):
        text_to_cluster[text] = cluster
    for value in texts:
        text = "" if value is None else str(value).strip()
        result.append(text_to_cluster.get(text, -1) if text else -1)
    return result


def _density(pattern: re.Pattern[str], text: str) -> float:
    tokens = max(1, len(text.split()))
    return len(pattern.findall(text)) / tokens


def _lexicon_density(text: str, lexicon: set[str]) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token.strip(".,!?;:\"'()[]") in lexicon)
    return hits / len(tokens)


def weak_label_frame(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Add ``label``, ``weak_score``, ``cluster_id``, and ``group_id`` columns."""
    settings = {**DEFAULT_CONFIG, **(config or {})}
    weights = {**DEFAULT_CONFIG["weights"], **settings.get("weights", {})}
    lexicon = {str(word).lower() for word in settings.get("lexicon", DEFAULT_CONFIG["lexicon"])}

    if "text" not in df.columns:
        raise KeyError("weak_label_frame requires a 'text' column")

    texts = df["text"].astype(str).tolist()
    raw_texts = (
        df["text_raw"].astype(str).tolist()
        if "text_raw" in df.columns
        else texts
    )
    clusters = near_duplicate_clusters(
        texts,
        threshold=float(settings["similarity_threshold"]),
        neighbors=int(settings["neighbors"]),
    )
    cluster_sizes = Counter(cluster for cluster in clusters if cluster >= 0)
    min_size = max(1, int(settings["min_cluster_size"]))
    weight_total = max(1e-9, sum(float(value) for value in weights.values()))

    scores: list[float] = []
    for text, raw_text, cluster in zip(texts, raw_texts, clusters):
        cluster_score = 1.0 if cluster >= 0 and cluster_sizes[cluster] >= min_size else 0.0
        hashtag_score = min(1.0, _density(_HASHTAG_RE, raw_text) * 10.0)
        link_score = 1.0 if _URL_RE.search(raw_text) else 0.0
        lexicon_score = min(1.0, _lexicon_density(raw_text, lexicon) * 5.0)
        score = (
            cluster_score * float(weights["cluster"])
            + hashtag_score * float(weights["hashtag"])
            + link_score * float(weights["link"])
            + lexicon_score * float(weights["lexicon"])
        ) / weight_total
        scores.append(score)

    result = df.copy()
    result["cluster_id"] = clusters
    result["weak_score"] = scores
    result["label"] = [
        int(score >= float(settings["score_threshold"])) for score in scores
    ]
    result["group_id"] = [
        f"c{cluster}" if cluster >= 0 else f"u{index}"
        for index, cluster in enumerate(clusters)
    ]
    return result

"""Public dataset sources and download helpers."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.cleaning import clean_text
from src.data.labels import harmonize_label
from src.utils.io import read_table
from src.utils.config import resolve_path

PUBLIC_SOURCES: dict[str, list[dict[str, Any]]] = {
    "cyberbullying": [
        {
            "name": "cahkanor_instagram",
            "url": "https://raw.githubusercontent.com/cahkanor/id-cyberbullying-instagram/master/id-cyberbullying-instagram.tsv",
            "filename": "cahkanor_instagram.tsv",
            "text_col": "text",
            "label_col": "label",
            "label_map": {"CB": 1, "Non_CB": 0},
            "domain": "instagram",
        },
        {
            "name": "rizalespe_instagram",
            "url": "https://raw.githubusercontent.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia/master/dataset_komentar_instagram_cyberbullying.csv",
            "filename": "rizalespe_instagram.csv",
            "text_col": "Instagram Comment Text",
            "label_col": "Sentiment",
            "label_map": {"negative": 1, "positive": 0},
            "domain": "instagram",
        },
        {
            "name": "rhiosutoyo_tiktok",
            "url": "https://raw.githubusercontent.com/rhiosutoyo/Indonesian-TikTok-Cyberbullying-Comments-Dataset/main/Dataset-Research.csv",
            "filename": "rhiosutoyo_tiktok.csv",
            "text_col": "comment",
            "label_col": "sentiment",
            "label_map": {"-1": 1, "1": 0},
            "domain": "tiktok",
        },
    ],
    "buzzer": [
        {
            "name": "rizalespe_pilkada_tweets",
            "url": "https://raw.githubusercontent.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia/master/dataset_tweet_sentiment_pilkada_DKI_2017.csv",
            "filename": "rizalespe_pilkada_tweets.csv",
            "text_col": "Text Tweet",
            "label_col": None,
            "domain": "twitter",
        },
    ],
}


def _download(url: str, destination: Path, timeout: int = 60) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "slm-buzzer-cyberbullying/1.0 (academic project)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            destination.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(
            f"Failed to download {url}. Place the file manually at {destination}."
        ) from error
    return destination


def download_public_sources(
    dest_dir: str | Path, task: str = "cyberbullying"
) -> dict[str, Path]:
    """Download all GitHub-hosted sources for a task.

    Kaggle sources are not downloaded automatically because they require
    authentication; see ``data/DATASET_CARD.md`` for manual instructions.
    """
    if task not in PUBLIC_SOURCES:
        raise ValueError(f"Unknown task: {task!r}")
    output_dir = resolve_path(dest_dir)
    downloaded: dict[str, Path] = {}
    for spec in PUBLIC_SOURCES[task]:
        destination = output_dir / spec["filename"]
        downloaded[spec["name"]] = _download(spec["url"], destination)
    return downloaded


def load_source(spec: dict[str, Any], task: str) -> pd.DataFrame:
    """Load one source and normalize it to ``text``/``label`` columns.

    ``spec`` may contain either ``url`` or ``path``. A per-source ``label_map``
    takes precedence over the generic harmonizer.
    """
    if "path" in spec and spec["path"]:
        path = resolve_path(spec["path"])
    elif "url" in spec and spec["url"]:
        raw_dir = spec.get("raw_dir", f"data/raw/{task}")
        path = _download(spec["url"], resolve_path(raw_dir) / spec["filename"])
    else:
        raise ValueError(f"Source {spec.get('name', '?')} has neither path nor url")

    frame = read_table(path)
    text_col = spec["text_col"]
    if text_col not in frame.columns:
        raise KeyError(
            f"Column {text_col!r} not found in {path}. Columns: {list(frame.columns)}"
        )

    result = pd.DataFrame()
    result["text_raw"] = frame[text_col].astype(object)
    result["source"] = spec.get("name", path.stem)
    result["domain"] = spec.get("domain", "unknown")

    label_col = spec.get("label_col")
    if label_col is None:
        result["label"] = pd.NA
    else:
        if label_col not in frame.columns:
            raise KeyError(
                f"Column {label_col!r} not found in {path}. "
                f"Columns: {list(frame.columns)}"
            )
        raw_labels = frame[label_col]
        label_map = spec.get("label_map")
        if label_map:
            normalized_map = {str(key).strip().lower(): int(value) for key, value in label_map.items()}
            result["label"] = [
                normalized_map.get(str(value).strip().lower())
                for value in raw_labels
            ]
        else:
            result["label"] = [harmonize_label(value, task) for value in raw_labels]

    if "group_id" in spec and spec["group_id"] in frame.columns:
        result["group_id"] = frame[spec["group_id"]].astype(str)
    else:
        result["group_id"] = pd.NA

    keep_emoji = bool(spec.get("keep_emoji", True))
    result["text"] = [clean_text(value, keep_emoji=keep_emoji) for value in result["text_raw"]]
    return result

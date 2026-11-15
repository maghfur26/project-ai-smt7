"""Robust table and JSON IO helpers.

The public datasets used in this project are inconsistent: some are UTF-8 with
a BOM, one is Latin-1 encoded, and delimiters vary between comma and tab.
``read_table`` hides those differences behind a single function.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")
_DELIMITERS = ",;\t|"


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV/TSV/JSON/JSONL/XLSX file with encoding fallback."""
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Table not found: {table_path}")

    suffix = table_path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(table_path, lines=True)
    if suffix == ".json":
        with table_path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return pd.json_normalize(payload)
        return pd.DataFrame(payload)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(table_path)

    last_error: Exception | None = None
    for encoding in _ENCODINGS:
        try:
            if suffix in {".tsv", ".tab"}:
                return pd.read_csv(table_path, sep="\t", encoding=encoding)
            sample = table_path.read_text(encoding=encoding)[:16384]
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=_DELIMITERS).delimiter
            except csv.Error:
                delimiter = ","
            return pd.read_csv(table_path, sep=delimiter, encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError) as error:  # pragma: no cover
            last_error = error
    raise ValueError(f"Could not read {table_path}: {last_error}")


def write_json(payload: Any, path: str | Path) -> None:
    """Write JSON with UTF-8 encoding, creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def copy_json_artifacts(source_dir: str | Path, dest_dir: str | Path) -> list[Path]:
    """Copy small JSON artefacts (not model weights) into a backup directory."""
    source = Path(source_dir)
    destination = Path(dest_dir)
    copied: list[Path] = []
    for path in sorted(source.glob("*.json")):
        target = destination / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return copied

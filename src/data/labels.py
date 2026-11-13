"""Canonical label schemes and raw-label harmonization."""

from __future__ import annotations

from typing import Any

TASK_LABELS: dict[str, dict[int, str]] = {
    "cyberbullying": {0: "non_cyberbullying", 1: "cyberbullying"},
    "buzzer": {0: "non_buzzer", 1: "buzzer_like"},
}

_CYBER_POSITIVE = {
    "cb",
    "cyber",
    "cyberbullying",
    "cyber_bullying",
    "cyber-bullying",
    "bully",
    "bullying",
    "negative",
    "negatif",
    "neg",
    "hate",
    "abusive",
    "toxic",
}
_CYBER_NEGATIVE = {
    "non_cb",
    "noncb",
    "non-cb",
    "non_cb",
    "noncyberbullying",
    "non_cyberbullying",
    "non-cyberbullying",
    "not_cb",
    "notcb",
    "normal",
    "positive",
    "positif",
    "pos",
    "neutral",
    "netral",
    "non-bullying",
    "nonbullying",
}
_BUZZER_POSITIVE = {"buzzer", "buzzer_like", "buzzer-like", "buzzerlike", "1", "true"}
_BUZZER_NEGATIVE = {"non_buzzer", "non-buzzer", "nonbuzzer", "0", "false", "organic"}


def _normalize_raw(raw: Any) -> str:
    return str(raw).strip().lower().replace(" ", "_")


def harmonize_label(raw: Any, task: str) -> int | None:
    """Map a raw label to 0/1 for the given task, or ``None`` if unknown.

    Numeric labels are deliberately not guessed because different sources use
    opposite conventions (``-1`` can mean cyberbullying in one dataset while
    ``1`` means cyberbullying in another). Numeric sources should define an
    explicit ``label_map`` in their data config.
    """
    if raw is None:
        return None
    token = _normalize_raw(raw)
    if token in {"", "nan", "none"}:
        return None

    if task == "cyberbullying":
        if token in _CYBER_POSITIVE:
            return 1
        if token in _CYBER_NEGATIVE:
            return 0
        return None
    if task == "buzzer":
        if token in _BUZZER_POSITIVE:
            return 1
        if token in _BUZZER_NEGATIVE:
            return 0
        return None
    raise ValueError(f"Unknown task: {task!r}. Expected 'cyberbullying' or 'buzzer'.")

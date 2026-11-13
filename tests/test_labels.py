from __future__ import annotations

import pytest

from src.data.labels import harmonize_label


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CB", 1),
        ("Cyberbullying", 1),
        ("negative", 1),
        ("Non_CB", 0),
        ("non-cyberbullying", 0),
        ("positive", 0),
        ("netral", 0),
        ("tidak diketahui", None),
        ("-1", None),
        (None, None),
    ],
)
def test_harmonize_cyberbullying(raw, expected):
    assert harmonize_label(raw, "cyberbullying") == expected


def test_harmonize_buzzer():
    assert harmonize_label("buzzer", "buzzer") == 1
    assert harmonize_label("organic", "buzzer") == 0
    assert harmonize_label("unknown", "buzzer") is None


def test_harmonize_rejects_unknown_task():
    with pytest.raises(ValueError):
        harmonize_label("CB", "sentiment")

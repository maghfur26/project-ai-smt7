from __future__ import annotations

from src.data.cleaning import clean_text, normalize_slang


def test_clean_text_removes_urls_mentions_hashtags_and_lowercases():
    raw = "HALO @user cek https://example.com #viral SEKARANG"
    assert clean_text(raw) == "halo cek sekarang"


def test_clean_text_normalizes_slang():
    assert clean_text("gk tau yg bener") == "tidak tau yang benar"


def test_normalize_slang_keeps_unknown_words():
    assert normalize_slang("makan nasi") == "makan nasi"


def test_clean_text_removes_repeated_characters():
    assert "baguss" in clean_text("bagusss banget")


def test_clean_text_can_remove_emoji():
    assert clean_text("keren 😍🔥", keep_emoji=False).strip() == "keren"


def test_clean_text_keeps_emoji_by_default():
    assert "😍" in clean_text("keren 😍")


def test_clean_text_handles_empty_values():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text("   ") == ""

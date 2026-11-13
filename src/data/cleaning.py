"""Text cleaning utilities for Indonesian social-media comments."""

from __future__ import annotations

import html
import re
import unicodedata

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_RT_RE = re.compile(r"^\s*rt\s+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "]+"
)

_SLANG_MAP = {
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "tdk": "tidak",
    "tak": "tidak",
    "yg": "yang",
    "dgn": "dengan",
    "utk": "untuk",
    "krn": "karena",
    "karna": "karena",
    "sdh": "sudah",
    "udh": "sudah",
    "blm": "belum",
    "bgt": "banget",
    "bgtt": "banget",
    "aja": "saja",
    "aj": "saja",
    "jg": "juga",
    "jd": "jadi",
    "klo": "kalau",
    "kl": "kalau",
    "gw": "saya",
    "gue": "saya",
    "gua": "saya",
    "lo": "kamu",
    "lu": "kamu",
    "loe": "kamu",
    "bikin": "buat",
    "dapet": "dapat",
    "gimana": "bagaimana",
    "gmn": "bagaimana",
    "knp": "kenapa",
    "kenapa": "kenapa",
    "dr": "dari",
    "dri": "dari",
    "sm": "sama",
    "sma": "sama",
    "tp": "tapi",
    "tpi": "tapi",
    "hrs": "harus",
    "bs": "bisa",
    "bisa": "bisa",
    "org": "orang",
    "orng": "orang",
    "kayak": "seperti",
    "kaya": "seperti",
    "kyk": "seperti",
    "bener": "benar",
    "bnr": "benar",
    "sgt": "sangat",
    "sangat": "sangat",
    "skrg": "sekarang",
    "skr": "sekarang",
    "td": "tadi",
    "kmrn": "kemarin",
    "bsk": "besok",
}

_SLANG_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _SLANG_MAP), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def normalize_slang(text: str) -> str:
    """Replace common Indonesian informal abbreviations with standard forms."""
    if not text:
        return ""

    def replace(match: re.Match[str]) -> str:
        return _SLANG_MAP[match.group(0).lower()]

    return _SLANG_PATTERN.sub(replace, text)


def clean_text(text: object, keep_emoji: bool = True, expand_slang: bool = True) -> str:
    """Normalize a raw social-media comment.

    Steps: unicode normalization, HTML unescape, URL/mention/hashtag removal,
    optional emoji removal, optional slang normalization, repeated-character
    trimming, and whitespace collapsing.
    """
    if text is None:
        return ""
    value = str(text)
    if value.lower() == "nan":
        return ""

    value = html.unescape(unicodedata.normalize("NFKC", value))
    value = value.lower()
    value = _URL_RE.sub(" ", value)
    value = _MENTION_RE.sub(" ", value)
    value = _HASHTAG_RE.sub(" ", value)
    value = _RT_RE.sub(" ", value)
    if not keep_emoji:
        value = _EMOJI_RE.sub(" ", value)
    if expand_slang:
        value = normalize_slang(value)
    value = _REPEATED_CHAR_RE.sub(r"\1\1", value)
    value = _WHITESPACE_RE.sub(" ", value).strip()
    return value

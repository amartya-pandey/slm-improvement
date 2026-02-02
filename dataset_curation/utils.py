"""
Shared preprocessing utilities for dataset curation.
"""

from __future__ import annotations

import hashlib
import html
import re
from typing import Iterable

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities."""
    if not text:
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_text(text: str) -> str:
    """Basic normalization: strip HTML, normalize whitespace."""
    return strip_html(text)


def is_english_text(text: str, min_alpha_ratio: float = 0.7) -> bool:
    """
    Heuristic language filter: checks ratio of ASCII letters to all letters.
    This is a lightweight alternative when language metadata is unavailable.
    """
    if not text:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_letters = [c for c in letters if "a" <= c.lower() <= "z"]
    ratio = len(ascii_letters) / len(letters)
    return ratio >= min_alpha_ratio


def text_too_short(text: str, min_chars: int = 200) -> bool:
    """Filter out short or low-information samples."""
    return len(text.strip()) < min_chars


def hash_text(text: str) -> str:
    """Stable hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate_texts(texts: Iterable[str]) -> list[str]:
    """Deduplicate an iterable of texts (in-memory)."""
    seen = set()
    unique = []
    for t in texts:
        h = hash_text(t)
        if h in seen:
            continue
        seen.add(h)
        unique.append(t)
    return unique

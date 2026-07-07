from __future__ import annotations

import math
import re
from collections import Counter


DEFAULT_REFERENCE = (
    "Machine Learning is a subset of artificial intelligence that allows systems "
    "to learn patterns from data and improve performance without being explicitly "
    "programmed."
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def _bag_of_words_similarity(text_a: str, text_b: str) -> float:
    a = Counter(_tokenize(text_a))
    b = Counter(_tokenize(text_b))
    vocab = set(a) | set(b)
    if not vocab:
        return 0.0
    dot = sum(a[word] * b[word] for word in vocab)
    mag_a = math.sqrt(sum(value * value for value in a.values()))
    mag_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


def semantic_similarity(student_text: str, reference_text: str = DEFAULT_REFERENCE) -> float:
    """Compute a lightweight cosine similarity for free-tier deployment."""
    score = _bag_of_words_similarity(student_text, reference_text)
    return round(max(0.0, min(score, 1.0)), 4)

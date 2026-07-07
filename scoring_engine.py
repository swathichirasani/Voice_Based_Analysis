from __future__ import annotations

import re
from typing import Dict, Tuple


FILLER_WORDS = {
    "um",
    "uh",
    "erm",
    "hmm",
    "like",
    "actually",
    "basically",
    "literally",
    "you know",
    "i mean",
    "sort of",
    "kind of",
}


def filler_word_ratio(transcript: str) -> float:
    words = re.findall(r"[a-zA-Z']+", transcript.lower())
    if not words:
        return 0.0

    single_fillers = {word for word in FILLER_WORDS if " " not in word}
    filler_count = sum(1 for word in words if word in single_fillers)
    lowered = " ".join(words)
    filler_count += sum(lowered.count(phrase) for phrase in FILLER_WORDS if " " in phrase)
    return round(filler_count / len(words), 4)


def evaluate_understanding(
    similarity: float, filler_ratio: float, audio: Dict[str, float]
) -> Tuple[int, str, str]:
    score = 0
    score += 50 if similarity > 0.7 else 30 if similarity > 0.4 else 10
    score += 20 if filler_ratio < 0.05 else 10
    score += 15 if audio.get("pause_ratio", 1.0) < 0.25 else 5
    score += 15 if audio.get("rms_energy", 0.0) > 0.01 else 5

    if score >= 80:
        return score, "Strong Understanding", "#22cc71"
    if score >= 50:
        return score, "Moderate Understanding", "#f39c12"
    return score, "Poor Understanding", "#e74c3c"

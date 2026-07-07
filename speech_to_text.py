from __future__ import annotations

import os
from pathlib import Path


def speech_to_text(audio_path: str | Path) -> str:
    """Transcribe audio with Whisper only when explicitly enabled."""
    if os.getenv("ENABLE_WHISPER", "").lower() not in {"1", "true", "yes"}:
        return ""

    try:
        import whisper

        model_name = os.getenv("WHISPER_MODEL", "tiny")
        model = whisper.load_model(model_name)
        result = model.transcribe(str(audio_path), fp16=False)
        text = str(result.get("text", "")).strip()
        return text
    except Exception:
        return ""

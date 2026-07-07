from __future__ import annotations

from pathlib import Path
from typing import Dict
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np


MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", "180"))
MAX_WAVEFORM_POINTS = 6000


def audio_duration(audio_path: str | Path) -> float:
    import soundfile as sf

    info = sf.info(str(audio_path))
    if not info.samplerate:
        return 0.0
    return float(info.frames / info.samplerate)


def load_audio(audio_path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio with SoundFile, using a small optional resampler when available."""
    path = str(audio_path)
    import soundfile as sf

    duration = audio_duration(path)
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"Audio is {duration:.0f} seconds long. Please upload audio up to "
            f"{MAX_AUDIO_SECONDS} seconds for this deployment."
        )

    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    if sr != target_sr:
        try:
            from scipy.signal import resample_poly

            gcd = np.gcd(sr, target_sr)
            data = resample_poly(data, target_sr // gcd, sr // gcd)
            data = np.asarray(data, dtype=np.float32)
            sr = target_sr
        except Exception:
            pass
    return data, int(sr)


def extract_audio_features_from_samples(y: np.ndarray, sr: int) -> Dict[str, float]:
    if y.size == 0:
        return {
            "pause_ratio": 1.0,
            "rms_energy": 0.0,
            "zero_crossing_rate": 0.0,
            "duration_sec": 0.0,
        }

    duration = float(len(y) / sr)
    rms = float(np.sqrt(np.mean(np.square(y))))
    threshold = max(rms * 0.35, 0.01)
    pause_ratio = float(np.mean(np.abs(y) < threshold))
    zero_crossings = np.mean(np.abs(np.diff(np.signbit(y))))

    return {
        "pause_ratio": round(pause_ratio, 4),
        "rms_energy": round(rms, 4),
        "zero_crossing_rate": round(float(zero_crossings), 4),
        "duration_sec": round(duration, 2),
    }


def extract_audio_features(audio_path: str | Path) -> Dict[str, float]:
    y, sr = load_audio(audio_path)
    return extract_audio_features_from_samples(y, sr)


def save_waveform_from_samples(y: np.ndarray, sr: int, output_path: str | Path) -> str:
    if y.size > MAX_WAVEFORM_POINTS:
        step = int(np.ceil(y.size / MAX_WAVEFORM_POINTS))
        y = y[::step]
    else:
        step = 1

    times = np.arange(len(y)) / sr if y.size else np.array([0])
    times = times * step
    samples = y if y.size else np.array([0])

    plt.figure(figsize=(8, 2.8), dpi=100)
    plt.plot(times, samples, color="#2e8ac8", linewidth=1)
    plt.title("Audio Waveform", fontsize=10)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.ylim(-1.05, 1.05)
    plt.grid(alpha=0.15)
    plt.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, facecolor="white")
    plt.close()
    return str(output)


def save_waveform(audio_path: str | Path, output_path: str | Path) -> str:
    y, sr = load_audio(audio_path)
    return save_waveform_from_samples(y, sr, output_path)

from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np

from voice_pipeline.inference.wav import read_wav

from .base import MetricResult


_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def evaluate_prosody(audio: Path, target_text: str) -> MetricResult:
    sample_rate, waveform = read_wav(audio)
    duration = waveform.size / sample_rate
    clipping_ratio = float(np.mean(np.abs(waveform) >= 0.999))
    import librosa

    frame_length = min(1024, max(256, 2 ** int(np.floor(np.log2(max(waveform.size, 256))))))
    hop_length = frame_length // 4
    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
    all_silent = bool(rms.size == 0 or float(np.max(rms)) < 1e-4)
    pause_ratio = float(np.mean(rms < 1e-3)) if rms.size else 1.0
    voiced = rms >= 1e-3
    f0_values = np.empty(0, dtype=np.float32)
    if not all_silent and frame_length > sample_rate / 50:
        f0 = librosa.yin(
            waveform,
            fmin=50,
            fmax=min(600, sample_rate / 2 - 1),
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        usable = min(f0.size, voiced.size)
        f0_values = f0[:usable][voiced[:usable] & np.isfinite(f0[:usable])]
    details = {
        "duration_seconds": float(duration),
        "f0_median": _stat(f0_values, "median"),
        "f0_range": _f0_range(f0_values),
        "f0_variance": _stat(f0_values, "var"),
        "voiced_ratio": float(np.mean(voiced)) if voiced.size else 0.0,
        "speaking_rate": _speech_units(target_text) / duration if duration else 0.0,
        "pause_ratio": pause_ratio,
        "rms_mean": float(np.mean(rms)) if rms.size else 0.0,
        "rms_std": float(np.std(rms)) if rms.size else 0.0,
        "clipping_ratio": clipping_ratio,
        "all_silent": all_silent,
        "severe_clipping": clipping_ratio > 0.05,
    }
    value = 0.0 if all_silent or details["severe_clipping"] else 1.0
    return MetricResult("prosody", value, details, True)


def _speech_units(text: str) -> int:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    if any(0x3040 <= ord(character) <= 0x30FF or 0x3400 <= ord(character) <= 0x9FFF for character in normalized):
        return sum(unicodedata.category(character)[0] in {"L", "N"} for character in normalized)
    return len(_WORD.findall(normalized))


def _stat(values: np.ndarray, operation: str) -> float | None:
    if not values.size:
        return None
    return float(np.median(values) if operation == "median" else np.var(values))


def _f0_range(values: np.ndarray) -> float | None:
    if not values.size:
        return None
    return float(np.percentile(values, 95) - np.percentile(values, 5))


__all__ = ["evaluate_prosody"]

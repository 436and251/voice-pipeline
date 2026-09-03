from __future__ import annotations

from pathlib import Path
import uuid
import wave

import numpy as np


def write_wav_atomic(path: str | Path, waveform: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    samples = np.asarray(waveform, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("waveform must be a non-empty finite mono array")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    pcm = np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with wave.open(str(temporary), "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(sample_rate)
            file.writeframes(pcm.tobytes())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as file:
            if file.getnchannels() != 1 or file.getsampwidth() != 2 or file.getcomptype() != "NONE":
                raise ValueError(f"WAV must be uncompressed mono PCM16: {path}")
            sample_rate = file.getframerate()
            frames = file.readframes(file.getnframes())
    except (EOFError, wave.Error) as error:
        raise ValueError(f"invalid WAV file {path}: {error}") from error
    waveform = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767
    return sample_rate, np.clip(waveform, -1, 1)


__all__ = ["read_wav", "write_wav_atomic"]

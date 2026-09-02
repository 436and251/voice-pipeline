from __future__ import annotations

from dataclasses import dataclass
import subprocess
from pathlib import Path

import numpy as np
import torch


class InvalidSampleAudio(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    wav32_int16: np.ndarray
    hubert_source_32k: torch.Tensor


def load_audio_32k(path: str | Path) -> torch.Tensor:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    process = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            str(source),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            "32000",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if process.returncode:
        message = process.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"failed to decode audio {source}: {message}")
    return torch.frombuffer(bytearray(process.stdout), dtype=torch.float32).clone()


def prepare_audio_from_source(path: str | Path) -> PreparedAudio:
    waveform = load_audio_32k(path)
    raw = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
    if raw.ndim != 1 or raw.size == 0:
        raise InvalidSampleAudio("decoded audio must be a non-empty mono waveform")
    if not np.isfinite(raw).all():
        raise InvalidSampleAudio("decoded audio contains non-finite samples")
    max_abs = float(np.max(np.abs(raw)))
    if max_abs == 0:
        raise InvalidSampleAudio("decoded audio is silent")
    if max_abs > 2.2:
        raise InvalidSampleAudio(f"decoded audio amplitude exceeds 2.2: {max_abs:g}")

    wav32 = raw / max_abs * (0.95 * 0.5 * 32768) + 0.5 * 32768 * raw
    hubert_source = raw / max_abs * (0.95 * 0.5 * 1145.14) + 0.5 * 1145.14 * raw
    return PreparedAudio(
        wav32.astype(np.int16),
        torch.from_numpy(hubert_source.astype(np.float32)),
    )

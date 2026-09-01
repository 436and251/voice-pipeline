from __future__ import annotations

import subprocess
from pathlib import Path

import torch


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

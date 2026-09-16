from __future__ import annotations

import numpy as np


TERMINAL_SILENCE_MS = 300


def assemble_waveforms(
    waveforms: list[np.ndarray], sample_rate: int, pause_ms: int
) -> np.ndarray:
    pause = np.zeros(round(sample_rate * pause_ms / 1000), dtype=np.float32)
    terminal = np.zeros(round(sample_rate * TERMINAL_SILENCE_MS / 1000), dtype=np.float32)
    assembled: list[np.ndarray] = []
    for index, waveform in enumerate(waveforms):
        if index and pause.size:
            assembled.append(pause)
        assembled.append(waveform)
    assembled.append(terminal)
    return np.ascontiguousarray(np.concatenate(assembled))


__all__ = ["TERMINAL_SILENCE_MS", "assemble_waveforms"]

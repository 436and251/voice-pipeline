from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class InferenceResult:
    waveform: np.ndarray
    sample_rate: int
    seed: int


__all__ = ["InferenceResult"]

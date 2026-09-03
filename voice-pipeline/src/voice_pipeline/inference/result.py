from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class InferenceIdentity:
    model_name: str
    s1_sha256: str
    s2_sha256: str
    reference_sha256: str
    reference_text: str | None
    reference_language: str


@dataclass(frozen=True, slots=True)
class InferenceResult:
    waveform: np.ndarray
    sample_rate: int
    seed: int


__all__ = ["InferenceIdentity", "InferenceResult"]

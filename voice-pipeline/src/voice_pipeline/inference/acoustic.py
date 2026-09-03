from __future__ import annotations

import numpy as np
import torch

from .reference import ReferenceCondition
from .semantic import SemanticResult


def decode_waveform(
    semantic: SemanticResult,
    *,
    s2,
    reference: ReferenceCondition,
    noise_scale: float,
    speed: float,
) -> np.ndarray:
    with torch.inference_mode():
        output = s2.decode(
            semantic.codes,
            semantic.target_phones,
            [reference.spectrogram],
            noise_scale=noise_scale,
            speed=speed,
            sv_emb=[reference.speaker_embedding],
        )
    if output.ndim != 3 or output.shape[0:2] != (1, 1) or output.shape[-1] == 0:
        raise RuntimeError("S2 returned an invalid waveform shape")
    waveform = output[0, 0].detach().float().cpu()
    if not torch.isfinite(waveform).all():
        raise RuntimeError("S2 returned non-finite waveform samples")
    peak = waveform.abs().max()
    if peak > 1:
        waveform = waveform / peak
    return np.ascontiguousarray(waveform.numpy(), dtype=np.float32)


__all__ = ["decode_waveform"]

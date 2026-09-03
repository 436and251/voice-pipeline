from __future__ import annotations

import numpy as np

from .result import InferenceResult
from .text_chunker import TextChunker


def synthesize_text(
    session,
    text: str,
    language: str,
    *,
    pause_ms: int = 10,
    max_chars: int | None = None,
    seed: int = 0,
    top_k: int = 5,
    top_p: float = 1.0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.35,
    noise_scale: float = 0.5,
    speed: float = 1.0,
) -> InferenceResult:
    if isinstance(pause_ms, bool) or not isinstance(pause_ms, int) or pause_ms < 0:
        raise ValueError("pause_ms must be a nonnegative integer")
    chunks = TextChunker(language, max_chars).chunk(text)
    waveforms: list[np.ndarray] = []
    sample_rate = None
    for index, chunk in enumerate(chunks):
        result = session.synthesize(
            chunk,
            language,
            seed=(seed + index) % 2**63,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            noise_scale=noise_scale,
            speed=speed,
        )
        waveform = np.asarray(result.waveform, dtype=np.float32)
        if waveform.ndim != 1:
            raise RuntimeError("inference result must be a mono waveform")
        if sample_rate is None:
            sample_rate = result.sample_rate
        elif result.sample_rate != sample_rate:
            raise RuntimeError("inference chunks returned inconsistent sample rates")
        waveforms.append(waveform)

    assert sample_rate is not None
    pause = np.zeros(round(sample_rate * pause_ms / 1000), dtype=np.float32)
    assembled: list[np.ndarray] = []
    for index, waveform in enumerate(waveforms):
        if index and pause.size:
            assembled.append(pause)
        assembled.append(waveform)
    return InferenceResult(np.ascontiguousarray(np.concatenate(assembled)), sample_rate, seed)


__all__ = ["synthesize_text"]

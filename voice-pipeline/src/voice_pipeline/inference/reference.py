from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torchaudio.functional import resample

from voice_pipeline.core.gpt_sovits.features.audio import load_audio_32k
from voice_pipeline.core.gpt_sovits.frontend.contract import FrontendResult
from voice_pipeline.core.gpt_sovits.s2_v2proplus.mel_processing import spectrogram_torch


@dataclass(frozen=True, slots=True)
class ReferenceCondition:
    prompt_semantic: torch.Tensor
    spectrogram: torch.Tensor
    speaker_embedding: torch.Tensor
    prompt_frontend: FrontendResult | None


def build_reference_condition(
    audio_path: str | Path,
    prompt_text: str | None,
    prompt_language: str,
    *,
    frontend,
    hubert,
    speaker,
    s2,
    device: torch.device,
    dtype: torch.dtype,
) -> ReferenceCondition:
    waveform = load_audio_32k(audio_path)
    if waveform.ndim != 1 or not torch.isfinite(waveform).all():
        raise ValueError("reference audio must decode to a finite mono waveform")
    if not 3 * 32_000 <= waveform.numel() <= 10 * 32_000:
        raise ValueError("reference audio must be between 3 and 10 seconds")

    waveform = waveform.float()
    conditioning_waveform = waveform
    peak = float(waveform.abs().max())
    if peak > 1:
        conditioning_waveform = waveform / min(2.0, peak)
    spectrogram = spectrogram_torch(
        conditioning_waveform.to(device=device, dtype=dtype).unsqueeze(0),
        2048,
        32_000,
        640,
        2048,
        center=False,
    )
    waveform_16k = resample(waveform, 32_000, 16_000)
    waveform_16k = torch.cat((waveform_16k, torch.zeros(9_600, dtype=waveform_16k.dtype)))
    content = hubert.extract(waveform_16k).to(device=device, dtype=dtype)
    with torch.inference_mode():
        prompt_semantic = s2.extract_latent(content)[0, 0]
    speaker_embedding = speaker.extract(conditioning_waveform).to(device=device, dtype=dtype)

    prompt_frontend = None
    if prompt_text is not None:
        text = prompt_text.strip()
        if not text:
            raise ValueError("reference text must not be empty")
        if text[-1] not in "，。？！,.?!~:：—…":
            text += "." if prompt_language == "en" else "。"
        prompt_frontend = frontend.process(text, prompt_language)

    return ReferenceCondition(prompt_semantic, spectrogram, speaker_embedding, prompt_frontend)


__all__ = ["ReferenceCondition", "build_reference_condition"]

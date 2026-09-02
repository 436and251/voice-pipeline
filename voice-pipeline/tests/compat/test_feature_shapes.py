from __future__ import annotations

import math
import os
import struct
import wave
from pathlib import Path

import pytest
import torch


def _required_asset(env_name: str) -> Path:
    value = os.environ.get(env_name)
    if not value:
        pytest.skip(f"set {env_name} for real feature compatibility tests")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"missing real feature test asset: {path}")
    return path


def test_load_audio_32k_returns_mono_float_tensor(tmp_path: Path) -> None:
    from voice_pipeline.core.gpt_sovits.features.audio import load_audio_32k

    source = tmp_path / "stereo-16k.wav"
    frames = bytearray()
    for index in range(4_000):
        left = int(8_000 * math.sin(2 * math.pi * 220 * index / 16_000))
        frames.extend(struct.pack("<hh", left, -left))
    with wave.open(str(source), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(frames)

    audio = load_audio_32k(source)

    assert audio.shape == (8_000,)
    assert audio.dtype == torch.float32
    assert audio.device.type == "cpu"


def test_real_cnhubert_extracts_official_content_shape() -> None:
    from voice_pipeline.core.gpt_sovits.features.cnhubert import CNHubertExtractor

    extractor = CNHubertExtractor(
        _required_asset("VOICE_PIPELINE_TEST_HUBERT_DIR"),
        device="cpu",
    )

    content = extractor.extract(torch.zeros(16_000))

    assert content.shape[0:2] == (1, 768)
    assert content.shape[2] > 0
    assert content.dtype == torch.float32
    assert content.device.type == "cpu"


def test_real_speaker_encoder_matches_forward3_shape() -> None:
    from voice_pipeline.core.gpt_sovits.features.speaker import SpeakerEncoder

    encoder = SpeakerEncoder(
        _required_asset("VOICE_PIPELINE_TEST_SV_CHECKPOINT"),
        device="cpu",
    )
    time = torch.arange(32_000, dtype=torch.float32) / 32_000
    waveform = torch.sin(2 * torch.pi * 220 * time).unsqueeze(0)

    embedding = encoder.extract(waveform)

    assert embedding.shape == (1, 20_480)
    assert embedding.dtype == torch.float32
    assert embedding.device.type == "cpu"


def test_cnhubert_to_float_changes_model_precision() -> None:
    from voice_pipeline.core.gpt_sovits.features.cnhubert import CNHubertExtractor

    extractor = CNHubertExtractor.__new__(CNHubertExtractor)
    extractor.model = torch.nn.Linear(2, 2).half()
    extractor.precision = "fp16"
    extractor.to_float()

    assert extractor.precision == "fp32"
    assert next(extractor.model.parameters()).dtype == torch.float32

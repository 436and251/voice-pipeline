import numpy as np
import pytest
import torch

from voice_pipeline.core.gpt_sovits.features import audio
from voice_pipeline.core.gpt_sovits.features.audio import InvalidSampleAudio


def test_prepare_audio_preserves_pinned_amplitude_mix(monkeypatch):
    raw = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float32)
    monkeypatch.setattr(audio, "load_audio_32k", lambda path: raw)
    prepared = audio.prepare_audio_from_source("ignored.wav")
    expected = raw.numpy() / 1.0 * (0.95 * 0.5 * 32768) + 0.5 * 32768 * raw.numpy()
    np.testing.assert_array_equal(prepared.wav32_int16, expected.astype(np.int16))
    assert prepared.hubert_source_32k.dtype == torch.float32


@pytest.mark.parametrize(
    "waveform",
    [torch.tensor([]), torch.zeros(32), torch.tensor([float("nan")]), torch.tensor([2.21])],
)
def test_prepare_audio_rejects_invalid_input(monkeypatch, waveform):
    monkeypatch.setattr(audio, "load_audio_32k", lambda path: waveform)
    with pytest.raises(InvalidSampleAudio):
        audio.prepare_audio_from_source("ignored.wav")

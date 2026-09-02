from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
import torch

from voice_pipeline.core.gpt_sovits.features.audio import InvalidSampleAudio, PreparedAudio
from voice_pipeline.profiles.v2proplus import V2PROPLUS
from voice_pipeline.training.experiment import Experiment
from voice_pipeline.training.manifest import ManifestItem, ManifestRecord
from voice_pipeline.training.preprocess.base import SampleFailure, StageContext
from voice_pipeline.training.preprocess import wav32k_stage
from voice_pipeline.training.preprocess.wav32k_stage import Wav32kStage


def record(path=Path("original.flac")):
    return ManifestRecord(1, "sample", ManifestItem(path, "speaker", "ja", "こんにちは"))


def context(tmp_path):
    return StageContext(Experiment.create("run", tmp_path), V2PROPLUS, {}, {})


def prepared():
    wav = np.array([0, 1000, -1000], dtype=np.int16)
    return PreparedAudio(wav, torch.from_numpy(wav.astype(np.float32)))


def test_wav32k_stage_writes_mono_int16_32khz(monkeypatch, tmp_path):
    monkeypatch.setattr(wav32k_stage, "prepare_audio_from_source", lambda path: prepared())
    result = Wav32kStage().run(record(), context(tmp_path))

    sample_rate, waveform = wavfile.read(result.output_paths[0])
    assert sample_rate == 32_000
    assert waveform.dtype == np.int16
    assert waveform.ndim == 1
    assert result.metadata["num_samples"] == 3


def test_wav32k_stage_ignores_abandoned_temporary_file(monkeypatch, tmp_path):
    monkeypatch.setattr(wav32k_stage, "prepare_audio_from_source", lambda path: prepared())
    output_dir = context(tmp_path).preprocess_dir / "wav32k"
    output_dir.mkdir(parents=True)
    (output_dir / ".sample.wav.abandoned.tmp").write_bytes(b"partial")

    result = Wav32kStage().run(record(), context(tmp_path))
    assert wavfile.read(result.output_paths[0])[0] == 32_000


def test_wav32k_stage_converts_invalid_audio_to_sample_failure(monkeypatch, tmp_path):
    def fail(path):
        raise InvalidSampleAudio("silent")

    monkeypatch.setattr(wav32k_stage, "prepare_audio_from_source", fail)
    with pytest.raises(SampleFailure, match="silent"):
        Wav32kStage().run(record(), context(tmp_path))

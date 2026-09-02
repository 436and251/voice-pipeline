from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
import torch

from voice_pipeline.core.gpt_sovits.features.audio import PreparedAudio
from voice_pipeline.profiles.v2proplus import V2PROPLUS
from voice_pipeline.training.experiment import Experiment
from voice_pipeline.training.manifest import ManifestItem, ManifestRecord
from voice_pipeline.training.preprocess.base import SampleFailure, StageContext
from voice_pipeline.training.preprocess import hubert_stage
from voice_pipeline.training.preprocess.hubert_stage import HubertStage
from voice_pipeline.training.preprocess.sv_stage import SVStage


class FakeHubert:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.to_float_calls = 0
        self.inputs = []

    def extract(self, waveform):
        self.inputs.append(waveform)
        return self.outputs.pop(0)

    def to_float(self):
        self.to_float_calls += 1


class FakeSpeaker:
    def __init__(self, output):
        self.output = output
        self.inputs = []

    def extract(self, waveform):
        self.inputs.append(waveform)
        return self.output


def record(audio_path=Path("original.flac")):
    return ManifestRecord(1, "sample", ManifestItem(audio_path, "speaker", "ja", "こんにちは"))


def context(tmp_path):
    return StageContext(
        Experiment.create("run", tmp_path),
        V2PROPLUS,
        {},
        {"hubert": "hubert-digest", "speaker": "speaker-digest"},
    )


def prepared():
    source = torch.sin(torch.arange(3_200) * 0.1)
    return PreparedAudio(np.zeros(3_200, dtype=np.int16), source)


def write_wav_dependency(ctx):
    path = ctx.preprocess_dir / "wav32k" / "sample.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, 32_000, np.arange(3_200, dtype=np.int16))
    return path


def test_hubert_stage_decodes_original_audio_not_wav32(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        hubert_stage,
        "prepare_audio_from_source",
        lambda path: calls.append(path) or prepared(),
    )
    source = tmp_path / "original.flac"
    result = HubertStage(FakeHubert([torch.ones(1, 768, 4)]), "fp32").run(
        record(source), context(tmp_path)
    )
    assert calls == [source]
    assert result.metadata["shape"] == [1, 768, 4]


def test_hubert_fp16_nonfinite_retries_once_in_fp32(monkeypatch, tmp_path):
    monkeypatch.setattr(hubert_stage, "prepare_audio_from_source", lambda path: prepared())
    extractor = FakeHubert(
        [torch.full((1, 768, 4), float("nan")), torch.ones(1, 768, 4)]
    )
    result = HubertStage(extractor, "fp16").run(record(), context(tmp_path))
    assert extractor.to_float_calls == 1
    assert len(extractor.inputs) == 2
    assert result.metadata["shape"] == [1, 768, 4]


@pytest.mark.parametrize(
    "outputs",
    [
        [torch.ones(1, 512, 4)],
        [torch.full((1, 768, 4), float("nan")), torch.full((1, 768, 4), float("nan"))],
    ],
)
def test_hubert_rejects_wrong_shape_or_persistently_nonfinite(monkeypatch, tmp_path, outputs):
    monkeypatch.setattr(hubert_stage, "prepare_audio_from_source", lambda path: prepared())
    with pytest.raises(SampleFailure):
        HubertStage(FakeHubert(outputs), "fp16").run(record(), context(tmp_path))


def test_sv_stage_requires_official_forward3_shape(tmp_path):
    ctx = context(tmp_path)
    write_wav_dependency(ctx)
    result = SVStage(FakeSpeaker(torch.ones(1, 20_480))).run(record(), ctx)
    assert result.metadata["shape"] == [1, 20_480]


@pytest.mark.parametrize(
    "embedding",
    [torch.ones(1, 512), torch.full((1, 20_480), float("nan"))],
)
def test_sv_rejects_wrong_width_or_nonfinite_tensor(tmp_path, embedding):
    ctx = context(tmp_path)
    write_wav_dependency(ctx)
    with pytest.raises(SampleFailure):
        SVStage(FakeSpeaker(embedding)).run(record(), ctx)

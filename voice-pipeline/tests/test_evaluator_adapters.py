from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from voice_pipeline.evaluation.asr import WhisperEvaluator
from voice_pipeline.evaluation.prosody import evaluate_prosody
from voice_pipeline.evaluation.speaker_similarity import WavLMSpeakerEvaluator
from voice_pipeline.inference.wav import write_wav_atomic


def test_s1_attention_import_does_not_patch_torch_globally() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import torch.nn.functional as f; original=f.multi_head_attention_forward; "
            "import voice_pipeline.core.gpt_sovits.s1.modules.activation; "
            "assert f.multi_head_attention_forward is original",
            str(Path(__file__).parents[1] / "src"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


class FakeWhisperModel:
    def transcribe(self, audio, *, language, **kwargs):
        text = " forced transcript" if language == "en" else " auto transcript"
        info = SimpleNamespace(language="ja" if language is None else language, language_probability=0.75)
        return iter((SimpleNamespace(text=text),)), info


def test_whisper_adapter_returns_structured_forced_and_auto_transcripts(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    wav.touch()
    evaluator = WhisperEvaluator.from_model(FakeWhisperModel())

    forced = evaluator.transcribe(wav, "en")
    automatic = evaluator.transcribe(wav, None)

    assert (forced.text, forced.language, forced.language_probability) == ("forced transcript", "en", 0.75)
    assert (automatic.text, automatic.language, automatic.language_probability) == ("auto transcript", "ja", 0.75)


class FakeFeatureExtractor:
    def __call__(self, waveform, *, sampling_rate, return_tensors):
        assert sampling_rate == 16000
        return {"input_values": torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)}


class FakeEmbeddingModel:
    def __call__(self, input_values):
        mean = input_values.mean(dim=1)
        return SimpleNamespace(embeddings=torch.stack((mean + 1, 2 - mean), dim=1))


def _constant_wav(path: Path, value: float, sample_rate: int = 16000) -> Path:
    write_wav_atomic(path, np.full(sample_rate // 10, value, dtype=np.float32), sample_rate)
    return path


def test_speaker_centroid_and_embeddings_are_normalized(tmp_path: Path) -> None:
    evaluator = WavLMSpeakerEvaluator.from_components(
        FakeFeatureExtractor(),
        FakeEmbeddingModel(),
        device="cpu",
    )
    reference_a = _constant_wav(tmp_path / "a.wav", 0.1)
    reference_b = _constant_wav(tmp_path / "b.wav", 0.5)

    embedding = evaluator.embedding(reference_a)
    centroid = evaluator.centroid((reference_a, reference_b))

    assert np.linalg.norm(embedding) == pytest.approx(1.0)
    assert np.linalg.norm(centroid) == pytest.approx(1.0)
    assert evaluator.similarity(reference_a, centroid) == pytest.approx(float(embedding @ centroid))


def test_speaker_evaluator_resamples_to_16khz(tmp_path: Path) -> None:
    evaluator = WavLMSpeakerEvaluator.from_components(
        FakeFeatureExtractor(),
        FakeEmbeddingModel(),
        device="cpu",
    )
    wav = _constant_wav(tmp_path / "32k.wav", 0.2, sample_rate=32000)

    embedding = evaluator.embedding(wav)

    assert embedding.shape == (2,)
    assert np.isfinite(embedding).all()


def test_silent_audio_sets_prosody_failure_flag(tmp_path: Path) -> None:
    wav = _constant_wav(tmp_path / "silent.wav", 0.0)

    result = evaluate_prosody(wav, "test")

    assert result.available is True
    assert result.details["all_silent"] is True
    assert result.value == 0.0


def test_sine_wave_has_finite_pitch_energy_and_duration(tmp_path: Path) -> None:
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    wav = tmp_path / "sine.wav"
    write_wav_atomic(wav, 0.25 * np.sin(2 * np.pi * 200 * time), sample_rate)

    result = evaluate_prosody(wav, "four test words here")

    assert result.value == pytest.approx(1.0)
    assert result.details["duration_seconds"] == pytest.approx(1.0)
    assert result.details["f0_median"] == pytest.approx(200, abs=5)
    assert result.details["rms_mean"] > 0
    assert result.details["clipping_ratio"] == 0.0

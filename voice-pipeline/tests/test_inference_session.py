from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from voice_pipeline.core.gpt_sovits.frontend.contract import FrontendResult
from voice_pipeline.inference.reference import ReferenceCondition, build_reference_condition
from voice_pipeline.inference.semantic import generate_semantic
from voice_pipeline.inference.session import InferenceSession
from voice_pipeline.inference import session as session_module


def _frontend_result(ids: list[int]) -> FrontendResult:
    return FrontendResult("text", ["a"] * len(ids), ids, None, torch.ones(1024, len(ids)))


def test_reference_condition_preserves_official_v2proplus_inputs(monkeypatch, tmp_path: Path):
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"placeholder")
    wav32 = torch.linspace(-2.0, 2.0, 96_000)
    calls = {}

    class Hubert:
        def extract(self, waveform):
            calls["hubert_length"] = waveform.numel()
            return torch.ones(1, 768, 12)

    class Speaker:
        def extract(self, waveform):
            calls["speaker_length"] = waveform.numel()
            calls["speaker_peak"] = float(waveform.abs().max())
            return torch.ones(1, 20_480)

    class S2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def extract_latent(self, features):
            calls["hubert_shape"] = tuple(features.shape)
            return torch.tensor([[[4, 5, 6]]])

    class Frontend:
        def process(self, text, language):
            calls["prompt"] = (text, language)
            return _frontend_result([10, 11])

    monkeypatch.setattr("voice_pipeline.inference.reference.load_audio_32k", lambda path: wav32)
    condition = build_reference_condition(
        audio,
        "reference text",
        "en",
        frontend=Frontend(),
        hubert=Hubert(),
        speaker=Speaker(),
        s2=S2(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert calls == {
        "hubert_length": 48_000 + 9_600,
        "speaker_length": 96_000,
        "speaker_peak": 1.0,
        "hubert_shape": (1, 768, 12),
        "prompt": ("reference text.", "en"),
    }
    assert condition.prompt_semantic.tolist() == [4, 5, 6]
    assert condition.spectrogram.shape[0:2] == (1, 1025)
    assert condition.speaker_embedding.shape == (1, 20_480)
    assert condition.prompt_frontend.phone_ids == [10, 11]


@pytest.mark.parametrize("samples", [95_999, 320_001])
def test_reference_audio_must_be_between_three_and_ten_seconds(monkeypatch, tmp_path: Path, samples: int):
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"placeholder")
    monkeypatch.setattr("voice_pipeline.inference.reference.load_audio_32k", lambda path: torch.ones(samples))
    with pytest.raises(ValueError, match="3.*10"):
        build_reference_condition(
            audio,
            None,
            "ja",
            frontend=object(),
            hubert=object(),
            speaker=object(),
            s2=torch.nn.Linear(1, 1),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_session_loads_once_caches_reference_and_synthesizes(monkeypatch, tmp_path: Path):
    bundle_root = tmp_path / "models" / "speaker"
    bundle_root.mkdir(parents=True)
    reference = SimpleNamespace(audio=Path("reference/default.wav"), text="prompt", language="ja")
    bundle = SimpleNamespace(
        root=bundle_root,
        profile="v2ProPlus",
        weights={"s1": Path("weights/s1.ckpt"), "s2": Path("weights/s2.pth")},
        reference=reference,
    )
    calls = {"s1_load": 0, "s2_load": 0, "reference": 0, "target": 0, "speaker_precision": None}

    class S1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def infer_panel(self, phones, lengths, prompt, bert, **options):
            assert phones.tolist() == [[10, 11, 20, 21, 22]]
            assert lengths.tolist() == [5]
            assert prompt.tolist() == [[7, 8]]
            assert tuple(bert.shape) == (1, 1024, 5)
            assert options == {
                "top_k": 5,
                "top_p": 1.0,
                "temperature": 1.0,
                "early_stop_num": 2850,
                "repetition_penalty": 1.35,
            }
            return torch.tensor([[99, 100, 101]]), 2

    class S2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def decode(self, codes, phones, references, *, noise_scale, speed, sv_emb):
            assert codes.tolist() == [[[100, 101]]]
            assert phones.tolist() == [[20, 21, 22]]
            assert len(references) == len(sv_emb) == 1
            assert noise_scale == 0.5 and speed == 1.0
            return torch.tensor([[[0.0, 2.0, -2.0]]])

    class Frontend:
        def __init__(self, *args, **kwargs):
            self.bert = SimpleNamespace(model=torch.nn.Linear(1, 1))

        def process(self, text, language):
            calls["target"] += 1
            assert (text, language) == ("target。", "zh")
            return _frontend_result([20, 21, 22])

    def load_s1(*args, **kwargs):
        calls["s1_load"] += 1
        return S1()

    def load_s2(*args, **kwargs):
        calls["s2_load"] += 1
        return S2()

    condition = ReferenceCondition(
        prompt_semantic=torch.tensor([7, 8]),
        spectrogram=torch.ones(1, 1025, 4),
        speaker_embedding=torch.ones(1, 20_480),
        prompt_frontend=_frontend_result([10, 11]),
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(session_module.ModelBundle, "load", lambda path: bundle)
    monkeypatch.setattr(session_module, "load_s1_checkpoint", load_s1)
    monkeypatch.setattr(session_module, "load_s2_generator", load_s2)
    monkeypatch.setattr(session_module, "MultilingualFrontend", Frontend)
    monkeypatch.setattr(session_module, "CNHubertExtractor", lambda *args, **kwargs: object())
    def load_speaker(*args, **kwargs):
        calls["speaker_precision"] = kwargs["precision"]
        return object()

    monkeypatch.setattr(session_module, "SpeakerEncoder", load_speaker)

    def build_reference(*args, **kwargs):
        calls["reference"] += 1
        return condition

    monkeypatch.setattr(session_module, "build_reference_condition", build_reference)

    session = InferenceSession.load(bundle_root, "cpu")
    first = session.synthesize("target", "zh", seed=123)
    second = session.synthesize("target", "zh", seed=123)

    assert calls == {
        "s1_load": 1,
        "s2_load": 1,
        "reference": 1,
        "target": 2,
        "speaker_precision": "fp32",
    }
    assert first.sample_rate == 32_000 and first.seed == 123
    assert first.waveform.dtype == np.float32
    assert first.waveform.tolist() == [0.0, 1.0, -1.0]
    assert np.array_equal(first.waveform, second.waveform)


def test_synthesize_validates_language_seed_and_decoding_parameters():
    session = object.__new__(InferenceSession)
    for language in ("fr", "auto", ""):
        with pytest.raises(ValueError, match="language"):
            session.synthesize("text", language)
    with pytest.raises(ValueError, match="seed"):
        session.synthesize("text", "en", seed=-1)
    with pytest.raises(ValueError, match="top_k"):
        session.synthesize("text", "en", top_k=0)
    with pytest.raises(ValueError, match="speed"):
        session.synthesize("text", "en", speed=0)


def test_reference_free_s1_zero_length_sentinel_keeps_all_generated_codes():
    class S1:
        def infer_panel(self, *args, **kwargs):
            assert args[2] is None
            return torch.tensor([[4, 5, 6]]), 0

    reference = ReferenceCondition(
        prompt_semantic=torch.tensor([1]),
        spectrogram=torch.ones(1, 1025, 4),
        speaker_embedding=torch.ones(1, 20_480),
        prompt_frontend=None,
    )
    semantic = generate_semantic(
        "target",
        "en",
        frontend=SimpleNamespace(process=lambda text, language: _frontend_result([20, 21])),
        s1=S1(),
        reference=reference,
        device=torch.device("cpu"),
        dtype=torch.float32,
        top_k=5,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=1.35,
        early_stop_num=2850,
    )

    assert semantic.codes.tolist() == [[[4, 5, 6]]]

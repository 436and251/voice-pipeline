import json
from pathlib import Path

import numpy as np
import pytest

from voice_pipeline.common.model_bundle import BundleReference
from voice_pipeline.evaluation import runner as runner_module
from voice_pipeline.evaluation.candidates import CandidatePair, CheckpointRef
from voice_pipeline.evaluation.config import EvaluationConfig, EvaluationPairing
from voice_pipeline.evaluation.runner import generate_candidate
from voice_pipeline.evaluation.suite import EvaluationCase, EvaluationSuite
from voice_pipeline.inference.result import InferenceIdentity, InferenceResult


class FakeSession:
    identity = InferenceIdentity("speaker", "1" * 64, "2" * 64, "3" * 64, "reference", "ja")

    def synthesize(self, text, language, *, seed, **options):
        if text == "fail":
            raise RuntimeError("synthetic failure")
        waveform = np.full(320, (seed + 1) / 100, dtype=np.float32)
        return InferenceResult(waveform, 32000, seed)


def _config(tmp_path: Path) -> EvaluationConfig:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference")
    return EvaluationConfig(
        run_dir=tmp_path / "runs" / "speaker",
        project_root=tmp_path,
        model_name="speaker",
        trained_languages=("ja",),
        validated_languages=("zh", "ja", "en"),
        reference=BundleReference(reference, "reference", "ja"),
        speaker_references=(reference,),
        suites={},
        asr_model="asr/model",
        speaker_model="speaker/model",
        cache_dir=tmp_path / "cache",
        pairing=EvaluationPairing(),
        device="cpu",
        precision="fp32",
    )


def _pair(tmp_path: Path, *, s2_hash: str = "2" * 64) -> CandidatePair:
    s1 = CheckpointRef("s1", tmp_path / "s1.pt", None, "1" * 64, True)
    s2 = CheckpointRef("s2", tmp_path / "s2.pt", 100, s2_hash, False)
    return CandidatePair("fixed-pair", s1, s2)


def _suite(text: str = "sample") -> EvaluationSuite:
    return EvaluationSuite(
        tuple(EvaluationCase(f"{language}-001", language, text) for language in ("zh", "ja", "en", "mixed"))
    )


def _fake_builder(destination: Path, **kwargs) -> Path:
    destination.mkdir(parents=True)
    return destination


def test_generation_manifest_resumes_completed_wavs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "build_candidate_bundle", _fake_builder)
    first = generate_candidate(_config(tmp_path), _pair(tmp_path), _suite(), session_factory=lambda *args, **kwargs: FakeSession())

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("a completed candidate must resume without loading inference")

    second = generate_candidate(_config(tmp_path), _pair(tmp_path), _suite(), session_factory=fail_if_loaded)

    assert first.generated == 4
    assert first.resumed == 0
    assert second.generated == 0
    assert second.resumed == 4
    assert [sample.case_id for sample in second.samples] == ["zh-001", "ja-001", "en-001", "mixed-001"]


def test_changed_checkpoint_hash_does_not_reuse_generation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "build_candidate_bundle", _fake_builder)
    generate_candidate(
        _config(tmp_path),
        _pair(tmp_path, s2_hash="a" * 64),
        _suite(),
        session_factory=lambda *args, **kwargs: FakeSession(),
    )

    with pytest.raises(ValueError, match="does not match"):
        generate_candidate(
            _config(tmp_path),
            _pair(tmp_path, s2_hash="b" * 64),
            _suite(),
            session_factory=lambda *args, **kwargs: FakeSession(),
        )


def test_tampered_sample_entry_is_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "build_candidate_bundle", _fake_builder)
    config = _config(tmp_path)
    pair = _pair(tmp_path)
    generate_candidate(config, pair, _suite(), session_factory=lambda *args, **kwargs: FakeSession())
    manifest_path = config.run_dir / "evaluation" / "generated" / pair.key / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"][0]["text"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest samples"):
        generate_candidate(config, pair, _suite(), session_factory=lambda *args, **kwargs: FakeSession())


def test_generation_records_one_failed_sample_and_continues(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "build_candidate_bundle", _fake_builder)
    suite = EvaluationSuite(
        (
            EvaluationCase("ja-001", "ja", "fail"),
            EvaluationCase("en-001", "en", "works"),
        )
    )

    result = generate_candidate(
        _config(tmp_path),
        _pair(tmp_path),
        suite,
        session_factory=lambda *args, **kwargs: FakeSession(),
    )

    assert result.status == "failed"
    assert result.generated == 1
    assert result.samples[0].status == "failed"
    assert result.samples[0].error_type == "RuntimeError"
    assert result.samples[1].status == "completed"
    assert result.samples[1].wav_path.is_file()

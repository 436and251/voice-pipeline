from __future__ import annotations

import json
from pathlib import Path
import wave

import numpy as np
import pytest

from voice_pipeline.inference.job import resolve_output_path, run_synthesis_job
from voice_pipeline.inference.result import InferenceIdentity, InferenceResult
from voice_pipeline.inference.wav import read_wav, write_wav_atomic


class FakeSession:
    sample_rate = 32_000

    def __init__(self, identity: InferenceIdentity | None = None):
        self.identity = identity or InferenceIdentity(
            "speaker_001", "1" * 64, "2" * 64, "3" * 64, "prompt", "ja"
        )
        self.calls: list[tuple[str, int]] = []

    def synthesize(self, text, language, **options):
        self.calls.append((text, options["seed"]))
        value = (options["seed"] % 10 + 1) / 10
        return InferenceResult(np.array([value, -value], dtype=np.float32), 32_000, options["seed"])


def test_output_path_is_namespaced_and_rejects_escape(tmp_path: Path):
    expected = (tmp_path / "speaker_001" / "folder" / "article.wav").resolve()
    assert resolve_output_path(tmp_path, "speaker_001", Path("folder/article.wav")) == expected

    for value in (Path("../escape.wav"), Path("C:/escape.wav"), Path("bad.mp3"), Path(".")):
        with pytest.raises(ValueError):
            resolve_output_path(tmp_path, "speaker_001", value)


def test_pcm16_wav_round_trip_is_atomic_and_mono(tmp_path: Path):
    path = tmp_path / "audio.wav"
    write_wav_atomic(path, np.array([0.0, 1.0, -1.0], dtype=np.float32), 32_000)

    with wave.open(str(path), "rb") as file:
        assert (file.getnchannels(), file.getsampwidth(), file.getframerate()) == (1, 2, 32_000)
    sample_rate, waveform = read_wav(path)
    assert sample_rate == 32_000
    assert waveform.dtype == np.float32
    assert np.allclose(waveform, [0.0, 1.0, -1.0], atol=1 / 32767)
    assert not list(tmp_path.glob(".*.tmp"))


def test_job_writes_manifest_resumes_and_repairs_only_bad_chunks(tmp_path: Path):
    session = FakeSession()
    output = resolve_output_path(tmp_path, session.identity.model_name, Path("article.wav"))

    first = run_synthesis_job(session, "aa。bb。cc。", "zh", output, max_chars=3, seed=8)

    assert first.generated_chunks == 3 and first.resumed_chunks == 0
    assert output.is_file()
    assert read_wav(output)[1].shape == (6 + 2 * 320,)
    work = output.with_suffix(".infer")
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["final"]["status"] == "completed"
    assert [entry["seed"] for entry in manifest["chunks"]] == [8, 9, 10]
    assert [entry["output"] for entry in manifest["chunks"]] == [
        "chunks/000001.wav", "chunks/000002.wav", "chunks/000003.wav"
    ]

    session.calls.clear()
    (work / "chunks" / "000002.wav").unlink()
    repaired = run_synthesis_job(session, "aa。bb。cc。", "zh", output, max_chars=3, seed=8)
    assert session.calls == [("bb。", 9)]
    assert (repaired.generated_chunks, repaired.resumed_chunks) == (1, 2)

    session.calls.clear()
    (work / "chunks" / "000003.wav").write_bytes(b"damaged")
    repaired = run_synthesis_job(session, "aa。bb。cc。", "zh", output, max_chars=3, seed=8)
    assert session.calls == [("cc。", 10)]
    assert (repaired.generated_chunks, repaired.resumed_chunks) == (1, 2)

    session.calls.clear()
    complete = run_synthesis_job(session, "aa。bb。cc。", "zh", output, max_chars=3, seed=8)
    assert session.calls == []
    assert (complete.generated_chunks, complete.resumed_chunks) == (0, 3)


@pytest.mark.parametrize(
    "changed",
    [
        {"text": "different"},
        {"language": "en"},
        {"pause_ms": 11},
        {"seed": 9},
        {"top_k": 6},
        {"top_p": 0.9},
        {"temperature": 0.9},
        {"repetition_penalty": 1.2},
        {"noise_scale": 0.4},
        {"speed": 1.1},
        {"max_chars": 4},
    ],
)
def test_job_rejects_changed_request_without_explicit_overwrite(tmp_path: Path, changed):
    session = FakeSession()
    output = resolve_output_path(tmp_path, session.identity.model_name, Path("article.wav"))
    base = {"text": "aa。bb。", "language": "zh", "max_chars": 3, "seed": 8}
    run_synthesis_job(session, output_path=output, **base)

    with pytest.raises(ValueError, match="--overwrite"):
        run_synthesis_job(session, output_path=output, **(base | changed))


def test_job_identity_change_requires_overwrite_and_overwrite_rebuilds(tmp_path: Path):
    first_session = FakeSession()
    output = resolve_output_path(tmp_path, first_session.identity.model_name, Path("article.wav"))
    run_synthesis_job(first_session, "text", "en", output)
    changed_identity = InferenceIdentity(
        "speaker_001", "4" * 64, "2" * 64, "3" * 64, "prompt", "ja"
    )
    second_session = FakeSession(changed_identity)

    with pytest.raises(ValueError, match="--overwrite"):
        run_synthesis_job(second_session, "text", "en", output)

    result = run_synthesis_job(second_session, "text", "en", output, overwrite=True)
    assert result.generated_chunks == 1
    assert second_session.calls == [("text", 0)]


def test_job_rejects_tampered_manifest_request(tmp_path: Path):
    session = FakeSession()
    output = resolve_output_path(tmp_path, session.identity.model_name, Path("article.wav"))
    run_synthesis_job(session, "text", "en", output)
    manifest_path = output.with_suffix(".infer") / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["request"]["text"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="--overwrite"):
        run_synthesis_job(session, "text", "en", output)

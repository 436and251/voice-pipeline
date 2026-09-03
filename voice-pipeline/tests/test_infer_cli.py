from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from voice_pipeline.cli import infer as infer_cli
from voice_pipeline.cli.main import app
from voice_pipeline.inference.job import JobResult


runner = CliRunner()


@pytest.fixture
def fake_runtime(tmp_path: Path, monkeypatch):
    bundle = tmp_path / "models" / "speaker_001"
    bundle.mkdir(parents=True)
    calls = {"loads": [], "jobs": []}
    session = SimpleNamespace(identity=SimpleNamespace(model_name="speaker_001"))

    def load(model, device, **reference):
        calls["loads"].append((model, device, reference))
        return session

    def run_job(loaded, text, language, output_path, **options):
        calls["jobs"].append((loaded, text, language, output_path, options))
        return JobResult(output_path, 1, 0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(infer_cli.InferenceSession, "load", load)
    monkeypatch.setattr(infer_cli, "run_synthesis_job", run_job)
    return bundle, calls


def test_infer_synthesize_forwards_defaults_and_namespaces_output(fake_runtime):
    bundle, calls = fake_runtime
    result = runner.invoke(
        app,
        [
            "infer", "synthesize", "--model", str(bundle), "--text", "hello",
            "--lang", "en", "--output", "folder/hello.wav", "--device", "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls["loads"]) == 1
    assert calls["loads"][0] == (bundle.resolve(), "cpu", {})
    _, text, language, output, options = calls["jobs"][0]
    assert (text, language) == ("hello", "en")
    project_root = bundle.parents[1]
    assert output == (project_root / "outputs/speaker_001/folder/hello.wav").resolve()
    assert options == {
        "overwrite": False,
        "pause_ms": 10,
        "max_chars": None,
        "seed": 0,
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 1.0,
        "repetition_penalty": 1.35,
        "noise_scale": 0.5,
        "speed": 1.0,
    }
    assert str(output) in result.stdout


@pytest.mark.parametrize("source_args", [[], ["--text", "a", "--text-file", "input.txt"]])
def test_infer_synthesize_requires_exactly_one_text_source(fake_runtime, source_args):
    bundle, calls = fake_runtime
    Path("input.txt").write_text("from file", encoding="utf-8")
    result = runner.invoke(
        app,
        ["infer", "synthesize", "--model", str(bundle), "--lang", "en", "--output", "x.wav", *source_args],
    )
    assert result.exit_code == 1
    assert "exactly one" in result.stderr
    assert calls["loads"] == []


def test_infer_synthesize_passes_reference_override(fake_runtime):
    bundle, calls = fake_runtime
    reference = Path("reference.wav")
    reference.write_bytes(b"audio")
    result = runner.invoke(
        app,
        [
            "infer", "synthesize", "--model", str(bundle), "--text", "hello", "--lang", "en",
            "--output", "x.wav", "--reference", str(reference), "--reference-lang", "ja",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls["loads"][0][2] == {
        "reference_audio": reference.resolve(),
        "reference_text": None,
        "reference_language": "ja",
    }


def test_infer_synthesize_reports_unsafe_output_as_controlled_error(fake_runtime):
    bundle, _ = fake_runtime
    result = runner.invoke(
        app,
        ["infer", "synthesize", "--model", str(bundle), "--text", "hello", "--lang", "en", "--output", "../x.wav"],
    )
    assert result.exit_code == 1
    assert "Error:" in result.stderr and "safe relative" in result.stderr


def test_infer_batch_loads_once_and_applies_job_overrides(fake_runtime):
    bundle, calls = fake_runtime
    Path("input.txt").write_text("second", encoding="utf-8")
    config = Path("infer.yaml")
    config.write_text(
        f"""
model: {bundle.as_posix()}
device: cpu
output_root: rendered
defaults:
  language: mixed
  pause_ms: 10
  seed: 4
  top_k: 5
jobs:
  - name: greeting
    text: hello
    language: en
  - name: nested/article
    text_file: input.txt
    pause_ms: 25
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["infer", "batch", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert len(calls["loads"]) == 1
    assert [(call[1], call[2]) for call in calls["jobs"]] == [("hello", "en"), ("second", "mixed")]
    assert calls["jobs"][0][3] == Path("rendered/speaker_001/greeting.wav").resolve()
    assert calls["jobs"][1][3] == Path("rendered/speaker_001/nested/article.wav").resolve()
    assert calls["jobs"][0][4]["pause_ms"] == 10
    assert calls["jobs"][1][4]["pause_ms"] == 25
    assert calls["jobs"][1][4]["seed"] == 4
    assert "completed 2 jobs" in result.stdout


@pytest.mark.parametrize(
    "yaml_text, message",
    [
        ("model: models/speaker_001\njobs: []\nunknown: true\n", "unknown"),
        (
            "model: models/speaker_001\ndefaults: {language: en}\njobs:\n"
            "  - {name: same, text: one}\n  - {name: same, text: two}\n",
            "duplicate",
        ),
        (
            "model: models/speaker_001\ndefaults: {language: en}\njobs:\n"
            "  - {name: one, text: hello, reference: bad.wav}\n",
            "unknown",
        ),
    ],
)
def test_infer_batch_rejects_invalid_strict_config(fake_runtime, yaml_text, message):
    _, calls = fake_runtime
    config = Path("infer.yaml")
    config.write_text(yaml_text, encoding="utf-8")
    result = runner.invoke(app, ["infer", "batch", "--config", str(config)])
    assert result.exit_code == 1
    assert message in result.stderr.lower()
    assert calls["loads"] == []


def test_infer_batch_stops_at_first_failed_job(fake_runtime, monkeypatch):
    bundle, calls = fake_runtime
    config = Path("infer.yaml")
    config.write_text(
        f"""model: {bundle.as_posix()}
defaults: {{language: en}}
jobs:
  - {{name: first, text: one}}
  - {{name: second, text: two}}
""",
        encoding="utf-8",
    )

    def fail(*args, **kwargs):
        calls["jobs"].append(args)
        raise RuntimeError("synthesis failed")

    monkeypatch.setattr(infer_cli, "run_synthesis_job", fail)
    result = runner.invoke(app, ["infer", "batch", "--config", str(config)])
    assert result.exit_code == 1
    assert "synthesis failed" in result.stderr
    assert len(calls["jobs"]) == 1

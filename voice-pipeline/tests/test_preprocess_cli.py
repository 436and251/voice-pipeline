from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from voice_pipeline.cli.main import app
from voice_pipeline.cli import preprocess as preprocess_cli
from voice_pipeline.training.preprocess.pipeline import PreprocessSummary, QuarantineEntry


runner = CliRunner()


def config_path(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"exists")
    (tmp_path / "data.list").write_text(f"{audio}|speaker|ja|text\n", encoding="utf-8")
    config = tmp_path / "pipeline.yaml"
    output_root = (tmp_path / "runs").as_posix()
    manifest = (tmp_path / "data.list").as_posix()
    config.write_text(
        f"""profile: {{name: v2ProPlus}}
experiment: {{name: test, output_root: '{output_root}'}}
device: {{device: cpu, precision: fp32}}
dataset: {{manifest: '{manifest}'}}
preprocess: {{resume: true}}
""",
        encoding="utf-8",
    )
    return config


class FakePipeline:
    def __init__(self, root, calls, quarantined=False):
        self.context = SimpleNamespace(preprocess_dir=root / "runs" / "test" / "preprocess")
        self.calls = calls
        self.quarantined = quarantined

    def run(self, records, issues, selected_stage=None):
        self.calls.append(selected_stage)
        quarantine = (
            [QuarantineEntry("bad", 2, "bad", "bad.wav", "text", "bad", "bad sample")]
            if self.quarantined
            else []
        )
        return PreprocessSummary([records[0].sample_id], quarantine, 2, 6)


def test_preprocess_cli_exposes_all_and_stage(monkeypatch, tmp_path):
    config = config_path(tmp_path)
    calls = []
    monkeypatch.setattr(
        preprocess_cli,
        "build_preprocess_pipeline",
        lambda parsed, selected_stage=None: FakePipeline(tmp_path, calls),
    )
    monkeypatch.setattr(preprocess_cli, "publish_training_indexes", lambda *args: [])

    assert runner.invoke(app, ["preprocess", "all", "-c", str(config)]).exit_code == 0
    assert runner.invoke(app, ["preprocess", "stage", "sv", "-c", str(config)]).exit_code == 0
    assert calls == [None, "sv"]


def test_unknown_stage_is_rejected_before_build(monkeypatch, tmp_path):
    config = config_path(tmp_path)
    built = []
    monkeypatch.setattr(preprocess_cli, "build_preprocess_pipeline", lambda *args, **kwargs: built.append(1))
    result = runner.invoke(app, ["preprocess", "stage", "cfm", "-c", str(config)])
    assert result.exit_code != 0
    assert not built


def test_tolerated_quarantine_prints_counts_and_report(monkeypatch, tmp_path):
    config = config_path(tmp_path)
    monkeypatch.setattr(
        preprocess_cli,
        "build_preprocess_pipeline",
        lambda parsed, selected_stage=None: FakePipeline(tmp_path, [], quarantined=True),
    )
    monkeypatch.setattr(preprocess_cli, "publish_training_indexes", lambda *args: [])
    result = runner.invoke(app, ["preprocess", "all", "-c", str(config)])
    assert result.exit_code == 0
    assert "bad=1" in result.stdout
    assert "allowed=2" in result.stdout
    assert "valid=1" in result.stdout
    assert "quarantine.jsonl" in result.stdout

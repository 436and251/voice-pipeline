import json
from pathlib import Path

from typer.testing import CliRunner

from voice_pipeline.cli.module import app
from voice_pipeline.module_api.runner import ModuleRunOutcome


runner = CliRunner()


def test_module_run_requires_jsonl_flag(tmp_path: Path):
    job = tmp_path / "job.json"
    job.write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["run", "--job", str(job)])

    assert result.exit_code != 0
    assert "--events-jsonl is required" in result.stderr


def test_module_run_keeps_stdout_machine_readable(tmp_path: Path, monkeypatch):
    job = tmp_path / "job.json"
    job.write_text("{}", encoding="utf-8")

    def fake_run(path, stream, *, diagnostics):
        assert path == job.resolve()
        stream.write(json.dumps({"protocol_version": 1, "type": "job_completed"}) + "\n")
        diagnostics.write("human diagnostic\n")
        return ModuleRunOutcome("job-001", "completed", ())

    monkeypatch.setattr("voice_pipeline.module_api.runner.run_module_job", fake_run)

    result = runner.invoke(app, ["run", "--job", str(job), "--events-jsonl"])

    assert result.exit_code == 0
    assert [json.loads(line)["type"] for line in result.stdout.splitlines()] == [
        "job_completed"
    ]
    assert "human diagnostic" in result.stderr


def test_module_infer_requires_jsonl_and_keeps_stdout_machine_readable(
    tmp_path: Path, monkeypatch
):
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")

    missing_flag = runner.invoke(app, ["infer", "--request", str(request)])
    assert missing_flag.exit_code == 2
    assert "--events-jsonl is required" in missing_flag.stderr

    def fake_run(path, stream, *, diagnostics):
        assert path == request.resolve()
        stream.write(json.dumps({"protocol_version": 1, "type": "inference_completed"}) + "\n")
        diagnostics.write("human diagnostic\n")
        return tmp_path / "output.wav"

    monkeypatch.setattr(
        "voice_pipeline.module_api.infer.run_module_inference", fake_run
    )
    result = runner.invoke(
        app, ["infer", "--request", str(request), "--events-jsonl"]
    )

    assert result.exit_code == 0
    assert [json.loads(line)["type"] for line in result.stdout.splitlines()] == [
        "inference_completed"
    ]
    assert "human diagnostic" in result.stderr

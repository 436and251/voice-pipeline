from __future__ import annotations

import hashlib
from io import StringIO
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from voice_pipeline.cli.module import app
from voice_pipeline.module_api.promote import ModulePromotionOutcome
from voice_pipeline.module_api.promote import promote_module_candidate


def _job_file(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    dataset = project / "dataset" / "dataset.list"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("clip.wav|speaker|ja|テスト\n", encoding="utf-8")
    output = project / "runs"
    output.mkdir()
    job_dir = project / "jobs" / "job-001"
    job_dir.mkdir(parents=True)
    job = {
        "protocol_version": 1,
        "job_id": "job-001",
        "project_name": "Acane",
        "project_root": str(project.resolve()),
        "output_root": str(output.resolve()),
        "dataset_list": str(dataset.resolve()),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "framework": "v2ProPlus",
        "stages": ["preprocess", "s2", "s1", "evaluate"],
        "device": "cuda:0",
        "precision": "fp16",
        "parameters": {},
        "reference": {
            "audio": str((project / "reference.wav").resolve()),
            "text": "参照音声です。",
            "language": "ja",
        },
        "job_dir": str(job_dir.resolve()),
    }
    (project / "reference.wav").write_bytes(b"reference")
    path = job_dir / "job.json"
    path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    return path, output / "Acane"


def _manifest(run: Path, *, corrupt_hash: bool = False) -> None:
    listening = run / "evaluation" / "listening"
    entries = []
    for candidate in ("candidate_A", "candidate_B"):
        samples = []
        for language in ("zh", "ja", "en"):
            wav = listening / candidate / f"{language}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.write_bytes(f"{candidate}-{language}".encode())
            digest = hashlib.sha256(wav.read_bytes()).hexdigest()
            samples.append(
                {
                    "language": language,
                    "text": f"{language} sample",
                    "wav": f"{candidate}/{language}.wav",
                    "sha256": (
                        "0" * 64
                        if corrupt_hash and not entries and language == "zh"
                        else digest
                    ),
                }
            )
        entries.append({"candidate": candidate, "samples": samples})
    (listening / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "candidates": entries}),
        encoding="utf-8",
    )


def test_promotion_accepts_only_manifest_id_and_emits_model_after_cleanup(tmp_path: Path) -> None:
    job, run = _job_file(tmp_path)
    _manifest(run)
    stream = StringIO()
    calls = []
    destination = job.parents[2] / "models" / "Acane"

    def validate(run_dir, project_root):
        calls.append(("validate", run_dir, project_root))

    def promote(run_dir, candidate_id, project_root):
        calls.append(("promote", run_dir, candidate_id, project_root))
        destination.mkdir(parents=True)
        return destination

    def cleanup(run_dir, project_root):
        calls.append(("cleanup", run_dir, project_root))

    outcome = promote_module_candidate(
        job,
        "candidate_A",
        stream,
        validate=validate,
        promote=promote,
        cleanup=cleanup,
    )

    assert [call[0] for call in calls] == ["validate", "promote", "cleanup"]
    assert outcome.promoted_model == destination.resolve()
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    artifact = next(event for event in events if event["type"] == "artifact")
    assert artifact["artifacts"] == [
        {"type": "promoted_model", "path": str(destination.resolve())}
    ]
    assert events[-1]["type"] == "promotion_completed"


@pytest.mark.parametrize("selection", ["A", "candidate_Z"])
def test_promotion_rejects_visible_or_unknown_id_before_copy_and_cleanup(
    tmp_path: Path, selection: str
) -> None:
    job, run = _job_file(tmp_path)
    _manifest(run)
    calls = []

    with pytest.raises(ValueError, match="listening manifest"):
        promote_module_candidate(
            job,
            selection,
            StringIO(),
            validate=lambda *_: None,
            promote=lambda *_: calls.append("promote"),
            cleanup=lambda *_: calls.append("cleanup"),
        )

    assert calls == []


def test_manifest_hash_failure_does_not_promote_or_cleanup(tmp_path: Path) -> None:
    job, run = _job_file(tmp_path)
    _manifest(run, corrupt_hash=True)
    calls = []

    def reject_invalid_artifacts(*_):
        raise ValueError("listening sample SHA-256 mismatch")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        promote_module_candidate(
            job,
            "candidate_A",
            StringIO(),
            validate=reject_invalid_artifacts,
            promote=lambda *_: calls.append("promote"),
            cleanup=lambda *_: calls.append("cleanup"),
        )

    assert calls == []


def test_failed_copy_preserves_run_and_skips_cleanup(tmp_path: Path) -> None:
    job, run = _job_file(tmp_path)
    _manifest(run)
    checkpoint = run / "training" / "s1" / "checkpoints" / "step.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"keep")
    before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    cleanup_calls = []

    def fail_copy(*_):
        raise OSError("copy failed")

    with pytest.raises(OSError, match="copy failed"):
        promote_module_candidate(
            job,
            "candidate_A",
            StringIO(),
            validate=lambda *_: None,
            promote=fail_copy,
            cleanup=lambda *_: cleanup_calls.append(True),
        )

    assert cleanup_calls == []
    assert {path: path.read_bytes() for path in run.rglob("*") if path.is_file()} == before


def test_promote_cli_requires_jsonl_and_keeps_stdout_machine_readable(
    tmp_path: Path, monkeypatch
) -> None:
    job = tmp_path / "job.json"
    job.write_text("{}", encoding="utf-8")
    runner = CliRunner()

    missing_flag = runner.invoke(
        app, ["promote", "--job", str(job), "--selection", "candidate_A"]
    )
    assert missing_flag.exit_code == 2
    assert "--events-jsonl is required" in missing_flag.stderr

    def fake_promote(path, selection, stream, *, diagnostics):
        assert path == job.resolve()
        assert selection == "candidate_A"
        event = {"protocol_version": 1, "type": "promotion_completed"}
        stream.write(json.dumps(event) + "\n")
        diagnostics.write("human diagnostic\n")
        return ModulePromotionOutcome("job-001", selection, tmp_path / "model")

    monkeypatch.setattr(
        "voice_pipeline.module_api.promote.promote_module_candidate", fake_promote
    )
    result = runner.invoke(
        app,
        [
            "promote",
            "--job",
            str(job),
            "--selection",
            "candidate_A",
            "--events-jsonl",
        ],
    )

    assert result.exit_code == 0
    assert [json.loads(line)["type"] for line in result.stdout.splitlines()] == [
        "promotion_completed"
    ]
    assert "human diagnostic" in result.stderr

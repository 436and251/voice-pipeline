from __future__ import annotations

import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from voice_pipeline.module_api.job import ModuleJob
from voice_pipeline.module_api.materialize import MaterializedJob, materialize_job
from voice_pipeline.module_api.runner import run_module_job
from voice_pipeline.pipeline.config import PipelineSpec
from voice_pipeline.pipeline.state import PipelineState


def _job_file(tmp_path: Path, stages: list[str]) -> Path:
    project = tmp_path / "project"
    dataset = project / "dataset" / "dataset.list"
    dataset.parent.mkdir(parents=True)
    (dataset.parent / "clip.wav").write_bytes(b"clip")
    dataset.write_text("clip.wav|speaker|ja|テスト\n", encoding="utf-8")
    reference = project / "reference.wav"
    reference.write_bytes(b"reference")
    output = project / "runs"
    output.mkdir()
    job_dir = project / "jobs" / "job-001"
    job_dir.mkdir(parents=True)
    job_path = job_dir / "job.json"
    payload = {
        "protocol_version": 1,
        "job_id": "job-001",
        "project_name": "Acane",
        "project_root": str(project.resolve()),
        "output_root": str(output.resolve()),
        "dataset_list": str(dataset.resolve()),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "framework": "v2ProPlus",
        "stages": stages,
        "device": "cuda:0",
        "precision": "fp16",
        "parameters": {},
        "reference": (
            {"audio": str(reference.resolve()), "text": "参照音声です。", "language": "ja"}
            if "evaluate" in stages
            else None
        ),
        "job_dir": str(job_dir.resolve()),
    }
    job_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return job_path


def _materialized(job_path: Path) -> MaterializedJob:
    project = job_path.parents[2]
    module_dir = job_path.parent / "module"
    module_dir.mkdir()
    training = module_dir / "train.yaml"
    pipeline = module_dir / "pipeline.yaml"
    training.write_text("training", encoding="utf-8")
    pipeline.write_text("pipeline", encoding="utf-8")
    return MaterializedJob(training, pipeline, project / "runs" / "Acane")


def _write_evaluation_artifacts(run_dir: Path) -> None:
    evaluation = run_dir / "evaluation"
    listening = evaluation / "listening"
    listening.mkdir(parents=True)
    (evaluation / "report.md").write_text("report", encoding="utf-8")
    (listening / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidates": [
                    {"candidate": "candidate_A", "samples": []},
                    {"candidate": "candidate_B", "samples": []},
                ],
            }
        ),
        encoding="utf-8",
    )
    for candidate in ("candidate_A", "candidate_B"):
        bundle = run_dir / "export" / "candidates" / candidate
        bundle.mkdir(parents=True)
        (bundle / "model.yaml").write_text("model", encoding="utf-8")


def test_runner_materializes_before_execution_and_declares_documented_artifacts(tmp_path):
    job_path = _job_file(tmp_path, ["preprocess", "s2", "s1", "evaluate"])
    stream = StringIO()
    diagnostics = StringIO()
    calls = []

    def materialize(job):
        calls.append("materialize")
        return _materialized(job_path)

    def orchestrate(path, root, *, event_sink, cancellation):
        calls.append("orchestrate")
        assert calls == ["materialize", "orchestrate"]
        assert path == job_path.parent / "module" / "pipeline.yaml"
        print("legacy human log")
        event_sink({"type": "stage_started", "stage": "preprocess"})
        run_dir = root / "runs" / "Acane"
        run_dir.mkdir(parents=True)
        state = run_dir / "pipeline-state.json"
        state.write_text("{}", encoding="utf-8")
        _write_evaluation_artifacts(run_dir)
        return SimpleNamespace(state_path=state)

    outcome = run_module_job(
        job_path,
        stream,
        diagnostics=diagnostics,
        materialize=materialize,
        orchestrator=orchestrate,
    )

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert all(isinstance(event, dict) for event in events)
    assert [event["type"] for event in events] == [
        "job_started",
        "stage_started",
        "artifact",
        "artifact",
        "artifact",
        "artifact",
        "artifact",
        "job_completed",
    ]
    artifacts = [event["artifacts"][0] for event in events if event["type"] == "artifact"]
    assert [artifact["type"] for artifact in artifacts] == [
        "pipeline_state",
        "evaluation_report",
        "listening_manifest",
        "candidate_bundle",
        "candidate_bundle",
    ]
    assert [Path(artifact["path"]).name for artifact in artifacts[-2:]] == [
        "candidate_A",
        "candidate_B",
    ]
    assert outcome.status == "completed"
    assert "legacy human log" in diagnostics.getvalue()
    assert (job_path.parent / "events.jsonl").read_text(encoding="utf-8") == stream.getvalue()


def test_runner_maps_cancellation_marker_to_job_cancelled(tmp_path):
    job_path = _job_file(tmp_path, ["preprocess"])
    stream = StringIO()
    diagnostics = StringIO()
    (job_path.parent / "cancel.requested").touch()

    def cancelled(*args, cancellation, **kwargs):
        assert cancellation.requested is True
        cancellation.raise_if_requested()

    outcome = run_module_job(
        job_path,
        stream,
        diagnostics=diagnostics,
        materialize=lambda job: _materialized(job_path),
        orchestrator=cancelled,
    )

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [event["type"] for event in events] == ["job_started", "job_cancelled"]
    assert outcome.status == "cancelled"
    assert "cancelled" in diagnostics.getvalue().lower()


def test_python_module_run_stdout_contains_only_jsonl_events(tmp_path):
    job_path = _job_file(tmp_path, ["preprocess"])
    job = ModuleJob.load(job_path)
    materialized = materialize_job(job)
    spec = PipelineSpec.load(materialized.pipeline_spec, job.project_root)
    state = PipelineState.load_or_create(
        materialized.run_dir / "pipeline-state.json", spec
    )
    state.start("preprocess")
    state.complete("preprocess")
    source_root = Path(__file__).parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(source_root), environment.get("PYTHONPATH")])
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "voice_pipeline",
            "module",
            "run",
            "--job",
            str(job_path),
            "--events-jsonl",
        ],
        cwd=job.project_root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert [event["type"] for event in events] == [
        "job_started",
        "stage_cache_hit",
        "pipeline_completed",
        "artifact",
        "job_completed",
    ]
    assert "running job job-001" in result.stderr

from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _fixture(tmp_path: Path, *, text: str | None = "hello", text_file: str | None = None):
    project = tmp_path / "project"
    dataset = project / "dataset" / "dataset.list"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("clip.wav|speaker|ja|テスト\n", encoding="utf-8")
    reference = project / "reference.wav"
    reference.write_bytes(b"reference")
    output_root = project / "runs"
    output_root.mkdir()
    job_dir = project / "jobs" / "job-001"
    job_dir.mkdir(parents=True)
    job = {
        "protocol_version": 2,
        "job_id": "job-001",
        "module_id": "gpt-sovits-v2proplus",
        "project_name": "Acane",
        "project_root": str(project.resolve()),
        "output_root": str(output_root.resolve()),
        "training_data": {"path": str(dataset.resolve()), "kind": "file"},
        "framework": "v2ProPlus",
        "stages": ["preprocess", "s2", "s1", "evaluate"],
        "device": "cuda:0",
        "precision": "fp16",
        "parameters": {},
        "reference": {
            "audio": str(reference.resolve()),
            "text": "reference",
            "language": "ja",
        },
        "job_dir": str(job_dir.resolve()),
    }
    (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
    model = project / "models" / "Acane"
    model.mkdir(parents=True)
    events = [
        {
            "protocol_version": 1,
            "job_id": "job-001",
            "type": "artifact",
            "timestamp": "2026-09-18T00:00:00+00:00",
            "artifacts": [{"type": "promoted_model", "path": str(model.resolve())}],
        },
        {
            "protocol_version": 1,
            "job_id": "job-001",
            "type": "promotion_completed",
            "timestamp": "2026-09-18T00:00:01+00:00",
        },
    ]
    (job_dir / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    request_dir = job_dir / "inference" / "request-001"
    request_dir.mkdir(parents=True)
    output = project / "outputs" / "Acane" / "gui" / "sample.wav"
    payload = {
        "protocol_version": 2,
        "request_id": "request-001",
        "project_root": str(project.resolve()),
        "project_name": "Acane",
        "model": str(model.resolve()),
        "text": text,
        "text_file": text_file,
        "language": "en",
        "device": "cpu",
        "output": str(output.resolve()),
    }
    request = request_dir / "request.json"
    request.write_text(json.dumps(payload), encoding="utf-8")
    return request, payload, job_dir, model, output


def _rewrite(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_request_loads_strict_inline_contract(tmp_path):
    from voice_pipeline.module_api.infer import ModuleInferenceRequest

    request, _, job_dir, model, output = _fixture(tmp_path)

    loaded = ModuleInferenceRequest.load(request)

    assert loaded.request_id == "request-001"
    assert loaded.job.job_id == "job-001"
    assert loaded.model == model.resolve()
    assert loaded.text == "hello"
    assert loaded.language == "en"
    assert loaded.output == output.resolve()
    assert loaded.request_dir == request.parent.resolve()
    assert loaded.job.job_dir == job_dir.resolve()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda payload, root: payload.update(protocol_version=1), "protocol_version"),
        (lambda payload, root: payload.update(request_id="../bad"), "request_id"),
        (lambda payload, root: payload.update(extra=True), "unknown inference request"),
        (lambda payload, root: payload.update(project_name="Other"), "project_name"),
        (lambda payload, root: payload.update(language="auto"), "language"),
        (lambda payload, root: payload.update(text=None), "exactly one"),
        (lambda payload, root: payload.update(text_file=str(root / "input.txt")), "exactly one"),
        (lambda payload, root: payload.update(model=str(root / "other-model")), "promoted_model"),
        (lambda payload, root: payload.update(output=str(root / "bad.mp3")), "output"),
        (lambda payload, root: payload.update(output=str(root.parent / "escape.wav")), "output"),
    ],
)
def test_request_rejects_invalid_contract(tmp_path, change, message):
    from voice_pipeline.module_api.infer import ModuleInferenceRequest

    request, payload, _, _, _ = _fixture(tmp_path)
    change(payload, Path(payload["project_root"]))
    _rewrite(request, payload)

    with pytest.raises(ValueError, match=message):
        ModuleInferenceRequest.load(request)


def test_request_reads_only_contained_utf8_txt_snapshot(tmp_path):
    from voice_pipeline.module_api.infer import ModuleInferenceRequest

    request, payload, _, _, _ = _fixture(tmp_path, text=None)
    snapshot = request.parent / "input.txt"
    snapshot.write_text("from file", encoding="utf-8")
    payload["text_file"] = str(snapshot.resolve())
    _rewrite(request, payload)

    loaded = ModuleInferenceRequest.load(request)

    assert loaded.text == "from file"
    assert loaded.text_file == snapshot.resolve()

    snapshot.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        ModuleInferenceRequest.load(request)


def test_request_rejects_txt_outside_request_snapshot_directory(tmp_path):
    from voice_pipeline.module_api.infer import ModuleInferenceRequest

    request, payload, _, _, _ = _fixture(tmp_path, text=None)
    outside = Path(payload["project_root"]) / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    payload["text_file"] = str(outside.resolve())
    _rewrite(request, payload)

    with pytest.raises(ValueError, match="request directory"):
        ModuleInferenceRequest.load(request)


def test_request_requires_latest_completed_promoted_model(tmp_path):
    from voice_pipeline.module_api.infer import ModuleInferenceRequest

    request, payload, job_dir, _, _ = _fixture(tmp_path)
    newer = Path(payload["project_root"]) / "models" / "newer"
    newer.mkdir()
    with (job_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "protocol_version": 1,
            "job_id": "job-001",
            "type": "artifact",
            "timestamp": "2026-09-18T00:01:00+00:00",
            "artifacts": [{"type": "promoted_model", "path": str(newer.resolve())}],
        }) + "\n")
        stream.write(json.dumps({
            "protocol_version": 1,
            "job_id": "job-001",
            "type": "promotion_completed",
            "timestamp": "2026-09-18T00:01:01+00:00",
        }) + "\n")

    with pytest.raises(ValueError, match="promoted_model"):
        ModuleInferenceRequest.load(request)
    payload["model"] = str(newer.resolve())
    _rewrite(request, payload)
    assert ModuleInferenceRequest.load(request).model == newer.resolve()


def test_run_module_inference_reuses_runtime_and_emits_jsonl(tmp_path, monkeypatch):
    from voice_pipeline.inference.job import JobResult
    from voice_pipeline.module_api import infer as infer_module

    request, _, _, model, output = _fixture(tmp_path)
    calls = []
    session = object()

    def load(path, device):
        calls.append(("load", path, device))
        return session

    def run(loaded, text, language, output_path):
        calls.append(("run", loaded, text, language, output_path))
        output_path.parent.mkdir(parents=True)
        output_path.write_bytes(b"wav")
        return JobResult(output_path, 1, 0)

    monkeypatch.setattr(infer_module.InferenceSession, "load", load)
    monkeypatch.setattr(infer_module, "run_synthesis_job", run)
    stream, diagnostics = StringIO(), StringIO()

    result = infer_module.run_module_inference(request, stream, diagnostics=diagnostics)

    assert result == output.resolve()
    assert calls == [
        ("load", model.resolve(), "cpu"),
        ("run", session, "hello", "en", output.resolve()),
    ]
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [event["type"] for event in events] == [
        "inference_started", "artifact", "inference_completed"
    ]
    assert events[1]["artifacts"] == [
        {"type": "inference_audio", "path": str(output.resolve())}
    ]
    assert "completed" in diagnostics.getvalue()


def test_runtime_failure_emits_failure_and_preserves_request(tmp_path, monkeypatch):
    from voice_pipeline.module_api import infer as infer_module

    request, payload, _, _, _ = _fixture(tmp_path, text=None)
    snapshot = request.parent / "input.txt"
    snapshot.write_text("from file", encoding="utf-8")
    payload["text_file"] = str(snapshot.resolve())
    _rewrite(request, payload)
    original = request.read_bytes()
    original_snapshot = snapshot.read_bytes()
    monkeypatch.setattr(infer_module.InferenceSession, "load", lambda *_: object())
    monkeypatch.setattr(
        infer_module,
        "run_synthesis_job",
        lambda *_: (_ for _ in ()).throw(RuntimeError("synthesis failed")),
    )
    stream = StringIO()

    with pytest.raises(RuntimeError, match="synthesis failed"):
        infer_module.run_module_inference(request, stream, diagnostics=StringIO())

    assert json.loads(stream.getvalue().splitlines()[-1])["type"] == "inference_failed"
    assert request.read_bytes() == original
    assert snapshot.read_bytes() == original_snapshot

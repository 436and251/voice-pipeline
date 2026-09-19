from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import TextIO

from voice_pipeline.inference.job import run_synthesis_job
from voice_pipeline.inference.session import InferenceSession
from voice_pipeline.module_api.events import ModuleEventEmitter
from voice_pipeline.module_api.job import ModuleJob


_FIELDS = {
    "protocol_version",
    "request_id",
    "project_root",
    "project_name",
    "model",
    "text",
    "text_file",
    "language",
    "device",
    "output",
}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_LANGUAGES = {"zh", "ja", "en", "mixed"}


@dataclass(frozen=True, slots=True)
class ModuleInferenceRequest:
    request_id: str
    job: ModuleJob
    request_dir: Path
    project_root: Path
    project_name: str
    model: Path
    text: str
    text_file: Path | None
    language: str
    device: str
    output: Path

    @classmethod
    def load(cls, path: Path) -> "ModuleInferenceRequest":
        request_file = Path(path).resolve()
        try:
            payload = json.loads(request_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid inference request JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError("inference request must be an object")
        unknown = set(payload) - _FIELDS
        missing = _FIELDS - set(payload)
        if unknown:
            raise ValueError(
                f"unknown inference request field: {', '.join(sorted(unknown))}"
            )
        if missing:
            raise ValueError(
                f"missing inference request field: {', '.join(sorted(missing))}"
            )
        if type(payload["protocol_version"]) is not int or payload["protocol_version"] != 2:
            raise ValueError("protocol_version must be integer 2")

        request_id = _text(payload["request_id"], "request_id")
        if _SAFE_ID.fullmatch(request_id) is None:
            raise ValueError("request_id must be a safe identifier")
        request_dir = request_file.parent
        if request_file.name != "request.json" or request_dir.name != request_id:
            raise ValueError("request_id must match the request directory")
        inference_dir = request_dir.parent
        if inference_dir.name != "inference":
            raise ValueError("request must be inside the job inference directory")
        job = ModuleJob.load(inference_dir.parent / "job.json")
        if not request_dir.is_relative_to(job.job_dir):
            raise ValueError("request directory must remain inside the source job")

        project_root = _absolute_path(payload["project_root"], "project_root")
        if project_root != job.project_root:
            raise ValueError("project_root must match the source job")
        project_name = _text(payload["project_name"], "project_name")
        if project_name != job.project_name:
            raise ValueError("project_name must match the source job")

        model = _absolute_path(payload["model"], "model")
        promoted_model = _latest_promoted_model(job)
        if promoted_model is None or model != promoted_model:
            raise ValueError("model must match the latest completed promoted_model")
        if not model.is_dir():
            raise ValueError("promoted_model directory does not exist")

        raw_text = payload["text"]
        raw_text_file = payload["text_file"]
        has_text = isinstance(raw_text, str) and bool(raw_text.strip())
        has_file = isinstance(raw_text_file, str) and bool(raw_text_file.strip())
        if has_text == has_file:
            raise ValueError("exactly one of text or text_file is required")
        if raw_text is not None and not has_text:
            raise ValueError("text must be null or a non-empty string")
        if raw_text_file is not None and not has_file:
            raise ValueError("text_file must be null or a non-empty absolute path")
        text_file = None
        if has_file:
            text_file = _absolute_path(raw_text_file, "text_file")
            if not text_file.is_relative_to(request_dir) or text_file.suffix.casefold() != ".txt":
                raise ValueError("text_file must be a .txt snapshot inside the request directory")
            if not text_file.is_file():
                raise ValueError("text_file does not exist")
            try:
                text = text_file.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("text_file must be valid UTF-8") from error
            if not text.strip():
                raise ValueError("text_file must contain non-empty UTF-8 text")
        else:
            text = raw_text.strip()

        language = _text(payload["language"], "language")
        if language not in _LANGUAGES:
            raise ValueError("language must be zh, ja, en, or mixed")
        device = _text(payload["device"], "device")
        output = _absolute_path(payload["output"], "output")
        output_namespace = (
            project_root / "outputs" / project_name / "gui"
        ).resolve()
        if output.suffix.casefold() != ".wav" or not output.is_relative_to(output_namespace):
            raise ValueError(
                "output must be a .wav inside outputs/<project_name>/gui"
            )

        return cls(
            request_id=request_id,
            job=job,
            request_dir=request_dir,
            project_root=project_root,
            project_name=project_name,
            model=model,
            text=text,
            text_file=text_file,
            language=language,
            device=device,
            output=output,
        )


def run_module_inference(
    request_path: Path,
    stream: TextIO,
    *,
    diagnostics: TextIO = sys.stderr,
) -> Path:
    request = ModuleInferenceRequest.load(Path(request_path).resolve())
    emitter = ModuleEventEmitter(
        request.job.job_id,
        stream,
        request.job.job_dir / "events.jsonl",
        allowed_roots=(request.project_root,),
    )
    emitter.emit(
        "inference_started",
        message_args={"request_id": request.request_id},
    )
    try:
        session = InferenceSession.load(request.model, request.device)
        result = run_synthesis_job(
            session,
            request.text,
            request.language,
            request.output,
            progress=lambda current, total: emitter.emit(
                "inference_progress", current=current, total=total
            ),
        )
        output = Path(result.output_path).resolve()
        if output != request.output or not output.is_file():
            raise ValueError("inference runtime returned an invalid output")
        emitter.emit(
            "artifact",
            artifacts=[{"type": "inference_audio", "path": output}],
        )
        emitter.emit(
            "inference_completed",
            message_args={"request_id": request.request_id},
        )
        print(f"[module] inference completed: {output}", file=diagnostics, flush=True)
        return output
    except Exception as error:
        emitter.emit(
            "inference_failed",
            message_key="inference.failed",
            message_args={"request_id": request.request_id, "error": str(error)},
        )
        print(f"[module] inference failed: {error}", file=diagnostics, flush=True)
        raise


def _latest_promoted_model(job: ModuleJob) -> Path | None:
    journal = job.job_dir / "events.jsonl"
    if not journal.is_file():
        return None
    pending: Path | None = None
    latest: Path | None = None
    for line in journal.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            not isinstance(event, dict)
            or event.get("protocol_version") != 1
            or event.get("job_id") != job.job_id
        ):
            continue
        if event.get("type") == "artifact":
            pending = _promoted_artifact(event.get("artifacts"), job.project_root)
        elif event.get("type") == "promotion_completed" and pending is not None:
            latest = pending
            pending = None
        elif event.get("type") == "promotion_failed":
            pending = None
    return latest


def _promoted_artifact(value: object, project_root: Path) -> Path | None:
    if not isinstance(value, list):
        return None
    paths = []
    for artifact in value:
        if not isinstance(artifact, dict) or artifact.get("type") != "promoted_model":
            continue
        raw_path = artifact.get("path")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            continue
        path = Path(raw_path).resolve()
        if path.is_relative_to(project_root) and path.is_dir():
            paths.append(path)
    return paths[-1] if paths else None


def _absolute_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip() or not Path(value).is_absolute():
        raise ValueError(f"{field} must be a non-empty absolute path")
    return Path(value).resolve()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


__all__ = ["ModuleInferenceRequest", "run_module_inference"]

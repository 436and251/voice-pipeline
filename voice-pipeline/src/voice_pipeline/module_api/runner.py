from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Callable, TextIO

from voice_pipeline.module_api.cancellation import FileCancellationToken, ModuleCancelled
from voice_pipeline.module_api.events import ModuleEventEmitter
from voice_pipeline.module_api.job import ModuleJob
from voice_pipeline.module_api.materialize import MaterializedJob, materialize_job
from voice_pipeline.pipeline.orchestrator import run_pipeline


_CANDIDATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


@dataclass(frozen=True, slots=True)
class ModuleArtifact:
    type: str
    path: Path


@dataclass(frozen=True, slots=True)
class ModuleRunOutcome:
    job_id: str
    status: str
    artifacts: tuple[ModuleArtifact, ...]


def run_module_job(
    job_path: Path,
    stream: TextIO,
    *,
    diagnostics: TextIO = sys.stderr,
    materialize: Callable[[ModuleJob], MaterializedJob] = materialize_job,
    orchestrator: Callable[..., object] = run_pipeline,
) -> ModuleRunOutcome:
    job = ModuleJob.load(Path(job_path).resolve())
    emitter = ModuleEventEmitter(
        job.job_id,
        stream,
        job.job_dir / "events.jsonl",
        allowed_roots=(job.project_root,),
    )
    cancellation = FileCancellationToken(job.job_dir / "cancel.requested")
    emitter.emit("job_started")
    print(f"[module] running job {job.job_id}", file=diagnostics, flush=True)

    try:
        materialized = materialize(job)
        with redirect_stdout(diagnostics):
            orchestrator(
                materialized.pipeline_spec,
                job.project_root,
                event_sink=lambda event: emitter.emit(**event),
                cancellation=cancellation,
            )
        artifacts = _collect_artifacts(job, materialized)
        for artifact in artifacts:
            emitter.emit(
                "artifact",
                artifacts=[{"type": artifact.type, "path": artifact.path}],
            )
        emitter.emit("job_completed")
        print(f"[module] completed job {job.job_id}", file=diagnostics, flush=True)
        return ModuleRunOutcome(job.job_id, "completed", artifacts)
    except ModuleCancelled:
        emitter.emit("job_cancelled")
        print(f"[module] cancelled job {job.job_id}", file=diagnostics, flush=True)
        return ModuleRunOutcome(job.job_id, "cancelled", ())
    except Exception as error:
        emitter.emit(
            "job_failed",
            message_key="job.failed",
            message_args={"error": str(error)},
        )
        print(f"[module] failed job {job.job_id}: {error}", file=diagnostics, flush=True)
        raise


def _collect_artifacts(
    job: ModuleJob, materialized: MaterializedJob
) -> tuple[ModuleArtifact, ...]:
    run_dir = materialized.run_dir.resolve()
    expected_run_dir = (job.output_root / job.project_name).resolve()
    if run_dir != expected_run_dir:
        raise ValueError("materialized run directory does not match the target project")
    state = _required_file(run_dir / "pipeline-state.json", run_dir)
    artifacts = [ModuleArtifact("pipeline_state", state)]
    if "evaluate" not in job.stages:
        return tuple(artifacts)

    evaluation = run_dir / "evaluation"
    report = _required_file(evaluation / "report.md", run_dir)
    manifest = _required_file(evaluation / "listening" / "manifest.json", run_dir)
    artifacts.extend(
        [
            ModuleArtifact("evaluation_report", report),
            ModuleArtifact("listening_manifest", manifest),
        ]
    )
    candidate_ids = _listening_candidates(manifest)
    bundle_root = run_dir / "export" / "candidates"
    exported = (
        {path.name for path in bundle_root.iterdir() if path.is_dir()}
        if bundle_root.is_dir()
        else set()
    )
    if exported != set(candidate_ids):
        raise ValueError("exported candidate bundles do not match the listening manifest")
    for candidate_id in candidate_ids:
        bundle = (bundle_root / candidate_id).resolve()
        if not bundle.is_relative_to(run_dir) or not bundle.is_dir():
            raise ValueError(f"candidate bundle does not exist: {candidate_id}")
        _required_file(bundle / "model.yaml", run_dir)
        artifacts.append(ModuleArtifact("candidate_bundle", bundle))
    return tuple(artifacts)


def _required_file(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"required module artifact does not exist: {path}")
    return resolved


def _listening_candidates(path: Path) -> tuple[str, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid listening manifest: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("invalid listening manifest")
    entries = payload.get("candidates")
    if payload.get("schema_version") != 1 or not isinstance(entries, list) or not entries:
        raise ValueError("invalid listening manifest")
    candidate_ids = tuple(
        entry.get("candidate") if isinstance(entry, dict) else None for entry in entries
    )
    if (
        any(
            not isinstance(candidate, str)
            or _CANDIDATE_ID.fullmatch(candidate) is None
            for candidate in candidate_ids
        )
        or len(set(candidate_ids)) != len(candidate_ids)
    ):
        raise ValueError("invalid listening candidate IDs")
    return candidate_ids


__all__ = ["ModuleArtifact", "ModuleRunOutcome", "run_module_job"]

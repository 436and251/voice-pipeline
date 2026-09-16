from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Callable, TextIO

from voice_pipeline.exporting.bundles import promote_candidate
from voice_pipeline.module_api.events import ModuleEventEmitter
from voice_pipeline.module_api.job import ModuleJob
from voice_pipeline.pipeline.cleanup import cleanup_successful_run, validate_successful_run


@dataclass(frozen=True, slots=True)
class ModulePromotionOutcome:
    job_id: str
    candidate_id: str
    promoted_model: Path


def promote_module_candidate(
    job_path: Path,
    selection: str,
    stream: TextIO,
    *,
    diagnostics: TextIO = sys.stderr,
    validate: Callable[[Path, Path], object] = validate_successful_run,
    promote: Callable[[Path, str, Path], Path] = promote_candidate,
    cleanup: Callable[[Path, Path], object] = cleanup_successful_run,
) -> ModulePromotionOutcome:
    job = ModuleJob.load(Path(job_path).resolve())
    run_dir = (job.output_root / job.project_name).resolve()
    emitter = ModuleEventEmitter(
        job.job_id,
        stream,
        job.job_dir / "events.jsonl",
        allowed_roots=(job.project_root,),
    )
    emitter.emit(
        "promotion_started",
        message_args={"candidate_id": selection},
    )
    try:
        if "evaluate" not in job.stages:
            raise ValueError("promotion requires an evaluated job")
        validate(run_dir, job.project_root)
        if selection not in _manifest_candidate_ids(run_dir):
            raise ValueError(f"candidate is not present in listening manifest: {selection}")
        promoted = Path(promote(run_dir, selection, job.project_root)).resolve()
        if not promoted.is_relative_to(job.project_root) or not promoted.is_dir():
            raise ValueError("promoted model must be a directory inside project_root")
        cleanup(run_dir, job.project_root)
        emitter.emit(
            "artifact",
            artifacts=[{"type": "promoted_model", "path": promoted}],
        )
        emitter.emit(
            "promotion_completed",
            message_args={"candidate_id": selection},
        )
        print(f"[module] promoted {selection} to {promoted}", file=diagnostics, flush=True)
        return ModulePromotionOutcome(job.job_id, selection, promoted)
    except Exception as error:
        emitter.emit(
            "promotion_failed",
            message_key="promotion.failed",
            message_args={"candidate_id": selection, "error": str(error)},
        )
        print(f"[module] promotion failed: {error}", file=diagnostics, flush=True)
        raise


def _manifest_candidate_ids(run_dir: Path) -> tuple[str, ...]:
    path = run_dir / "evaluation" / "listening" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(entry["candidate"] for entry in payload["candidates"])


__all__ = ["ModulePromotionOutcome", "promote_module_candidate"]

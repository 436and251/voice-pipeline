from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import uuid

from voice_pipeline.common.model_bundle import BundleLanguages, Candidate
from voice_pipeline.exporting.bundles import build_candidate_bundle
from voice_pipeline.inference.job import run_synthesis_job
from voice_pipeline.inference.session import InferenceSession

from .candidates import CandidatePair
from .config import EvaluationConfig
from .suite import EvaluationSuite


_SYNTHESIS_OPTIONS = {
    "pause_ms": 10,
    "max_chars": None,
    "top_k": 5,
    "top_p": 1.0,
    "temperature": 1.0,
    "repetition_penalty": 1.35,
    "noise_scale": 0.5,
    "speed": 1.0,
}


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    case_id: str
    language: str
    target_text: str
    wav_path: Path
    status: str
    sha256: str | None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateGeneration:
    pair_key: str
    status: str
    samples: tuple[GeneratedSample, ...]
    generated: int
    resumed: int


def generate_candidate(
    config: EvaluationConfig,
    pair: CandidatePair,
    suite: EvaluationSuite,
    *,
    session_factory=InferenceSession.load,
) -> CandidateGeneration:
    output_dir = config.run_dir / "evaluation" / "generated" / pair.key
    manifest_path = output_dir / "manifest.json"
    request = _request(config, pair, suite)
    signature = _json_sha256(request)
    manifest = _load_or_create_manifest(manifest_path, signature, request, suite)

    generated = 0
    resumed = 0
    pending = []
    for entry in manifest["samples"]:
        wav_path = output_dir / entry["wav"]
        if entry.get("status") == "completed" and _matches_hash(wav_path, entry.get("sha256")):
            resumed += 1
        else:
            pending.append((entry, wav_path))

    session = None
    if pending:
        bundle_path = config.run_dir / "evaluation" / "work" / pair.key / "bundle"
        if not bundle_path.exists():
            build_candidate_bundle(
                bundle_path,
                profile="v2ProPlus",
                model_name=config.model_name,
                reference=config.reference,
                languages=BundleLanguages(config.trained_languages, config.validated_languages),
                candidate=Candidate(pair.key, pair.s1.path, pair.s2.path),
                project_root=config.project_root,
            )
        session = session_factory(bundle_path, device=config.device)

    for entry, wav_path in pending:
        try:
            run_synthesis_job(
                session,
                entry["text"],
                entry["language"],
                wav_path,
                seed=entry["seed"],
                **_SYNTHESIS_OPTIONS,
            )
            entry.update(
                status="completed",
                sha256=_sha256(wav_path),
                error_type=None,
                error_message=None,
            )
            generated += 1
        except Exception as error:
            entry.update(
                status="failed",
                sha256=None,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        _write_json_atomic(manifest_path, manifest)

    samples = tuple(
        GeneratedSample(
            case_id=entry["case_id"],
            language=entry["language"],
            target_text=entry["text"],
            wav_path=output_dir / entry["wav"],
            status=entry["status"],
            sha256=entry.get("sha256"),
            error_type=entry.get("error_type"),
            error_message=entry.get("error_message"),
        )
        for entry in manifest["samples"]
    )
    status = "completed" if all(sample.status == "completed" for sample in samples) else "failed"
    manifest["status"] = status
    _write_json_atomic(manifest_path, manifest)
    return CandidateGeneration(pair.key, status, samples, generated, resumed)


def _request(config: EvaluationConfig, pair: CandidatePair, suite: EvaluationSuite) -> dict:
    return {
        "profile": "v2ProPlus",
        "pair": {
            "key": pair.key,
            "s1_sha256": pair.s1.sha256,
            "s2_sha256": pair.s2.sha256,
        },
        "reference_sha256": _sha256(config.reference.audio),
        "cases": [
            {"id": case.id, "language": case.language, "text": case.text, "seed": index}
            for index, case in enumerate(suite.cases)
        ],
        "synthesis": _SYNTHESIS_OPTIONS,
    }


def _load_or_create_manifest(path: Path, signature: str, request: dict, suite: EvaluationSuite) -> dict:
    if path.is_file():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid evaluation manifest {path}: {error}") from error
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("invalid evaluation manifest")
        if manifest.get("signature") != signature or manifest.get("request") != request:
            raise ValueError("existing evaluation manifest does not match current inputs")
        entries = manifest.get("samples")
        if not isinstance(entries, list) or len(entries) != len(suite.cases):
            raise ValueError("invalid evaluation manifest samples")
        for index, (entry, case) in enumerate(zip(entries, suite.cases, strict=True)):
            if not isinstance(entry, dict) or any(
                (
                    entry.get("case_id") != case.id,
                    entry.get("language") != case.language,
                    entry.get("text") != case.text,
                    entry.get("seed") != index,
                    entry.get("wav") != f"{case.id}.wav",
                    entry.get("status") not in {"pending", "completed", "failed"},
                )
            ):
                raise ValueError("invalid evaluation manifest samples")
        return manifest
    entries = [
        {
            "case_id": case.id,
            "language": case.language,
            "text": case.text,
            "seed": index,
            "wav": f"{case.id}.wav",
            "status": "pending",
            "sha256": None,
            "error_type": None,
            "error_message": None,
        }
        for index, case in enumerate(suite.cases)
    ]
    manifest = {
        "schema_version": 1,
        "signature": signature,
        "request": request,
        "status": "pending",
        "samples": entries,
    }
    _write_json_atomic(path, manifest)
    return manifest


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_sha256(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _matches_hash(path: Path, expected) -> bool:
    return isinstance(expected, str) and path.is_file() and _sha256(path) == expected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["CandidateGeneration", "GeneratedSample", "generate_candidate"]

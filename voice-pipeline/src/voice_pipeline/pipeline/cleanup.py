from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil

import yaml

from voice_pipeline.common.model_bundle import ModelBundle


_REPORTS = (
    "stage1-report.md",
    "stage1-results.json",
    "report.md",
    "results.json",
)
_LANGUAGES = {"zh", "ja", "en"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class CleanupResult:
    removed: tuple[Path, ...]


def cleanup_successful_run(run_dir: Path, project_root: Path) -> CleanupResult:
    run = Path(run_dir).resolve()
    root = Path(project_root).resolve()
    _validate_run_boundary(run, root)
    _validate_durable_artifacts(run, root)

    removed: list[Path] = []
    _remove_unless(run, run, {"pipeline-state.json", "evaluation", "export"}, removed)
    _remove_unless(
        run / "evaluation",
        run,
        {
            "stage1-report.md",
            "stage1-results.json",
            "report.md",
            "results.json",
            "shortlist.yaml",
            "listening",
        },
        removed,
    )
    _remove_unless(run / "export", run, {"candidates"}, removed)
    return CleanupResult(tuple(removed))


def _validate_run_boundary(run: Path, project_root: Path) -> None:
    filesystem_root = Path(run.anchor).resolve()
    if run == project_root:
        raise ValueError("run_dir must not equal project_root")
    if run == filesystem_root:
        raise ValueError("run_dir must not be a filesystem root")
    if not run.is_dir():
        raise ValueError(f"run_dir is not a directory: {run}")


def _validate_durable_artifacts(run: Path, project_root: Path) -> None:
    evaluation = run / "evaluation"
    for name in _REPORTS:
        if not (evaluation / name).is_file():
            raise ValueError(f"missing evaluation artifact: {name}")

    candidate_ids = _load_shortlist_candidate_ids(evaluation / "shortlist.yaml")
    expected = set(candidate_ids)
    candidates_root = run / "export" / "candidates"
    if not candidates_root.is_dir():
        raise ValueError("missing exported candidates")
    children = tuple(candidates_root.iterdir())
    if {child.name for child in children} != expected or any(
        not child.is_dir() for child in children
    ):
        raise ValueError("exported candidate IDs do not match shortlist")
    for candidate_id in candidate_ids:
        bundle = ModelBundle.load(candidates_root / candidate_id)
        if bundle.metadata.get("candidate_id") != candidate_id:
            raise ValueError("candidate metadata id does not match its directory")

    _validate_listening_manifest(evaluation / "listening", expected)


def _load_shortlist_candidate_ids(path: Path) -> tuple[str, ...]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid shortlist: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("shortlist must be a mapping")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("shortlist.schema_version must be 1")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("shortlist.candidates must be a non-empty list")
    ids: list[str] = []
    for candidate in candidates:
        candidate_id = candidate.get("id") if isinstance(candidate, dict) else None
        if not isinstance(candidate_id, str) or not _SAFE_ID.fullmatch(candidate_id):
            raise ValueError("shortlist candidate id must be a safe name")
        if candidate_id in ids:
            raise ValueError(f"duplicate shortlist candidate id: {candidate_id}")
        ids.append(candidate_id)
    return tuple(ids)


def _validate_listening_manifest(listening: Path, expected: set[str]) -> None:
    manifest_path = listening / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid listening manifest: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "candidates"}:
        raise ValueError("invalid listening manifest fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("listening manifest schema_version must be 1")
    entries = payload["candidates"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("listening manifest candidates must be a non-empty list")

    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"candidate", "samples"}:
            raise ValueError("invalid listening candidate fields")
        candidate_id = entry["candidate"]
        if not isinstance(candidate_id, str) or candidate_id in seen:
            raise ValueError("listening candidate IDs must be unique strings")
        seen.add(candidate_id)
        _validate_samples(listening, entry["samples"])
    if seen != expected:
        raise ValueError("listening candidate IDs do not match shortlist")


def _validate_samples(listening: Path, samples: object) -> None:
    if not isinstance(samples, list) or len(samples) != 3:
        raise ValueError("listening candidate must contain exactly three samples")
    seen: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {
            "language",
            "text",
            "wav",
            "sha256",
        }:
            raise ValueError("invalid listening sample fields")
        language = sample["language"]
        if language not in _LANGUAGES or language in seen:
            raise ValueError("listening samples must contain unique zh, ja, and en")
        seen.add(language)
        if not isinstance(sample["text"], str) or not sample["text"].strip():
            raise ValueError("listening sample text must be non-empty")
        wav = sample["wav"]
        if not isinstance(wav, str) or not wav.strip():
            raise ValueError("listening sample wav must be a relative path")
        relative = Path(wav)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".wav":
            raise ValueError("listening sample wav must remain inside listening directory")
        path = (listening / relative).resolve()
        if not path.is_relative_to(listening.resolve()):
            raise ValueError("listening sample wav must remain inside listening directory")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"listening sample wav is missing or empty: {wav}")
        expected_hash = sample["sha256"]
        if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
            raise ValueError(f"listening sample SHA-256 mismatch: {wav}")
    if seen != _LANGUAGES:
        raise ValueError("listening samples must contain exactly zh, ja, and en")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_unless(
    directory: Path,
    run: Path,
    keep: set[str],
    removed: list[Path],
) -> None:
    for child in tuple(directory.iterdir()):
        if child.name in keep:
            continue
        child.relative_to(run)
        if child.is_symlink() or not child.is_dir():
            child.unlink()
        else:
            shutil.rmtree(child)
        removed.append(child)


__all__ = ["CleanupResult", "cleanup_successful_run"]

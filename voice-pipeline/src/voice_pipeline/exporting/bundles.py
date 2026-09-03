from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import uuid

from voice_pipeline.common.model_bundle import BundleReference, ModelBundle, Shortlist
from voice_pipeline.exporting.checkpoints import export_s1_checkpoint, export_s2_checkpoint
from voice_pipeline.profiles.registry import ProfileRegistry


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def export_candidates(
    shortlist: Shortlist,
    run_dir: Path,
    project_root: Path,
    overwrite: bool = False,
) -> list[Path]:
    run_dir = Path(run_dir).resolve()
    project_root = Path(project_root).resolve()
    if shortlist.project_root != project_root:
        raise ValueError("shortlist and export project roots must match")
    destination = run_dir / "export" / "candidates"
    if destination.exists() and not overwrite:
        raise ValueError(f"candidate export already exists: {destination}")
    export_root = destination.parent
    export_root.mkdir(parents=True, exist_ok=True)
    temporary = export_root / f".candidates.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        profile = ProfileRegistry.get(shortlist.profile)
        base_s1 = project_root / profile.s1_relative_path
        base_s2 = project_root / profile.s2g_relative_path
        for candidate in shortlist.candidates:
            _build_candidate(temporary / candidate.id, shortlist, candidate, base_s1, base_s2)
        _publish_tree(temporary, destination, overwrite)
    finally:
        _remove_tree(temporary)
    return [destination / candidate.id for candidate in shortlist.candidates]


def promote_candidate(
    run_dir: Path,
    candidate_id: str,
    project_root: Path,
    overwrite: bool = False,
    model_root: Path | None = None,
) -> Path:
    if not isinstance(candidate_id, str) or not _SAFE_NAME.fullmatch(candidate_id):
        raise ValueError("candidate id must be a safe name")
    run_dir = Path(run_dir).resolve()
    project_root = Path(project_root).resolve()
    source = run_dir / "export" / "candidates" / candidate_id
    if not source.is_dir():
        raise ValueError(f"candidate does not exist: {candidate_id}")
    bundle = ModelBundle.load(source)
    if bundle.metadata.get("candidate_id") != candidate_id:
        raise ValueError("candidate metadata id does not match its directory")
    model_name = bundle.metadata.get("model_name")
    if not isinstance(model_name, str) or not _SAFE_NAME.fullmatch(model_name):
        raise ValueError("candidate metadata contains an invalid model_name")
    model_root = (Path(model_root).resolve() if model_root is not None else project_root / "models")
    destination = model_root / model_name
    if destination.exists() and not overwrite:
        raise ValueError(f"model bundle already exists: {destination}")
    model_root.mkdir(parents=True, exist_ok=True)
    temporary = model_root / f".{model_name}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copytree(source, temporary)
        ModelBundle.load(temporary)
        _publish_tree(temporary, destination, overwrite)
    finally:
        _remove_tree(temporary)
    return destination


def _build_candidate(root, shortlist, candidate, base_s1, base_s2) -> None:
    weights = root / "weights"
    reference = root / "reference"
    weights.mkdir(parents=True)
    reference.mkdir()
    s1_metadata = export_s1_checkpoint(candidate.s1, base_s1, weights / "s1.ckpt")
    s2_metadata = export_s2_checkpoint(candidate.s2, base_s2, weights / "s2.pth")
    shutil.copy2(shortlist.reference.audio, reference / "default.wav")
    reference_payload = {"language": shortlist.reference.language}
    if shortlist.reference.text is not None:
        reference_payload["text"] = shortlist.reference.text
    (reference / "default.json").write_text(
        json.dumps(reference_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bundle = ModelBundle(
        root=root,
        profile=shortlist.profile,
        weights={"s1": Path("weights/s1.ckpt"), "s2": Path("weights/s2.pth")},
        reference=BundleReference(Path("reference/default.wav"), shortlist.reference.text, shortlist.reference.language),
        languages=shortlist.languages,
        metadata={
            "candidate_id": candidate.id,
            "model_name": shortlist.model_name,
            "profile": shortlist.profile,
            "checkpoints": {"s1": s1_metadata, "s2": s2_metadata},
        },
    )
    bundle.write()
    ModelBundle.load(root)


def _publish_tree(temporary: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise ValueError(f"destination already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.bak")
    moved_existing = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved_existing = True
        os.replace(temporary, destination)
    except Exception:
        if moved_existing and not destination.exists() and backup.exists():
            os.replace(backup, destination)
        raise
    else:
        _remove_tree(backup)


def _remove_tree(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


__all__ = ["export_candidates", "promote_candidate"]

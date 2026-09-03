from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from voice_pipeline.common.model_bundle import (
    BundleLanguages,
    BundleReference,
    Candidate,
    ModelBundle,
    Shortlist,
)


def _write_shortlist(run_dir: Path, project_root: Path, **changes) -> Path:
    reference = project_root / "data" / "reference.wav"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes(b"RIFF")
    candidates = []
    for name in ("A", "B", "C"):
        s1 = project_root / "runs" / f"s1-{name}.ckpt"
        s2 = project_root / "runs" / f"s2-{name}.pth"
        s1.parent.mkdir(parents=True, exist_ok=True)
        s1.write_bytes(b"s1")
        s2.write_bytes(b"s2")
        candidates.append(
            {"id": f"candidate_{name}", "s1": s1.relative_to(project_root).as_posix(), "s2": s2.relative_to(project_root).as_posix()}
        )
    payload = {
        "schema_version": 1,
        "profile": "v2ProPlus",
        "model_name": "speaker_name",
        "reference": {"audio": "data/reference.wav", "text": "こんにちは。", "language": "ja"},
        "languages": {"trained": ["ja"], "validated": ["zh", "ja", "en"]},
        "candidates": candidates,
    }
    payload.update(changes)
    path = run_dir / "evaluation" / "shortlist.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_load_three_candidate_shortlist(tmp_path: Path):
    run_dir = tmp_path / "runs" / "speaker"
    _write_shortlist(run_dir, tmp_path)

    shortlist = Shortlist.load(run_dir, tmp_path)

    assert shortlist.model_name == "speaker_name"
    assert shortlist.reference.language == "ja"
    assert shortlist.languages.validated == ("zh", "ja", "en")
    assert [candidate.id for candidate in shortlist.candidates] == ["candidate_A", "candidate_B", "candidate_C"]
    assert shortlist.candidates[0].s1 == (tmp_path / "runs" / "s1-A.ckpt").resolve()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"profile": "v2"}, "v2ProPlus"),
        ({"candidates": []}, "at least one"),
        ({"candidates": [{"id": "../bad", "s1": "runs/s1-A.ckpt", "s2": "runs/s2-A.pth"}]}, "candidate id"),
        ({"candidates": [{"id": "same", "s1": "runs/s1-A.ckpt", "s2": "runs/s2-A.pth"}, {"id": "same", "s1": "runs/s1-B.ckpt", "s2": "runs/s2-B.pth"}]}, "duplicate"),
        ({"languages": {"trained": ["fr"], "validated": ["ja"]}}, "zh, ja, or en"),
        ({"unexpected": True}, "unknown shortlist field"),
    ],
)
def test_rejects_invalid_shortlist(tmp_path: Path, changes: dict, message: str):
    run_dir = tmp_path / "run"
    _write_shortlist(run_dir, tmp_path, **changes)
    with pytest.raises(ValueError, match=message):
        Shortlist.load(run_dir, tmp_path)


@pytest.mark.parametrize("source", ["../outside.ckpt", "C:/outside.ckpt"])
def test_rejects_escaping_or_absolute_checkpoint_paths(tmp_path: Path, source: str):
    run_dir = tmp_path / "run"
    _write_shortlist(
        run_dir,
        tmp_path,
        candidates=[{"id": "candidate_A", "s1": source, "s2": "runs/s2-A.pth"}],
    )
    with pytest.raises(ValueError, match="safe project-relative path"):
        Shortlist.load(run_dir, tmp_path)


def _bundle(root: Path, *, text: str | None = "こんにちは。") -> ModelBundle:
    (root / "weights").mkdir(parents=True)
    (root / "reference").mkdir()
    (root / "weights" / "s1.ckpt").write_bytes(b"s1")
    (root / "weights" / "s2.pth").write_bytes(b"s2")
    (root / "reference" / "default.wav").write_bytes(b"RIFF")
    (root / "reference" / "default.json").write_text("{}", encoding="utf-8")
    return ModelBundle(
        root=root,
        profile="v2ProPlus",
        weights={"s1": Path("weights/s1.ckpt"), "s2": Path("weights/s2.pth")},
        reference=BundleReference(Path("reference/default.wav"), text, "ja"),
        languages=BundleLanguages(("ja",), ("zh", "ja", "en")),
        metadata={"candidate_id": "candidate_A"},
    )


def test_model_bundle_write_load_and_validate(tmp_path: Path):
    root = tmp_path / "candidate_A"
    bundle = _bundle(root, text=None)
    bundle.write()

    loaded = ModelBundle.load(root)

    assert loaded.reference.text is None
    assert loaded.weights["s2"] == Path("weights/s2.pth")
    assert json.loads((root / "metadata.json").read_text(encoding="utf-8"))["candidate_id"] == "candidate_A"


def test_model_bundle_rejects_unsafe_or_missing_files(tmp_path: Path):
    root = tmp_path / "candidate_A"
    bundle = _bundle(root)
    bundle.weights["s1"] = Path("../outside.ckpt")
    with pytest.raises(ValueError, match="safe bundle-relative path"):
        bundle.write()

    bundle.weights["s1"] = Path("weights/missing.ckpt")
    with pytest.raises(ValueError, match="does not exist"):
        bundle.write()


def test_model_bundle_manifests_do_not_leak_absolute_paths(tmp_path: Path):
    root = tmp_path / "candidate_A"
    bundle = _bundle(root)
    bundle.write()
    manifests = (root / "model.yaml").read_text(encoding="utf-8") + (root / "metadata.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in manifests

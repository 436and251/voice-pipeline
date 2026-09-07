from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from voice_pipeline.common.model_bundle import (
    BundleLanguages,
    BundleReference,
    ModelBundle,
)
from voice_pipeline.pipeline.cleanup import cleanup_successful_run


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bundle(root: Path, candidate_id: str) -> None:
    (root / "weights").mkdir(parents=True)
    (root / "reference").mkdir()
    (root / "weights" / "s1.ckpt").write_bytes(b"s1-export")
    (root / "weights" / "s2.pth").write_bytes(b"s2-export")
    (root / "reference" / "default.wav").write_bytes(b"RIFF-reference")
    (root / "reference" / "default.json").write_text(
        json.dumps({"language": "ja", "text": "参照音声です。"}),
        encoding="utf-8",
    )
    bundle = ModelBundle(
        root=root,
        profile="v2ProPlus",
        weights={"s1": Path("weights/s1.ckpt"), "s2": Path("weights/s2.pth")},
        reference=BundleReference(
            Path("reference/default.wav"), "参照音声です。", "ja"
        ),
        languages=BundleLanguages(("ja",), ("zh", "ja", "en")),
        metadata={
            "candidate_id": candidate_id,
            "model_name": "speaker",
            "profile": "v2ProPlus",
            "checkpoints": {
                "s1": {
                    "source_sha256": "0" * 64,
                    "exported_sha256": _sha256(b"s1-export"),
                    "source_kind": "training_checkpoint",
                    "optimizer_step": 10,
                },
                "s2": {
                    "source_sha256": "1" * 64,
                    "exported_sha256": _sha256(b"s2-export"),
                    "source_kind": "training_checkpoint",
                    "optimizer_step": 20,
                },
            },
        },
    )
    bundle.write()


def _valid_run(project_root: Path, run_dir: Path | None = None) -> Path:
    run = run_dir or project_root / "runs" / "speaker"
    reference = project_root / "data" / "reference.wav"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes(b"RIFF-source")
    s1 = run / "training" / "s1" / "checkpoints" / "step.pt"
    s2 = run / "training" / "s2" / "checkpoints" / "step.pt"
    s1.parent.mkdir(parents=True, exist_ok=True)
    s2.parent.mkdir(parents=True, exist_ok=True)
    s1.write_bytes(b"raw-s1")
    s2.write_bytes(b"raw-s2")
    evaluation = run / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    shortlist = {
        "schema_version": 1,
        "profile": "v2ProPlus",
        "model_name": "speaker",
        "reference": {
            "audio": reference.relative_to(project_root).as_posix(),
            "text": "参照音声です。",
            "language": "ja",
        },
        "languages": {"trained": ["ja"], "validated": ["zh", "ja", "en"]},
        "candidates": [
            {
                "id": "candidate_A",
                "s1": s1.relative_to(project_root).as_posix(),
                "s2": s2.relative_to(project_root).as_posix(),
            }
        ],
    }
    (evaluation / "shortlist.yaml").write_text(
        yaml.safe_dump(shortlist, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    for name in ("stage1-report.md", "stage1-results.json", "report.md", "results.json"):
        (evaluation / name).write_text(f"valid {name}\n", encoding="utf-8")

    candidate = run / "export" / "candidates" / "candidate_A"
    _write_bundle(candidate, "candidate_A")
    listening = evaluation / "listening"
    samples = []
    for language in ("zh", "ja", "en"):
        data = f"RIFF-{language}".encode()
        wav = listening / "candidate_A" / f"{language}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(data)
        samples.append(
            {
                "language": language,
                "text": f"{language} sample",
                "wav": f"candidate_A/{language}.wav",
                "sha256": _sha256(data),
            }
        )
    (listening / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidates": [{"candidate": "candidate_A", "samples": samples}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (run / "preprocess").mkdir(parents=True, exist_ok=True)
    (run / "preprocess" / "features.pt").write_bytes(b"features")
    (evaluation / "generated").mkdir()
    (evaluation / "generated" / "raw.wav").write_bytes(b"raw")
    (evaluation / "work" / "bundle").mkdir(parents=True)
    (evaluation / "work" / "bundle" / "model.yaml").write_text(
        "temporary", encoding="utf-8"
    )
    (run / "unneeded.log").write_text("remove me", encoding="utf-8")
    (run / "pipeline-state.json").write_text("{}", encoding="utf-8")
    return run


def test_cleanup_validates_then_preserves_only_durable_artifacts(tmp_path: Path) -> None:
    run = _valid_run(tmp_path)

    result = cleanup_successful_run(run, tmp_path)

    assert not (run / "preprocess").exists()
    assert not (run / "training").exists()
    assert not (run / "evaluation" / "generated").exists()
    assert not (run / "evaluation" / "work").exists()
    assert not (run / "unneeded.log").exists()
    ModelBundle.load(run / "export" / "candidates" / "candidate_A")
    assert (run / "evaluation" / "report.md").is_file()
    assert (run / "evaluation" / "listening" / "candidate_A" / "zh.wav").is_file()
    assert (run / "pipeline-state.json").is_file()
    assert result.removed


def test_cleanup_is_idempotent_after_raw_checkpoints_are_removed(tmp_path: Path) -> None:
    run = _valid_run(tmp_path)
    cleanup_successful_run(run, tmp_path)

    second = cleanup_successful_run(run, tmp_path)

    assert second.removed == ()
    ModelBundle.load(run / "export" / "candidates" / "candidate_A")


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _rewrite_listening_manifest(run: Path, change) -> None:
    path = run / "evaluation" / "listening" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    "damage",
    [
        "missing_report",
        "invalid_bundle",
        "export_id_mismatch",
        "listening_id_mismatch",
        "hash_mismatch",
        "path_escape",
    ],
)
def test_cleanup_validation_failure_performs_zero_deletions(
    tmp_path: Path, damage: str
) -> None:
    run = _valid_run(tmp_path)
    if damage == "missing_report":
        (run / "evaluation" / "report.md").unlink()
    elif damage == "invalid_bundle":
        (run / "export" / "candidates" / "candidate_A" / "weights" / "s1.ckpt").write_bytes(
            b"tampered"
        )
    elif damage == "export_id_mismatch":
        (run / "export" / "candidates" / "candidate_A").rename(
            run / "export" / "candidates" / "candidate_B"
        )
    elif damage == "listening_id_mismatch":
        _rewrite_listening_manifest(
            run,
            lambda payload: payload["candidates"][0].update(candidate="candidate_B"),
        )
    elif damage == "hash_mismatch":
        _rewrite_listening_manifest(
            run,
            lambda payload: payload["candidates"][0]["samples"][0].update(
                sha256="0" * 64
            ),
        )
    else:
        _rewrite_listening_manifest(
            run,
            lambda payload: payload["candidates"][0]["samples"][0].update(
                wav="../outside.wav"
            ),
        )
    before = _snapshot(tmp_path)

    with pytest.raises(ValueError):
        cleanup_successful_run(run, tmp_path)

    assert _snapshot(tmp_path) == before


def test_cleanup_before_evaluation_performs_zero_deletions(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "speaker"
    raw = run / "training" / "s2" / "checkpoints" / "step.pt"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"keep")
    before = _snapshot(tmp_path)

    with pytest.raises(ValueError, match="missing evaluation artifact"):
        cleanup_successful_run(run, tmp_path)

    assert _snapshot(tmp_path) == before


def test_cleanup_rejects_project_root_without_deleting_files(tmp_path: Path) -> None:
    _valid_run(tmp_path, tmp_path)
    before = _snapshot(tmp_path)

    with pytest.raises(ValueError, match="must not equal project_root"):
        cleanup_successful_run(tmp_path, tmp_path)

    assert _snapshot(tmp_path) == before

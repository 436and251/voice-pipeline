from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from voice_pipeline.common.errors import EvaluationError
from voice_pipeline.common.model_bundle import BundleLanguages, BundleReference, ModelBundle, Shortlist
from voice_pipeline.evaluation.config import EvaluationConfig, EvaluationConstraints, EvaluationPairing
from voice_pipeline.evaluation.pipeline import EvaluationServices, run_evaluation
from voice_pipeline.evaluation.runner import CandidateEvaluation, CandidateGeneration
from voice_pipeline.inference.wav import write_wav_atomic


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path, *, all_fail: bool = False) -> EvaluationConfig:
    profile = tmp_path / "models" / "pretrained" / "v2proplus"
    (profile / "s1").mkdir(parents=True)
    (profile / "s2").mkdir()
    (profile / "s1" / "s1v3.ckpt").write_bytes(b"base-s1")
    (profile / "s2" / "s2Gv2ProPlus.pth").write_bytes(b"base-s2")
    run_dir = tmp_path / "runs" / "speaker"
    for kind, step_field, weights_field in (
        ("s1", "optimizer_step", "model"),
        ("s2", "global_step", "net_g"),
    ):
        directory = run_dir / "training" / kind / "checkpoints"
        directory.mkdir(parents=True)
        for step in (100, 200):
            torch.save(
                {
                    "format_version": 1,
                    "profile": "v2ProPlus",
                    step_field: step,
                    weights_field: {"x": torch.ones(1)},
                },
                directory / f"step-{step:08d}.pt",
            )
    reference = tmp_path / "data" / "reference.wav"
    write_wav_atomic(reference, np.full(1600, 0.1, dtype=np.float32), 16000)
    suites = {}
    for language in ("zh", "ja", "en", "mixed"):
        suite = tmp_path / "configs" / f"{language}.txt"
        suite.parent.mkdir(exist_ok=True)
        suite.write_text(f"{language} sample\n", encoding="utf-8")
        suites[language] = suite
    threshold = 0.99 if all_fail else 0.5
    return EvaluationConfig(
        run_dir=run_dir,
        project_root=tmp_path,
        model_name="speaker",
        trained_languages=("ja",),
        validated_languages=("zh", "ja", "en"),
        reference=BundleReference(reference, "reference", "ja"),
        speaker_references=(reference,),
        suites=suites,
        asr_model="asr/model",
        speaker_model="speaker/model",
        cache_dir=tmp_path / "cache",
        pairing=EvaluationPairing(s2_keep=1, shortlist_size=3),
        device="cpu",
        precision="fp32",
        constraints=EvaluationConstraints(
            min_speaker_similarity={"zh": threshold, "ja": threshold, "en": threshold},
            max_cer={"zh": 0.2, "ja": 0.2},
            max_wer={"en": 0.2},
            min_language_consistency={"zh": 0.5, "ja": 0.5, "en": 0.5, "mixed": 0.5},
        ),
    )


def _write_fake_bundle(root: Path, shortlist: Shortlist, candidate) -> Path:
    (root / "weights").mkdir(parents=True)
    (root / "reference").mkdir()
    (root / "weights" / "s1.ckpt").write_bytes(b"s1")
    (root / "weights" / "s2.pth").write_bytes(b"s2")
    (root / "reference" / "default.wav").write_bytes(shortlist.reference.audio.read_bytes())
    (root / "reference" / "default.json").write_text(
        json.dumps({"language": "ja", "text": "reference"}), encoding="utf-8"
    )
    def metadata(source: Path, exported: Path) -> dict:
        is_base = source.name in {"s1v3.ckpt", "s2Gv2ProPlus.pth"}
        step = None if is_base else int(source.stem.removeprefix("step-"))
        return {
            "source_sha256": _sha256(source),
            "exported_sha256": _sha256(exported),
            "source_kind": "official_base" if is_base else "training_checkpoint",
            "optimizer_step": step,
        }

    bundle = ModelBundle(
        root,
        "v2ProPlus",
        {"s1": Path("weights/s1.ckpt"), "s2": Path("weights/s2.pth")},
        BundleReference(Path("reference/default.wav"), "reference", "ja"),
        BundleLanguages(("ja",), ("zh", "ja", "en")),
        {
            "candidate_id": candidate.id,
            "model_name": "speaker",
            "profile": "v2ProPlus",
            "checkpoints": {
                "s1": metadata(candidate.s1, root / "weights/s1.ckpt"),
                "s2": metadata(candidate.s2, root / "weights/s2.pth"),
            },
        },
    )
    bundle.write()
    return root


def _services(seen: list[tuple[int | None, int | None]]) -> EvaluationServices:
    pairs = {}

    def generate(config, pair, suite):
        pairs[pair.key] = pair
        seen.append((pair.s1.step, pair.s2.step))
        return CandidateGeneration(pair.key, "completed", (), 4, 0)

    def evaluate(generation, **kwargs):
        pair = pairs[generation.pair_key]
        similarity = {None: 0.81, 100: 0.90, 200: 0.85}[pair.s2.step]
        if pair.s1.step == 100:
            similarity = 0.93
        elif pair.s1.step == 200:
            similarity = 0.91
        by_language = {
            language: {
                "speaker_similarity": similarity,
                "pronunciation": 0.05,
                "language_consistency": 1.0,
                "prosody": 1.0,
            }
            for language in ("zh", "ja", "en", "mixed")
        }
        return CandidateEvaluation(
            pair.key,
            "completed",
            (),
            by_language,
            {"worst_similarity": similarity},
        )

    def export(shortlist, run_dir, project_root, overwrite):
        destination = run_dir / "export" / "candidates"
        destination.mkdir(parents=True)
        return [_write_fake_bundle(destination / candidate.id, shortlist, candidate) for candidate in shortlist.candidates]

    def preview(bundle_path, cases, output_dir, device):
        paths = []
        for language, text in cases.items():
            path = output_dir / f"{language}.wav"
            write_wav_atomic(path, np.full(320, 0.2, dtype=np.float32), 32000)
            paths.append(path)
        return tuple(paths)

    class Speaker:
        def centroid(self, paths):
            return np.array([1.0], dtype=np.float32)

    return EvaluationServices(
        load_asr=lambda config: object(),
        load_speaker=lambda config: Speaker(),
        generate=generate,
        evaluate=evaluate,
        export=export,
        preview=preview,
    )


def test_pipeline_filters_s2_before_crossing_s1_and_exports_shortlist(tmp_path: Path) -> None:
    seen = []

    outcome = run_evaluation(_config(tmp_path), services=_services(seen))

    assert seen == [(None, None), (None, 100), (None, 200), (100, 100), (200, 100)]
    shortlist = Shortlist.load(outcome.run_dir, tmp_path)
    assert [candidate.id for candidate in shortlist.candidates] == ["candidate_A", "candidate_B", "candidate_C"]
    assert [candidate.s1.name for candidate in shortlist.candidates] == [
        "step-00000100.pt", "step-00000200.pt", "s1v3.ckpt"
    ]
    for candidate in shortlist.candidates:
        ModelBundle.load(outcome.run_dir / "export" / "candidates" / candidate.id)
        listening = outcome.run_dir / "evaluation" / "listening" / candidate.id
        assert sorted(path.name for path in listening.glob("*.wav")) == ["en.wav", "ja.wav", "zh.wav"]


def test_pipeline_does_not_write_shortlist_when_all_fail(tmp_path: Path) -> None:
    seen = []
    config = _config(tmp_path, all_fail=True)

    with pytest.raises(EvaluationError, match="no S2 candidate passed"):
        run_evaluation(config, services=_services(seen))

    assert not (config.run_dir / "evaluation" / "shortlist.yaml").exists()
    assert (config.run_dir / "evaluation" / "stage1-results.json").is_file()

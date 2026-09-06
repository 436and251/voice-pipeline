from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Callable

import yaml

from voice_pipeline.common.errors import EvaluationError
from voice_pipeline.common.model_bundle import Shortlist
from voice_pipeline.exporting.bundles import export_candidates
from voice_pipeline.inference.job import run_synthesis_job
from voice_pipeline.inference.session import InferenceSession

from .asr import WhisperEvaluator
from .candidates import CandidatePair, discover_checkpoints, stage_one_pairs, stage_two_pairs
from .config import EvaluationConfig
from .ranking import RankedCandidate, assign_anonymous_ids, rank_candidates
from .report import write_report, write_results
from .runner import evaluate_generation, generate_candidate
from .speaker_similarity import WavLMSpeakerEvaluator
from .suite import load_evaluation_suite


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    run_dir: Path
    shortlisted: tuple[str, ...]


def _load_asr(config: EvaluationConfig):
    return WhisperEvaluator.load(
        config.asr_model,
        cache_dir=config.cache_dir,
        device=config.device,
        precision=config.precision,
    )


def _load_speaker(config: EvaluationConfig):
    return WavLMSpeakerEvaluator.load(
        config.speaker_model,
        cache_dir=config.cache_dir,
        device=config.device,
    )


def _preview_bundle(bundle_path: Path, cases: dict[str, str], output_dir: Path, device: str) -> tuple[Path, ...]:
    session = InferenceSession.load(bundle_path, device=device)
    paths = []
    for language, text in cases.items():
        output = output_dir / f"{language}.wav"
        run_synthesis_job(session, text, language, output, seed=0, pause_ms=10)
        work = output.with_suffix(".infer")
        if work.is_dir():
            shutil.rmtree(work)
        paths.append(output)
    return tuple(paths)


@dataclass(frozen=True, slots=True)
class EvaluationServices:
    load_asr: Callable = _load_asr
    load_speaker: Callable = _load_speaker
    generate: Callable = generate_candidate
    evaluate: Callable = evaluate_generation
    export: Callable = export_candidates
    preview: Callable = _preview_bundle


def run_evaluation(
    config: EvaluationConfig,
    *,
    services: EvaluationServices | None = None,
) -> EvaluationOutcome:
    services = services or EvaluationServices()
    evaluation_dir = config.run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    suite = load_evaluation_suite(config)
    s1_refs, s2_refs = discover_checkpoints(config)
    asr = services.load_asr(config)
    speaker = services.load_speaker(config)
    centroid = speaker.centroid(config.speaker_references)

    evaluations = {}
    pairs = {}
    stage_one = stage_one_pairs(s1_refs[0], s2_refs)
    for pair in stage_one:
        pairs[pair.key] = pair
        generation = services.generate(config, pair, suite)
        evaluations[pair.key] = services.evaluate(
            generation,
            asr=asr,
            speaker=speaker,
            speaker_centroid=centroid,
        )
    ranked_s2 = rank_candidates(
        tuple(evaluations[pair.key] for pair in stage_one),
        config.constraints,
        config.ranking,
    )
    write_results(evaluation_dir / "stage1-results.json", ranked_s2)
    write_report(evaluation_dir / "stage1-report.md", ranked_s2)
    retained = tuple(candidate for candidate in ranked_s2 if candidate.eligible)[: config.pairing.s2_keep]
    if not retained:
        raise EvaluationError("no S2 candidate passed the evaluation constraints")
    retained_s2 = tuple(pairs[candidate.pair_key].s2 for candidate in retained)

    stage_two = stage_two_pairs(s1_refs, retained_s2)
    for pair in stage_two:
        pairs[pair.key] = pair
        if pair.key in evaluations:
            continue
        generation = services.generate(config, pair, suite)
        evaluations[pair.key] = services.evaluate(
            generation,
            asr=asr,
            speaker=speaker,
            speaker_centroid=centroid,
        )
    ranked = rank_candidates(
        tuple(evaluations[pair.key] for pair in stage_two),
        config.constraints,
        config.ranking,
    )
    write_results(evaluation_dir / "results.json", ranked)
    write_report(evaluation_dir / "report.md", ranked)
    shortlisted = assign_anonymous_ids(ranked, limit=config.pairing.shortlist_size)
    if not shortlisted:
        raise EvaluationError("no candidate passed the final evaluation constraints")

    shortlist = _write_shortlist(config, shortlisted, pairs)
    exported = services.export(shortlist, config.run_dir, config.project_root, True)
    _write_listening(config, suite, shortlisted, exported, services.preview)
    _remove_work_tree(evaluation_dir / "work", evaluation_dir)
    return EvaluationOutcome(config.run_dir, tuple(candidate.public_id for candidate in shortlisted))


def _write_shortlist(
    config: EvaluationConfig,
    shortlisted: tuple[RankedCandidate, ...],
    pairs: dict[str, CandidatePair],
) -> Shortlist:
    reference = {
        "audio": _project_relative(config.reference.audio, config.project_root),
        "language": config.reference.language,
    }
    if config.reference.text is not None:
        reference["text"] = config.reference.text
    payload = {
        "schema_version": 1,
        "profile": "v2ProPlus",
        "model_name": config.model_name,
        "reference": reference,
        "languages": {
            "trained": list(config.trained_languages),
            "validated": list(config.validated_languages),
        },
        "candidates": [
            {
                "id": candidate.public_id,
                "s1": _project_relative(pairs[candidate.pair_key].s1.path, config.project_root),
                "s2": _project_relative(pairs[candidate.pair_key].s2.path, config.project_root),
            }
            for candidate in shortlisted
        ],
    }
    path = config.run_dir / "evaluation" / "shortlist.yaml"
    _write_text_atomic(path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    return Shortlist.load(config.run_dir, config.project_root)


def _write_listening(config, suite, shortlisted, exported, preview) -> None:
    evaluation_dir = config.run_dir / "evaluation"
    destination = evaluation_dir / "listening"
    temporary = evaluation_dir / f".listening.{uuid.uuid4().hex}.tmp"
    cases = {
        language: next(case.text for case in suite.cases if case.language == language)
        for language in ("zh", "ja", "en")
    }
    entries = []
    try:
        temporary.mkdir()
        for candidate, bundle_path in zip(shortlisted, exported, strict=True):
            candidate_dir = temporary / candidate.public_id
            paths = preview(bundle_path, cases, candidate_dir, config.device)
            entries.append(
                {
                    "candidate": candidate.public_id,
                    "samples": [
                        {
                            "language": language,
                            "text": cases[language],
                            "wav": path.relative_to(temporary).as_posix(),
                            "sha256": _sha256(path),
                        }
                        for language, path in zip(cases, paths, strict=True)
                    ],
                }
            )
        _write_text_atomic(
            temporary / "manifest.json",
            json.dumps({"schema_version": 1, "candidates": entries}, ensure_ascii=False, indent=2) + "\n",
        )
        _publish_tree(temporary, destination)
    finally:
        if temporary.is_dir():
            shutil.rmtree(temporary)


def _project_relative(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError as error:
        raise EvaluationError(f"evaluation path must remain inside project root: {path}") from error


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_tree(temporary: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.bak")
    try:
        if destination.exists():
            os.replace(destination, backup)
        os.replace(temporary, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.is_dir():
            shutil.rmtree(backup)


def _remove_work_tree(path: Path, evaluation_dir: Path) -> None:
    path = path.resolve()
    if path.parent != evaluation_dir.resolve():
        raise EvaluationError("refusing to remove evaluation work outside its run")
    if path.is_dir():
        shutil.rmtree(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["EvaluationOutcome", "EvaluationServices", "run_evaluation"]

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import EvaluationConstraints, EvaluationRanking
from .runner import CandidateEvaluation


@dataclass(frozen=True, slots=True)
class ConstraintFailure:
    metric: str
    actual: float | str | None
    threshold: float | str
    reason: str


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    pair_key: str
    evaluation: CandidateEvaluation
    eligible: bool
    failures: tuple[ConstraintFailure, ...]
    score: float | None
    components: dict[str, float]
    public_id: str | None = None


def rank_candidates(
    evaluations: tuple[CandidateEvaluation, ...],
    constraints: EvaluationConstraints,
    ranking: EvaluationRanking,
) -> tuple[RankedCandidate, ...]:
    ranked = []
    for evaluation in evaluations:
        failures = _failures(evaluation, constraints)
        components = _components(evaluation) if not failures else {}
        score = (
            components["speaker"] * ranking.speaker
            + components["pronunciation"] * ranking.pronunciation
            + components["language_consistency"] * ranking.language_consistency
            + components["prosody"] * ranking.prosody
            if components
            else None
        )
        ranked.append(
            RankedCandidate(evaluation.pair_key, evaluation, not failures, tuple(failures), score, components)
        )
    return tuple(
        sorted(
            ranked,
            key=lambda candidate: (
                not candidate.eligible,
                -(candidate.score if candidate.score is not None else -1),
                candidate.pair_key,
            ),
        )
    )


def assign_anonymous_ids(
    ranked: tuple[RankedCandidate, ...],
    *,
    limit: int,
) -> tuple[RankedCandidate, ...]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 26:
        raise ValueError("anonymous candidate limit must be between 1 and 26")
    eligible = [candidate for candidate in ranked if candidate.eligible][:limit]
    return tuple(
        replace(candidate, public_id=f"candidate_{chr(ord('A') + index)}")
        for index, candidate in enumerate(eligible)
    )


def _failures(evaluation: CandidateEvaluation, constraints: EvaluationConstraints) -> list[ConstraintFailure]:
    failures: list[ConstraintFailure] = []
    if evaluation.status != "completed":
        failures.append(ConstraintFailure("candidate.status", evaluation.status, "completed", "candidate evaluation failed"))
    for language, threshold in constraints.min_speaker_similarity.items():
        _minimum(failures, evaluation, language, "speaker_similarity", threshold)
    for language, threshold in constraints.max_cer.items():
        _maximum(failures, evaluation, language, "pronunciation", threshold)
    for language, threshold in constraints.max_wer.items():
        _maximum(failures, evaluation, language, "pronunciation", threshold)
    for language, threshold in constraints.min_language_consistency.items():
        if language in evaluation.by_language:
            _minimum(failures, evaluation, language, "language_consistency", threshold)
    audio_failures = set()
    for sample in evaluation.samples:
        prosody = sample.metrics.get("prosody")
        if prosody is None:
            continue
        for flag in ("all_silent", "severe_clipping"):
            if prosody.details.get(flag) and flag not in audio_failures:
                failures.append(ConstraintFailure(f"audio.{flag}", 1.0, 0.0, f"audio flag {flag} is set"))
                audio_failures.add(flag)
    return failures


def _minimum(failures, evaluation, language: str, metric: str, threshold: float) -> None:
    actual = evaluation.by_language.get(language, {}).get(metric)
    if actual is None or actual < threshold:
        failures.append(ConstraintFailure(f"{metric}.{language}", actual, threshold, "value is below minimum"))


def _maximum(failures, evaluation, language: str, metric: str, threshold: float) -> None:
    actual = evaluation.by_language.get(language, {}).get(metric)
    if actual is None or actual > threshold:
        failures.append(ConstraintFailure(f"{metric}.{language}", actual, threshold, "value exceeds maximum"))


def _components(evaluation: CandidateEvaluation) -> dict[str, float]:
    language_metrics = tuple(evaluation.by_language.values())
    similarities = [metrics["speaker_similarity"] for metrics in language_metrics if "speaker_similarity" in metrics]
    worst = evaluation.summary["worst_similarity"]
    speaker = (2 * worst + sum(similarities) / len(similarities)) / 3
    errors = [metrics["pronunciation"] for metrics in language_metrics if "pronunciation" in metrics]
    language = [metrics["language_consistency"] for metrics in language_metrics if "language_consistency" in metrics]
    prosody = [metrics["prosody"] for metrics in language_metrics if "prosody" in metrics]
    return {
        "speaker": _clamp(speaker),
        "pronunciation": sum(1 - min(max(error, 0), 1) for error in errors) / len(errors),
        "language_consistency": sum(_clamp(value) for value in language) / len(language),
        "prosody": sum(_clamp(value) for value in prosody) / len(prosody),
    }


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


__all__ = ["ConstraintFailure", "RankedCandidate", "assign_anonymous_ids", "rank_candidates"]

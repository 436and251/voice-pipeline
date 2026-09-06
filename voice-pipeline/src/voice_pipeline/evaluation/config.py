from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re

from voice_pipeline.common.model_bundle import BundleReference


LANGUAGES = ("zh", "ja", "en", "mixed")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_DEFAULT_ASR = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
_DEFAULT_SPEAKER = "microsoft/wavlm-base-plus-sv"


@dataclass(frozen=True, slots=True)
class EvaluationPairing:
    s2_keep: int = 2
    shortlist_size: int = 3


@dataclass(frozen=True, slots=True)
class EvaluationConstraints:
    min_speaker_similarity: dict[str, float] = field(
        default_factory=lambda: {"zh": 0.70, "ja": 0.75, "en": 0.70}
    )
    max_cer: dict[str, float] = field(default_factory=lambda: {"zh": 0.30, "ja": 0.25})
    max_wer: dict[str, float] = field(default_factory=lambda: {"en": 0.35})
    min_language_consistency: dict[str, float] = field(
        default_factory=lambda: {"zh": 0.50, "ja": 0.50, "en": 0.50, "mixed": 0.50}
    )


@dataclass(frozen=True, slots=True)
class EvaluationRanking:
    speaker: float = 0.60
    pronunciation: float = 0.25
    language_consistency: float = 0.10
    prosody: float = 0.05


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    run_dir: Path
    project_root: Path
    model_name: str
    trained_languages: tuple[str, ...]
    validated_languages: tuple[str, ...]
    reference: BundleReference
    speaker_references: tuple[Path, ...]
    suites: dict[str, Path]
    asr_model: str
    speaker_model: str
    cache_dir: Path
    pairing: EvaluationPairing
    device: str
    precision: str
    constraints: EvaluationConstraints = field(default_factory=EvaluationConstraints)
    ranking: EvaluationRanking = field(default_factory=EvaluationRanking)


def parse_evaluation_config(
    value,
    *,
    root: Path,
    run_dir: Path,
    model_name: str,
    objective,
    device: str,
    precision: str,
) -> EvaluationConfig | None:
    if value is None:
        return None
    section = _mapping(value, "evaluation")
    _strict(
        section,
        "evaluation",
        {"enabled", "reference", "speaker_references", "suites", "models", "pairing", "constraints", "ranking"},
    )
    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("evaluation.enabled must be boolean")
    if not enabled:
        return None
    if not _SAFE_NAME.fullmatch(model_name):
        raise ValueError("experiment.name must be a safe name when evaluation is enabled")

    objective = _mapping(objective, "objective")
    trained = _languages(objective.get("training_languages"), "objective.training_languages")
    validated = _languages(objective.get("target_languages"), "objective.target_languages")

    reference_data = _mapping(section.get("reference"), "evaluation.reference")
    _strict(reference_data, "evaluation.reference", {"audio", "text", "language"})
    language = reference_data.get("language")
    if language not in {"zh", "ja", "en"}:
        raise ValueError("evaluation.reference.language must be zh, ja, or en")
    audio = _path(root, reference_data.get("audio"), "evaluation.reference.audio")
    text = reference_data.get("text")
    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise ValueError("evaluation.reference.text must be a non-empty string")
    reference = BundleReference(audio, text.strip() if text is not None else None, language)

    raw_references = section.get("speaker_references", [reference.audio])
    if not isinstance(raw_references, list) or not raw_references:
        raise ValueError("evaluation.speaker_references must be a non-empty list")
    speaker_references = tuple(
        _path(root, item, f"evaluation.speaker_references[{index}]")
        for index, item in enumerate(raw_references)
    )

    suites_data = _mapping(section.get("suites"), "evaluation.suites")
    _strict(suites_data, "evaluation.suites", set(LANGUAGES))
    missing = set(LANGUAGES) - suites_data.keys()
    if missing:
        raise ValueError(f"evaluation.suites is missing: {', '.join(sorted(missing))}")
    suites = {language: _path(root, suites_data[language], f"evaluation.suites.{language}") for language in LANGUAGES}

    models = section.get("models", {})
    models = _mapping(models, "evaluation.models")
    _strict(models, "evaluation.models", {"asr", "speaker", "cache_dir"})
    asr_model = _model_id(models.get("asr", _DEFAULT_ASR), "evaluation.models.asr")
    speaker_model = _model_id(models.get("speaker", _DEFAULT_SPEAKER), "evaluation.models.speaker")
    cache_dir = _path(root, models.get("cache_dir", "models/evaluators"), "evaluation.models.cache_dir")

    pairing_data = section.get("pairing", {})
    pairing_data = _mapping(pairing_data, "evaluation.pairing")
    _strict(pairing_data, "evaluation.pairing", {"s2_keep", "shortlist_size"})
    pairing = EvaluationPairing(
        _positive_integer(pairing_data.get("s2_keep", 2), "evaluation.pairing.s2_keep"),
        _positive_integer(pairing_data.get("shortlist_size", 3), "evaluation.pairing.shortlist_size"),
    )
    if pairing.shortlist_size > 26:
        raise ValueError("evaluation.pairing.shortlist_size must be at most 26")
    constraints = _constraints(section.get("constraints", {}))
    ranking = _ranking(section.get("ranking", {}))
    return EvaluationConfig(
        run_dir=run_dir,
        project_root=root,
        model_name=model_name,
        trained_languages=trained,
        validated_languages=validated,
        reference=reference,
        speaker_references=speaker_references,
        suites=suites,
        asr_model=asr_model,
        speaker_model=speaker_model,
        cache_dir=cache_dir,
        pairing=pairing,
        device=device,
        precision=precision,
        constraints=constraints,
        ranking=ranking,
    )


def _mapping(value, field: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} keys must be strings")
    return value


def _strict(value: dict, field: str, allowed: set[str]) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {field} field: {', '.join(sorted(unknown))}")


def _languages(value, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(language not in {"zh", "ja", "en"} for language in value):
        raise ValueError(f"{field} must contain only zh, ja, or en")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must not contain duplicates")
    return tuple(value)


def _path(root: Path, value, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _model_id(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(character.isspace() for character in value):
        raise ValueError(f"{field} must be a non-empty model ID or path")
    return value


def _positive_integer(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _constraints(value) -> EvaluationConstraints:
    section = _mapping(value, "evaluation.constraints")
    _strict(
        section,
        "evaluation.constraints",
        {"min_speaker_similarity", "max_cer", "max_wer", "min_language_consistency"},
    )
    defaults = EvaluationConstraints()
    return EvaluationConstraints(
        _thresholds(
            section.get("min_speaker_similarity", defaults.min_speaker_similarity),
            "evaluation.constraints.min_speaker_similarity",
            {"zh", "ja", "en"},
            minimum=-1,
            maximum=1,
        ),
        _thresholds(
            section.get("max_cer", defaults.max_cer),
            "evaluation.constraints.max_cer",
            {"zh", "ja"},
            minimum=0,
        ),
        _thresholds(
            section.get("max_wer", defaults.max_wer),
            "evaluation.constraints.max_wer",
            {"en"},
            minimum=0,
        ),
        _thresholds(
            section.get("min_language_consistency", defaults.min_language_consistency),
            "evaluation.constraints.min_language_consistency",
            {"zh", "ja", "en", "mixed"},
            minimum=0,
            maximum=1,
        ),
    )


def _ranking(value) -> EvaluationRanking:
    section = _mapping(value, "evaluation.ranking")
    names = {"speaker", "pronunciation", "language_consistency", "prosody"}
    _strict(section, "evaluation.ranking", names)
    defaults = EvaluationRanking()
    weights = {
        name: _finite_number(section.get(name, getattr(defaults, name)), f"evaluation.ranking.{name}")
        for name in names
    }
    if any(weight < 0 for weight in weights.values()) or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("evaluation.ranking weights must be nonnegative and sum to 1")
    return EvaluationRanking(**weights)


def _thresholds(value, field: str, allowed: set[str], *, minimum: float, maximum: float | None = None) -> dict[str, float]:
    section = _mapping(value, field)
    _strict(section, field, allowed)
    missing = allowed - section.keys()
    if missing:
        raise ValueError(f"{field} is missing: {', '.join(sorted(missing))}")
    result = {language: _finite_number(raw, f"{field}.{language}") for language, raw in section.items()}
    if any(number < minimum or maximum is not None and number > maximum for number in result.values()):
        bounds = f"[{minimum}, {maximum}]" if maximum is not None else f"at least {minimum}"
        raise ValueError(f"{field} values must be {bounds}")
    return result


def _finite_number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


__all__ = [
    "EvaluationConfig",
    "EvaluationConstraints",
    "EvaluationPairing",
    "EvaluationRanking",
    "LANGUAGES",
    "parse_evaluation_config",
]

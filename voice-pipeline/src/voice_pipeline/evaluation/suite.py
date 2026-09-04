from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import EvaluationConfig, LANGUAGES


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    language: str
    text: str


@dataclass(frozen=True, slots=True)
class EvaluationSuite:
    cases: tuple[EvaluationCase, ...]


def load_evaluation_suite(config: EvaluationConfig) -> EvaluationSuite:
    cases: list[EvaluationCase] = []
    for language in LANGUAGES:
        texts = _parse_suite_file(config.suites[language], language)
        cases.extend(
            EvaluationCase(f"{language}-{index:03d}", language, text)
            for index, text in enumerate(texts, 1)
        )
    return EvaluationSuite(tuple(cases))


def _parse_suite_file(path: Path, language: str) -> tuple[str, ...]:
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"invalid {language} evaluation suite {path}: {error}") from error
    texts = tuple(line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#"))
    if not texts:
        raise ValueError(f"{language} evaluation suite must contain at least one sentence")
    if len(set(texts)) != len(texts):
        raise ValueError(f"duplicate sentence in {language} evaluation suite")
    return texts


__all__ = ["EvaluationCase", "EvaluationSuite", "load_evaluation_suite"]

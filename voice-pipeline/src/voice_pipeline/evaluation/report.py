from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import uuid

from .ranking import RankedCandidate


def write_results(path: Path, ranked: tuple[RankedCandidate, ...]) -> Path:
    payload = {"schema_version": 1, "candidates": [asdict(candidate) for candidate in ranked]}
    _write_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return Path(path)


def write_report(path: Path, ranked: tuple[RankedCandidate, ...]) -> Path:
    lines = ["# Automatic Evaluation Report", ""]
    for index, candidate in enumerate(ranked, 1):
        score = "n/a" if candidate.score is None else f"{candidate.score:.6f}"
        lines.extend(
            [
                f"## {index}. {candidate.pair_key}",
                "",
                f"- Eligible: {'yes' if candidate.eligible else 'no'}",
                f"- Score: {score}",
            ]
        )
        for language, metrics in candidate.evaluation.by_language.items():
            rendered = ", ".join(f"{name}={value:.6f}" for name, value in sorted(metrics.items()))
            lines.append(f"- {language}: {rendered}")
        for failure in candidate.failures:
            lines.append(
                f"- Failure: {failure.metric} actual={failure.actual} threshold={failure.threshold} ({failure.reason})"
            )
        lines.append("")
    _write_atomic(path, "\n".join(lines))
    return Path(path)


def _write_atomic(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["write_report", "write_results"]

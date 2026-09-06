from __future__ import annotations

import math
import unicodedata

from .base import MetricResult


_SCRIPT_ORDER = ("han", "kana", "latin")


def language_consistency(
    target: str,
    transcript: str,
    language: str,
    *,
    detected_language: str | None,
    probability: float | None,
) -> MetricResult:
    if language == "mixed":
        required = _scripts(target)
        present = _scripts(transcript)
        if not required:
            raise ValueError("mixed target text contains no supported script")
        retained = required & present
        return MetricResult(
            "language_consistency",
            len(retained) / len(required),
            {
                "required_scripts": _ordered(required),
                "present_scripts": _ordered(present),
                "detected_language": detected_language,
            },
            True,
        )
    if language not in {"zh", "ja", "en"}:
        raise ValueError("language must be zh, ja, en, or mixed")
    if detected_language is None or probability is None:
        return MetricResult("language_consistency", None, {}, False, "language detection unavailable")
    if not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("language probability must be between 0 and 1")
    value = float(probability) if detected_language == language else 0.0
    return MetricResult("language_consistency", value, {"detected_language": detected_language}, True)


def _scripts(text: str) -> set[str]:
    scripts: set[str] = set()
    for character in unicodedata.normalize("NFKC", text):
        codepoint = ord(character)
        if 0x4E00 <= codepoint <= 0x9FFF or 0x3400 <= codepoint <= 0x4DBF:
            scripts.add("han")
        elif 0x3040 <= codepoint <= 0x30FF:
            scripts.add("kana")
        elif "LATIN" in unicodedata.name(character, ""):
            scripts.add("latin")
    return scripts


def _ordered(scripts: set[str]) -> list[str]:
    return [script for script in _SCRIPT_ORDER if script in scripts]


__all__ = ["language_consistency"]

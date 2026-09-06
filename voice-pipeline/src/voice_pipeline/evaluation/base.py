from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: str | None
    language_probability: float | None


@dataclass(frozen=True, slots=True)
class MetricResult:
    name: str
    value: float | None
    details: dict[str, Any]
    available: bool
    error: str | None = None


__all__ = ["MetricResult", "Transcript"]

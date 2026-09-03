from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import yaml


_LANGUAGES = {"zh", "ja", "en"}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class BundleReference:
    audio: Path
    text: str | None
    language: str


@dataclass(frozen=True, slots=True)
class BundleLanguages:
    trained: tuple[str, ...]
    validated: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Candidate:
    id: str
    s1: Path
    s2: Path


@dataclass(frozen=True, slots=True)
class Shortlist:
    run_dir: Path
    project_root: Path
    profile: str
    model_name: str
    reference: BundleReference
    languages: BundleLanguages
    candidates: tuple[Candidate, ...]

    @classmethod
    def load(cls, run_dir: Path, project_root: Path) -> "Shortlist":
        run_dir = Path(run_dir).resolve()
        project_root = Path(project_root).resolve()
        payload = _load_yaml(run_dir / "evaluation" / "shortlist.yaml", "shortlist")
        _strict(payload, "shortlist", {"schema_version", "profile", "model_name", "reference", "languages", "candidates"})
        if payload.get("schema_version") != 1:
            raise ValueError("shortlist.schema_version must be 1")
        if payload.get("profile") != "v2ProPlus":
            raise ValueError("shortlist.profile must be v2ProPlus")
        model_name = _safe_name(payload.get("model_name"), "model_name")
        reference = _parse_reference(payload.get("reference"), project_root, bundle=False)
        languages = _parse_languages(payload.get("languages"))
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError("shortlist.candidates must contain at least one candidate")
        candidates: list[Candidate] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_candidates):
            raw = _mapping(raw, f"candidates[{index}]")
            _strict(raw, f"candidates[{index}]", {"id", "s1", "s2"})
            candidate_id = _safe_name(raw.get("id"), "candidate id")
            if candidate_id in seen:
                raise ValueError(f"duplicate candidate id: {candidate_id}")
            seen.add(candidate_id)
            candidates.append(
                Candidate(
                    candidate_id,
                    _safe_existing_path(project_root, raw.get("s1"), f"{candidate_id}.s1", "project"),
                    _safe_existing_path(project_root, raw.get("s2"), f"{candidate_id}.s2", "project"),
                )
            )
        return cls(run_dir, project_root, "v2ProPlus", model_name, reference, languages, tuple(candidates))


@dataclass(slots=True)
class ModelBundle:
    root: Path
    profile: str
    weights: dict[str, Path]
    reference: BundleReference
    languages: BundleLanguages
    metadata: dict[str, Any]

    @classmethod
    def load(cls, root: Path) -> "ModelBundle":
        root = Path(root).resolve()
        payload = _load_yaml(root / "model.yaml", "model bundle")
        _strict(payload, "model bundle", {"schema_version", "profile", "weights", "reference", "languages"})
        if payload.get("schema_version") != 1:
            raise ValueError("model bundle schema_version must be 1")
        weights = _mapping(payload.get("weights"), "weights")
        _strict(weights, "weights", {"s1", "s2"})
        metadata_path = root / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid model bundle metadata: {error}") from error
        if not isinstance(metadata, dict):
            raise ValueError("model bundle metadata must be a mapping")
        bundle = cls(
            root,
            payload.get("profile"),
            {"s1": Path(weights.get("s1", "")), "s2": Path(weights.get("s2", ""))},
            _parse_reference(payload.get("reference"), root, bundle=True),
            _parse_languages(payload.get("languages")),
            metadata,
        )
        bundle.validate()
        return bundle

    def validate(self) -> None:
        if self.profile != "v2ProPlus":
            raise ValueError("model bundle profile must be v2ProPlus")
        if set(self.weights) != {"s1", "s2"}:
            raise ValueError("model bundle weights must contain exactly s1 and s2")
        for name, path in self.weights.items():
            _safe_existing_path(self.root, path, f"weights.{name}", "bundle")
        _safe_existing_path(self.root, self.reference.audio, "reference.audio", "bundle")
        _safe_existing_path(self.root, Path("reference/default.json"), "reference metadata", "bundle")
        _validate_reference_values(self.reference)
        _validate_languages(self.languages)
        if not isinstance(self.metadata, dict):
            raise ValueError("model bundle metadata must be a mapping")

    def write(self, root: Path | None = None) -> None:
        if root is not None:
            self.root = Path(root).resolve()
        else:
            self.root = Path(self.root).resolve()
        self.validate()
        reference: dict[str, Any] = {
            "audio": self.reference.audio.as_posix(),
            "language": self.reference.language,
        }
        if self.reference.text is not None:
            reference["text"] = self.reference.text
        payload = {
            "schema_version": 1,
            "profile": self.profile,
            "weights": {name: self.weights[name].as_posix() for name in ("s1", "s2")},
            "reference": reference,
            "languages": {
                "trained": list(self.languages.trained),
                "validated": list(self.languages.validated),
            },
        }
        (self.root / "model.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        (self.root / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _load_yaml(path: Path, name: str) -> dict:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid {name} YAML {path}: {error}") from error
    return _mapping(payload, name)


def _mapping(value: Any, field: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} keys must be strings")
    return value


def _strict(mapping: dict, field: str, allowed: set[str]) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"unknown {field} field: {', '.join(sorted(unknown))}")


def _safe_name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"{field} must be a safe name")
    return value


def _safe_existing_path(root: Path, value: Any, field: str, scope: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{field} must be a safe {scope}-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must be a safe {scope}-relative path")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{field} must be a safe {scope}-relative path")
    if not resolved.is_file():
        raise ValueError(f"{field} does not exist: {relative.as_posix()}")
    return relative if scope == "bundle" else resolved


def _parse_reference(value: Any, root: Path, *, bundle: bool) -> BundleReference:
    value = _mapping(value, "reference")
    _strict(value, "reference", {"audio", "text", "language"})
    audio = _safe_existing_path(root, value.get("audio"), "reference.audio", "bundle" if bundle else "project")
    reference = BundleReference(audio, value.get("text"), value.get("language"))
    _validate_reference_values(reference)
    return reference


def _validate_reference_values(reference: BundleReference) -> None:
    if reference.text is not None and (not isinstance(reference.text, str) or not reference.text.strip()):
        raise ValueError("reference.text must be a non-empty string when present")
    if reference.language not in _LANGUAGES:
        raise ValueError("reference.language must be zh, ja, or en")


def _parse_languages(value: Any) -> BundleLanguages:
    value = _mapping(value, "languages")
    _strict(value, "languages", {"trained", "validated"})
    languages = BundleLanguages(_language_list(value.get("trained"), "trained"), _language_list(value.get("validated"), "validated"))
    _validate_languages(languages)
    return languages


def _language_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"languages.{field} must be a non-empty list")
    if any(not isinstance(language, str) or language not in _LANGUAGES for language in value):
        raise ValueError(f"languages.{field} must contain only zh, ja, or en")
    if len(set(value)) != len(value):
        raise ValueError(f"languages.{field} must not contain duplicates")
    return tuple(value)


def _validate_languages(languages: BundleLanguages) -> None:
    for field, values in (("trained", languages.trained), ("validated", languages.validated)):
        if not values or len(set(values)) != len(values) or any(value not in _LANGUAGES for value in values):
            raise ValueError(f"languages.{field} must contain unique zh, ja, or en values")


__all__ = ["BundleLanguages", "BundleReference", "Candidate", "ModelBundle", "Shortlist"]

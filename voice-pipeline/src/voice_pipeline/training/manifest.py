import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from voice_pipeline.common.errors import ManifestError


ALLOWED_LANGUAGES = frozenset({"zh", "ja", "en", "mixed"})


@dataclass(frozen=True, slots=True)
class ManifestItem:
    audio_path: Path
    speaker: str
    language: str
    text: str


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    line_no: int
    sample_id: str
    item: ManifestItem


@dataclass(frozen=True, slots=True)
class ManifestIssue:
    line_no: int
    raw: str
    category: str
    message: str


@dataclass(frozen=True, slots=True)
class ManifestReadResult:
    records: list[ManifestRecord]
    issues: list[ManifestIssue]
    total_records: int


def stable_sample_id(audio_path: str, speaker: str, language: str, text: str) -> str:
    canonical = "\0".join((Path(audio_path).as_posix(), speaker, language, text))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def allowed_bad_records(total_records: int) -> int:
    if total_records < 1:
        return 0
    return min(5, math.ceil(total_records * 0.20))


def read_manifest_records(path: Path) -> ManifestReadResult:
    if not path.is_file():
        raise ManifestError(f"manifest does not exist: {path}")

    records: list[ManifestRecord] = []
    issues: list[ManifestIssue] = []
    seen: set[str] = set()
    total_records = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not raw.strip():
            continue
        total_records += 1
        parts = raw.split("|")
        if len(parts) != 4:
            issues.append(ManifestIssue(line_no, raw, "malformed", "expected 4 fields separated by '|'"))
            continue

        audio, speaker, language, text = (part.strip() for part in parts)
        if not speaker:
            issues.append(ManifestIssue(line_no, raw, "empty_speaker", "speaker is empty"))
            continue
        if language not in ALLOWED_LANGUAGES:
            issues.append(
                ManifestIssue(line_no, raw, "unsupported_language", f"unsupported language: {language or '<empty>'}")
            )
            continue
        if not text:
            issues.append(ManifestIssue(line_no, raw, "empty_text", "text is empty"))
            continue

        audio_path = Path(audio)
        if not audio_path.is_absolute():
            audio_path = (path.parent / audio_path).resolve()
        if not audio_path.is_file():
            issues.append(ManifestIssue(line_no, raw, "missing_audio", f"audio file does not exist: {audio_path}"))
            continue

        sample_id = stable_sample_id(str(audio_path), speaker, language, text)
        if sample_id in seen:
            issues.append(ManifestIssue(line_no, raw, "duplicate", f"duplicate manifest record: {sample_id}"))
            continue
        seen.add(sample_id)
        records.append(ManifestRecord(line_no, sample_id, ManifestItem(audio_path, speaker, language, text)))

    return ManifestReadResult(records, issues, total_records)


def load_manifest(path: Path) -> list[ManifestItem]:
    if not path.is_file():
        raise ManifestError(f"manifest does not exist: {path}")
    items: list[ManifestItem] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split("|")
        if len(parts) != 4:
            raise ManifestError(f"line {line_no}: expected 4 fields separated by '|'")
        audio, speaker, language, text = (p.strip() for p in parts)
        audio_path = Path(audio)
        if not audio_path.is_file():
            raise ManifestError(f"line {line_no}: audio file does not exist: {audio_path}")
        if not speaker:
            raise ManifestError(f"line {line_no}: speaker is empty")
        if not language:
            raise ManifestError(f"line {line_no}: language is empty")
        if not text:
            raise ManifestError(f"line {line_no}: text is empty")
        items.append(ManifestItem(audio_path, speaker, language, text))
    if not items:
        raise ManifestError("manifest contains no usable entries")
    return items

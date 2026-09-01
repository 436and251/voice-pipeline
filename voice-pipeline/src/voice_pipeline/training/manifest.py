from dataclasses import dataclass
from pathlib import Path
from voice_pipeline.common.errors import ManifestError


@dataclass(frozen=True, slots=True)
class ManifestItem:
    audio_path: Path
    speaker: str
    language: str
    text: str


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

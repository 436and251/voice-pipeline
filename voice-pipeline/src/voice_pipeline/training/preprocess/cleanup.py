from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CleanupReport:
    removed_temporary: int
    removed_quarantined: int
    retained_valid: int


def cleanup_after_training(preprocess_dir: Path, training_succeeded: bool) -> CleanupReport:
    if not training_succeeded:
        return CleanupReport(0, 0, 0)

    root = preprocess_dir.resolve()
    valid_ids = _sample_ids(preprocess_dir / "valid_samples.jsonl", allow_null=False)
    quarantined_ids = _sample_ids(preprocess_dir / "quarantine.jsonl", allow_null=True) - valid_ids

    temporary = [path for path in preprocess_dir.rglob("*.tmp") if path.is_file()]
    quarantined: list[Path] = []
    suffixes = {
        "text": (".json", ".bert.pt"),
        "wav32k": (".wav",),
        "hubert": (".pt",),
        "sv": (".pt",),
        "semantic": (".pt",),
    }
    for sample_id in quarantined_ids:
        if Path(sample_id).name != sample_id or "/" in sample_id or "\\" in sample_id:
            raise ValueError(f"cleanup candidate is outside preprocess root: {sample_id}")
        for stage, stage_suffixes in suffixes.items():
            quarantined.extend(preprocess_dir / stage / f"{sample_id}{suffix}" for suffix in stage_suffixes)

    candidates = temporary + quarantined
    for candidate in candidates:
        if not candidate.resolve().is_relative_to(root):
            raise ValueError(f"cleanup candidate is outside preprocess root: {candidate}")

    removed_temporary = 0
    for path in temporary:
        if path.exists():
            path.unlink()
            removed_temporary += 1
    removed_quarantined = 0
    for path in quarantined:
        if path.is_file():
            path.unlink()
            removed_quarantined += 1
    return CleanupReport(removed_temporary, removed_quarantined, len(valid_ids))


def _sample_ids(path: Path, *, allow_null: bool) -> set[str]:
    if not path.is_file():
        raise ValueError(f"cleanup report does not exist: {path}")
    sample_ids: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
            sample_id = row["sample_id"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"invalid cleanup report {path} line {line_no}: {error}") from error
        if sample_id is None and allow_null:
            continue
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"invalid sample_id in {path} line {line_no}")
        sample_ids.add(sample_id)
    return sample_ids

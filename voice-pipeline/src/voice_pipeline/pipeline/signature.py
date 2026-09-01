import hashlib
import json
from pathlib import Path


def _file_digest(path: Path) -> dict[str, object]:
    stat = path.stat()
    payload: dict[str, object] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size <= 4 * 1024 * 1024:
        payload["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload


def compute_stage_signature(
    stage: str,
    config: dict[str, object],
    inputs: list[Path],
    profile: str,
    implementation_version: str,
) -> str:
    payload = {
        "stage": stage,
        "config": config,
        "inputs": [_file_digest(path) for path in inputs],
        "profile": profile,
        "implementation_version": implementation_version,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

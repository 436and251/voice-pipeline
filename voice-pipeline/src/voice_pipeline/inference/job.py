from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid

import numpy as np

from .session import validate_synthesis_options
from .text_chunker import TextChunker
from .wav import read_wav, write_wav_atomic


_SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class JobResult:
    output_path: Path
    generated_chunks: int
    resumed_chunks: int


def resolve_output_path(
    output_root: str | Path,
    model_name: str,
    relative_output: str | Path,
) -> Path:
    if not isinstance(model_name, str) or not _SAFE_MODEL_NAME.fullmatch(model_name):
        raise ValueError("model_name must be a safe name")
    relative = Path(relative_output)
    unsafe_component = any(
        ":" in part
        or part != part.rstrip(" .")
        or part.rstrip(" .").casefold().endswith(".infer")
        for part in relative.parts
    )
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.suffix.lower() != ".wav"
        or unsafe_component
    ):
        raise ValueError("output must be a safe relative .wav path")
    namespace = (Path(output_root).resolve() / model_name).resolve()
    resolved = (namespace / relative).resolve()
    if not resolved.is_relative_to(namespace):
        raise ValueError("output must remain inside the model output directory")
    return resolved


def run_synthesis_job(
    session,
    text: str,
    language: str,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    pause_ms: int = 10,
    max_chars: int | None = None,
    seed: int = 0,
    top_k: int = 5,
    top_p: float = 1.0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.35,
    noise_scale: float = 0.5,
    speed: float = 1.0,
) -> JobResult:
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be boolean")
    if isinstance(pause_ms, bool) or not isinstance(pause_ms, int) or pause_ms < 0:
        raise ValueError("pause_ms must be a nonnegative integer")
    validate_synthesis_options(
        text, language, seed, top_k, top_p, temperature, repetition_penalty, noise_scale, speed
    )
    output_path = Path(output_path).resolve()
    if output_path.suffix.lower() != ".wav":
        raise ValueError("output_path must end in .wav")
    work_dir = output_path.with_suffix(".infer")
    manifest_path = work_dir / "manifest.json"
    chunks = TextChunker(language, max_chars).chunk(text)
    options = {
        "language": language,
        "pause_ms": pause_ms,
        "max_chars": max_chars,
        "seed": seed,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "noise_scale": noise_scale,
        "speed": speed,
    }
    signature_material = {
        "identity": asdict(session.identity),
        "text": text.strip(),
        "chunks": chunks,
        "options": options,
    }
    signature = _json_sha256(signature_material)

    if overwrite:
        if output_path.is_file():
            output_path.unlink()
        if work_dir.is_dir():
            shutil.rmtree(work_dir)
        elif work_dir.is_file():
            work_dir.unlink()

    if manifest_path.is_file():
        manifest = _load_manifest(manifest_path)
        if manifest.get("signature") != signature or manifest.get("request") != signature_material:
            raise ValueError("existing inference job does not match; use --overwrite")
    else:
        if output_path.exists() or work_dir.exists():
            raise ValueError("existing inference output has no matching manifest; use --overwrite")
        entries = [
            {
                "index": index,
                "text": chunk,
                "seed": (seed + index) % 2**63,
                "status": "pending",
                "output": f"chunks/{index + 1:06d}.wav",
                "sha256": None,
            }
            for index, chunk in enumerate(chunks)
        ]
        manifest = {
            "schema_version": 1,
            "signature": signature,
            "request": signature_material,
            "chunks": entries,
            "final": {"status": "pending", "sha256": None},
        }
        _write_json_atomic(manifest_path, manifest)

    entries = manifest.get("chunks")
    if not isinstance(entries, list) or len(entries) != len(chunks):
        raise ValueError("invalid inference manifest; use --overwrite")

    generated = 0
    resumed = 0
    for index, (chunk, entry) in enumerate(zip(chunks, entries, strict=True)):
        expected_relative = f"chunks/{index + 1:06d}.wav"
        if not isinstance(entry, dict) or any(
            (
                entry.get("index") != index,
                entry.get("text") != chunk,
                entry.get("seed") != (seed + index) % 2**63,
                entry.get("output") != expected_relative,
            )
        ):
            raise ValueError("invalid inference manifest; use --overwrite")
        chunk_path = work_dir / expected_relative
        if entry.get("status") == "completed" and _matches_hash(chunk_path, entry.get("sha256")):
            resumed += 1
            continue
        result = session.synthesize(
            chunk,
            language,
            seed=(seed + index) % 2**63,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            noise_scale=noise_scale,
            speed=speed,
        )
        if result.sample_rate != 32_000:
            raise RuntimeError("inference chunks must use a 32000 Hz sample rate")
        write_wav_atomic(chunk_path, result.waveform, result.sample_rate)
        entry.update(status="completed", sha256=_sha256(chunk_path))
        _write_json_atomic(manifest_path, manifest)
        generated += 1

    final = manifest.get("final")
    if generated == 0 and isinstance(final, dict) and final.get("status") == "completed":
        if _matches_hash(output_path, final.get("sha256")):
            return JobResult(output_path, 0, resumed)

    waveforms: list[np.ndarray] = []
    for entry in entries:
        sample_rate, waveform = read_wav(work_dir / entry["output"])
        if sample_rate != 32_000:
            raise ValueError("chunk WAV must use a 32000 Hz sample rate")
        waveforms.append(waveform)
    pause = np.zeros(round(32_000 * pause_ms / 1000), dtype=np.float32)
    assembled: list[np.ndarray] = []
    for index, waveform in enumerate(waveforms):
        if index and pause.size:
            assembled.append(pause)
        assembled.append(waveform)
    write_wav_atomic(output_path, np.concatenate(assembled), 32_000)
    manifest["final"] = {"status": "completed", "sha256": _sha256(output_path)}
    _write_json_atomic(manifest_path, manifest)
    return JobResult(output_path, generated, resumed)


def _load_manifest(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid inference manifest; use --overwrite") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("invalid inference manifest; use --overwrite")
    return payload


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _matches_hash(path: Path, expected) -> bool:
    return isinstance(expected, str) and path.is_file() and _sha256(path) == expected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["JobResult", "resolve_output_path", "run_synthesis_job"]

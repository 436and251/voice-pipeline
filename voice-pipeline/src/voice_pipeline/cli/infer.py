from __future__ import annotations

import math
from pathlib import Path
import re
from time import perf_counter

import typer
import yaml

from voice_pipeline.inference.job import resolve_output_path, run_synthesis_job
from voice_pipeline.inference.long_text import synthesize_text
from voice_pipeline.inference.session import InferenceSession
from voice_pipeline.inference.text_source import resolve_text_source


app = typer.Typer(help="Run standalone GPT-SoVITS inference.")
_LANGUAGES = {"zh", "ja", "en", "mixed"}
_JOB_KEYS = {
    "name", "text", "text_file", "language", "pause_ms", "max_chars", "seed",
    "top_k", "top_p", "temperature", "repetition_penalty", "noise_scale", "speed",
}
_OPTION_KEYS = _JOB_KEYS - {"name", "text", "text_file"}
_BUILTIN_OPTIONS = {
    "pause_ms": 10,
    "max_chars": None,
    "seed": 0,
    "top_k": 5,
    "top_p": 1.0,
    "temperature": 1.0,
    "repetition_penalty": 1.35,
    "noise_scale": 0.5,
    "speed": 1.0,
}
_SAFE_JOB_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@app.command()
def synthesize(
    model: Path = typer.Option(..., "--model", exists=True, file_okay=False),
    text: str | None = typer.Option(None, "--text"),
    text_file: Path | None = typer.Option(None, "--text-file"),
    language: str = typer.Option(..., "--lang"),
    output: Path = typer.Option(..., "--output"),
    device: str = typer.Option("cuda:0", "--device"),
    output_root: Path = typer.Option(Path("outputs"), "--output-root", file_okay=False),
    reference: Path | None = typer.Option(None, "--reference"),
    reference_text: str | None = typer.Option(None, "--reference-text"),
    reference_language: str | None = typer.Option(None, "--reference-lang"),
    pause_ms: int = typer.Option(10, "--pause-ms"),
    max_chars: int | None = typer.Option(None, "--max-chars"),
    seed: int = typer.Option(0, "--seed"),
    top_k: int = typer.Option(5, "--top-k"),
    top_p: float = typer.Option(1.0, "--top-p"),
    temperature: float = typer.Option(1.0, "--temperature"),
    repetition_penalty: float = typer.Option(1.35, "--repetition-penalty"),
    noise_scale: float = typer.Option(0.5, "--noise-scale"),
    speed: float = typer.Option(1.0, "--speed"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Synthesize inline or UTF-8 TXT input to a resumable WAV job."""
    try:
        source = resolve_text_source(text, _resolve_optional_path(text_file))
        _validate_options(language, pause_ms, max_chars, seed, top_k, top_p, temperature, repetition_penalty, noise_scale, speed)
        reference_options = _reference_options(reference, reference_text, reference_language)
        session = InferenceSession.load(model.resolve(), device, **reference_options)
        destination = resolve_output_path(output_root.resolve(), session.identity.model_name, output)
        result = run_synthesis_job(
            session,
            source,
            language,
            destination,
            overwrite=overwrite,
            pause_ms=pause_ms,
            max_chars=max_chars,
            seed=seed,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            noise_scale=noise_scale,
            speed=speed,
        )
        typer.echo(
            f"wrote {result.output_path} "
            f"(generated {result.generated_chunks}, resumed {result.resumed_chunks} chunks)"
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        _fail(error)


@app.command()
def benchmark(
    model: Path = typer.Option(..., "--model", exists=True, file_okay=False),
    text: str | None = typer.Option(None, "--text"),
    text_file: Path | None = typer.Option(None, "--text-file"),
    language: str = typer.Option(..., "--lang"),
    device: str = typer.Option("cuda:0", "--device"),
    reference: Path | None = typer.Option(None, "--reference"),
    reference_text: str | None = typer.Option(None, "--reference-text"),
    reference_language: str | None = typer.Option(None, "--reference-lang"),
    pause_ms: int = typer.Option(10, "--pause-ms"),
    max_chars: int | None = typer.Option(None, "--max-chars"),
    seed: int = typer.Option(0, "--seed"),
    top_k: int = typer.Option(5, "--top-k"),
    top_p: float = typer.Option(1.0, "--top-p"),
    temperature: float = typer.Option(1.0, "--temperature"),
    repetition_penalty: float = typer.Option(1.35, "--repetition-penalty"),
    noise_scale: float = typer.Option(0.5, "--noise-scale"),
    speed: float = typer.Option(1.0, "--speed"),
    warmup: int = typer.Option(1, "--warmup"),
    runs: int = typer.Option(3, "--runs"),
) -> None:
    """Measure complete in-memory synthesis without writing artifacts."""
    try:
        if isinstance(warmup, bool) or warmup < 0:
            raise ValueError("warmup must be a nonnegative integer")
        if isinstance(runs, bool) or runs <= 0:
            raise ValueError("runs must be a positive integer")
        source = resolve_text_source(text, _resolve_optional_path(text_file))
        _validate_options(language, pause_ms, max_chars, seed, top_k, top_p, temperature, repetition_penalty, noise_scale, speed)
        reference_options = _reference_options(reference, reference_text, reference_language)
        session = InferenceSession.load(model.resolve(), device, **reference_options)
        options = {
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
        result = None
        for _ in range(warmup):
            result = synthesize_text(session, source, language, **options)
        elapsed = []
        for _ in range(runs):
            started = perf_counter()
            result = synthesize_text(session, source, language, **options)
            elapsed.append(perf_counter() - started)
        assert result is not None
        audio_seconds = result.waveform.size / result.sample_rate
        if audio_seconds <= 0:
            raise RuntimeError("benchmark produced empty audio")
        average = sum(elapsed) / len(elapsed)
        typer.echo(f"audio_seconds: {audio_seconds:.3f}")
        typer.echo(f"average_seconds: {average:.3f}")
        typer.echo(f"fastest_seconds: {min(elapsed):.3f}")
        typer.echo(f"rtf: {average / audio_seconds:.3f}")
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        _fail(error)


@app.command("batch")
def batch_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Run a strict YAML batch with one cached model session."""
    try:
        payload = _load_batch(config)
        session = InferenceSession.load(payload["model"], payload["device"], **payload["reference"])
        for index, job in enumerate(payload["jobs"], start=1):
            options = dict(job["options"])
            language = options.pop("language")
            destination = resolve_output_path(
                payload["output_root"], session.identity.model_name, Path(f"{job['name']}.wav")
            )
            result = run_synthesis_job(
                session,
                job["text"],
                language,
                destination,
                overwrite=overwrite,
                **options,
            )
            typer.echo(
                f"[{index}/{len(payload['jobs'])}] {result.output_path} "
                f"(generated {result.generated_chunks}, resumed {result.resumed_chunks})"
            )
        typer.echo(f"completed {len(payload['jobs'])} jobs")
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError, yaml.YAMLError) as error:
        _fail(error)


def _load_batch(path: Path) -> dict:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError as error:
        raise ValueError(f"batch config must be valid UTF-8: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("batch config must be a mapping")
    _reject_unknown("batch", payload, {"model", "device", "output_root", "reference", "defaults", "jobs"})
    if not isinstance(payload.get("model"), str) or not payload["model"].strip():
        raise ValueError("batch model must be a non-empty path")
    device = payload.get("device", "cuda:0")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("batch device must be a string")
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("batch jobs must be a non-empty list")
    project_root = Path.cwd().resolve()
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("batch defaults must be a mapping")
    _reject_unknown("batch defaults", defaults, _OPTION_KEYS)
    merged_defaults = _BUILTIN_OPTIONS | defaults
    reference = _parse_batch_reference(payload.get("reference"), project_root)
    jobs = []
    names: set[str] = set()
    for index, raw_job in enumerate(raw_jobs):
        if not isinstance(raw_job, dict):
            raise ValueError(f"batch jobs[{index}] must be a mapping")
        _reject_unknown(f"batch jobs[{index}]", raw_job, _JOB_KEYS)
        name = _safe_job_name(raw_job.get("name"))
        folded = name.casefold()
        if folded in names:
            raise ValueError(f"duplicate batch job name: {name}")
        names.add(folded)
        source = resolve_text_source(
            raw_job.get("text"),
            _project_path(project_root, raw_job.get("text_file")) if raw_job.get("text_file") is not None else None,
        )
        options = merged_defaults | {key: raw_job[key] for key in _OPTION_KEYS if key in raw_job}
        language = options.get("language")
        _validate_options(
            language,
            options["pause_ms"],
            options["max_chars"],
            options["seed"],
            options["top_k"],
            options["top_p"],
            options["temperature"],
            options["repetition_penalty"],
            options["noise_scale"],
            options["speed"],
        )
        jobs.append({"name": name, "text": source, "options": options})
    return {
        "model": _project_path(project_root, payload["model"]),
        "device": device,
        "output_root": _project_path(project_root, payload.get("output_root", "outputs")),
        "reference": reference,
        "jobs": jobs,
    }


def _parse_batch_reference(value, project_root: Path) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("batch reference must be a mapping")
    _reject_unknown("batch reference", value, {"audio", "text", "language"})
    return _reference_options(
        _project_path(project_root, value.get("audio")) if value.get("audio") is not None else None,
        value.get("text"),
        value.get("language"),
    )


def _reference_options(audio, text, language) -> dict:
    if audio is None and text is None and language is None:
        return {}
    if audio is None:
        raise ValueError("reference text/language cannot be used without reference audio")
    if language is None:
        raise ValueError("reference audio requires reference language")
    if language not in {"zh", "ja", "en"}:
        raise ValueError("reference language must be zh, ja, or en")
    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise ValueError("reference text must be a non-empty string when present")
    resolved_audio = Path(audio).resolve()
    if not resolved_audio.is_file():
        raise ValueError(f"reference audio does not exist: {resolved_audio}")
    return {
        "reference_audio": resolved_audio,
        "reference_text": text,
        "reference_language": language,
    }


def _validate_options(language, pause_ms, max_chars, seed, top_k, top_p, temperature, repetition_penalty, noise_scale, speed):
    if language not in _LANGUAGES:
        raise ValueError("language must be zh, ja, en, or mixed")
    if isinstance(pause_ms, bool) or not isinstance(pause_ms, int) or pause_ms < 0:
        raise ValueError("pause_ms must be a nonnegative integer")
    if max_chars is not None and (isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0):
        raise ValueError("max_chars must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be a nonnegative 64-bit integer")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    _positive("top_p", top_p, upper=1.0)
    _positive("temperature", temperature)
    _positive("repetition_penalty", repetition_penalty)
    _positive("speed", speed)
    if isinstance(noise_scale, bool) or not isinstance(noise_scale, (int, float)) or not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("noise_scale must be a finite nonnegative number")


def _positive(name: str, value, *, upper: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    if upper is not None and value > upper:
        raise ValueError(f"{name} must be at most {upper}")


def _safe_job_name(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("batch job name must be a safe relative name")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or any(not _SAFE_JOB_PART.fullmatch(part) for part in path.parts):
        raise ValueError("batch job name must be a safe relative name")
    return path.as_posix()


def _project_path(root: Path, value) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("path must be a non-empty string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_optional_path(path: Path | None) -> Path | None:
    return path.resolve() if path is not None else None


def _reject_unknown(name: str, mapping: dict, allowed: set[str]) -> None:
    if any(not isinstance(key, str) for key in mapping):
        raise ValueError(f"{name} keys must be strings")
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"unknown {name} field: {', '.join(sorted(unknown))}")


def _fail(error: Exception) -> None:
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=1) from error


__all__ = ["app"]

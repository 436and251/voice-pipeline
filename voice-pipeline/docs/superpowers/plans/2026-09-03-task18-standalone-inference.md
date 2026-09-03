# Task 18 Standalone Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable long-text inference, resumable WAV jobs, and thin synthesize/batch/benchmark CLI commands around the Task 17 v2ProPlus session.

**Architecture:** The Python inference package owns model loading, reference overrides, text chunking, waveform assembly, WAV persistence, and resumable manifests. Typer only parses CLI/config inputs and calls that package, so a desktop application or later service can use the same in-memory API without importing CLI code.

**Tech Stack:** Python 3.12, PyTorch, NumPy, standard-library `wave`/`json`/`hashlib`, Typer, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-task18-inference-cli-design.md`

## Global Constraints

- Work directly on `main`; do not create a branch, worktree, Conda environment, or ZIP archive.
- Run Python with `D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe`.
- Run pytest with `--basetemp .pytest_cache\codex-temp` or another child of that directory.
- Public v2ProPlus assets resolve from the profile's fixed paths relative to `Path.cwd()`.
- Inference must run with only a valid ModelBundle and public assets; it must not inspect training runs.
- Outputs are always namespaced as `<output_root>/<model_name>/...`.
- Default inter-chunk pause is 10 ms and may be overridden with a nonnegative integer.
- Do not add HTTP, streaming, GUI, evaluation, candidate selection, profiler, VRAM, or concurrent batch features.
- Use tests with fake sessions; no weight copying, model training, or multi-step GPU work.
- Preserve manifest and chunk WAV files after success; remove only test and bytecode caches at task completion.

---

### Task 1: Reference overrides and a service-safe session

**Files:**
- Modify: `src/voice_pipeline/inference/result.py`
- Modify: `src/voice_pipeline/inference/session.py`
- Modify: `tests/test_inference_session.py`

**Interfaces:**
- Consumes: `ModelBundle.load(root)`, `build_reference_condition(...)`, and the existing two-positional-argument `InferenceSession.load(bundle_path, device)` call.
- Produces: `InferenceIdentity`; `InferenceSession.identity`; and optional keyword-only `reference_audio`, `reference_text`, `reference_language` arguments on `InferenceSession.load`.

- [ ] **Step 1: Write failing identity, override-validation, and serialization tests**

Add tests that monkeypatch model/frontend loaders, then assert:

```python
session = InferenceSession.load(
    bundle_root,
    "cpu",
    reference_audio=override_wav,
    reference_language="en",
)
assert captured_reference == (override_wav.resolve(), None, "en")
assert session.identity.model_name == "speaker_001"
assert session.identity.reference_sha256 == hashlib.sha256(b"override").hexdigest()

with pytest.raises(ValueError, match="reference_audio.*reference_language"):
    InferenceSession.load(bundle_root, "cpu", reference_audio=override_wav)
with pytest.raises(ValueError, match="without reference_audio"):
    InferenceSession.load(bundle_root, "cpu", reference_text="prompt")
```

Use two threads with a fake S1/S2 pair that records its active call count and
assert the maximum is one, demonstrating that a session serializes inference.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest voice-pipeline/tests/test_inference_session.py --basetemp .pytest_cache\codex-temp-task18-1 -q
```

Expected: failures because override keywords, `InferenceIdentity`, and the session lock do not exist.

- [ ] **Step 3: Implement the minimal identity, override, and lock behavior**

Add the immutable value type:

```python
@dataclass(frozen=True, slots=True)
class InferenceIdentity:
    model_name: str
    s1_sha256: str
    s2_sha256: str
    reference_sha256: str
    reference_text: str | None
    reference_language: str
```

Validate override combinations before loading heavyweight models. Resolve and
hash the selected audio file. Read model name and exported hashes from the
already-validated bundle metadata. Keep the old load call valid. Create
`self._lock = threading.Lock()` and wrap only the fixed-seed semantic/acoustic
section of `synthesize`:

```python
with self._lock, _fixed_seed(seed, self.device):
    semantic = generate_semantic(...)
    waveform = decode_waveform(...)
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command. Expected: all `test_inference_session.py` tests pass.

- [ ] **Step 5: Commit the service-safe session**

```powershell
git add -- voice-pipeline/src/voice_pipeline/inference/result.py voice-pipeline/src/voice_pipeline/inference/session.py voice-pipeline/tests/test_inference_session.py
git commit -m "feat: support standalone inference references"
```

---

### Task 2: In-memory long-text inference

**Files:**
- Create: `src/voice_pipeline/inference/long_text.py`
- Modify: `src/voice_pipeline/inference/__init__.py`
- Create: `tests/test_long_text_inference.py`

**Interfaces:**
- Consumes: `TextChunker(language, max_chars)`, `InferenceSession.synthesize(...)`, and `InferenceResult`.
- Produces: `synthesize_text(session, text, language, *, pause_ms=10, max_chars=None, seed=0, top_k=5, top_p=1.0, temperature=1.0, repetition_penalty=1.35, noise_scale=0.5, speed=1.0) -> InferenceResult`.

- [ ] **Step 1: Write failing chunk, pause, seed, and validation tests**

Use a fake session whose `synthesize` returns a two-sample float32 waveform and
records calls. Force three chunks with a small `max_chars`, then assert:

```python
result = synthesize_text(session, "aa。bb。cc。", "zh", max_chars=3, seed=8)
assert [call.seed for call in session.calls] == [8, 9, 10]
assert result.sample_rate == 32_000
assert len(result.waveform) == 6 + 2 * 320  # 10 ms at 32 kHz between chunks
```

Also assert `pause_ms=0` inserts no samples, `pause_ms=25` inserts 800 samples,
empty text is rejected by the shared chunker, and boolean/negative pause or
invalid `max_chars` is rejected before session access.

- [ ] **Step 2: Run the new test file and verify RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest voice-pipeline/tests/test_long_text_inference.py --basetemp .pytest_cache\codex-temp-task18-2 -q
```

Expected: collection fails because `voice_pipeline.inference.long_text` does not exist.

- [ ] **Step 3: Implement the minimal in-memory function**

Validate pause and max length, call `TextChunker.chunk`, derive each chunk seed
with `(seed + index) % 2**63`, and forward decoding parameters unchanged. Reject
inconsistent sample rates or non-one-dimensional waveforms from the session.
Use `np.concatenate` with a float32 zero array of
`round(sample_rate * pause_ms / 1000)` samples only between chunks.

- [ ] **Step 4: Run focused and Task 17 regression tests**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest voice-pipeline/tests/test_long_text_inference.py voice-pipeline/tests/test_inference_session.py --basetemp .pytest_cache\codex-temp-task18-2-green -q
```

Expected: both files pass.

- [ ] **Step 5: Commit the in-memory application API**

```powershell
git add -- voice-pipeline/src/voice_pipeline/inference/long_text.py voice-pipeline/src/voice_pipeline/inference/__init__.py voice-pipeline/tests/test_long_text_inference.py
git commit -m "feat: add in-memory long-text inference"
```

---

### Task 3: Atomic WAV files and resumable chunk jobs

**Files:**
- Create: `src/voice_pipeline/inference/wav.py`
- Create: `src/voice_pipeline/inference/job.py`
- Create: `tests/test_inference_job.py`

**Interfaces:**
- Consumes: `InferenceIdentity`, `TextChunker`, `InferenceSession.synthesize`, and Task 2 decoding defaults.
- Produces: `write_wav_atomic(path, waveform, sample_rate)`, `read_wav(path) -> tuple[int, np.ndarray]`, `resolve_output_path(output_root, model_name, relative_output) -> Path`, and `run_synthesis_job(session, text, language, output_path, *, overwrite=False, pause_ms=10, max_chars=None, seed=0, top_k=5, top_p=1.0, temperature=1.0, repetition_penalty=1.35, noise_scale=0.5, speed=1.0) -> JobResult`.

- [ ] **Step 1: Write failing output-path and WAV tests**

Assert a safe name resolves beneath the model namespace and unsafe paths fail:

```python
assert resolve_output_path(root, "speaker_001", Path("folder/article.wav")) == (
    root / "speaker_001/folder/article.wav"
).resolve()
for value in (Path("../escape.wav"), Path("C:/escape.wav"), Path("bad.mp3")):
    with pytest.raises(ValueError):
        resolve_output_path(root, "speaker_001", value)
```

Round-trip `[0.0, 1.0, -1.0]` through the WAV functions and assert mono, 32 kHz,
PCM16, finite float32 output with quantization tolerance of `1 / 32767`.

- [ ] **Step 2: Write failing manifest/resume tests**

Use a fake session with a fixed `InferenceIdentity`. Assert the first run writes
chunk WAVs, a schema-version-1 manifest, and the final WAV. Delete one completed
chunk and rerun; assert only that chunk is synthesized again. Corrupt another
chunk and assert its hash mismatch also causes only that chunk to regenerate.

Change each signature-bearing input in parameterized cases (text, language,
pause, seed, decode option, max chars, identity) and assert reuse fails with a
message containing `--overwrite`. With `overwrite=True`, assert a fresh job
replaces the conflicting work state. Assert a valid completed final file makes
the next call a no-op.

- [ ] **Step 3: Run the new tests and verify RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest voice-pipeline/tests/test_inference_job.py --basetemp .pytest_cache\codex-temp-task18-3 -q
```

Expected: collection fails because the WAV/job modules do not exist.

- [ ] **Step 4: Implement WAV conversion and safe output resolution**

Clip float samples to `[-1, 1]`, convert to little-endian signed 16-bit PCM, and
write a temporary sibling before `Path.replace`. Read only mono 16-bit PCM and
return float32. Resolve user paths and verify `resolved.is_relative_to((root /
model_name).resolve())`.

- [ ] **Step 5: Implement the strict resumable manifest**

Create deterministic JSON signature material from identity, resolved source
text, chunk list, language, max chars, pause, seed, and every decoding option;
hash canonical `json.dumps(..., sort_keys=True, separators=(",", ":"))` bytes.
Use zero-padded one-based chunk filenames. Save the manifest atomically after
each completed chunk. Verify completed chunk hashes before skipping them.
Concatenate verified PCM chunk arrays with the configured silence, atomically
write the final WAV, hash it, and mark the final status complete.

Return the immutable summary:

```python
@dataclass(frozen=True, slots=True)
class JobResult:
    output_path: Path
    generated_chunks: int
    resumed_chunks: int
```

For overwrite, resolve both the final file and work directory first, verify both
remain beneath `<output_root>/<model_name>`, then remove only those exact targets.

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run the Step 3 command. Expected: all job tests pass.

- [ ] **Step 7: Commit resumable WAV jobs**

```powershell
git add -- voice-pipeline/src/voice_pipeline/inference/wav.py voice-pipeline/src/voice_pipeline/inference/job.py voice-pipeline/tests/test_inference_job.py
git commit -m "feat: add resumable inference wav jobs"
```

---

### Task 4: Synthesize and batch CLI

**Files:**
- Create: `src/voice_pipeline/cli/infer.py`
- Modify: `src/voice_pipeline/cli/main.py`
- Create: `configs/infer.example.yaml`
- Create: `tests/test_infer_cli.py`

**Interfaces:**
- Consumes: `InferenceSession.load`, `resolve_text_source`, `resolve_output_path`, and `run_synthesis_job`.
- Produces: `voice-pipeline infer synthesize` and `voice-pipeline infer batch`.

- [ ] **Step 1: Write failing synthesize CLI contract tests**

Use `CliRunner` and monkeypatch session/job calls. Verify `--text` and
`--text-file` are mutually exclusive, explicit language is required, reference
audio requires reference language, unsafe output is rejected, a controlled
inference error exits 1, and a valid invocation forwards all default values:

```python
result = runner.invoke(app, [
    "infer", "synthesize", "--model", str(bundle), "--text", "hello",
    "--lang", "en", "--output", "hello.wav",
])
assert result.exit_code == 0
assert captured["pause_ms"] == 10
assert captured["output"] == (project / "outputs/speaker_001/hello.wav").resolve()
```

- [ ] **Step 2: Run the synthesize tests and verify RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest voice-pipeline/tests/test_infer_cli.py -k synthesize --basetemp .pytest_cache\codex-temp-task18-4a -q
```

Expected: failure because the `infer` Typer group is absent.

- [ ] **Step 3: Implement the thin synthesize command**

Define one Typer app in `cli/infer.py`, add it to `cli/main.py`, resolve text
through `resolve_text_source`, load one session with optional reference fields,
resolve the target-person path, and call `run_synthesis_job`. Catch only expected
filesystem/config/model/inference value errors, print `Error:`, and exit 1.

- [ ] **Step 4: Run synthesize tests and verify GREEN**

Run Step 2 again. Expected: synthesize tests pass.

- [ ] **Step 5: Write failing strict batch tests**

Write YAML fixtures proving unknown fields and duplicate names fail, job values
override defaults, relative model/text/reference/output-root paths resolve from
the project root (`Path.cwd()`), the session loads exactly once, jobs run in
order, and execution stops at the first failed job. Confirm a whole-batch
reference override is passed once and per-job reference fields are rejected.

- [ ] **Step 6: Run batch tests and verify RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest voice-pipeline/tests/test_infer_cli.py -k batch --basetemp .pytest_cache\codex-temp-task18-4b -q
```

Expected: failures because batch parsing and command behavior are absent.

- [ ] **Step 7: Implement strict YAML parsing and sequential batch execution**

Allow exactly top-level `model`, `device`, `output_root`, `reference`,
`defaults`, and `jobs`. Allow defaults/job keys exactly as defined in the spec.
Merge built-in defaults, YAML defaults, and job overrides in that order. Require
safe unique job names and exactly one text source. Load the session once, then
call the same output resolver and job runner used by synthesize.

Create `configs/infer.example.yaml` with two jobs and comments stating that one
reference condition applies to the entire batch.

- [ ] **Step 8: Run all CLI tests and verify GREEN**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest voice-pipeline/tests/test_infer_cli.py voice-pipeline/tests/test_cli.py --basetemp .pytest_cache\codex-temp-task18-4-green -q
```

Expected: all CLI tests pass.

- [ ] **Step 9: Commit synthesize and batch CLI**

```powershell
git add -- voice-pipeline/src/voice_pipeline/cli/infer.py voice-pipeline/src/voice_pipeline/cli/main.py voice-pipeline/configs/infer.example.yaml voice-pipeline/tests/test_infer_cli.py
git commit -m "feat: add inference synthesize and batch cli"
```

---

### Task 5: Benchmark, documentation, and final verification

**Files:**
- Modify: `src/voice_pipeline/cli/infer.py`
- Modify: `tests/test_infer_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `synthesize_text` and the same shared CLI parameter builders used by synthesize.
- Produces: `voice-pipeline infer benchmark` and documented standalone Python/CLI usage.

- [ ] **Step 1: Write failing benchmark tests**

Monkeypatch `InferenceSession.load`, `synthesize_text`, and `time.perf_counter`.
Assert defaults produce exactly four calls (one warm-up plus three measured),
model loading occurs before the first timed call, the same seed/options reach
each run, and stdout contains audio duration, average seconds, fastest seconds,
and RTF. Assert `--warmup -1` and `--runs 0` fail, and no WAV/manifest is created.

- [ ] **Step 2: Run benchmark tests and verify RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest voice-pipeline/tests/test_infer_cli.py -k benchmark --basetemp .pytest_cache\codex-temp-task18-5 -q
```

Expected: failure because the benchmark command is absent.

- [ ] **Step 3: Implement minimal benchmark reporting**

Validate warm-up/run counts before loading. Resolve text and reference exactly
as synthesize does. Load once, perform untimed warm-ups, time each complete
`synthesize_text` call, calculate duration from returned sample count/rate, and
print four labeled values. RTF is mean elapsed seconds divided by audio seconds.

- [ ] **Step 4: Update README architecture and examples**

Document:

```text
training/export → ModelBundle
                       ↓
CLI / desktop backend / future service → reusable inference package → audio
```

State that inference can run independently of training runs, show the
`outputs/<model_name>` layout, correct the reference override rules, add a
Python `InferenceSession` plus `synthesize_text` example, describe manifest
resume/overwrite behavior, and list synthesize/batch/benchmark examples. Keep
HTTP and streaming explicitly outside Task 18.

- [ ] **Step 5: Run focused CLI and inference tests**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest voice-pipeline/tests/test_infer_cli.py voice-pipeline/tests/test_inference_session.py voice-pipeline/tests/test_long_text_inference.py voice-pipeline/tests/test_inference_job.py --basetemp .pytest_cache\codex-temp-task18-focused -q
```

Expected: all focused tests pass.

- [ ] **Step 6: Request code review and resolve every actionable finding**

Review the complete Task 18 diff against the approved spec. For every valid
finding, first add or adjust a regression test, observe the expected failure,
then apply the smallest fix and rerun the focused suite.

- [ ] **Step 7: Run fresh final verification**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest voice-pipeline/tests -m 'not gpu' --basetemp .pytest_cache\codex-temp-task18-final -q
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m compileall -q voice-pipeline/src/voice_pipeline
git diff --check
```

Expected: zero test failures, compile exit code 0, and no whitespace errors.

- [ ] **Step 8: Remove generated test and bytecode caches**

Resolve every `.pytest_cache` and `__pycache__` target, verify it is below the
repository root, and remove only those directories. Do not remove models,
formal preprocessing results, manifests, or inference outputs.

- [ ] **Step 9: Commit documentation and benchmark**

```powershell
git add -- voice-pipeline/src/voice_pipeline/cli/infer.py voice-pipeline/tests/test_infer_cli.py voice-pipeline/README.md
git commit -m "feat: add inference benchmark and documentation"
git status --short
```

Expected: commit succeeds and the worktree is clean.

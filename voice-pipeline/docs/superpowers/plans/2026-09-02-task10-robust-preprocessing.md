# Task 10 Robust v2ProPlus Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert an existing official-format `data.list` into validated, resumable, per-sample v2ProPlus text, wav32k, HuBERT, SV, and base-S2G semantic artifacts with bounded global quarantine.

**Architecture:** Extend the existing manifest, state, signature, profile, and graph foundations instead of creating a second orchestration framework. Concrete stages write per-sample artifacts atomically; a single pipeline owns global quarantine and publishes canonical training indexes only after every required stage has a compatible result.

**Tech Stack:** Python 3.12, Typer, PyYAML, PyTorch, torchaudio, SciPy, Transformers, pyopenjtalk, G2PW/ONNX Runtime, NLTK/g2p_en, fast-langdetect, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-task10-robust-preprocessing-design.md`

## Global Constraints

- Work directly on `main`; do not create a branch or worktree.
- Use only `D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe`; never use Conda.
- Do not create a ZIP.
- V1 implements only `v2ProPlus`: `SynthesizerTrn + MultiPeriodDiscriminator + SV + GAN`; no V3/V4, CFM, LoRA, or external vocoder.
- Input is an existing `audio_path|speaker|language|text` manifest. Do not rewrite it, run ASR, slice audio, augment data, or validate speaker count.
- Apply `% -> -` and `￥ -> ,` at the shared training/inference frontend entry.
- HuBERT input must be decoded from original audio, never from saved int16 wav32k.
- Semantic extraction must use profile base `s2Gv2ProPlus.pth`; no fine-tuned override exists.
- Global bad-record allowance is `min(5, ceil(total_nonempty_records * 0.20))`, with at least one valid sample.
- Any sample failure globally quarantines that sample from every S1/S2 index.
- Keep valid preprocessing artifacts after successful training; cleanup only temporary files, transient run caches, and quarantined orphan artifacts.
- Use `--basetemp .pytest_cache\codex-temp` for pytest because the default Windows temp pytest root has incompatible ACLs.
- Keep dependency-file changes deferred until final environment consolidation, per user instruction.
- Finish the complete Task 10 Part, run verification, commit, and stop for user review before Task 11.

---

### Task 1: Harden manifest identity and complete the v2ProPlus asset contract

**Files:**
- Modify: `src/voice_pipeline/training/manifest.py`
- Modify: `src/voice_pipeline/profiles/base.py`
- Modify: `src/voice_pipeline/profiles/v2proplus.py`
- Modify: `src/voice_pipeline/common/assets.py`
- Modify: `tests/test_manifest.py`
- Modify: `tests/test_assets.py`

**Interfaces:**
- Produces: `stable_sample_id(audio_path: str, speaker: str, language: str, text: str) -> str`
- Produces: `read_manifest_records(path: Path) -> ManifestReadResult`
- Produces: `allowed_bad_records(total_records: int) -> int`
- Produces: `ManifestRecord(line_no, sample_id, item)` and `ManifestIssue(line_no, raw, category, message)`
- Extends: `ModelProfile` with `g2pw_relative_path`, `nltk_data_relative_path`, and `langdetect_relative_path`

- [ ] **Step 1: Write failing tolerant-manifest and allowance tests**

```python
def test_preprocess_manifest_keeps_valid_rows_and_reports_bad_rows(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    manifest = tmp_path / "data.list"
    manifest.write_text(
        f"{wav}|speaker|ja|こんにちは。\n"
        "broken|row\n"
        f"{wav}|speaker|ko|안녕\n",
        encoding="utf-8",
    )
    result = read_manifest_records(manifest)
    assert result.total_records == 3
    assert [record.item.language for record in result.records] == ["ja"]
    assert [issue.category for issue in result.issues] == ["malformed", "unsupported_language"]


@pytest.mark.parametrize(
    ("total", "allowed"), [(1, 1), (6, 2), (10, 2), (21, 5), (100, 5)]
)
def test_bad_record_allowance_rounds_up_and_caps_at_five(total, allowed):
    assert allowed_bad_records(total) == allowed


def test_exact_duplicate_is_reported_without_speaker_count_validation(tmp_path):
    result = read_manifest_records(_manifest_with_two_identical_valid_lines(tmp_path))
    assert len(result.records) == 1
    assert result.issues[0].category == "duplicate"
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest tests\test_manifest.py tests\test_assets.py `
  --basetemp .pytest_cache\codex-temp -q
```

Expected: import/attribute failures for the new manifest and profile fields.

- [ ] **Step 3: Implement canonical records and asset fields**

```python
ALLOWED_LANGUAGES = frozenset({"zh", "ja", "en", "mixed"})


def stable_sample_id(audio_path: str, speaker: str, language: str, text: str) -> str:
    canonical = "\0".join((Path(audio_path).as_posix(), speaker, language, text))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def allowed_bad_records(total_records: int) -> int:
    if total_records < 1:
        return 0
    return min(5, math.ceil(total_records * 0.20))
```

`read_manifest_records` counts non-empty physical records, parses exactly four
pipe-separated fields, preserves `speaker` without uniqueness checks, reports bad
rows instead of raising, verifies audio existence, validates the four allowed
languages, and converts a repeated stable ID into one `duplicate` issue. Keep the
existing fail-fast `load_manifest()` API for its current callers and tests. Resolve
relative audio paths against the directory containing `data.list`; do not resolve
them against the process working directory.

Add these profile paths exactly:

```python
g2pw_relative_path = "models/pretrained/v2proplus/g2pw/G2PWModel"
nltk_data_relative_path = "models/pretrained/v2proplus/g2p/en/nltk_data"
langdetect_relative_path = "models/pretrained/v2proplus/langdetect"
```

Include them in `verify_profile_assets` as `g2pw`, `nltk`, and `langdetect`.

- [ ] **Step 4: Run GREEN**

Run the Step 2 command. Expected: all manifest and asset tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- src/voice_pipeline/training/manifest.py src/voice_pipeline/profiles `
  src/voice_pipeline/common/assets.py tests/test_manifest.py tests/test_assets.py
git commit -m "feat: harden preprocessing inputs and assets"
```

---

### Task 2: Add full asset digests, atomic artifact writes, and rich stage state

**Files:**
- Create: `src/voice_pipeline/training/preprocess/__init__.py`
- Create: `src/voice_pipeline/training/preprocess/artifacts.py`
- Modify: `src/voice_pipeline/pipeline/signature.py`
- Modify: `src/voice_pipeline/common/state.py`
- Create: `tests/test_preprocess_artifacts.py`
- Modify: `tests/test_signature.py`
- Modify: `tests/test_state.py`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`
- Produces: `sha256_tree(path: Path) -> str`
- Produces: `atomic_write_text(path: Path, text: str) -> None`
- Produces: `atomic_torch_save(path: Path, value: object) -> None`
- Produces: `write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None`
- Extends: `StageState` with `signature`, `outputs`, `started_at`, `finished_at`, `warning_count`, and `error`

- [ ] **Step 1: Write failing digest, atomic-write, and persistence tests**

```python
def test_large_file_signature_changes_when_content_changes_but_size_and_mtime_do_not(tmp_path):
    asset = tmp_path / "weight.pth"
    asset.write_bytes(b"a" * (4 * 1024 * 1024 + 1))
    original_mtime = asset.stat().st_mtime_ns
    first = sha256_file(asset)
    asset.write_bytes(b"b" * asset.stat().st_size)
    os.utime(asset, ns=(original_mtime, original_mtime))
    assert sha256_file(asset) != first


def test_atomic_torch_save_leaves_no_temp_file(tmp_path):
    target = tmp_path / "feature.pt"
    atomic_torch_save(target, torch.ones(2, 3))
    assert torch.equal(torch.load(target, weights_only=True), torch.ones(2, 3))
    assert list(tmp_path.glob("*.tmp")) == []


def test_stage_metadata_round_trips(tmp_path):
    state = RunState(tmp_path / "state.json")
    state.start("text", signature="abc")
    state.complete("text", outputs=["preprocess/text/index.jsonl"], warning_count=2)
    loaded = RunState(tmp_path / "state.json").get("text")
    assert (loaded.signature, loaded.warning_count) == ("abc", 2)
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest tests\test_preprocess_artifacts.py tests\test_signature.py `
  tests\test_state.py --basetemp .pytest_cache\codex-temp -q
```

- [ ] **Step 3: Implement atomic writes and full hashing**

Use destination-local unique temporary names so `Path.replace()` remains atomic:

```python
def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
```

`atomic_torch_save` follows the same pattern. `write_jsonl` serializes rows with
`ensure_ascii=False`, sorted keys, and one trailing newline. `sha256_tree` hashes
sorted relative paths plus every file's full SHA-256. Change `_file_digest` so it
always contains SHA-256 and never falls back to size/mtime identity.

Add `RunState.start`, `complete`, `fail`, and `invalidate` convenience methods while
retaining legal transition enforcement and atomic JSON replacement. Loading malformed
JSON raises a message naming the state path.

- [ ] **Step 4: Run GREEN and existing graph/signature regression**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest tests\test_preprocess_artifacts.py tests\test_signature.py `
  tests\test_state.py tests\test_stage_graph.py `
  --basetemp .pytest_cache\codex-temp -q
```

- [ ] **Step 5: Commit**

```powershell
git add -- src/voice_pipeline/training/preprocess src/voice_pipeline/pipeline/signature.py `
  src/voice_pipeline/common/state.py tests/test_preprocess_artifacts.py `
  tests/test_signature.py tests/test_state.py
git commit -m "feat: add durable preprocessing artifacts"
```

---

### Task 3: Implement fake-stage orchestration and the global quarantine barrier

**Files:**
- Create: `src/voice_pipeline/training/preprocess/base.py`
- Create: `src/voice_pipeline/training/preprocess/pipeline.py`
- Modify: `src/voice_pipeline/pipeline/graph.py`
- Create: `tests/test_preprocess_pipeline.py`
- Modify: `tests/test_stage_graph.py`

**Interfaces:**
- Produces: `StageContext(experiment: Experiment, profile: ModelProfile, config: Mapping[str, object], asset_digests: Mapping[str, str])`
- Produces exception: `SampleFailure(stage: str, category: str, message: str)`
- Produces: `StageSampleResult(sample_id: str, output_paths: list[Path], metadata: dict[str, object])`
- Produces: `QuarantineEntry(key: str, line_no: int, sample_id: str | None, audio_path: str | None, stage: str, category: str, message: str)`
- Produces: `PreprocessSummary(valid_sample_ids: list[str], quarantined: list[QuarantineEntry], allowed_bad: int, total_records: int)`
- Produces: `PreprocessPipeline(stages, graph, state, context)`
- Produces: `PreprocessPipeline.run(records, initial_issues, selected_stage=None) -> PreprocessSummary`
- Produces: `StageGraph.topological_order(target: str | None = None) -> list[str]`
- Stage duck type: `name: str`, `dependencies: set[str]`,
  `signature(record, context) -> str`, `run(record, context) -> StageSampleResult`,
  and `validate_cached(record, entry, context) -> bool`

- [ ] **Step 1: Write failing orchestration tests with fake stages**

```python
def test_pipeline_runs_in_dependency_order_and_resumes_completed_samples(tmp_path):
    calls = []
    stages = {
        "text": FakeStage("text", set(), calls),
        "wav32k": FakeStage("wav32k", set(), calls),
        "hubert": FakeStage("hubert", {"wav32k"}, calls),
        "sv": FakeStage("sv", {"wav32k"}, calls),
        "semantic": FakeStage("semantic", {"hubert"}, calls),
    }
    pipeline = make_pipeline(tmp_path, stages)
    pipeline.run(two_records(), [])
    assert [name for name, _ in calls[:5]] == ["text", "text", "wav32k", "wav32k", "hubert"]
    calls.clear()
    pipeline.run(two_records(), [])
    assert calls == []


def test_one_stage_failure_quarantines_sample_from_all_downstream_indexes(tmp_path):
    pipeline = make_pipeline(tmp_path, stages_with_failure("hubert", sample="bad"))
    summary = pipeline.run(six_records(), [])
    assert summary.valid_sample_ids == five_good_ids()
    assert summary.quarantined[0].sample_id == "bad"
    assert "bad" not in read_all_stage_training_ids(tmp_path)


def test_pipeline_fails_on_third_bad_record_out_of_ten(tmp_path):
    pipeline = make_pipeline(tmp_path, stages_with_three_failures())
    with pytest.raises(QuarantineLimitExceeded, match="3 > 2"):
        pipeline.run(ten_records(), [])
    assert not (tmp_path / "valid_samples.jsonl").exists()
```

Also test six records allow two, zero valid records fail, malformed manifest issues
consume the same global allowance, a changed dependency signature invalidates only
downstream samples, abandoned `*.tmp` files never create cache hits, a target stage
includes its dependency closure, unknown dependencies are rejected, and a graph cycle
is rejected with the participating stage names.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest tests\test_preprocess_pipeline.py `
  --basetemp .pytest_cache\codex-temp -q
```

- [ ] **Step 3: Implement the minimal concrete runner**

Extend the existing `StageGraph` with deterministic topological sorting and graph
validation, then use it for both complete runs and target dependency closure. The
pipeline owns one
quarantine dictionary keyed by sample ID (or `line-<n>` for unparsed records), skips
quarantined samples in every later stage, and rewrites `quarantine.jsonl` atomically.
A cached sample is valid only when its signature matches, every declared output exists,
and its recorded metadata validates.

Publish `valid_samples.jsonl` only after all five required stages are current:

```python
valid_ids = [record.sample_id for record in records if record.sample_id not in quarantine]
if not valid_ids:
    raise QuarantineLimitExceeded("preprocessing left no valid samples")
if len(quarantine) > allowed_bad_records(total_records):
    raise QuarantineLimitExceeded(
        f"bad record limit exceeded: {len(quarantine)} > {allowed_bad_records(total_records)}"
    )
write_jsonl(context.preprocess_dir / "valid_samples.jsonl", valid_rows)
```

Do not catch configuration/model-construction failures inside a sample loop.

- [ ] **Step 4: Run GREEN**

Run the Step 2 command. Expected: all fake-stage, resume, invalidation, and quarantine
tests pass without loading a model.

- [ ] **Step 5: Commit**

```powershell
git add -- src/voice_pipeline/training/preprocess/base.py `
  src/voice_pipeline/training/preprocess/pipeline.py src/voice_pipeline/pipeline/graph.py `
  tests/test_preprocess_pipeline.py tests/test_stage_graph.py
git commit -m "feat: add resumable preprocessing orchestration"
```

---

### Task 4: Put official symbol cleanup at the shared frontend and add the text stage

**Files:**
- Modify: `src/voice_pipeline/core/gpt_sovits/frontend/multilingual.py`
- Create: `src/voice_pipeline/training/preprocess/text_stage.py`
- Create: `tests/compat/test_frontend_input_cleanup.py`
- Create: `tests/test_preprocess_text_stage.py`

**Interfaces:**
- Produces: `sanitize_frontend_text(text: str) -> str`
- Produces: `TextStage(frontend: MultilingualFrontend)`
- `TextStage.run(record, context) -> StageSampleResult`

- [ ] **Step 1: Write failing cleanup and stage contract tests**

```python
def test_shared_frontend_applies_official_training_symbol_cleanup(fake_frontend):
    assert sanitize_frontend_text("成功率50%￥500") == "成功率50-,500"


def test_text_stage_saves_metadata_and_aligned_bert(tmp_path):
    frontend = FakeFrontend(
        FrontendResult("こんにちは。", ["k", "o", "."], [1, 2, 3], None, torch.zeros(1024, 3))
    )
    result = TextStage(frontend).run(ja_record(), context(tmp_path))
    metadata = json.loads(Path(result.output_paths[0]).read_text(encoding="utf-8"))
    bert = torch.load(result.output_paths[1], weights_only=True)
    assert metadata["phone_ids"] == [1, 2, 3]
    assert bert.shape == (1024, 3)
```

Also assert unsupported language and empty phones become `SampleFailure`, and mixed
text calls the same frontend once with `language="mixed"`.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest tests\compat\test_frontend_input_cleanup.py `
  tests\test_preprocess_text_stage.py --basetemp .pytest_cache\codex-temp -q
```

- [ ] **Step 3: Implement shared cleanup and atomic text outputs**

```python
def sanitize_frontend_text(text: str) -> str:
    return text.replace("%", "-").replace("￥", ",")


def process(self, text: str, language: str) -> FrontendResult:
    text = sanitize_frontend_text(text)
    # retain the reviewed routing and concatenation body unchanged
```

`TextStage` validates `FrontendResult`, writes JSON without the tensor, writes BERT
with `atomic_torch_save`, and records `(1024, phone_count)` plus tensor dtype in the
stage result.

- [ ] **Step 4: Run GREEN plus all frontend compatibility tests**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest tests\compat tests\test_preprocess_text_stage.py `
  -k 'frontend or chinese or japanese or english or symbols or tone or text_stage' `
  --basetemp .pytest_cache\codex-temp -q
```

- [ ] **Step 5: Commit**

```powershell
git add -- src/voice_pipeline/core/gpt_sovits/frontend/multilingual.py `
  src/voice_pipeline/training/preprocess/text_stage.py `
  tests/compat/test_frontend_input_cleanup.py tests/test_preprocess_text_stage.py
git commit -m "feat: add multilingual text preprocessing stage"
```

---

### Task 5: Implement upstream-compatible raw-audio preparation and wav32k

**Files:**
- Modify: `src/voice_pipeline/core/gpt_sovits/features/audio.py`
- Create: `src/voice_pipeline/training/preprocess/wav32k_stage.py`
- Create: `tests/compat/test_wav32k_preparation.py`
- Create: `tests/test_preprocess_wav32k_stage.py`

**Interfaces:**
- Produces: `PreparedAudio(wav32_int16: np.ndarray, hubert_source_32k: torch.Tensor)`
- Produces: `prepare_audio_from_source(path: str | Path) -> PreparedAudio`
- Produces: `Wav32kStage()`

- [ ] **Step 1: Write failing numerical and invalid-audio tests**

```python
def test_prepare_audio_preserves_pinned_amplitude_mix(monkeypatch):
    raw = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float32)
    monkeypatch.setattr(audio, "load_audio_32k", lambda path: raw)
    prepared = audio.prepare_audio_from_source("ignored.wav")
    expected = raw.numpy() / 1.0 * (0.95 * 0.5 * 32768) + 0.5 * 32768 * raw.numpy()
    np.testing.assert_array_equal(prepared.wav32_int16, expected.astype(np.int16))


@pytest.mark.parametrize("waveform", [torch.zeros(32), torch.tensor([float("nan")])])
def test_prepare_audio_rejects_silent_or_nonfinite_input(monkeypatch, waveform):
    monkeypatch.setattr(audio, "load_audio_32k", lambda path: waveform)
    with pytest.raises(InvalidSampleAudio):
        audio.prepare_audio_from_source("ignored.wav")
```

Add a stage test that loads the written WAV and asserts mono int16/32 kHz, plus a
test that an existing `.tmp` is ignored and replaced.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest tests\compat\test_wav32k_preparation.py `
  tests\test_preprocess_wav32k_stage.py --basetemp .pytest_cache\codex-temp -q
```

- [ ] **Step 3: Implement the pinned preparation formula and stage**

Reject empty arrays, non-finite values, `max_abs == 0`, and `max_abs > 2.2`. Convert
the tensor returned by `load_audio_32k` to a CPU NumPy float32 array, then preserve
both pinned scaling branches:

```python
raw = load_audio_32k(path).cpu().numpy().astype(np.float32, copy=False)
wav32 = raw / max_abs * (0.95 * 0.5 * 32768) + 0.5 * 32768 * raw
hubert_source = raw / max_abs * (0.95 * 0.5 * 1145.14) + 0.5 * 1145.14 * raw
return PreparedAudio(wav32.astype(np.int16), torch.from_numpy(hubert_source.astype(np.float32)))
```

`Wav32kStage` calls this helper from the original manifest audio and writes with
`scipy.io.wavfile.write` through a destination-local temporary file and atomic replace.

- [ ] **Step 4: Run GREEN plus audio compatibility tests**

Run the Step 2 command followed by:

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest tests\compat\test_feature_shapes.py `
  --basetemp .pytest_cache\codex-temp -q
```

Expected: both commands pass.

- [ ] **Step 5: Commit**

```powershell
git add -- src/voice_pipeline/core/gpt_sovits/features/audio.py `
  src/voice_pipeline/training/preprocess/wav32k_stage.py `
  tests/compat/test_wav32k_preparation.py tests/test_preprocess_wav32k_stage.py
git commit -m "feat: add official wav32k preprocessing"
```

---

### Task 6: Add HuBERT and SV stages without changing their source paths

**Files:**
- Modify: `src/voice_pipeline/core/gpt_sovits/features/cnhubert.py`
- Create: `src/voice_pipeline/training/preprocess/hubert_stage.py`
- Create: `src/voice_pipeline/training/preprocess/sv_stage.py`
- Create: `tests/test_preprocess_feature_stages.py`
- Modify: `tests/compat/test_feature_shapes.py`

**Interfaces:**
- Extends: `CNHubertExtractor.extract(wav_16k)` with configured precision and
  `CNHubertExtractor.to_float() -> None` recovery
- Produces: `HubertStage(extractor, precision)`
- Produces: `SVStage(encoder)`

- [ ] **Step 1: Write failing source, retry, and shape tests**

```python
def test_hubert_stage_decodes_original_audio_not_wav32(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(hubert_stage, "prepare_audio_from_source", lambda path: calls.append(path) or prepared())
    HubertStage(FakeHubert()).run(record(audio_path=tmp_path / "original.flac"), context(tmp_path))
    assert calls == [tmp_path / "original.flac"]


def test_hubert_fp16_nonfinite_retries_once_in_fp32(tmp_path):
    extractor = FakeHubert(outputs=[torch.full((1, 768, 4), float("nan")), torch.ones(1, 768, 4)])
    result = HubertStage(extractor, precision="fp16").run(record(), context(tmp_path))
    assert extractor.to_float_calls == 1
    assert result.metadata["shape"] == [1, 768, 4]


def test_sv_stage_requires_official_forward3_shape(tmp_path):
    result = SVStage(FakeSpeaker(torch.ones(1, 20480))).run(record(), context_with_wav(tmp_path))
    assert result.metadata["shape"] == [1, 20480]
```

Also test fp32 retry still non-finite becomes a sample failure, HuBERT rejects a
wrong channel dimension, and SV rejects a non-finite or wrong-width tensor.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest tests\test_preprocess_feature_stages.py `
  tests\compat\test_feature_shapes.py --basetemp .pytest_cache\codex-temp -q
```

- [ ] **Step 3: Implement both stages**

`HubertStage` invokes `prepare_audio_from_source(record.item.audio_path)` itself,
resamples `hubert_source_32k` directly to 16 kHz, extracts, validates
`(1, 768, frames)`, retries fp16 non-finite output once after moving the extractor
to float32, then saves CPU tensors atomically.

`SVStage` loads the committed 32 kHz WAV dependency with torchaudio, asserts 32 kHz
mono, invokes the migrated `SpeakerEncoder`, validates `(1, 20480)`, and atomically
saves a CPU tensor.

- [ ] **Step 4: Run GREEN and real feature-shape compatibility tests**

Set `VOICE_PIPELINE_TEST_HUBERT_DIR` and `VOICE_PIPELINE_TEST_SV_CHECKPOINT` to the
known local assets, then run the Step 2 command. Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add -- src/voice_pipeline/core/gpt_sovits/features/cnhubert.py `
  src/voice_pipeline/training/preprocess/hubert_stage.py `
  src/voice_pipeline/training/preprocess/sv_stage.py `
  tests/test_preprocess_feature_stages.py tests/compat/test_feature_shapes.py
git commit -m "feat: add hubert and speaker preprocessing stages"
```

---

### Task 7: Add strictly base-S2G semantic extraction

**Files:**
- Create: `src/voice_pipeline/training/preprocess/semantic_stage.py`
- Create: `tests/test_preprocess_semantic_stage.py`
- Create: `tests/compat/test_semantic_extraction.py`

**Interfaces:**
- Produces: `SemanticExtractor(base_s2g_path: Path, device, precision)`
- Produces: `SemanticExtractor.extract(ssl: torch.Tensor) -> torch.Tensor`
- Produces: `SemanticStage(extractor)`

- [ ] **Step 1: Write failing strict-source and tensor tests**

```python
def test_semantic_extractor_has_no_finetuned_checkpoint_override():
    assert list(inspect.signature(SemanticExtractor).parameters) == [
        "base_s2g_path", "device", "precision"
    ]


def test_semantic_stage_saves_integer_25hz_tokens(tmp_path):
    extractor = FakeSemantic(torch.tensor([[[3, 7, 9]]], dtype=torch.long))
    result = SemanticStage(extractor).run(record(), context_with_hubert(tmp_path))
    saved = torch.load(result.output_paths[0], weights_only=True)
    assert saved.tolist() == [3, 7, 9]
    assert result.metadata == {"shape": [3], "dtype": "torch.int64", "frame_rate": "25hz"}
```

The real compatibility test loads the official S2G with `load_s2_generator`, feeds a
small finite `(1, 768, frames)` SSL tensor, compares the stage tokens to direct
`model.extract_latent(ssl)[0, 0]`, and asserts token IDs are within `0..1023`.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest tests\test_preprocess_semantic_stage.py `
  tests\compat\test_semantic_extraction.py --basetemp .pytest_cache\codex-temp -q
```

- [ ] **Step 3: Implement strict base-model extraction**

```python
class SemanticExtractor:
    def __init__(self, base_s2g_path: Path, device="cpu", precision="fp32"):
        self.model = load_s2_generator(base_s2g_path, device=device).eval()
        if precision == "fp16" and torch.device(device).type == "cuda":
            self.model.half()

    def extract(self, ssl: torch.Tensor) -> torch.Tensor:
        dtype = next(self.model.parameters()).dtype
        with torch.inference_mode():
            codes = self.model.extract_latent(ssl.to(next(self.model.parameters()).device, dtype=dtype))
        return codes[0, 0].to(device="cpu", dtype=torch.long)
```

Validate non-empty rank-one integer output, finite source SSL, 25 Hz profile, and
token range before atomic save.

- [ ] **Step 4: Run GREEN with the real base S2G**

Set `VOICE_PIPELINE_TEST_S2_DIR` to the known local v2Pro directory and run the Step 2
command. Expected: all pass with the official `s2Gv2ProPlus.pth`.

- [ ] **Step 5: Commit**

```powershell
git add -- src/voice_pipeline/training/preprocess/semantic_stage.py `
  tests/test_preprocess_semantic_stage.py tests/compat/test_semantic_extraction.py
git commit -m "feat: add base-s2g semantic preprocessing"
```

---

### Task 8: Build real stages from YAML, publish official indexes, and add CLI

**Files:**
- Create: `src/voice_pipeline/training/preprocess/config.py`
- Create: `src/voice_pipeline/training/preprocess/factory.py`
- Create: `src/voice_pipeline/training/preprocess/indexes.py`
- Create: `src/voice_pipeline/cli/preprocess.py`
- Modify: `src/voice_pipeline/cli/main.py`
- Create: `tests/test_preprocess_config.py`
- Create: `tests/test_preprocess_indexes.py`
- Create: `tests/test_preprocess_cli.py`

**Interfaces:**
- Produces: `PreprocessConfig(profile, experiment_name, output_root, manifest, project_root, device, precision, resume)`
- Produces: `PreprocessConfig.from_yaml(path: Path, project_root: Path | None = None) -> PreprocessConfig`
- Produces: `build_preprocess_pipeline(config: PreprocessConfig, selected_stage: str | None = None) -> PreprocessPipeline`
- Produces: `publish_training_indexes(preprocess_dir: Path, records, valid_ids) -> list[Path]`
- Produces CLI: `voice-pipeline preprocess all -c <config>`
- Produces CLI: `voice-pipeline preprocess stage <name> -c <config>`

- [ ] **Step 1: Write failing config, index, and CLI tests**

```python
def test_config_reads_existing_readme_shape(tmp_path):
    config = PreprocessConfig.from_yaml(write_config(tmp_path, profile="v2ProPlus"))
    assert config.profile.name == "v2ProPlus"
    assert config.manifest.name == "data.list"
    assert config.precision == "fp16"


def test_indexes_share_exact_valid_sample_membership(tmp_path):
    outputs = publish_training_indexes(tmp_path, records(), {"a", "c"})
    assert ids_from_name2text(outputs[0]) == {"a", "c"}
    assert ids_from_name2semantic(outputs[1]) == {"a", "c"}


def test_preprocess_cli_exposes_all_and_stage(monkeypatch, config_path):
    monkeypatch.setattr(preprocess_cli, "build_preprocess_pipeline", fake_builder)
    assert runner.invoke(app, ["preprocess", "all", "-c", str(config_path)]).exit_code == 0
    assert runner.invoke(app, ["preprocess", "stage", "sv", "-c", str(config_path)]).exit_code == 0
```

Also assert unknown stage, non-v2ProPlus profile, missing required asset, and malformed
YAML fail before any stage output is created. A tolerated quarantine must print bad
count, allowance, remaining valid count, and report path while exiting zero.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest tests\test_preprocess_config.py tests\test_preprocess_indexes.py `
  tests\test_preprocess_cli.py --basetemp .pytest_cache\codex-temp -q
```

- [ ] **Step 3: Implement config, factory, index publication, and commands**

Read only the existing README fields:

```yaml
profile: v2ProPlus
experiment:
  name: speaker_001
  output_root: runs
device:
  device: cuda:0
  precision: fp16
dataset:
  manifest: D:/dataset/data.list
preprocess:
  resume: true
```

Resolve assets through the selected `ModelProfile` and project root. The factory
uses `Path.cwd()` when `project_root` is omitted. Relative output and model paths are
resolved from that project root; relative manifest audio paths remain relative to the
manifest directory. The factory constructs each heavy model at most once per CLI
process, creates the reviewed dependency graph, writes full asset digests to
`assets.json`, and injects concrete stages into `PreprocessPipeline`. A selected stage
constructs and verifies only the models required by its dependency closure. For
example, `stage semantic` executes `wav32k -> hubert -> semantic`, while `stage text`
does not require S2G, HuBERT, or SV assets.

`publish_training_indexes` sorts by original manifest line, reads only validated
per-sample artifacts, atomically writes the official tab-separated text and semantic
views, and refuses mismatched sample sets.

- [ ] **Step 4: Run GREEN plus root CLI tests**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest tests\test_preprocess_config.py tests\test_preprocess_indexes.py `
  tests\test_preprocess_cli.py tests\test_cli.py `
  --basetemp .pytest_cache\codex-temp -q
```

- [ ] **Step 5: Commit**

```powershell
git add -- src/voice_pipeline/training/preprocess src/voice_pipeline/cli `
  tests/test_preprocess_config.py tests/test_preprocess_indexes.py `
  tests/test_preprocess_cli.py
git commit -m "feat: expose v2proplus preprocessing cli"
```

---

### Task 9: Add safe post-training cleanup ownership

**Files:**
- Create: `src/voice_pipeline/training/preprocess/cleanup.py`
- Create: `tests/test_preprocess_cleanup.py`

**Interfaces:**
- Produces: `cleanup_after_training(preprocess_dir: Path, training_succeeded: bool) -> CleanupReport`
- Produces: `CleanupReport(removed_temporary, removed_quarantined, retained_valid)`

- [ ] **Step 1: Write failing cleanup-scope tests**

```python
def test_success_cleanup_removes_temp_and_quarantined_but_keeps_valid(tmp_path):
    layout = make_cleanup_fixture(tmp_path, valid={"good"}, quarantined={"bad"})
    report = cleanup_after_training(layout.preprocess_dir, training_succeeded=True)
    assert not layout.temp.exists()
    assert not layout.bad_hubert.exists()
    assert layout.good_hubert.exists()
    assert report.retained_valid == 1


def test_failed_training_removes_nothing(tmp_path):
    layout = make_cleanup_fixture(tmp_path, valid={"good"}, quarantined={"bad"})
    cleanup_after_training(layout.preprocess_dir, training_succeeded=False)
    assert layout.temp.exists() and layout.bad_hubert.exists() and layout.good_hubert.exists()


def test_cleanup_rejects_output_path_outside_preprocess_root(tmp_path):
    layout = make_malicious_quarantine_path_fixture(tmp_path)
    with pytest.raises(ValueError, match="outside preprocess root"):
        cleanup_after_training(layout.preprocess_dir, training_succeeded=True)
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest tests\test_preprocess_cleanup.py `
  --basetemp .pytest_cache\codex-temp -q
```

- [ ] **Step 3: Implement conservative cleanup**

Read `valid_samples.jsonl` and `quarantine.jsonl`, resolve every candidate path, and
require `candidate.resolve().is_relative_to(preprocess_dir.resolve())`. When
`training_succeeded` is false, return zero removals without touching disk. When true,
remove destination-local `*.tmp` files and only known stage artifacts whose sample IDs
are quarantined and absent from the valid set. Never remove stage indexes, aggregate
indexes, state, signatures, reports, or valid artifacts.

- [ ] **Step 4: Run GREEN**

Run the Step 2 command. Expected: all cleanup boundary tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- src/voice_pipeline/training/preprocess/cleanup.py `
  tests/test_preprocess_cleanup.py
git commit -m "feat: add safe post-training cache cleanup"
```

---

### Task 10: Run end-to-end real preprocessing and close the architecture audit

**Files:**
- Create: `tests/integration/test_real_preprocess.py`
- Modify: `src/voice_pipeline/core/gpt_sovits/UPSTREAM.md`
- Modify: `README.md`
- Modify: `.sdd-progress.md`

**Interfaces:**
- Verifies all Task 10 public interfaces; adds no second production path.

- [ ] **Step 1: Add a real-fixture integration test**

Create a short temporary official-format manifest pointing to an existing local audio
fixture. Build the real pipeline with explicit test asset paths and assert:

```python
summary = pipeline.run(read.records, read.issues)
assert summary.quarantined == []
assert len(summary.valid_sample_ids) == 1
assert load_text_bert(summary).shape[0] == 1024
assert load_hubert(summary).shape[1] == 768
assert load_sv(summary).shape == (1, 20480)
assert load_semantic(summary).dtype == torch.int64
assert s1_index_ids(summary) == s2_index_ids(summary)
```

If no committed audio fixture exists, generate a deterministic sine-wave WAV inside
pytest's `tmp_path`; do not add generated audio to the repository.

- [ ] **Step 2: Run the full real-asset regression**

Set these variables to the known local assets:

```powershell
$env:VOICE_PIPELINE_TEST_S1_CHECKPOINT='D:\AI-Training\voice-clone\GPT-SoVITS\GPT_SoVITS\pretrained_models\s1v3.ckpt'
$env:VOICE_PIPELINE_TEST_S2_DIR='D:\AI-Training\voice-clone\GPT-SoVITS\GPT_SoVITS\pretrained_models\v2Pro'
$env:VOICE_PIPELINE_TEST_HUBERT_DIR='D:\AI-Training\voice-clone\GPT-SoVITS\GPT_SoVITS\pretrained_models\chinese-hubert-base'
$env:VOICE_PIPELINE_TEST_SV_CHECKPOINT='D:\AI-Training\voice-clone\GPT-SoVITS\GPT_SoVITS\pretrained_models\sv\pretrained_eres2netv2w24s4ep4.ckpt'
$env:VOICE_PIPELINE_TEST_BERT_DIR='D:\AI-Training\voice-clone\GPT-SoVITS\GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large'
$env:VOICE_PIPELINE_TEST_G2PW_DIR='D:\AI-Training\voice-clone\voice-pipeline\voice-pipeline\models\pretrained\v2proplus\g2pw\G2PWModel'
$env:VOICE_PIPELINE_TEST_NLTK_DATA='D:\AI-Training\voice-clone\voice-pipeline\voice-pipeline\models\pretrained\v2proplus\g2p\en\nltk_data'
$env:VOICE_PIPELINE_TEST_LANGDETECT_DIR='D:\AI-Training\voice-clone\GPT-SoVITS\GPT_SoVITS\pretrained_models\fast_langdetect'
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest --basetemp .pytest_cache\codex-temp -q
```

Expected: zero asset-gated skips and all tests pass.

- [ ] **Step 3: Run architecture, compilation, coupling, and whitespace checks**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m compileall -q src tests
rg -n "s2_train_v3|SynthesizerTrnV3|CFM|f5_tts|sys\.path|GPT_SoVITS" `
  src/voice_pipeline/training/preprocess src/voice_pipeline/cli/preprocess.py
git diff --check
```

Expected: compilation and whitespace checks pass; coupling scan returns no production
match. Re-run profile route tests explicitly:

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest tests\compat\test_v2proplus_route.py tests\compat\test_s2_core.py `
  --basetemp .pytest_cache\codex-temp -q
```

- [ ] **Step 4: Record exact implementation and cleanup provenance**

Update `UPSTREAM.md` with every adapted preprocessing source path, the raw-audio
HuBERT boundary, base-S2G semantic enforcement, and differences from upstream
(atomic per-sample files and bounded quarantine). Update README commands and artifact
layout. Record exact final pass/skip counts in `.sdd-progress.md`.

- [ ] **Step 5: Commit and stop for review**

```powershell
git add -- tests/integration/test_real_preprocess.py `
  src/voice_pipeline/core/gpt_sovits/UPSTREAM.md README.md .sdd-progress.md
git commit -m "test: verify robust v2proplus preprocessing"
git status --short
```

Report the commits, exact tests, real assets exercised, quarantine behavior, warnings,
and any environment-only dependencies observed. Do not start Task 11 and do not create
a ZIP.

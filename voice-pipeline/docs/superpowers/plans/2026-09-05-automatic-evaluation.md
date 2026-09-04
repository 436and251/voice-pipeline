# Automatic Checkpoint Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic cross-language evaluation that filters S2 first, evaluates retained S2 against S1 checkpoints, and exports an anonymous human-listening shortlist without automatically selecting a final model.

**Architecture:** Evaluation is a lazy-loaded subsystem that consumes training checkpoints and the existing inference/export APIs. Task19 establishes configuration, suites, candidates, temporary bundles, deterministic audio generation, and resumable manifests; Task20 adds Faster-Whisper, independent WavLM speaker verification, text-error/language checks, and local prosody; Task21 applies hard constraints and weighted ranking, writes reports/shortlist, exports every shortlisted CandidateBundle, and generates anonymous ZH/JA/EN previews.

**Tech Stack:** Python 3.12, dataclasses, pathlib, JSON/YAML, NumPy, PyTorch, Transformers, faster-whisper, librosa, Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-automatic-evaluation-design.md`

## Global Constraints

- Work directly on `main`; do not create branches, worktrees, Conda environments, or ZIP archives.
- Use `D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe` for Python and pytest.
- ASR is `mobiuslabsgmbh/faster-whisper-large-v3-turbo`; speaker evaluation is `microsoft/wavlm-base-plus-sv`.
- Evaluator models remain outside training and standalone inference imports.
- Speaker preservation has the highest ranking weight; hard constraints run before ranking.
- Use `Base S1 × all S2`, retain `s2_keep`, then `(Base S1 + all S1) × retained S2`.
- Do not evaluate the full S1×S2 Cartesian product.
- Do not add an automatic final-model promotion interface.
- Every shortlisted pair is exported and receives ZH/JA/EN listening previews.
- Preserve complete raw metrics and failure reasons; do not emit a shortlist when no candidate passes.
- Reuse existing dependencies; defer dependency declaration cleanup until the whole project is complete.
- Follow strict test-first RED → GREEN → REFACTOR cycles.

---

## Task19-A: Evaluation Configuration and Fixed Suites

**Files:**
- Create: `src/voice_pipeline/evaluation/__init__.py`
- Create: `src/voice_pipeline/evaluation/config.py`
- Create: `src/voice_pipeline/evaluation/suite.py`
- Create: `configs/eval/zh.txt`
- Create: `configs/eval/ja.txt`
- Create: `configs/eval/en.txt`
- Create: `configs/eval/mixed.txt`
- Modify: `src/voice_pipeline/training/config.py`
- Modify: `configs/train.example.yaml`
- Test: `tests/test_eval_suite.py`
- Test: `tests/test_training_config.py`

**Interfaces:**
- Consumes: `TrainingConfig.from_yaml(path: Path, project_root: Path | None) -> TrainingConfig` and profile paths from `ModelProfile`.
- Produces: `EvaluationConfig`, `EvaluationSuite`, `EvaluationCase`, and `load_evaluation_suite(config: EvaluationConfig) -> EvaluationSuite`.
- `TrainingConfig.evaluation` is `EvaluationConfig | None`.

- [ ] **Step 1: Write failing suite and configuration tests**

```python
def test_suite_uses_file_language_and_stable_ids(tmp_path: Path):
    zh = tmp_path / "zh.txt"
    zh.write_text("# comment\n你好，欢迎参加测试。\n\n今天是二〇二六年。\n", encoding="utf-8")
    config = evaluation_config(tmp_path, suites={"zh": zh})
    suite = load_evaluation_suite(config)
    assert [(case.id, case.language, case.text) for case in suite.cases] == [
        ("zh-001", "zh", "你好，欢迎参加测试。"),
        ("zh-002", "zh", "今天是二〇二六年。"),
    ]

def test_training_config_resolves_evaluator_paths(tmp_path: Path):
    config = TrainingConfig.from_yaml(write_training_yaml(tmp_path), tmp_path)
    assert config.evaluation is not None
    assert config.evaluation.cache_dir == (tmp_path / "hf-cache").resolve()
    assert config.evaluation.pairing.s2_keep == 2
    assert config.evaluation.pairing.shortlist_size == 3
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/test_eval_suite.py tests/test_training_config.py -q --basetemp ..\.pytest_cache\task19-a-red
```

Expected: import/attribute failures because the evaluation config and suite types do not exist.

- [ ] **Step 3: Implement strict immutable configuration**

```python
@dataclass(frozen=True, slots=True)
class EvaluationPairing:
    s2_keep: int = 2
    shortlist_size: int = 3

@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    run_dir: Path
    project_root: Path
    model_name: str
    trained_languages: tuple[str, ...]
    validated_languages: tuple[str, ...]
    reference: BundleReference
    speaker_references: tuple[Path, ...]
    suites: dict[str, Path]
    asr_model: str
    speaker_model: str
    cache_dir: Path
    pairing: EvaluationPairing
    device: str
    precision: str
```

Validate unknown fields, required ZH/JA/EN/Mixed suite paths, reference file values, positive pairing sizes, safe model IDs, device, precision, and path resolution. Derive `model_name` from `experiment.name`, `trained_languages` from `objective.training_languages`, and `validated_languages` from `objective.target_languages`; require those objective fields when evaluation is enabled. Preserve `evaluation: {enabled: false}` as `TrainingConfig.evaluation is None`.

- [ ] **Step 4: Implement suite parsing and checked-in text suites**

```python
@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    language: str
    text: str

@dataclass(frozen=True, slots=True)
class EvaluationSuite:
    cases: tuple[EvaluationCase, ...]

def load_evaluation_suite(config: EvaluationConfig) -> EvaluationSuite:
    cases = []
    for language in ("zh", "ja", "en", "mixed"):
        texts = parse_suite_file(config.suites[language])
        cases.extend(EvaluationCase(f"{language}-{i:03d}", language, text) for i, text in enumerate(texts, 1))
    return EvaluationSuite(tuple(cases))
```

Each suite must contain at least one effective line, strip UTF-8 BOM/whitespace, ignore blank/comment lines, and reject duplicate normalized sentences within one language.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Step 2 command with `task19-a-green`. Expected: all selected tests pass.

- [ ] **Step 6: Commit Task19-A**

```powershell
git add src/voice_pipeline/evaluation src/voice_pipeline/training/config.py configs/eval configs/train.example.yaml tests/test_eval_suite.py tests/test_training_config.py
git commit -m "feat: add evaluation config and suites"
```

---

## Task19-B: Checkpoint Discovery and One-Candidate Bundle Materialization

**Files:**
- Create: `src/voice_pipeline/evaluation/candidates.py`
- Modify: `src/voice_pipeline/exporting/bundles.py`
- Test: `tests/test_eval_candidates.py`
- Test: `tests/test_bundle.py`

**Interfaces:**
- Consumes: S1/S2 checkpoint envelopes, `ProfileRegistry`, `BundleReference`, `BundleLanguages`, and checkpoint export functions.
- Produces: `CheckpointRef(kind: str, path: Path, step: int | None, sha256: str, is_base: bool)`, `CandidatePair`, `discover_checkpoints(config: EvaluationConfig) -> tuple[tuple[CheckpointRef, ...], tuple[CheckpointRef, ...]]`, `stage_one_pairs(...)`, `stage_two_pairs(...)`, and public `build_candidate_bundle(...) -> Path`.

- [ ] **Step 1: Write failing discovery and pairing tests**

```python
def test_stage_two_only_crosses_s1_with_retained_s2(tmp_path: Path):
    base_s1, s1_100, s1_200 = refs("s1", None, 100, 200)
    s2_100, s2_200, s2_300 = refs("s2", 100, 200, 300)
    pairs = stage_two_pairs((base_s1, s1_100, s1_200), (s2_100, s2_300))
    assert {(pair.s1.step, pair.s2.step) for pair in pairs} == {
        (None, 100), (100, 100), (200, 100),
        (None, 300), (100, 300), (200, 300),
    }
    assert all(pair.s2 != s2_200 for pair in pairs)

def test_discovery_rejects_filename_step_mismatch(tmp_path: Path):
    write_s1_checkpoint(tmp_path / "step-00000200.pt", embedded_step=100)
    with pytest.raises(ValueError, match="filename does not match embedded step"):
        discover_s1_checkpoints(tmp_path)
```

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/test_eval_candidates.py tests/test_bundle.py -q --basetemp ..\.pytest_cache\task19-b-red
```

Expected: missing candidate discovery API.

- [ ] **Step 3: Implement validated discovery and deterministic pairing**

```python
@dataclass(frozen=True, slots=True)
class CandidatePair:
    key: str
    s1: CheckpointRef
    s2: CheckpointRef

def stage_one_pairs(base_s1, s2_refs):
    return tuple(CandidatePair(pair_key(base_s1, s2), base_s1, s2) for s2 in s2_refs)

def stage_two_pairs(s1_refs, retained_s2):
    return tuple(
        CandidatePair(pair_key(s1, s2), s1, s2)
        for s2 in retained_s2
        for s1 in s1_refs
    )
```

Load each internal checkpoint on CPU, validate envelope version/profile/embedded step/filename agreement, hash its bytes, sort by `(is_base desc, step asc)`, and reject duplicate steps. Add the official base S1/S2 as explicit refs with `step=None`.

- [ ] **Step 4: Expose minimal one-bundle export API**

Refactor existing `_build_candidate` without changing `export_candidates` behavior:

```python
def build_candidate_bundle(
    destination: Path,
    *,
    profile: str,
    model_name: str,
    reference: BundleReference,
    languages: BundleLanguages,
    candidate: Candidate,
    project_root: Path,
) -> Path:
    # atomically build and validate one ModelBundle
```

The function must reject an existing destination, use the same S1/S2 conversion functions as normal export, and leave no temporary tree after failure.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Step 2 command with `task19-b-green`. Expected: all tests pass, including existing bundle regressions.

- [ ] **Step 6: Commit Task19-B**

```powershell
git add src/voice_pipeline/evaluation/candidates.py src/voice_pipeline/exporting/bundles.py tests/test_eval_candidates.py tests/test_bundle.py
git commit -m "feat: discover evaluation checkpoint pairs"
```

---

## Task19-C: Deterministic Generation Runner and Resume Manifest

**Files:**
- Create: `src/voice_pipeline/evaluation/runner.py`
- Modify: `src/voice_pipeline/evaluation/__init__.py`
- Test: `tests/test_eval_runner.py`

**Interfaces:**
- Consumes: `EvaluationConfig`, `EvaluationSuite`, `CandidatePair`, `build_candidate_bundle`, `InferenceSession.load`, `run_synthesis_job`.
- Produces: `GeneratedSample`, `CandidateGeneration`, and `generate_candidate(config, pair, suite, session_factory=InferenceSession.load) -> CandidateGeneration`.

- [ ] **Step 1: Write failing observable runner tests**

```python
def test_generation_manifest_resumes_completed_wavs(tmp_path: Path):
    first = generate_candidate(config(tmp_path), pair(tmp_path), suite(), session_factory=fake_session_factory)
    second = generate_candidate(config(tmp_path), pair(tmp_path), suite(), session_factory=fail_if_loaded)
    assert first.generated == 4
    assert second.generated == 0
    assert second.resumed == 4
    assert [sample.case_id for sample in second.samples] == ["zh-001", "ja-001", "en-001", "mixed-001"]

def test_changed_checkpoint_hash_does_not_reuse_generation(tmp_path: Path):
    generate_candidate(config(tmp_path), pair(tmp_path, s2_hash="a" * 64), suite(), session_factory=fake_session_factory)
    with pytest.raises(ValueError, match="does not match"):
        generate_candidate(config(tmp_path), pair(tmp_path, s2_hash="b" * 64), suite(), session_factory=fake_session_factory)
```

- [ ] **Step 2: Run focused test and verify RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/test_eval_runner.py -q --basetemp ..\.pytest_cache\task19-c-red
```

Expected: missing runner API.

- [ ] **Step 3: Implement deterministic generation**

```python
@dataclass(frozen=True, slots=True)
class GeneratedSample:
    case_id: str
    language: str
    target_text: str
    wav_path: Path
    sha256: str

def generate_candidate(config, pair, suite, session_factory=InferenceSession.load):
    # build <run>/evaluation/work/<pair.key>/bundle once
    # synthesize one WAV per case with fixed seed/options
    # atomically update manifest after every completed WAV
    # validate hashes before reporting a resumed sample
```

Manifest signature includes suite text/language, pair checkpoint hashes, reference hash, profile, fixed seed, and all synthesis options. Reuse `run_synthesis_job`; do not duplicate chunking or WAV assembly. Store relative paths inside the manifest.

- [ ] **Step 4: Add per-sample failure recording**

Catch synthesis failures at the case boundary, record `{status: failed, error_type, message}`, continue remaining cases, and return candidate status `failed`. Do not catch `KeyboardInterrupt` or `SystemExit`.

- [ ] **Step 5: Run Task19 tests and regression slice**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/test_eval_suite.py tests/test_eval_candidates.py tests/test_eval_runner.py tests/test_bundle.py tests/test_inference_job.py -q --basetemp ..\.pytest_cache\task19-green
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task19-C and stop for review**

```powershell
git add src/voice_pipeline/evaluation tests/test_eval_runner.py
git commit -m "feat: generate resumable evaluation audio"
```

Task19 review evidence must include focused test counts, `git diff --check`, and confirmation that no evaluator model was imported by standalone inference.

---

## Task20-A: Metric Contracts and Text Error Rates

**Files:**
- Create: `src/voice_pipeline/evaluation/base.py`
- Create: `src/voice_pipeline/evaluation/pronunciation.py`
- Create: `src/voice_pipeline/evaluation/language_consistency.py`
- Test: `tests/test_evaluator_metrics.py`

**Interfaces:**
- Consumes: `GeneratedSample` and structured ASR results.
- Produces: `EvaluationSample`, `MetricResult`, `Transcript`, `character_error_rate`, `word_error_rate`, and language-consistency scoring.

- [ ] **Step 1: Write failing hand-calculated metric tests**

```python
def test_chinese_cer_is_hand_calculated():
    assert character_error_rate("你好世界", "你好世") == pytest.approx(0.25)

def test_english_wer_is_hand_calculated():
    assert word_error_rate("the quick brown fox", "the quick fox") == pytest.approx(0.25)

def test_mixed_language_does_not_require_one_detected_language():
    result = language_consistency("你好 hello", "你好 hello", "mixed", detected_language="zh")
    assert result.available is True
    assert result.value == pytest.approx(1.0)
```

- [ ] **Step 2: Verify RED, implement minimal normalizers/edit distance, verify GREEN**

Use one standard-library dynamic-programming edit-distance helper. Normalize Unicode with `unicodedata.normalize("NFKC", text)`, strip punctuation/space for ZH/JA CER, and lowercase/tokenize English words with a Unicode regex already available in the environment.

Run:

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/test_evaluator_metrics.py -q --basetemp ..\.pytest_cache\task20-a
```

- [ ] **Step 3: Commit Task20-A**

```powershell
git add src/voice_pipeline/evaluation/base.py src/voice_pipeline/evaluation/pronunciation.py src/voice_pipeline/evaluation/language_consistency.py tests/test_evaluator_metrics.py
git commit -m "feat: add evaluation metric contracts"
```

---

## Task20-B: Faster-Whisper, WavLM, and Prosody Adapters

**Files:**
- Create: `src/voice_pipeline/evaluation/asr.py`
- Create: `src/voice_pipeline/evaluation/speaker_similarity.py`
- Create: `src/voice_pipeline/evaluation/prosody.py`
- Modify: `src/voice_pipeline/evaluation/runner.py`
- Test: `tests/test_evaluator_adapters.py`
- Test: `tests/test_eval_runner.py`

**Interfaces:**
- Consumes: evaluator model IDs/cache, generated WAVs, reference WAVs.
- Produces: `WhisperEvaluator.transcribe(path, language) -> Transcript`, `WavLMSpeakerEvaluator.centroid(paths) -> np.ndarray`, `WavLMSpeakerEvaluator.similarity(path, centroid) -> float`, `evaluate_prosody(path, target_text) -> MetricResult`, and `evaluate_generation(...)`.

- [ ] **Step 1: Write failing adapter-boundary tests**

Use tiny real WAV fixtures and narrow fake model objects; assertions target normalized returned values and audio preparation, not mock call counts.

```python
def test_speaker_centroid_is_normalized():
    evaluator = WavLMSpeakerEvaluator.from_components(feature_extractor, embedding_model)
    centroid = evaluator.centroid((reference_a, reference_b))
    assert np.linalg.norm(centroid) == pytest.approx(1.0)

def test_silent_audio_fails_prosody_gate(tmp_path: Path):
    wav = write_wav(tmp_path / "silent.wav", np.zeros(16000, dtype=np.float32), 16000)
    result = evaluate_prosody(wav, "test")
    assert result.available is True
    assert result.details["all_silent"] is True
```

- [ ] **Step 2: Verify RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/test_evaluator_adapters.py -q --basetemp ..\.pytest_cache\task20-b-red
```

- [ ] **Step 3: Implement lazy Faster-Whisper adapter**

Resolve the configured model through `huggingface_hub.snapshot_download(repo_id, cache_dir=...)`, which reuses the current complete cache and downloads only when absent. Instantiate `WhisperModel` on configured device/precision only inside evaluator construction. Force `zh`, `ja`, or `en` for pronunciation transcription; perform a separate short auto-detection request only where language consistency requires it.

- [ ] **Step 4: Implement lazy WavLM adapter**

Load `AutoFeatureExtractor` and `WavLMForXVector` only inside evaluator construction. Read mono audio, resample to 16 kHz, reject empty/non-finite samples, infer under `torch.inference_mode()`, L2-normalize embeddings, average reference vectors, and normalize the centroid again.

- [ ] **Step 5: Implement local prosody and aggregate metrics**

Use frame RMS for pause/silence/energy and librosa pitch extraction for finite voiced F0 values. Return raw details even when a hard-audio flag is raised. Aggregate ASR, speaker, language, and prosody per case and per language into JSON-safe primitives.

- [ ] **Step 6: Verify GREEN and lazy import isolation**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/test_evaluator_metrics.py tests/test_evaluator_adapters.py tests/test_eval_runner.py -q --basetemp ..\.pytest_cache\task20-b-green
$env:PYTHONPATH = 'src'
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -c "import sys; import voice_pipeline.inference; assert 'faster_whisper' not in sys.modules; assert 'voice_pipeline.evaluation.asr' not in sys.modules; assert 'voice_pipeline.evaluation.speaker_similarity' not in sys.modules"
```

- [ ] **Step 7: Commit Task20-B and stop for review**

```powershell
git add src/voice_pipeline/evaluation tests/test_evaluator_adapters.py tests/test_eval_runner.py
git commit -m "feat: score cross-language evaluation audio"
```

---

## Task21-A: Hard Constraints, Ranking, and Reports

**Files:**
- Create: `src/voice_pipeline/evaluation/ranking.py`
- Create: `src/voice_pipeline/evaluation/report.py`
- Test: `tests/test_ranking.py`

**Interfaces:**
- Consumes: candidate aggregate results and `EvaluationConfig` constraints/ranking weights.
- Produces: `ConstraintFailure`, `RankedCandidate`, `rank_candidates(results, config) -> tuple[RankedCandidate, ...]`, `assign_anonymous_ids(...)`, `write_results(...)`, and `write_report(...)`.

- [ ] **Step 1: Write failing hard-gate and priority tests**

```python
def test_high_average_cannot_hide_failed_english_similarity():
    strong_average_bad_en = result("pair-a", sim={"zh": .95, "ja": .96, "en": .60})
    balanced = result("pair-b", sim={"zh": .84, "ja": .86, "en": .83})
    ranked = rank_candidates((strong_average_bad_en, balanced), config(min_sim={"zh": .80, "ja": .80, "en": .80}))
    assert ranked[0].pair_key == "pair-b"
    assert ranked[1].eligible is False
    assert ranked[1].failures[0].metric == "speaker_similarity.en"

def test_anonymous_ids_are_stable_for_same_ranking():
    assert [item.public_id for item in assign_anonymous_ids(ranked_results())] == ["candidate_A", "candidate_B", "candidate_C"]
```

- [ ] **Step 2: Verify RED, implement gates/ranking/report, verify GREEN**

Validate configured weights sum to 1.0. Apply all hard constraints before scoring. Normalize error metrics as `1 - min(error, 1)`, prioritize `WorstSim` inside the speaker component, preserve every raw metric, and use pair key as the final deterministic tie-breaker.

Run:

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/test_ranking.py -q --basetemp ..\.pytest_cache\task21-a
```

- [ ] **Step 3: Commit Task21-A**

```powershell
git add src/voice_pipeline/evaluation/ranking.py src/voice_pipeline/evaluation/report.py tests/test_ranking.py
git commit -m "feat: rank evaluation candidates"
```

---

## Task21-B: Full Orchestrator, Shortlist, Candidate Export, and Listening Audio

**Files:**
- Create: `src/voice_pipeline/evaluation/pipeline.py`
- Create: `src/voice_pipeline/cli/evaluate.py`
- Modify: `src/voice_pipeline/cli/main.py`
- Modify: `src/voice_pipeline/evaluation/__init__.py`
- Test: `tests/test_evaluation_pipeline.py`
- Test: `tests/test_evaluate_cli.py`

**Interfaces:**
- Consumes: all Task19/20/21-A interfaces plus `Shortlist`, `export_candidates`, `ModelBundle`, and `run_synthesis_job`.
- Produces: `run_evaluation(config: EvaluationConfig) -> EvaluationOutcome` and CLI `voice-pipeline evaluate --config PATH --project-root PATH`.

- [ ] **Step 1: Write failing end-to-end orchestration tests**

```python
def test_pipeline_filters_s2_before_crossing_s1_and_exports_shortlist(tmp_path: Path):
    outcome = run_evaluation(config(tmp_path), services=fake_services())
    assert outcome.evaluated_pair_steps == [
        (None, None), (None, 100), (None, 200),
        (100, 100), (200, 100),
    ]
    assert [candidate.id for candidate in Shortlist.load(outcome.run_dir, tmp_path).candidates] == [
        "candidate_A", "candidate_B", "candidate_C"
    ]
    for candidate in ("candidate_A", "candidate_B", "candidate_C"):
        ModelBundle.load(outcome.run_dir / "export" / "candidates" / candidate)
        assert sorted(path.name for path in (outcome.run_dir / "evaluation" / "listening" / candidate).glob("*.wav")) == ["en.wav", "ja.wav", "zh.wav"]

def test_pipeline_does_not_write_shortlist_when_all_fail(tmp_path: Path):
    with pytest.raises(EvaluationError, match="no candidate passed"):
        run_evaluation(config(tmp_path), services=all_failed_services())
    assert not (run_dir(tmp_path) / "evaluation" / "shortlist.yaml").exists()
```

- [ ] **Step 2: Verify RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/test_evaluation_pipeline.py tests/test_evaluate_cli.py -q --basetemp ..\.pytest_cache\task21-b-red
```

- [ ] **Step 3: Implement staged orchestration**

Run stage one, rank eligible S2 pairs, retain exactly `min(s2_keep, eligible_count)`, run/reuse stage-two pairs, rank all eligible final pairs, and choose exactly `min(shortlist_size, eligible_count)`. Persist `results.json` and `report.md` before attempting shortlist export.

- [ ] **Step 4: Write strict shortlist atomically**

Write schema version 1 using project-relative S1/S2 paths, explicit reference/languages, and stable public IDs. Immediately reload with `Shortlist.load` before continuing.

- [ ] **Step 5: Export every shortlisted bundle and generate previews**

Call existing `export_candidates` once. Load every resulting `ModelBundle` and generate the first ZH/JA/EN suite case as `listening/<candidate>/<language>.wav`. Write an anonymous listening manifest containing public IDs, text, language, relative WAV path, and hash; keep the private checkpoint mapping only in the full evaluator report.

- [ ] **Step 6: Add evaluate CLI and concise progress output**

```python
def evaluate_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    project_root: Path = typer.Option(Path.cwd(), "--project-root", exists=True, file_okay=False),
) -> None:
    training = TrainingConfig.from_yaml(config, project_root.resolve())
    if training.evaluation is None:
        raise ValueError("evaluation must be enabled")
    outcome = run_evaluation(training.evaluation)
    typer.echo(f"shortlisted {len(outcome.shortlisted)} candidates in {outcome.run_dir / 'evaluation'}")
```

- [ ] **Step 7: Verify GREEN**

Run the Step 2 command with `task21-b-green`, then run CLI help:

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m voice_pipeline.cli.main evaluate --help
```

- [ ] **Step 8: Commit Task21-B**

```powershell
git add src/voice_pipeline/evaluation src/voice_pipeline/cli tests/test_evaluation_pipeline.py tests/test_evaluate_cli.py
git commit -m "feat: produce human evaluation shortlist"
```

---

## Task21-C: Documentation, Real-Model Smoke, Full Regression, and Cleanup

**Files:**
- Modify: `README.md`
- Modify: `README_使用指南.md`
- Modify: `configs/train.example.yaml`
- Create: `docs/evaluation.md`
- Modify tests only if the real smoke reveals a reproducible product defect; add the failing regression test before the fix.

**Interfaces:**
- Consumes: final evaluate/export CLI.
- Produces: Chinese operator workflow from training checkpoints through human selection.

- [ ] **Step 1: Document exact commands and result interpretation**

Document cache path, evaluator download behavior, VRAM expectations, thresholds, resume behavior, report locations, anonymous listening, and the final human command:

```powershell
voice-pipeline evaluate --config configs/train.yaml --project-root .
voice-pipeline export --run runs/<speaker> --project-root . --select candidate_A
```

- [ ] **Step 2: Run real evaluator model smoke**

Use the configured Hugging Face cache to load Faster-Whisper locally. Download WavLM into the same cache when absent. Run transcription on a generated or checked-in small speech fixture and speaker embedding on a real reference WAV; assert finite outputs and unit-normalized embeddings.

- [ ] **Step 3: Run one real v2ProPlus evaluation smoke**

Use real pretrained/training resources with the smallest fixed suite that exercises ZH/JA/EN/Mixed. Confirm valid WAVs, raw metrics, report, shortlist, CandidateBundles, and ZH/JA/EN previews. This smoke may use one inference sample per language; it must not train extra steps or leave additional test weights.

- [ ] **Step 4: Run full regression and structural checks**

```powershell
New-Item -ItemType Directory -Force '..\.pytest_cache' | Out-Null
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest -q --basetemp ..\.pytest_cache\full-evaluation
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m compileall -q src tests
git diff --check
```

Expected: no failures or skips attributable to missing evaluator functionality, and no whitespace errors.

- [ ] **Step 5: Clean generated caches and test artifacts**

Remove only verified project-local `.pytest_cache`, `__pycache__`, temporary evaluation work trees, and additional test weights. Preserve official weights, evaluator caches, formal evaluation results, shortlisted CandidateBundles, and listening WAVs.

- [ ] **Step 6: Commit documentation and finish**

```powershell
git add README.md README_使用指南.md configs/train.example.yaml docs/evaluation.md
git commit -m "docs: add automatic evaluation workflow"
git status --short
```

Expected: clean working tree.

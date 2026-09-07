# Pipeline YAML Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable `voice-pipeline run` command that executes the valid v2ProPlus stages in order, preserves raw training artifacts for human review, and cleans them only after an explicit human-selected candidate is successfully promoted.

**Architecture:** A strict YAML parser defines the referenced training config and canonical stage subsequence. An atomic JSON state file records stage transitions, an in-process orchestrator calls existing core APIs through one stage dispatcher, and a separately tested cleanup module validates all durable evaluation/candidate artifacts before deleting exact run-owned paths.

**Tech Stack:** Python 3.12, dataclasses, pathlib, JSON, PyYAML, Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-09-07-pipeline-orchestration-design.md`

## Global Constraints

- Work directly on `main`; do not create branches, worktrees, Conda environments, or ZIP archives.
- Use `D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe` for Python and pytest.
- Accepted stages are exactly `preprocess`, `s2`, `s1`, and `evaluate`, in that canonical order with no duplicates.
- Do not infer checkpoints; S1/S2 recovery remains explicit through `resume_from` in the referenced training YAML.
- Never automate human candidate selection or promotion.
- `run` and `evaluate` always keep preprocessing and raw checkpoints for human review.
- Strong cleanup runs only after explicit `export --select` promotion succeeds and all preserved artifacts validate.
- Never delete dataset inputs, official pretrained weights, evaluator weights, source files, or promoted ModelBundles.
- Use strict RED → GREEN → REFACTOR cycles and stop for user review after each implementation task.

## File Structure

- Create `src/voice_pipeline/pipeline/config.py`: strict pipeline YAML parsing and canonical stage validation.
- Create `src/voice_pipeline/pipeline/state.py`: atomic state creation, validation, and transitions.
- Create `src/voice_pipeline/pipeline/orchestrator.py`: default in-process stage dispatch and ordered execution.
- Create `src/voice_pipeline/pipeline/cleanup.py`: durable-artifact validation and exact run cleanup.
- Create `src/voice_pipeline/pipeline/__init__.py`: small public surface.
- Create `src/voice_pipeline/cli/run.py`: Typer adapter only.
- Modify `src/voice_pipeline/cli/main.py`: register `run`.
- Create `configs/pipeline.example.yaml`: minimal operator example.
- Create `tests/test_pipeline_config.py`: parser and state contract tests.
- Create `tests/test_orchestrator.py`: order, stop, skip, and retry behavior.
- Create `tests/test_pipeline_cleanup.py`: preservation and destructive-boundary tests.
- Create `tests/test_run_cli.py`: CLI registration and exit behavior.
- Modify `README.md` and `README_使用指南.md`: exact command, state semantics, and cleanup warning.

---

### Task22-A: Strict Pipeline Config and Atomic State

**Files:**
- Create: `src/voice_pipeline/pipeline/__init__.py`
- Create: `src/voice_pipeline/pipeline/config.py`
- Create: `src/voice_pipeline/pipeline/state.py`
- Create: `tests/test_pipeline_config.py`

**Interfaces:**
- Produces: `PipelineSpec.load(path: Path, project_root: Path) -> PipelineSpec`.
- Produces: `PipelineSpec.path`, `PipelineSpec.training_config`, and `PipelineSpec.stages`.
- Produces: `PipelineState.load_or_create(path: Path, spec: PipelineSpec) -> PipelineState`.
- Produces: `PipelineState.status(stage)`, `start(stage)`, `complete(stage)`, and `fail(stage, error)`; every mutation is atomically persisted.

- [ ] **Step 1: Write failing parser tests**

Create tests that express the public API before production files exist:

```python
from pathlib import Path
import pytest

from voice_pipeline.pipeline.config import PipelineSpec


def _write_pipeline(tmp_path: Path, stages: str) -> Path:
    (tmp_path / "train.yaml").write_text("training", encoding="utf-8")
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        "schema_version: 1\nconfig: train.yaml\nstages:\n" + stages,
        encoding="utf-8",
    )
    return path


def test_pipeline_spec_resolves_config_and_preserves_canonical_order(tmp_path: Path):
    spec = PipelineSpec.load(
        _write_pipeline(tmp_path, "  - preprocess\n  - s2\n  - evaluate\n"),
        tmp_path,
    )
    assert spec.training_config == (tmp_path / "train.yaml").resolve()
    assert spec.stages == ("preprocess", "s2", "evaluate")


@pytest.mark.parametrize(
    "stages, message",
    [
        ("", "non-empty"),
        ("  - preprocess\n  - preprocess\n", "duplicate"),
        ("  - preprocess\n  - export\n", "unknown"),
        ("  - s1\n  - s2\n", "canonical order"),
    ],
)
def test_pipeline_spec_rejects_invalid_stages(tmp_path: Path, stages: str, message: str):
    with pytest.raises(ValueError, match=message):
        PipelineSpec.load(_write_pipeline(tmp_path, stages), tmp_path)
```

Also test a non-mapping root, schema versions other than integer `1`, unknown top-level keys,
missing/non-file training config, and a training config path that escapes `project_root`.

- [ ] **Step 2: Run parser tests and verify RED**

Run:

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/test_pipeline_config.py -q --basetemp ..\.pytest_cache\task22-a-red
```

Expected: collection fails because `voice_pipeline.pipeline.config` does not exist.

- [ ] **Step 3: Implement the strict parser**

Implement a frozen slots dataclass and a three-field strict parser:

```python
CANONICAL_STAGES = ("preprocess", "s2", "s1", "evaluate")


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    path: Path
    training_config: Path
    stages: tuple[str, ...]

    @classmethod
    def load(cls, path: Path, project_root: Path) -> "PipelineSpec":
        path = Path(path).resolve()
        root = Path(project_root).resolve()
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ValueError(f"invalid pipeline YAML {path}: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError("pipeline YAML root must be a mapping")
        unknown = set(payload) - {"schema_version", "config", "stages"}
        if unknown:
            raise ValueError(f"unknown pipeline field: {', '.join(sorted(unknown))}")
        if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
            raise ValueError("pipeline.schema_version must be 1")
        raw_config = payload.get("config")
        if not isinstance(raw_config, str) or not raw_config.strip():
            raise ValueError("pipeline.config must be a non-empty path")
        training_config = (root / raw_config).resolve()
        try:
            training_config.relative_to(root)
        except ValueError as error:
            raise ValueError("pipeline.config must remain inside project_root") from error
        if not training_config.is_file():
            raise FileNotFoundError(training_config)
        stages = payload.get("stages")
        if not isinstance(stages, list) or not stages or any(not isinstance(stage, str) for stage in stages):
            raise ValueError("pipeline.stages must be a non-empty list of strings")
        if len(set(stages)) != len(stages):
            raise ValueError("pipeline.stages must not contain duplicates")
        unknown_stages = set(stages) - set(CANONICAL_STAGES)
        if unknown_stages:
            raise ValueError(f"unknown pipeline stage: {', '.join(sorted(unknown_stages))}")
        indexes = [CANONICAL_STAGES.index(stage) for stage in stages]
        if indexes != sorted(indexes):
            raise ValueError("pipeline.stages must follow canonical order")
        return cls(path, training_config, tuple(stages))
```

- [ ] **Step 4: Write failing state tests**

Append tests using a real `PipelineSpec` and state file:

```python
from voice_pipeline.pipeline.state import PipelineState


def test_pipeline_state_persists_transitions_and_failure(tmp_path: Path):
    spec = PipelineSpec.load(
        _write_pipeline(tmp_path, "  - preprocess\n  - s2\n"), tmp_path
    )
    path = tmp_path / "run" / "pipeline-state.json"
    state = PipelineState.load_or_create(path, spec)
    state.start("preprocess")
    state.complete("preprocess")
    state.start("s2")
    state.fail("s2", RuntimeError("CUDA overflow"))

    loaded = PipelineState.load_or_create(path, spec)
    assert loaded.status("preprocess") == "completed"
    assert loaded.status("s2") == "failed"
    assert loaded.failure == {"stage": "s2", "type": "RuntimeError", "message": "CUDA overflow"}


def test_pipeline_state_rejects_identity_mismatch(tmp_path: Path):
    first = PipelineSpec.load(
        _write_pipeline(tmp_path, "  - preprocess\n  - s2\n"), tmp_path
    )
    state_path = tmp_path / "run" / "pipeline-state.json"
    PipelineState.load_or_create(state_path, first)
    changed_path = tmp_path / "changed.yaml"
    changed_path.write_text(
        "schema_version: 1\nconfig: train.yaml\nstages: [preprocess, s2, s1]\n",
        encoding="utf-8",
    )
    changed = PipelineSpec.load(changed_path, tmp_path)
    with pytest.raises(ValueError, match="identity"):
        PipelineState.load_or_create(state_path, changed)
```

Monkeypatch `os.replace` in a separate test and assert every state mutation writes a sibling
temporary file and publishes it through `os.replace`.

- [ ] **Step 5: Run state tests and verify RED**

Run the Step 2 command with basetemp `task22-a-state-red`.

Expected: parser tests pass and state tests fail because `pipeline.state` does not exist.

- [ ] **Step 6: Implement atomic state**

Use one JSON schema and no history/event log:

```json
{
  "schema_version": 1,
  "identity": {
    "pipeline": "D:/project/configs/pipeline.yaml",
    "config": "D:/project/configs/train.yaml",
    "stages": ["preprocess", "s2", "s1", "evaluate"]
  },
  "stages": {
    "preprocess": "completed",
    "s2": "failed",
    "s1": "pending",
    "evaluate": "pending"
  },
  "failure": {"stage": "s2", "type": "RuntimeError", "message": "CUDA overflow"}
}
```

Write UTF-8 JSON to `.<filename>.<uuid>.tmp`, flush by closing the file, and publish with
`os.replace`. Validate every field when loading. `start`, `complete`, and `fail` reject stage
names not declared by the spec. `start` clears the previous failure; `complete` leaves it null.

- [ ] **Step 7: Run Task22-A GREEN tests**

Run the Step 2 command with basetemp `task22-a-green`.

Expected: all `tests/test_pipeline_config.py` tests pass.

- [ ] **Step 8: Commit Task22-A**

```powershell
git add -- src/voice_pipeline/pipeline tests/test_pipeline_config.py
git diff --cached --check
git commit -m "feat: add pipeline config and state"
```

Stop for user review.

---

### Task22-B: Ordered In-Process Stage Orchestration

**Files:**
- Create: `src/voice_pipeline/pipeline/orchestrator.py`
- Modify: `src/voice_pipeline/pipeline/__init__.py`
- Create: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `PipelineSpec` and `PipelineState` from Task22-A.
- Produces: `PipelineOutcome(executed, skipped, state_path, cleaned)`.
- Produces: `run_pipeline(path: Path, project_root: Path, *, execute_stage=execute_stage) -> PipelineOutcome`; Task22-C adds the cleanup injection after its implementation exists.
- Produces: `execute_stage(stage: str, training_config: Path, project_root: Path) -> None`.

- [ ] **Step 1: Write failing order, failure, and resume tests**

Use an injected callable so no model is loaded:

```python
def test_run_pipeline_executes_declared_stages_in_order(tmp_path: Path, monkeypatch):
    pipeline = _pipeline_fixture(tmp_path, ["preprocess", "s2", "s1"])
    calls = []
    monkeypatch.setattr(orchestrator, "_run_dir", lambda config, root: tmp_path / "runs" / "speaker")

    outcome = run_pipeline(
        pipeline,
        tmp_path,
        execute_stage=lambda stage, config, root: calls.append(stage),
    )

    assert calls == ["preprocess", "s2", "s1"]
    assert outcome.executed == ("preprocess", "s2", "s1")
    assert outcome.skipped == ()
    assert outcome.cleaned is False


def test_failure_stops_later_stages_and_rerun_skips_completed(tmp_path: Path, monkeypatch):
    pipeline = _pipeline_fixture(tmp_path, ["preprocess", "s2", "s1"])
    run_dir = tmp_path / "runs" / "speaker"
    monkeypatch.setattr(orchestrator, "_run_dir", lambda config, root: run_dir)
    first_calls = []

    def fail_s2(stage, config, root):
        first_calls.append(stage)
        if stage == "s2":
            raise RuntimeError("failed S2")

    with pytest.raises(PipelineStageError, match="s2.*failed S2"):
        run_pipeline(pipeline, tmp_path, execute_stage=fail_s2)
    assert first_calls == ["preprocess", "s2"]

    retry_calls = []
    outcome = run_pipeline(
        pipeline,
        tmp_path,
        execute_stage=lambda stage, config, root: retry_calls.append(stage),
    )
    assert retry_calls == ["s2", "s1"]
    assert outcome.skipped == ("preprocess",)
```

Add a test that manually changes a persisted `running` stage and confirms the next run retries it.

- [ ] **Step 2: Run orchestrator tests and verify RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/test_orchestrator.py -q --basetemp ..\.pytest_cache\task22-b-red
```

Expected: collection fails because `pipeline.orchestrator` does not exist.

- [ ] **Step 3: Implement ordered execution**

Add `PipelineStageError` to `common/errors.py` as a `VoicePipelineError` subclass.
`run_pipeline` must:

```python
spec = PipelineSpec.load(path, project_root)
run_dir = _run_dir(spec.training_config, project_root)
state = PipelineState.load_or_create(run_dir / "pipeline-state.json", spec)
for stage in spec.stages:
    if state.status(stage) == "completed":
        skipped.append(stage)
        continue
    state.start(stage)
    try:
        execute_stage(stage, spec.training_config, project_root)
    except Exception as error:
        state.fail(stage, error)
        raise PipelineStageError(f"stage {stage} failed: {error}") from error
    state.complete(stage)
    executed.append(stage)
```

Do not catch `KeyboardInterrupt` or `SystemExit`; an interrupted `running` state is intentionally
retried next time.

- [ ] **Step 4: Write failing default-dispatch tests**

Monkeypatch the imported core factories/trainers/evaluator and verify:

- `preprocess` reads the manifest, runs all preprocessing stages, and publishes indexes;
- `s2` constructs only `S2Trainer` with the configured `resume_from`;
- `s1` constructs only `S1Trainer` with the configured `resume_from`;
- `evaluate` rejects disabled evaluation and otherwise calls `run_evaluation`;
- an unknown dispatcher stage raises `ValueError` even though the YAML parser normally blocks it.

The tests assert calls on the actual dispatcher boundary, not Typer output.

- [ ] **Step 5: Run dispatcher tests and verify RED**

Run the Step 2 command with basetemp `task22-b-dispatch-red`.

Expected: orchestration tests pass, dispatcher tests fail because `execute_stage` lacks the core
stage behavior.

- [ ] **Step 6: Implement lazy in-process dispatch**

Import heavy training/evaluation modules inside the selected branch. Reuse the exact existing core
calls:

```python
if stage == "preprocess":
    config = PreprocessConfig.from_yaml(training_config)
    manifest = read_manifest_records(config.manifest)
    pipeline = build_preprocess_pipeline(config)
    summary = pipeline.run(manifest.records, manifest.issues)
    publish_training_indexes(pipeline.context.preprocess_dir, manifest.records, set(summary.valid_sample_ids))
elif stage in {"s2", "s1"}:
    config = TrainingConfig.from_yaml(training_config, project_root)
    # require selected stage enabled, then call the matching trainer with its explicit resume path
elif stage == "evaluate":
    config = TrainingConfig.from_yaml(training_config, project_root)
    # require config.evaluation, then call run_evaluation(config.evaluation)
else:
    raise ValueError(f"unknown pipeline stage: {stage}")
```

Do not call private Typer command functions and do not spawn subprocesses.

- [ ] **Step 7: Run Task22-B GREEN and regression slice**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/test_orchestrator.py tests/test_preprocess_cli.py tests/test_train_cli.py tests/test_evaluate_cli.py -q --basetemp ..\.pytest_cache\task22-b-green
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit Task22-B**

```powershell
git add -- src/voice_pipeline/common/errors.py src/voice_pipeline/pipeline tests/test_orchestrator.py
git diff --cached --check
git commit -m "feat: orchestrate pipeline stages"
```

Stop for user review.

---

### Task22-C: Validated Strong Cleanup

**Files:**
- Create: `src/voice_pipeline/pipeline/cleanup.py`
- Modify: `src/voice_pipeline/pipeline/orchestrator.py`
- Modify: `src/voice_pipeline/pipeline/__init__.py`
- Create: `tests/test_pipeline_cleanup.py`
- Modify: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: a completed run directory produced by `evaluate`.
- Produces: `CleanupResult(removed: tuple[Path, ...])`.
- Produces: `cleanup_successful_run(run_dir: Path, project_root: Path) -> CleanupResult`.

- [ ] **Step 1: Write failing validation-and-preservation test**

Create a real minimal valid `ModelBundle` under `run/export/candidates/candidate_A`, a strict
`shortlist.yaml` naming `candidate_A`, all four report/result files, and a listening manifest with
real SHA-256 values for `zh.wav`, `ja.wav`, and `en.wav`. Also create raw and debris files:

```text
run/preprocess/features.pt
run/training/s1/checkpoints/step.pt
run/training/s2/checkpoints/step.pt
run/evaluation/generated/raw.wav
run/evaluation/work/bundle/model.yaml
run/unneeded.log
run/evaluation/listening/candidate_A/{zh,ja,en}.wav
run/export/candidates/candidate_A/{model.yaml,metadata.json,weights/,reference/}
run/pipeline-state.json
```

Then assert:

```python
result = cleanup_successful_run(run, project_root)

assert not (run / "preprocess").exists()
assert not (run / "training").exists()
assert not (run / "evaluation" / "generated").exists()
assert not (run / "evaluation" / "work").exists()
assert not (run / "unneeded.log").exists()
ModelBundle.load(run / "export" / "candidates" / "candidate_A")
assert (run / "evaluation" / "report.md").is_file()
assert (run / "evaluation" / "listening" / "candidate_A" / "zh.wav").is_file()
assert (run / "pipeline-state.json").is_file()
assert result.removed
```

- [ ] **Step 2: Run cleanup test and verify RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/test_pipeline_cleanup.py -q --basetemp ..\.pytest_cache\task22-c-red
```

Expected: collection fails because `pipeline.cleanup` does not exist.

- [ ] **Step 3: Implement preserved-artifact validation**

Before deleting anything, validate all of the following in one read-only pass:

1. The four report/result files and `shortlist.yaml` are files.
2. The shortlist YAML has schema version 1 and a non-empty unique list of safe candidate IDs.
3. `export/candidates` contains exactly those candidate directories and every directory passes
   `ModelBundle.load` with matching `metadata.candidate_id`.
4. `listening/manifest.json` has schema version 1 and exactly those candidate IDs.
5. Every candidate has exactly `zh`, `ja`, and `en` samples; each relative WAV path remains under
   `evaluation/listening`, is non-empty, and its SHA-256 equals the manifest.

Raise `ValueError` before the first deletion when any invariant fails. Use `yaml.safe_load`,
`json.loads`, `hashlib.sha256`, and existing `ModelBundle.load`; add no dependency.

- [ ] **Step 4: Implement exact cleanup boundaries**

After validation, resolve `run_dir` and reject it if it equals `project_root`, its filesystem root,
or is not a directory. Use `Path.relative_to(run_dir)` before every removal. Preserve only:

```python
root_keep = {"pipeline-state.json", "evaluation", "export"}
evaluation_keep = {
    "stage1-report.md", "stage1-results.json", "report.md", "results.json",
    "shortlist.yaml", "listening",
}
export_keep = {"candidates"}
```

Delete every other direct child in those three scopes using `shutil.rmtree` for directories and
`Path.unlink` for files. This removes raw preprocessing/training checkpoints, generated evaluation
audio, `work`, temporary/backup trees, and run-local debris while never traversing dataset/model
locations outside `run_dir`.

- [ ] **Step 5: Write failing safety tests**

Add parameterized tests proving cleanup performs zero deletions when:

- a report is missing;
- a candidate bundle is invalid;
- candidate IDs differ between shortlist/export/listening;
- a preview hash differs;
- a preview path attempts `../` escape;
- `run_dir == project_root`;
- validation is invoked before an evaluation exists.

Snapshot all files before the call and assert the snapshot is unchanged after every failure.

- [ ] **Step 6: Run safety tests and make them GREEN**

Run the Step 2 command with basetemp `task22-c-green`.

Expected: all cleanup and no-deletion-on-validation-failure tests pass.

- [ ] **Step 7: Connect cleanup to explicit human promotion**

Keep `run_pipeline` free of cleanup so evaluation completion retains raw checkpoints for listening
and possible continuation. After `export --select <candidate>` successfully promotes the selected
candidate outside the run directory, call `cleanup_successful_run(run, project_root)`. Promotion
failure must never clean. A cleanup failure reports that promotion succeeded while preserving the
final model and any training resources not removed. Reject a final model root inside the run.

- [ ] **Step 8: Run Task22-C regression slice**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/test_pipeline_cleanup.py tests/test_orchestrator.py tests/test_bundle.py tests/test_evaluation_pipeline.py tests/test_export_cli.py -q --basetemp ..\.pytest_cache\task22-c-regression
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit Task22-C**

```powershell
git add -- src/voice_pipeline/pipeline tests/test_pipeline_cleanup.py tests/test_orchestrator.py
git diff --cached --check
git commit -m "feat: clean completed pipeline runs"
```

Stop for user review.

---

### Task22-D: CLI, Operator Documentation, and Final Verification

**Files:**
- Create: `src/voice_pipeline/cli/run.py`
- Modify: `src/voice_pipeline/cli/main.py`
- Create: `configs/pipeline.example.yaml`
- Create: `tests/test_run_cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`
- Modify: `README_使用指南.md`

**Interfaces:**
- Consumes: `run_pipeline` from Task22-B/C.
- Produces: CLI `voice-pipeline run PIPELINE --project-root PATH`.

- [ ] **Step 1: Write failing CLI tests**

```python
def test_run_cli_reports_executed_skipped_state_and_cleanup(tmp_path: Path, monkeypatch):
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        run_cli,
        "run_pipeline",
        lambda path, root: SimpleNamespace(
            executed=("s2", "s1", "evaluate"),
            skipped=("preprocess",),
            state_path=tmp_path / "runs" / "speaker" / "pipeline-state.json",
            cleaned=True,
        ),
    )
    result = runner.invoke(app, ["run", str(pipeline), "--project-root", str(tmp_path)])
    assert result.exit_code == 0
    assert "executed=s2,s1,evaluate" in result.output
    assert "skipped=preprocess" in result.output
    assert "cleaned=yes" in result.output


def test_run_cli_returns_one_for_pipeline_failure(tmp_path: Path, monkeypatch):
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        run_cli,
        "run_pipeline",
        lambda path, root: (_ for _ in ()).throw(PipelineStageError("stage s2 failed")),
    )
    result = runner.invoke(app, ["run", str(pipeline), "--project-root", str(tmp_path)])
    assert result.exit_code == 1
    assert "Error: stage s2 failed" in result.output
```

Also extend root help to assert the `run` command is listed.

- [ ] **Step 2: Run CLI tests and verify RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/test_run_cli.py tests/test_cli.py -q --basetemp ..\.pytest_cache\task22-d-red
```

Expected: collection fails because `voice_pipeline.cli.run` and the registered command do not exist.

- [ ] **Step 3: Implement the thin Typer command**

`run.py` accepts one existing file positional argument and `--project-root`. It calls
`run_pipeline`, prints one concise summary, and converts `FileNotFoundError`, `OSError`,
`RuntimeError`, `ValueError`, and `VoicePipelineError` into `Error: <message>` plus exit code 1. Register
it in `cli/main.py` as `run`.

- [ ] **Step 4: Run CLI GREEN tests and help**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/test_run_cli.py tests/test_cli.py -q --basetemp ..\.pytest_cache\task22-d-green
$env:PYTHONPATH='src'
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m voice_pipeline.cli.main run --help
```

Expected: tests pass and help shows positional `PIPELINE` plus `--project-root`.

- [ ] **Step 5: Add example and exact Chinese workflow**

Create `configs/pipeline.example.yaml` with the approved four-line stage sequence. Update both
READMEs with:

```powershell
Copy-Item configs/pipeline.example.yaml configs/pipeline.local.yaml
voice-pipeline run configs/pipeline.local.yaml --project-root .
```

Document state location, completed-stage skipping, explicit `resume_from`, retry behavior, the
post-evaluation human review window, and irreversible deletion of raw checkpoints only after the
human command succeeds:

```powershell
voice-pipeline export --run runs/<目标人> --project-root . --select candidate_A
```

- [ ] **Step 6: Run full fresh verification**

```powershell
New-Item -ItemType Directory -Force '..\.pytest_cache' | Out-Null
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest -q --basetemp ..\.pytest_cache\task22-full
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m compileall -q src tests
git diff --check
```

Expected: no failures attributable to Task22, compileall exit 0, and no whitespace errors.

- [ ] **Step 7: Clean development artifacts**

Resolve and verify exact absolute targets, then remove `..\.pytest_cache`, project-local
`.pytest_cache`, and every `__pycache__` below the Python project. Do not remove model caches,
pretrained weights, datasets, formal run candidates, or final ModelBundles.

- [ ] **Step 8: Commit Task22-D and verify clean main**

```powershell
git add -- src/voice_pipeline/cli configs/pipeline.example.yaml README.md README_使用指南.md tests/test_run_cli.py tests/test_cli.py
git diff --cached --check
git commit -m "feat: add pipeline run command"
git status --short
git branch --show-current
```

Expected: empty status and branch `main`. Stop for user review before Task23.

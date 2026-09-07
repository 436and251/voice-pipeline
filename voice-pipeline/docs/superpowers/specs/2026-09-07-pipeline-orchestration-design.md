# Task22 Pipeline YAML Orchestration Design

## Goal

Add one explicit command that runs the existing v2ProPlus stages in their valid order:

```powershell
voice-pipeline run configs/pipeline.yaml --project-root .
```

The orchestrator coordinates existing APIs. It does not reimplement preprocessing, training,
evaluation, checkpoint selection, or inference.

## Pipeline Configuration

The pipeline file contains only orchestration data and references the existing training YAML:

```yaml
schema_version: 1
config: configs/train.local.yaml
stages:
  - preprocess
  - s2
  - s1
  - evaluate
```

Paths are resolved from `--project-root`. The accepted stage names are exactly `preprocess`,
`s2`, `s1`, and `evaluate`. Stages must be unique and form a non-empty ordered subsequence of
that canonical order. Omitting a stage is allowed; reordering one is not.

## Execution

The implementation uses in-process stage adapters around the existing application APIs. It
does not spawn the project's own CLI as a child process. Each stage receives the referenced
training config path and project root.

The orchestrator stops immediately when a stage fails. It never runs human selection,
candidate promotion, standalone inference, or model export beyond the candidate export already
owned by `evaluate`.

## State and Resume

State is written atomically to `<run>/pipeline-state.json`, where `<run>` is resolved from the
referenced training configuration. The schema records:

- schema version;
- resolved pipeline and training-config paths;
- the ordered stage list;
- each stage status: `pending`, `running`, `completed`, or `failed`;
- the latest failure type and message when present.

Before a stage starts it is marked `running`; after success it is marked `completed`; caught
failures are marked `failed` before being re-raised. A process interruption may leave `running`.
On the next invocation, only `completed` stages are skipped; `running`, `failed`, and `pending`
stages execute again.

The state identity is the resolved pipeline path, resolved training-config path, and declared
stage list. The contents of the training YAML are intentionally not fingerprinted, because a
user must be able to add or change `resume_from` after an interrupted training stage. If the
pipeline identity changes, the existing state is rejected instead of guessed or silently reset.

Checkpoint-level recovery remains explicit in the training YAML through each stage's
`resume_from`. The orchestrator does not scan for or choose a checkpoint.

## Successful Cleanup

Strong cleanup runs only when the declared pipeline includes `evaluate`, every declared stage is
completed, the shortlist exists, every exported candidate passes `ModelBundle.load`, and the
listening manifest names exactly those candidates and every referenced preview WAV is non-empty
and matches its recorded SHA-256. A failed or interrupted pipeline does not run this cleanup.

After validation, the orchestrator preserves:

```text
<run>/pipeline-state.json
<run>/evaluation/{stage1-report.md,stage1-results.json,report.md,results.json,shortlist.yaml,listening/}
<run>/export/candidates/
<project>/models/<model_name>/   # only when separately promoted by a human
```

It removes these run-owned, reproducible or superseded artifacts:

```text
<run>/preprocess/
<run>/training/                  # including all raw S1/S2 recovery checkpoints
<run>/evaluation/generated/
<run>/evaluation/work/
temporary .tmp/.bak trees under the run
other run children not present in the preservation list
```

Deletion is constrained to resolved descendants of the exact run directory. Dataset inputs,
official pretrained weights, evaluator weights, the project source tree, and final promoted
ModelBundles are never cleanup targets. Development-time pytest and Python bytecode caches are
cleaned at Task22 handoff, not by the production command.

The raw checkpoints are intentionally deleted after successful evaluation: all shortlisted S1
and S2 weights have already been converted into self-contained CandidateBundles, and human
promotion reads only `<run>/export/candidates/<candidate_id>`.

## Errors and CLI Output

Configuration errors identify the invalid field or stage. Stage failures identify the stage and
retain its original exception as the cause. CLI exit code is non-zero for invalid configuration,
state mismatch, stage failure, candidate validation failure, or cleanup failure.

Successful output lists executed stages, skipped completed stages, the state path, and the
cleanup result. No output claims a final model was selected.

## Testing

Tests use injected stage callables and temporary run directories; they do not train models.
Coverage includes:

- valid canonical execution order and allowed omissions;
- rejection of unknown, duplicate, empty, and reordered stages;
- failure stops later stages and is persisted;
- rerun skips completed stages and retries a failed or interrupted stage;
- state identity mismatch is rejected;
- atomic state replacement;
- successful cleanup keeps validated reports, listening files, CandidateBundles, and state while
  removing preprocessing, raw checkpoints, generated evaluation audio, and run-local debris;
- cleanup does not run on failure, without `evaluate`, or before candidate validation succeeds;
- CLI registration and non-zero failure behavior.

The final verification runs Task22 tests, the full regression suite, compileall, CLI help, and a
clean-worktree/cache check on `main`.

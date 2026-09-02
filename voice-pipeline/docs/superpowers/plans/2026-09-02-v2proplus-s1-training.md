# v2ProPlus S1 Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained v2ProPlus S1 trainer that preserves the pinned GPT-SoVITS model, ScaledAdam, AMP, and learning-rate behavior while enforcing exact continuous four-mini-batch gradient accumulation and atomic resume.

**Architecture:** Copy the official S1 optimizer and scheduler core into `voice_pipeline.core.gpt_sovits.s1`, then add strict Task 10 artifact loading and an explicit PyTorch trainer under `voice_pipeline.training.s1`. The trainer owns deterministic sampling, mixed precision, accumulation, logging, checkpoint validation, exact resume, and success-only cleanup without Lightning or runtime imports from the external checkout.

**Tech Stack:** Python 3.12, PyTorch, pytest, committed tensor fixtures, existing `PipelineLogger` and preprocessing cleanup.

**Spec:** `docs/superpowers/specs/2026-09-02-s1-training-design.md`

## Global Constraints

- Work directly on `main`; do not create a branch, worktree, ZIP, CLI, or YAML schema.
- Use only `D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe`; do not use Conda.
- Every pytest invocation uses `--basetemp .pytest_cache\codex-temp`.
- Preserve `Text2SemanticDecoder.forward_old`, upstream `ScaledAdam`, default dynamic GradScaler, and the scheduler's actual first-update `0.01` then `0.002` behavior.
- Normalize only gradient accumulation: exactly four successful mini-batches per optimizer boundary, continuously across epochs, with no loss division.
- Tests use the fixed five-sample fixture; they never generate input artifacts or pretrained weights at test runtime.
- Real GPU smoke runs four mini-batches and one optimizer iteration, then deletes its large checkpoint in `finally`.
- No runtime code imports the external `GPT-SoVITS` checkout.
- Stop after Task 13 verification and review; do not begin Task 14.

---

### Task 1: Migrate the official ScaledAdam and effective S1 scheduler

**Files:**
- Create: `src/voice_pipeline/core/gpt_sovits/s1/optim.py`
- Create: `src/voice_pipeline/core/gpt_sovits/s1/lr_scheduler.py`
- Modify: `src/voice_pipeline/core/gpt_sovits/s1/__init__.py`
- Test: `tests/compat/test_s1_optimization.py`

**Interfaces:**
- Consumes: trainable iterables accepted by `torch.optim.Optimizer`.
- Produces: `ScaledAdam`, copied from pinned `GPT_SoVITS/AR/modules/optim.py`.
- Produces: `FixedS1LRSchedule(optimizer)`, with `step() -> float`, `state_dict() -> dict[str, int | float]`, and `load_state_dict(state) -> None`.

- [ ] **Step 1: Write failing optimizer and scheduler tests**

Add tests that construct two named `nn.Parameter` values of the same shape,
perform one ScaledAdam update, and assert finite changed parameters. Add a
scheduler test that proves the optimizer starts at `0.01`, changes to `0.002`
after the first scheduler step, remains `0.002`, and round-trips its step count:

```python
def test_fixed_s1_scheduler_preserves_actual_upstream_learning_rates():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    scheduler = FixedS1LRSchedule(optimizer)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert scheduler.step() == 0.002
    assert optimizer.param_groups[0]["lr"] == 0.002
    state = scheduler.state_dict()
    scheduler.step()
    restored = FixedS1LRSchedule(optimizer)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
```

- [ ] **Step 2: Run RED**

Run:

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/compat/test_s1_optimization.py -q --basetemp .pytest_cache\codex-temp
```

Expected: collection fails because the internal optimizer modules do not exist.

- [ ] **Step 3: Copy the pinned optimizer and implement the minimal scheduler wrapper**

Copy `ScaledAdam` and every helper it directly uses from repository revision
`48b1a0169a28582a8984402f82cf438d3bfa6aca` into `s1/optim.py`. Preserve its
defaults, batched-parameter context manager, clipping calculation, parameter
scale updates, state keys, and logging. Remove only the external-package import
path and executable/demo code.

Implement the actual upstream scheduler behavior without unused matplotlib,
warmup, or cosine scaffolding:

```python
class FixedS1LRSchedule:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self._current_step = 0

    def step(self) -> float:
        for group in self.optimizer.param_groups:
            group["lr"] = 0.002
        self._current_step += 1
        return 0.002

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, int]:
        return {"current_step": self._current_step}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"current_step"} or isinstance(state["current_step"], bool) or not isinstance(state["current_step"], int) or state["current_step"] < 0:
            raise ValueError("invalid S1 scheduler state")
        self._current_step = state["current_step"]
        if self._current_step:
            for group in self.optimizer.param_groups:
                group["lr"] = 0.002
```

Export both classes from the internal S1 package.

- [ ] **Step 4: Run GREEN**

Run the Task 1 command again. Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- voice-pipeline/src/voice_pipeline/core/gpt_sovits/s1 voice-pipeline/tests/compat/test_s1_optimization.py
git commit -m "feat: migrate official s1 optimizer"
```

---

### Task 2: Add strict S1 dataset, collate, deterministic sampling, and fixed artifacts

**Files:**
- Create: `src/voice_pipeline/training/sampler.py`
- Modify: `src/voice_pipeline/training/s2/data.py`
- Create: `src/voice_pipeline/training/s1/__init__.py`
- Create: `src/voice_pipeline/training/s1/data.py`
- Create: `tests/fixtures/s2_smoke/preprocess/text/s2-smoke-01.bert.pt`
- Create: `tests/fixtures/s2_smoke/preprocess/text/s2-smoke-02.bert.pt`
- Create: `tests/fixtures/s2_smoke/preprocess/text/s2-smoke-03.bert.pt`
- Create: `tests/fixtures/s2_smoke/preprocess/text/s2-smoke-04.bert.pt`
- Create: `tests/fixtures/s2_smoke/preprocess/text/s2-smoke-05.bert.pt`
- Create: `tests/fixtures/s2_smoke/preprocess/semantic/s2-smoke-01.pt`
- Create: `tests/fixtures/s2_smoke/preprocess/semantic/s2-smoke-02.pt`
- Create: `tests/fixtures/s2_smoke/preprocess/semantic/s2-smoke-03.pt`
- Create: `tests/fixtures/s2_smoke/preprocess/semantic/s2-smoke-04.pt`
- Create: `tests/fixtures/s2_smoke/preprocess/semantic/s2-smoke-05.pt`
- Modify: `tests/fixtures/s2_smoke/checksums.json`
- Test: `tests/training/test_s1_data.py`
- Test: `tests/training/test_s2_data.py`

**Interfaces:**
- Produces: `DeterministicEpochSampler(dataset: Sized, seed: int)` in the shared training package.
- Produces: `S1Dataset(preprocess_dir: Path, *, max_sec=57, hz=25, min_ps_ratio=3, max_ps_ratio=25)`.
- Produces: `S1Collate(pad_value=1024)` returning `dict[str, Tensor | list[str]]`.
- Consumes: Task 10 canonical `valid_samples.jsonl`, text metadata/BERT tensors, and semantic tensors.

- [ ] **Step 1: Extend the fixed fixture once and update its checksum manifest**

Create the ten binary input artifacts outside the test runtime. Each BERT file
must be a committed finite float32 tensor shaped `(1024, 8)` matching its JSON
phone count. Each semantic file must be a committed non-empty int64 vector in
`[0, 1023]` long enough to satisfy the official `[3, 25]` phone/sec ratio at
25 Hz. Recompute `checksums.json` over every file below `preprocess/`.

- [ ] **Step 2: Write failing dataset and sampler tests**

Cover five physical samples, 100 repeated logical samples, unchanged behavior
for 100+ items, strict BERT/semantic validation with the failing sample ID,
duration and phone/sec admission, EOS padding, and deterministic epochs. The
main contract assertion is:

```python
dataset = S1Dataset(FIXTURE / "preprocess")
assert dataset.sample_count == 5
assert len(dataset) == 100
batch = S1Collate()([dataset[0], dataset[1]])
assert batch["phoneme_ids"].dtype == torch.int64
assert batch["semantic_ids"].dtype == torch.int64
assert batch["bert_feature"].shape == (2, 1024, 8)
assert batch["sample_ids"] == ["s2-smoke-01", "s2-smoke-02"]
```

Move the existing sampler tests to import the shared class while retaining the
S2 public re-export so existing callers do not break.

- [ ] **Step 3: Run RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/training/test_s1_data.py tests/training/test_s2_data.py -q --basetemp .pytest_cache\codex-temp
```

Expected: S1 imports fail; the updated fixed-fixture checksum test already
passes and proves the binary inputs are stable.

- [ ] **Step 4: Implement the strict loader and collator**

Parse `valid_samples.jsonl` with the same safe sample-ID checks as S2. Load
`phone_ids` from JSON, BERT with `weights_only=True`, and semantic with
`weights_only=True`. Reject corrupt types, shapes, ranges, non-finite values,
and official admission-bound violations with the sample ID in the exception.

Store physical samples separately from logical repeated indices:

```python
repeat_count = max(2, int(100 / count)) if count < 100 else 1
self._indices = list(range(count)) * repeat_count
```

Implement zero phone padding, `1024` semantic padding, and zero BERT padding.
Move the existing deterministic sampler implementation unchanged to
`training/sampler.py`, importing and re-exporting it from S2 data.

- [ ] **Step 5: Run GREEN**

Run the Task 2 test command again. Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add -- voice-pipeline/src/voice_pipeline/training/sampler.py voice-pipeline/src/voice_pipeline/training/s1 voice-pipeline/src/voice_pipeline/training/s2/data.py voice-pipeline/tests/fixtures/s2_smoke voice-pipeline/tests/training/test_s1_data.py voice-pipeline/tests/training/test_s2_data.py
git commit -m "feat: add strict s1 training dataset"
```

---

### Task 3: Add S1 configuration, construction, and four-batch AMP step

**Files:**
- Create: `src/voice_pipeline/training/s1/config.py`
- Create: `src/voice_pipeline/training/s1/optim.py`
- Create: `src/voice_pipeline/training/s1/step.py`
- Modify: `src/voice_pipeline/training/s1/__init__.py`
- Delete: `src/voice_pipeline/training/s1_runtime.py`
- Delete: `tests/training/test_s1_upstream_gate.py`
- Test: `tests/training/test_s1_config.py`
- Test: `tests/training/test_s1_step.py`

**Interfaces:**
- Produces: frozen `S1TrainConfig` with the exact fields in the approved spec.
- Produces: `build_optimizer(model, config) -> ScaledAdam`.
- Produces: `build_scheduler(optimizer) -> FixedS1LRSchedule`.
- Produces: `backward_s1_minibatch(batch, model, scaler, config) -> S1MiniBatchResult`.
- Produces: `finish_s1_optimizer_step(model, optimizer, scheduler, scaler) -> S1OptimizerResult`.

- [ ] **Step 1: Write failing validation and accumulation-step tests**

Test missing/wrong base name, invalid device/precision, non-positive values,
`gradient_accumulation != 4`, complete named optimizer parameters, forward_old
selection, no loss division, default scaler use, and step order. Use a tiny
trainable model whose `forward_old` returns a differentiable scalar and accuracy.

The key test calls backward four times and asserts no parameter change before
`finish_s1_optimizer_step`, then one change afterward:

```python
optimizer.zero_grad()
for _ in range(4):
    result = backward_s1_minibatch(batch, model, scaler, config)
    assert result.loss == pytest.approx(expected_loss)
assert torch.equal(parameter_before, model.weight.detach())
finish_s1_optimizer_step(model, optimizer, scheduler, scaler)
assert not torch.equal(parameter_before, model.weight.detach())
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/training/test_s1_config.py tests/training/test_s1_step.py -q --basetemp .pytest_cache\codex-temp
```

Expected: failures because config, optimizer builders, and step functions do not
exist.

- [ ] **Step 3: Implement configuration and explicit step functions**

Validate `base_s1_path.name == "s1v3.ckpt"`, supported `fp16|fp32`, CUDA
availability, paths, positive budgets, `num_workers >= 0`, and accumulation 4.
Construct ScaledAdam with the official values and exact named parameters.

`backward_s1_minibatch` moves the five model tensors to the configured device,
runs only `forward_old` inside CUDA fp16 autocast when enabled, and calls
`scaler.scale(loss).backward()` without division. `finish_s1_optimizer_step`
must perform exactly:

```python
scaler.unscale_(optimizer)
scaler.step(optimizer)
scaler.update()
learning_rate = float(optimizer.param_groups[0]["lr"])
scheduler.step()
optimizer.zero_grad()
```

Return both the update LR (the first is `0.01`) and post-update scaler scale for
logging. Delete the obsolete anomaly-preserving `S1StepController` runtime and
its test.

- [ ] **Step 4: Run GREEN**

Run the Task 3 tests plus `tests/compat/test_s1_optimization.py`. Expected: all
pass.

- [ ] **Step 5: Commit**

```powershell
git add -- voice-pipeline/src/voice_pipeline/training/s1 voice-pipeline/src/voice_pipeline/training/s1_runtime.py voice-pipeline/tests/training/test_s1_upstream_gate.py voice-pipeline/tests/training/test_s1_config.py voice-pipeline/tests/training/test_s1_step.py
git commit -m "feat: accumulate official s1 gradients explicitly"
```

---

### Task 4: Add atomic S1 checkpoint validation and restore

**Files:**
- Create: `src/voice_pipeline/training/s1/checkpoint.py`
- Modify: `src/voice_pipeline/training/s1/__init__.py`
- Test: `tests/training/test_s1_checkpoint.py`

**Interfaces:**
- Produces: frozen `S1TrainingCursor(optimizer_step=0, epoch=0, next_batch_index=0, accumulation_position=0)`.
- Produces: `checkpoint_path(output_dir: Path, step: int) -> Path`.
- Produces: `save_checkpoint(path, *, model, optimizer, scheduler, scaler, cursor) -> None`.
- Produces: `load_checkpoint(path, *, model, optimizer, scheduler, scaler, target_optimizer_steps) -> S1TrainingCursor`.

- [ ] **Step 1: Write failing round-trip and corruption tests**

Cover atomic naming, no `.tmp`, exact model/optimizer/scheduler/scaler/cursor/RNG
round-trip, filename mismatch, wrong profile/version, target exceeded, nonzero
accumulation position, malformed ScaledAdam state, and a late corruption that
must not partially mutate the live model.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/training/test_s1_checkpoint.py -q --basetemp .pytest_cache\codex-temp
```

Expected: import failure because the checkpoint module does not exist.

- [ ] **Step 3: Implement the atomic envelope and preflight**

Use format version 1, profile `v2ProPlus`, and filenames
`training/s1/checkpoints/step-{step:08d}.pt`. Save to a UUID-suffixed temporary
file beside the destination and publish with `os.replace`.

Require exactly these payload keys:

```python
{
    "format_version", "profile", "optimizer_step", "epoch",
    "next_batch_index", "accumulation_position", "model", "optimizer",
    "scheduler", "scaler", "python_rng", "torch_rng", "cuda_rng",
}
```

Before loading anything, validate cursor integers and zero accumulation,
model keys/shapes, optimizer group count/parameter IDs/state tensor shapes,
scheduler/scaler structures, and every RNG state using temporary generators.
Only after the complete preflight may live objects and global RNG be mutated.

- [ ] **Step 4: Run GREEN**

Run the Task 4 command again. Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- voice-pipeline/src/voice_pipeline/training/s1 voice-pipeline/tests/training/test_s1_checkpoint.py
git commit -m "feat: checkpoint s1 training atomically"
```

---

### Task 5: Implement the bounded S1 trainer lifecycle and exact resume

**Files:**
- Create: `src/voice_pipeline/training/s1/trainer.py`
- Modify: `src/voice_pipeline/training/s1/__init__.py`
- Test: `tests/training/test_s1_trainer.py`

**Interfaces:**
- Produces: `S1Trainer.from_pretrained(config, *, resume_from=None, logger=None) -> S1Trainer`.
- Produces: `S1Trainer.train() -> S1TrainingCursor`.
- Consumes: Tasks 1-4 model, data, optimizer, scheduler, step, checkpoint, logger, and cleanup interfaces.

- [ ] **Step 1: Write failing lifecycle tests**

Use injected tiny training objects and fixed batches. Cover:

- eight mini-batches produce exactly two optimizer events
- a three-batch epoch plus the next epoch's first batch produces one update
- cursor advances after every successful mini-batch but checkpoints only at zero accumulation
- resume skips saved deterministic batches and reproduces uninterrupted weights
- mini-batch logs contain loss, top-3 accuracy, accumulation position, and step
- optimizer logs contain update LR and scaler scale
- final checkpoint is always present at the target
- exceptions create no new checkpoint and perform no cleanup
- cleanup runs only after target plus successful final checkpoint

The principal assertion is:

```python
cursor = trainer.train()
assert cursor.optimizer_step == 2
events = read_events(log_path)
assert len([event for event in events if event["event"] == "mini_batch"]) == 8
assert len([event for event in events if event["event"] == "optimizer"]) == 2
```

- [ ] **Step 2: Run RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/training/test_s1_trainer.py -q --basetemp .pytest_cache\codex-temp
```

Expected: import failure because `S1Trainer` does not exist.

- [ ] **Step 3: Implement construction and continuous accumulation**

Seed Python/Torch/CUDA, strictly load `s1v3.ckpt`, build optimizer/scheduler and
default GradScaler, create the strict dataset/sampler/loader with its own seeded
DataLoader generator, then optionally restore the internal checkpoint.

The loop increments `accumulation_position` after each successful backward,
wraps the batch cursor deterministically at epoch boundaries, and performs an
optimizer boundary only at position four. After that boundary it resets
accumulation to zero, increments optimizer step, logs, writes interval/final
checkpoints, and checks the target. Scheduler steps once per boundary, not per
epoch. Call cleanup only after the final checkpoint succeeds.

- [ ] **Step 4: Run GREEN**

Run all S1 ordinary tests:

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/compat/test_s1_optimization.py tests/training/test_s1_data.py tests/training/test_s1_config.py tests/training/test_s1_step.py tests/training/test_s1_checkpoint.py tests/training/test_s1_trainer.py -q --basetemp .pytest_cache\codex-temp
```

Expected: all pass without real model weights.

- [ ] **Step 5: Commit**

```powershell
git add -- voice-pipeline/src/voice_pipeline/training/s1 voice-pipeline/tests/training/test_s1_trainer.py
git commit -m "feat: train s1 to an explicit step budget"
```

---

### Task 6: Verify the real v2ProPlus S1 path and document provenance

**Files:**
- Create: `tests/integration/test_s1_gpu_smoke.py`
- Modify: `src/voice_pipeline/core/gpt_sovits/UPSTREAM.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the public Task 13 S1 trainer and local `models/pretrained/v2proplus/s1/s1v3.ckpt`.
- Produces: one separately marked real CUDA smoke and accurate architecture/provenance documentation.

- [ ] **Step 1: Write the real GPU smoke test**

Copy the committed fixture into `tmp_path`, build config with batch size one,
accumulation four, target one, and checkpoint interval one. Run the real model,
assert four mini-batch logs, one optimizer log, finite loss/accuracy, first
update LR `0.01`, dynamic scaler backoff if any recorded gradient-related metric
is non-finite, checkpoint restore into fresh objects, and no `.tmp` file.

Put deletion in `finally` so both pass and failure avoid retained weight copies:

```python
checkpoint = checkpoint_path(output, 1)
try:
    assert trainer.train().optimizer_step == 1
    restored = S1Trainer.from_pretrained(config, resume_from=checkpoint)
    assert restored.cursor.optimizer_step == 1
finally:
    shutil.rmtree(output, ignore_errors=True)
    gc.collect()
    torch.cuda.empty_cache()
```

- [ ] **Step 2: Run the real CUDA smoke**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest tests/integration/test_s1_gpu_smoke.py -m gpu -q --basetemp .pytest_cache\codex-temp
```

Expected: `1 passed`, no skip on the user's CUDA machine, one four-mini-batch AMP
iteration, successful restore, and no retained real-model checkpoint under the
test output.

- [ ] **Step 3: Document the exact upstream and framework boundary**

Add revision and source paths, copied ScaledAdam behavior, first-update `0.01`,
subsequent fixed `0.002`, `forward_old`, no loss division, continuous four-batch
normalization, checkpoint semantics, and test cleanup to `UPSTREAM.md`. Update
README current status without exposing the unimplemented Task 14 CLI/YAML.

- [ ] **Step 4: Run complete verification**

Run all ordinary tests with the real asset environment variables already used
by the project, excluding only the separately executed GPU marker:

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m pytest -m 'not gpu' -q --basetemp .pytest_cache\codex-temp
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' -m compileall -q src tests
rg -n "^(from|import).*GPT[-_]SoVITS" src tests -g "*.py"
git diff --check
```

Expected: ordinary suite passes with zero skips, compileall succeeds, forbidden
runtime import scan has no matches, and diff check is clean.

- [ ] **Step 5: Request correctness review and fix findings with focused RED/GREEN tests**

Review must check strict upstream math, ScaledAdam state restore, AMP ordering,
continuous accumulation, checkpoint preflight, deterministic cursor behavior,
cleanup safety, external-import isolation, and GPU artifact deletion. Every code
fix receives a reproducing failing test before implementation.

- [ ] **Step 6: Commit**

```powershell
git add -- voice-pipeline/README.md voice-pipeline/src/voice_pipeline/core/gpt_sovits/UPSTREAM.md voice-pipeline/tests/integration/test_s1_gpu_smoke.py
git commit -m "test: verify real v2proplus s1 training"
```

- [ ] **Step 7: Stop at the Task 13 review checkpoint**

Report ordinary and GPU test counts, compile/import/diff evidence, reviewer
findings, commits, and transient checkpoint cleanup. Do not begin Task 14 until
the user explicitly approves Task 13.

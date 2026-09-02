# v2ProPlus S1 Training Design

## Purpose

Task 13 adds a self-contained, single-device S1 trainer for the v2ProPlus
`s1v3.ckpt`. It consumes Task 10's canonical preprocessing artifacts, copies the
pinned GPT-SoVITS S1 training core into the internal package, and replaces only
the Lightning orchestration with explicit PyTorch control.

The intended few-shot strategy remains conservative: S2 is the primary speaker
adaptation stage, while S1 is a short optional adaptation whose checkpoints can
later be compared with the unchanged base S1. Task 13 does not change the S1
model, loss, optimizer, or effective scheduler learning rate.

## Frozen architecture boundary

The supported path is:

```text
Task 10 valid preprocessing artifacts
-> deterministic S1 dataset and collate
-> Text2SemanticDecoder.forward_old
-> four-mini-batch accumulated backward
-> ScaledAdam update
-> official scheduler step, locking subsequent updates to 0.002
-> atomic internal resume checkpoint
```

Task 13 excludes DPO, validation-driven optimization, S2, DDP, multi-GPU launch,
Lightning, TensorBoard, CLI/YAML integration, model selection, inference, and
user-facing half-precision S1 export. Task 14 owns the training CLI and top-level
configuration mapping. A later export task owns conversion into a deployable S1
`.ckpt`.

## Upstream migration boundary

Training behavior is copied from the pinned GPT-SoVITS revision, primarily:

- `GPT_SoVITS/AR/models/t2s_lightning_module.py`
- `GPT_SoVITS/AR/modules/optim.py`
- `GPT_SoVITS/AR/modules/lr_schedulers.py`
- `GPT_SoVITS/AR/data/dataset.py`
- `GPT_SoVITS/s1_train.py`

The internal core receives the upstream `ScaledAdam` implementation and the
effective scheduler behavior needed by S1 training. The trainer reuses the
already migrated `Text2SemanticDecoder`; it never imports the external
GPT-SoVITS checkout at runtime.

Only orchestration-specific code is replaced: Lightning/DDP setup, callbacks,
TensorBoard, filesystem discovery, and upstream checkpoint lifecycle are not
copied. Provenance and deliberate differences are recorded in `UPSTREAM.md`.

## Production inputs

`S1Dataset` reads one completed preprocessing directory:

```text
preprocess/
├── valid_samples.jsonl
├── quarantine.jsonl
├── text/<sample_id>.json
├── text/<sample_id>.bert.pt
└── semantic/<sample_id>.pt
```

For every ID declared valid, dataset construction validates all required files.
It uses `phone_ids` produced by the per-record language frontend and never runs
G2P again. BERT must be a finite floating tensor shaped
`(1024, phone_count)`. Semantic tokens must be a non-empty `int64` vector in
`[0, 1023]`.

The official S1 admission bounds remain in force: semantic duration is limited
by the v2ProPlus maximum duration and 25 Hz frame rate, phone length is bounded
relative to maximum duration, and the phone-per-second ratio remains within
`[3, 25]`. Task 10 has already quarantined bad source records; therefore a
missing, corrupt, misaligned, or out-of-range artifact for a canonical valid ID
is an explicit sample-scoped error rather than another silent skip.

For fewer than 100 admitted samples, the list is repeated with the official S1
rule:

```python
repeat_count = max(2, int(100 / admitted_sample_count))
```

At 100 or more samples, it is not repeated. The repeated list is shuffled by a
deterministic epoch sampler. An empty admitted set is a configuration error.

## Collation contract

The collator returns the upstream five training fields:

```text
phoneme_ids       int64   (B, max_phone_length), zero padded
phoneme_ids_len   int64   (B,)
semantic_ids      int64   (B, max_semantic_length), EOS/PAD 1024 padded
semantic_ids_len  int64   (B,)
bert_feature      float   (B, 1024, max_phone_length), zero padded
```

Sample IDs may accompany the internal batch for diagnostics but are not passed
to the model. The collator preserves tensor dtypes and does not perform frontend,
semantic, or BERT recomputation.

## Configuration and construction

`S1TrainConfig` owns only Task 13 settings:

- `preprocess_dir`, `output_dir`, and `base_s1_path`
- `device="cuda:0"`, `precision="fp16"`, `batch_size=2`, and `num_workers=0`
- `gradient_accumulation=4`, `target_optimizer_steps`,
  `checkpoint_every_steps`, and `seed=1234`
- v2ProPlus data bounds: 25 Hz, maximum 57 seconds, phone/sec `[3, 25]`

The first implementation accepts only accumulation 4 because that is the
approved official training contract, not a speculative tuning surface. It
rejects invalid step/batch/checkpoint values, unsupported precision, missing
inputs, a non-v2ProPlus base checkpoint, and fp16 on a non-CUDA device before
creating training output.

`S1Trainer.from_pretrained(...)` loads the official checkpoint config and model
weights strictly through the internal compatibility layer. Continuation accepts
only the internal Task 13 resume format; a user-facing fine-tuned checkpoint is
not treated as optimizer state.

## Model, loss, optimizer, and scheduler

The trainer calls `Text2SemanticDecoder.forward_old(...)`. The DPO path remains
disabled. The returned cross-entropy loss and top-3 accuracy retain the migrated
model's exact behavior.

`ScaledAdam` is copied from upstream with these official construction values:

```text
lr=0.01
betas=(0.9, 0.95)
clipping_scale=2.0
clipping_update_period=1000
show_dominant_parameters=False
```

The optimizer receives the model's complete trainable parameter list and the
matching names required by its batched-parameter implementation. No extra
PyTorch gradient clipping is introduced. Although the checkpoint config contains
warmup and cosine values, the pinned upstream scheduler forces every parameter
group to learning rate `0.002` when it is stepped. Consequently the first
optimizer update uses ScaledAdam's construction LR `0.01`, and subsequent
updates use `0.002`. Task 13 preserves that actual behavior rather than silently
repairing the unused schedule fields.

## Explicit gradient accumulation and AMP

CUDA fp16 uses PyTorch autocast and the default dynamic `GradScaler`. Each
mini-batch computes one forward and scaled backward. The loss is not divided by
four, matching the upstream sum of four mini-batch gradients.

An optimizer update occurs after exactly four successful mini-batches:

1. Accumulate scaled gradients for mini-batches 1 through 4.
2. Unscale once at the complete boundary.
3. Call `scaler.step(optimizer)`.
4. Call `scaler.update()`.
5. Step the effective official scheduler once.
6. Clear gradients and increment the optimizer-step counter once.

The accumulation counter is continuous across DataLoader epoch boundaries. An
epoch tail of one to three mini-batches is neither flushed early nor discarded;
the next epoch completes the same four-batch window.

This deliberately replaces the upstream condition
`batch_idx > 0 and batch_idx % 4 == 0`, whose first update consumes five batches
and whose reset at every epoch does not express a stable four-batch contract.
Eight successful physical mini-batches always correspond to two optimizer-step
boundaries.

As with the official mixed-precision runtime, an overflow may cause GradScaler
to skip the underlying ScaledAdam update while reducing its scale. The completed
four-batch AMP iteration still advances the framework optimizer-step counter.
The trainer never forces a low initial scale.

## Step budget, logging, and failure behavior

`S1Trainer.train()` cycles deterministic epochs until the configured optimizer
step target is reached. It logs a mini-batch event after every successful
backward with loss, top-3 accuracy, epoch/batch position, accumulation position,
and current optimizer step. At every complete accumulation boundary it logs an
optimizer event with the new step, effective learning rate, and GradScaler scale.

An exception during a mini-batch does not advance its cursor. An exception before
the fourth backward leaves no published checkpoint. A completed boundary is the
smallest resumable unit; Task 13 does not serialize partial gradients.

The trainer calls `cleanup_after_training(preprocess_dir, True)` only after the
target optimizer step and final checkpoint both succeed. Failure or interruption
performs no preprocessing cleanup.

## Atomic checkpoint and exact resume

Internal checkpoints are named by completed optimizer step:

```text
training/s1/checkpoints/step-00000001.pt
```

Each destination-local atomic archive contains:

- format version and `v2ProPlus` profile
- model state
- ScaledAdam and scheduler state
- GradScaler state
- completed optimizer step
- epoch and next deterministic batch index
- zero accumulation position, asserted at save time
- Python, CPU torch, and all active CUDA RNG states

Checkpoint loading validates the complete envelope, filename/embedded step,
model keys and shapes, optimizer structure, scheduler/scaler structure, cursor,
and RNG state before mutating live objects. It refuses a checkpoint beyond the
requested target. Resume reconstructs the deterministic loader and continues at
the next unprocessed mini-batch with an empty accumulation window.

The archive is intentionally optimized for exact continuation, not size or
deployment. Tests remove real-model checkpoint copies after restore validation.

## Testing

The existing five-sample committed smoke fixture is extended with fixed
`text/*.bert.pt` and `semantic/*.pt` artifacts. Tests do not synthesize WAV,
BERT, semantic, text, manifest, or pretrained-weight inputs dynamically.

Ordinary tests use injected tiny models or small fixed tensors and cover:

- strict artifact validation, S1 admission bounds, and official repetition
- exact collate shapes, padding, dtypes, and BERT alignment
- copied ScaledAdam construction and effective `0.002` scheduler behavior
- `8 mini-batches = 2 optimizer steps`
- a four-batch accumulation window spanning an epoch boundary
- mini-batch and optimizer-step logging
- no checkpoint or success cleanup on failure
- atomic checkpoint validation, cursor/RNG restoration, and exact continuation

A separately marked GPU smoke loads the real project
`models/pretrained/v2proplus/s1/s1v3.ckpt`, runs four fixed mini-batches and one
CUDA fp16 optimizer iteration, verifies finite loss metrics and dynamic scaler
behavior, restores the checkpoint into fresh training objects, and deletes the
large generated checkpoint in a `finally` block. It does not increase the step
count or retain diagnostic weight copies.

## Acceptance criteria

- All Task 13 unit tests pass without loading real weights.
- The fixed real-model CUDA smoke completes four mini-batches and one optimizer
  iteration on the user's GPU, restores successfully, and leaves no large test
  checkpoint behind.
- Existing Task 10 and Task 12 regressions remain green.
- Compile, diff, and forbidden external-import scans remain clean.
- Upstream provenance and the accumulation normalization are documented.
- The implementation is committed directly on `main` and stops before Task 14
  for user review.

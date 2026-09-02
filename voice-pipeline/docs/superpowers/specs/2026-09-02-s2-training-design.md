# v2ProPlus S2 Training Design

## Purpose

Task 12 adds a self-contained, single-device S2 training subsystem that consumes
the validated Task 10 preprocessing artifacts and preserves the pinned
GPT-SoVITS `s2_train.py` behavior for `v2ProPlus`. Unit tests execute one tiny
injected D+G update. A separate GPU smoke test uses five committed samples in
the production artifact format and executes one update with the real
`s2Gv2ProPlus.pth` and `s2Dv2ProPlus.pth` models.

The implementation targets the user's single NVIDIA RTX 4060 Laptop GPU. It does
not read or modify the user's real dataset during testing. The five test samples
are stable repository fixtures; changing the preprocessing contract may require
an intentional fixture update.

## Frozen architecture boundary

The only supported route is:

```text
v2ProPlus
-> SynthesizerTrn
-> MultiPeriodDiscriminator
-> speaker embedding conditioning
-> adversarial discriminator update
-> adversarial generator update
```

Task 12 excludes DDP, multi-GPU launch, S1, CLI/YAML integration, TensorBoard,
evaluation, LoRA, CFM, F5-TTS, V3/V4, external vocoders, and changes to model,
loss, mel, semantic, or speaker-conditioning algorithms. Task 14 owns the
training CLI and top-level YAML mapping.

## Production inputs

`S2Dataset` reads only one completed experiment's preprocessing directory:

```text
preprocess/
├── valid_samples.jsonl
├── quarantine.jsonl
├── text/<sample_id>.json
├── wav32k/<sample_id>.wav
├── hubert/<sample_id>.pt
└── sv/<sample_id>.pt
```

For each ID in `valid_samples.jsonl`, construction validates that all four S2
artifacts exist. It reads `phone_ids` from the text JSON, not from a second G2P
pass. It accepts only mono int16 32 kHz WAVs with duration strictly between 0.6
and 54 seconds, HuBERT tensors shaped `(1, 768, frames)`, and finite speaker
features shaped `(1, 20480)`. A missing, corrupt, mismatched, or non-finite
artifact is a fail-fast dataset error; the trainer must never substitute zeros
for a sample already declared valid.

The normal upstream frame boundary is preserved: when HuBERT is exactly one
frame shorter than the computed spectrogram, its final frame is replicated once
on the right. Equal lengths are used unchanged; every other temporal mismatch is
a fail-fast artifact error. This is the official loader's alignment behavior,
not a bad-sample zero fallback.

Audio is converted to normalized float and its linear spectrogram is computed
with the migrated `spectrogram_torch(..., center=False)` using the pinned
v2ProPlus values `filter_length=2048`, `hop_length=640`, `win_length=2048`, and
`sampling_rate=32000`.

The dataset preserves the official few-shot repetition rule:

```python
repeat_count = max(2, int(100 / valid_sample_count))
```

The repeated sample list is shuffled deterministically per epoch. The committed
five-sample smoke fixture therefore provides 100 logical entries.

## Collation contract

`S2Collate` returns the exact v2ProPlus nine-tensor batch:

```text
ssl, ssl_lengths,
spec, spec_lengths,
wav, wav_lengths,
text, text_lengths,
sv_emb
```

It sorts by descending spectrogram length, right-pads SSL/spec/wav/text with
zeros, retains integer text IDs, and flattens each `(1, 20480)` SV artifact into
one `(20480,)` row. SSL and spectrogram time dimensions preserve the pinned
collator's next-even sizing rule `2 * ((maximum // 2) + 1)`.

## Configuration and construction

`S2TrainConfig` owns only Task 12 settings:

- `preprocess_dir`, `output_dir`, `base_s2g_path`, and `base_s2d_path`
- `device="cuda:0"`, `precision="fp16"`, `batch_size=2`, `num_workers=0`
- `target_optimizer_steps`, `checkpoint_every_steps`, and `seed=1234`
- pinned defaults: learning rate `1e-4`, text low-LR rate `0.4`, betas
  `(0.8, 0.99)`, epsilon `1e-9`, decay `0.999875`, segment size `20480`,
  mel coefficient `45`, and KL coefficient `1.0`

It rejects non-positive step/batch/checkpoint values, unsupported precision,
non-v2ProPlus model paths, missing inputs, and `fp16` on a non-CUDA device before
constructing an output checkpoint.

`S2Trainer.from_pretrained(...)` strictly loads the existing base G and D through
the migrated checkpoint loaders. No fine-tuned checkpoint can be passed as a
second initialization source; continuation uses only the internal resume format.

## Optimizers and schedulers

The generator AdamW contains four disjoint and exhaustive trainable groups:

```text
all remaining trainable parameters  -> 1.0 * base LR
enc_p.text_embedding                -> 0.4 * base LR
enc_p.encoder_text                  -> 0.4 * base LR
enc_p.mrte                          -> 0.4 * base LR
```

Construction verifies that no trainable generator parameter is duplicated or
omitted. The discriminator uses one AdamW group at base LR. Both preserve the
pinned betas and epsilon. Each has an `ExponentialLR(gamma=0.999875)` scheduler,
stepped after a complete deterministic DataLoader epoch, matching upstream's
epoch-level decay.

## One S2 optimizer step

One global S2 update is complete only after this sequence succeeds on the same
batch:

1. Move the nine tensors to the configured device; SSL never requires gradients.
2. Run `SynthesizerTrn(ssl, spec, spec_lengths, text, text_lengths, sv_emb)`.
3. Compute the pinned source mel, sliced target mel, generated mel, and waveform
   slice using the returned `ids_slice`.
4. Run D on `(real, generated.detach())`; extract `discriminator_loss(...)[0]`.
5. D: `zero_grad -> scaled backward -> unscale -> clip -> scaler.step`.
6. Run D again on `(real, generated)` and compute adversarial generator,
   feature-matching, mel, `kl_ssl`, and KL losses.
7. G: `zero_grad -> scaled backward -> unscale -> clip -> scaler.step`.
8. Call `scaler.update()` and only then increment `global_step` once.

This follows upstream iteration semantics. If dynamic loss scaling detects an
overflow, `scaler.step(...)` may skip the underlying optimizer update and lower
its scale; the completed D+G AMP iteration still advances `global_step`. The
trainer does not override GradScaler's default initial scale or adaptation.

The total generator loss remains exactly:

```text
loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl
```

Autocast is enabled only for CUDA fp16. Loss aggregation remains outside autocast
as in the pinned trainer. Reaching the target is checked after the completed G
update; a completed D update followed by a failed G update never advances the
global step or publishes a success checkpoint.

## Step budget, logging, and lifecycle

`S2Trainer.train()` cycles deterministic epochs until `global_step` equals
`target_optimizer_steps`. It records `batch` and `checkpoint` events with the
Task 11 `PipelineLogger`, including D total, G total, feature, mel, KL-SSL, KL,
gradient norms, learning rate, mini step, and optimizer step as plain numeric
metrics.

The trainer calls `cleanup_after_training(preprocess_dir, True)` only after the
target step and final checkpoint both succeed. Any exception or interruption
propagates and performs no preprocessing cleanup.

## Atomic checkpoint and exact resume

Internal resume checkpoints are named by completed global update:

```text
training/s2/checkpoints/step-00000001.pt
```

Each destination-local atomic torch archive contains:

- format version and profile name
- generator and discriminator state dictionaries
- both optimizer and scheduler states
- GradScaler state
- completed `global_step`
- current epoch and next batch index
- Python RNG, CPU torch RNG, and all CUDA RNG states when CUDA is active

The trainer always writes a checkpoint on the configured interval and a final
checkpoint at the target if that step was not already written. Resume accepts one
explicit internal checkpoint, validates its version/profile/step fields, strictly
loads both models, restores all training states and RNG, and resumes from the next
unprocessed deterministic batch. It refuses a checkpoint whose completed step is
greater than the requested target.

The internal archive is for exact training continuation. Later export code owns
conversion of its generator state into the user-facing SoVITS `06` checkpoint;
Task 12 does not introduce a second export format.

## Testing

Unit tests cover:

- five-artifact dataset validation and official repetition
- exact nine-tensor collation and padding
- disjoint/exhaustive generator LR groups
- one tiny D-before-G update and one global-step increment
- no increment/checkpoint/cleanup when G fails after D
- atomic checkpoint round-trip and resume cursor/RNG restoration
- interval/final checkpoint naming and success-only cleanup

The GPU smoke test reads five fixed production-format samples committed under
`tests/fixtures/s2_smoke/`, constructs the real v2ProPlus G and D from the project
model directory, uses batch size one to bound VRAM, runs one complete update on
CUDA fp16, and checks that the resulting checkpoint can be restored. The fixture
contains its fixed manifest views, text JSON, 32 kHz WAV, HuBERT tensors, and SV
tensors; tests never regenerate it. The smoke is explicitly GPU/asset gated and
remains separate from ordinary unit tests.

## Acceptance criteria

- All Task 12 unit tests pass without loading real weights.
- The five-sample real-model CUDA smoke completes one D+G update on the user's GPU.
- Existing real-asset regression remains green.
- Compile and forbidden-coupling scans remain clean.
- No runtime import references the external GPT-SoVITS checkout.
- The Part is committed on `main` and stops before Task 13 for user review.

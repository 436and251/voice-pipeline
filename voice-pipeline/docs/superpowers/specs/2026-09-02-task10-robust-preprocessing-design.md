# Task 10 Robust v2ProPlus Preprocessing Design

**Date:** 2026-09-02  
**Status:** written spec pending user review; implementation not started  
**Profile:** v2ProPlus only  
**Upstream baseline:** GPT-SoVITS `48b1a0169a28582a8984402f82cf438d3bfa6aca`

## 1. Purpose

Task 10 turns an already prepared official-format `data.list` into the complete
v2ProPlus training feature set. It is also an architecture checkpoint: the new
pipeline must preserve the selected v2ProPlus GAN route and the cross-language
frontend contract while adding reliable caching, recovery, quarantine, and output
validation.

The input format remains:

```text
audio_path|speaker|language|text
```

Task 10 does not create, rewrite, transcribe, slice, or automatically repair the
manifest. `speaker` remains metadata. The pipeline does not perform a single- or
multi-speaker consistency check; the user guarantees a single-speaker source.

## 2. Architecture invariants

The implementation must keep these existing decisions unchanged:

- Profile is `v2ProPlus`.
- S2 remains `SynthesizerTrn + MultiPeriodDiscriminator + SV conditioning` with
  adversarial G/D training. Task 10 must not introduce V3/V4, CFM, LoRA, or an
  external vocoder.
- Training and inference use the same `MultilingualFrontend` and the same explicit
  `zh`, `ja`, `en`, or `mixed` routing.
- Text language comes from each manifest record. The dominant language of the
  speaker's dataset does not override the record language.
- Semantic tokens come from the profile's base `s2Gv2ProPlus.pth`, never a
  fine-tuned experiment checkpoint.
- HuBERT's 16 kHz input is produced from the original audio, never by resampling the
  saved int16 wav32k artifact.
- SV follows `wav32k -> 16 kHz -> 80-bin Kaldi fbank -> ERes2NetV2 forward3()`.
- S1 and S2 indexes are generated from one final valid-sample set so their sample
  membership cannot diverge.

Architecture compatibility assertions remain part of the regression suite rather
than being inferred from successful feature extraction.

## 3. Chosen storage model

The pipeline uses per-sample atomic artifacts and generated aggregate indexes. It
does not use a database or a task queue.

```text
runs/<experiment>/preprocess/
├── text/
│   ├── <sample-id>.json
│   ├── <sample-id>.bert.pt
│   └── index.jsonl
├── wav32k/
│   ├── <sample-id>.wav
│   └── index.jsonl
├── hubert/
│   ├── <sample-id>.pt
│   └── index.jsonl
├── sv/
│   ├── <sample-id>.pt
│   └── index.jsonl
├── semantic/
│   ├── <sample-id>.pt
│   └── index.jsonl
├── assets.json
├── valid_samples.jsonl
├── quarantine.jsonl
├── 2-name2text.txt
└── 6-name2semantic.tsv
```

The generated `2-name2text.txt` and `6-name2semantic.tsv` are official-compatible
aggregate views. The internal trainers consume `valid_samples.jsonl` plus the
per-stage artifacts, avoiding duplicate tensor storage.

Each non-empty manifest record receives a stable sample ID derived from its
canonical record fields. An exact duplicate record is a sample-level validation
error rather than a second training example. Reordering otherwise unchanged records
does not change their IDs.

## 4. Stage data flow

### 4.1 Text

Before the shared frontend is called, the official two-character compatibility
rule is applied at the common frontend entry used by both training and inference:

```python
text = text.replace("%", "-").replace("￥", ",")
```

The stage calls `MultilingualFrontend.process(text, language)` and saves:

- normalized text;
- phone sequence and phone IDs;
- `word2ph` for pure Chinese, otherwise `None`;
- a BERT tensor shaped `(1024, phone_count)`;
- speaker and language metadata.

Chinese and Chinese spans in mixed text contain real RoBERTa features. Japanese and
English columns remain aligned zeros as defined by the reviewed frontend.

### 4.2 wav32k

The original audio is decoded to mono 32 kHz float32. The stage preserves the
pinned upstream amplitude-mixing formula and writes a 32 kHz int16 WAV. Empty,
silent, non-finite, undecodable, or upstream-over-amplitude inputs are sample
failures. Task 10 does not invent loudness normalization, trimming, augmentation,
or duration limits absent from the pinned algorithm.

### 4.3 CN-HuBERT

This stage independently decodes the original audio for its 16 kHz model input. It
uses the already migrated local-only `CNHubertExtractor` and saves content features
with upstream layout `(1, 768, frames)`. In fp16 mode, a non-finite result is retried
once in fp32 before the sample is quarantined.

The operational graph may require wav32k to complete first, but the HuBERT waveform
must not be derived from the wav32k file.

### 4.4 SV

The stage reads the valid wav32k artifact and uses `SpeakerEncoder.extract()`. The
output must be finite and shaped `(1, 20480)`. No speaker-count or speaker-identity
classifier is added.

### 4.5 Semantic

The stage consumes the saved HuBERT tensor and a strictly loaded base v2ProPlus S2G.
It calls the pinned `extract_latent()` path and saves integer 25 Hz semantic tokens.
The constructor takes the S2G path only from the selected profile asset set; there is
no fine-tuned-checkpoint override in this stage API.

## 5. Atomicity, state, and cache signatures

Each artifact is written to a temporary file in its destination directory, validated,
and committed with an atomic replace. A temporary file never counts as a successful
output. A stage index is also atomically rewritten after its artifact commits.

Each stage index records per sample:

- sample ID and signature;
- status;
- output paths;
- output shape and dtype where applicable;
- error category and message when failed;
- start and finish timestamps.

The run-level state is hardened to record the stage signature, outputs, timestamps,
warning count, and failure summary in addition to the existing status enum.
`completed` with a non-zero warning count is valid; no new
`completed_with_warnings` status is introduced.

At run initialization, every relevant model file receives a full SHA-256 digest.
Model directories receive a deterministic tree digest of their required files. The
result is stored in `assets.json` and reused by stage signatures. This replaces the
current weak large-file identity based only on size and mtime.

Per-sample signatures cover the canonical manifest record, original audio identity,
relevant stage configuration, profile, implementation version, dependency signature,
and relevant asset digest. A stage reruns only missing, invalid, or signature-changed
samples. Downstream invalidation follows the reviewed graph:

```text
text
wav32k -> hubert -> semantic
       -> sv
```

Text remains independent of audio-stage invalidation.

## 6. Quarantine policy and global validity barrier

The maximum tolerated bad-record count is:

```python
allowed_bad = min(5, ceil(total_nonempty_manifest_records * 0.20))
```

Examples: 6 records allow 2; 10 allow 2; 21 allow 5; 100 still allow 5. At
least one valid sample must remain.

Sample-level failures include malformed or duplicate records, missing or unreadable
audio, unsupported language, empty frontend output, invalid audio, non-finite
features, and output contract violations. Configuration-level failures such as a
missing model, failed model construction, invalid profile, corrupt state file, or
unwritable output directory fail immediately and do not consume quarantine budget.

Failure in any stage quarantines the entire sample. Downstream stages skip it and it
is absent from every final training index. Earlier valid artifacts are retained but
become unreferenced; this permits reuse if the record is later repaired. The
deterministically rewritten `quarantine.jsonl` records manifest line, sample ID when
available, audio path, failing stage, category, and message.

The stage fails as soon as the bad-record count exceeds the allowance. A failed or
interrupted run does not publish a new `valid_samples.jsonl` or aggregate training
index.

## 7. Commands and execution

Task 10 adds:

```text
voice-pipeline preprocess all -c <config>
voice-pipeline preprocess stage <name> -c <config>
```

`all` executes the existing dependency graph in deterministic topological order.
The first version remains single-process; GPU model reuse within a command is allowed,
but parallel workers and distributed preprocessing are out of scope.

Running one stage publishes only that stage's validated artifacts and index. Final
valid-sample and training indexes are published only when every required v2ProPlus
preprocessing stage has a current compatible result.

## 8. Post-training cleanup lifecycle

Preprocessing itself does not delete artifacts after completion. The later training
orchestrator invokes the cleanup primitive only after the complete requested training
run succeeds.

Default successful-training cleanup removes:

- temporary files;
- unreferenced artifacts belonging to currently quarantined samples;
- transient runtime caches owned by the run.

It retains every artifact referenced by `valid_samples.jsonl`, all aggregate indexes,
state, signatures, and reports so retraining and resume remain possible. Failed,
cancelled, or interrupted training performs no cleanup. Deleting valid preprocessing
artifacts requires a future explicit destructive option and is not part of Task 10.

## 9. Error reporting

CLI errors distinguish configuration failure from tolerated sample quarantine. A
successful command with quarantined samples exits successfully but prints the exact
count, allowance, report path, and remaining valid count. Exceeding the allowance
returns failure and leaves the last fully published valid index unchanged.

Exceptions are not swallowed. User-facing reports include the manifest line, sample
ID when present, stage, source path, and concise root cause without dumping model
weights or unrelated environment data.

## 10. Verification gates

Implementation is accepted only after all of the following pass:

1. Unit tests for stable IDs, duplicate handling, the rounded 20%/five-item limit,
   global quarantine, atomic commits, cache hits, and recursive invalidation.
2. Compatibility tests for text/BERT, wav32k, HuBERT, SV, and semantic shape, dtype,
   and pinned key behavior.
3. Integration tests showing an allowed number of bad samples continues, exceeding
   the allowance fails, and S1/S2 indexes contain exactly the same sample IDs.
4. Recovery tests showing abandoned temporary files are ignored and only missing or
   invalid samples rerun.
5. Real-asset preprocessing of a small fixture with the existing BERT, HuBERT, SV,
   base S2G, G2PW, NLTK, pyopenjtalk, and fast-langdetect assets.
6. Architecture assertions for the v2ProPlus GAN/SV route, base-S2G semantic source,
   shared multilingual frontend, and raw-audio HuBERT branch.
7. Full project regression, compilation, source-coupling scan, and whitespace checks.

## 11. Explicit non-goals

Task 10 does not add ASR, slicing, augmentation, multi-speaker validation, V3/V4,
CFM, LoRA, an external vocoder, model downloading, a database, a task queue,
distributed workers, or speculative profile abstractions.

# Task 18: Standalone Inference, CLI, and Resume Design

## Scope

Task 18 turns the single-utterance `InferenceSession` from Task 17 into a
standalone inference application layer and a thin CLI. Training is not a
runtime dependency: a valid `ModelBundle`, the profile's public assets, and
text are sufficient.

This task includes:

- in-memory long-text synthesis;
- optional reference-condition override;
- mono PCM16 WAV persistence;
- resumable chunk jobs;
- `infer synthesize`, `infer batch`, and `infer benchmark`;
- documentation for embedding inference in another Python process.

It does not include HTTP, authentication, streaming audio, GUI integration,
process supervision, evaluation, or candidate selection. Those remain later
adapters around the same application layer.

## Runtime boundaries

The dependency direction is:

```text
desktop application / future service / CLI
                    ↓
       inference application functions
                    ↓
            InferenceSession
                    ↓
 ModelBundle + v2ProPlus public model assets
```

The reusable inference modules must not import Typer. The CLI parses input,
calls those modules, prints results, and translates expected failures into a
nonzero exit code.

`InferenceSession.load(bundle_path, device)` remains valid. It gains optional
keyword-only `reference_audio`, `reference_text`, and `reference_language`
arguments:

- with no override fields, the bundle's reference is used;
- `reference_audio` and `reference_language` must be provided together;
- `reference_text` is optional when reference audio is overridden;
- omitting override text selects the official reference-free S1 path;
- override text or language without override audio is rejected.

The reference condition is still built once when the session loads. A batch
therefore uses one reference condition. Jobs requiring another reference use
another batch/session.

## Service-ready session behavior

`InferenceSession` remains a long-lived object and caches all loaded models and
reference features. A per-session lock covers RNG seeding plus S1/S2 inference,
so calls from multiple threads are deterministic and safe but execute
serially. GPU concurrency should later use multiple worker processes or model
replicas rather than sharing mutable model/RNG state inside one session.

No global working directory changes, environment mutations, training run
lookups, or CLI state are allowed in the reusable inference path. Public model
assets keep the already-approved profile paths relative to `Path.cwd()`.

## In-memory long-text synthesis

The application layer exposes a function equivalent to:

```python
synthesize_text(
    session,
    text,
    language,
    *,
    pause_ms=10,
    seed=0,
    **decoding_options,
) -> InferenceResult
```

It uses the Task 16 `TextChunker`, synthesizes chunks in source order, inserts
silence only between chunks, and returns one 32 kHz mono float32 waveform.
`pause_ms` must be a nonnegative integer and defaults to 10 ms. Each chunk uses
`(base_seed + chunk_index) % 2**63`; this avoids resetting every chunk to an
identical random sequence while allowing any chunk to be reproduced after a
restart.

The callable is independent of files and is the intended entry point for a
desktop assistant or future request handler whose input is text and whose
output is audio samples.

## WAV and resumable job format

WAV files are mono, 32 kHz, signed PCM16. Writing uses the Python standard
library and atomic temporary-file replacement; Task 18 adds no audio-writing
dependency.

The CLI's `--output` is a safe relative WAV path. Absolute paths, `..`, empty
names, and non-`.wav` suffixes are rejected. The final destination is always
namespaced by the bundle's validated `metadata.model_name`:

```text
<output_root>/
└── <model_name>/
    ├── article.wav
    └── article.infer/
        ├── manifest.json
        └── chunks/
            ├── 000001.wav
            └── ...
```

Nested safe output names are allowed and retain the same layout beside the
final WAV. `output_root` defaults to `<project_root>/outputs`.

The manifest is JSON schema version 1. It stores:

- a request signature;
- model name and exported S1/S2 hashes from bundle metadata;
- reference audio content hash and reference text/language;
- input language, complete resolved source text, pause, base seed, and all
  decoding parameters;
- ordered chunk text, derived seed, status, relative output path, and completed
  WAV SHA-256;
- final output status and SHA-256.

Each chunk entry contains at least:

```json
{
  "index": 0,
  "text": "...",
  "seed": 0,
  "status": "completed",
  "output": "chunks/000001.wav",
  "sha256": "..."
}
```

Manifest writes and chunk WAV writes are atomic. A chunk is marked completed
only after its WAV is present and hashed. On resume, completed chunks are
skipped only when their file and hash still match; missing or damaged chunk
files are regenerated.

The request signature covers every value that can change output. If an
existing work directory has a different signature, the command fails with an
instruction to choose another output or pass `--overwrite`. Only explicit
`--overwrite` may replace a conflicting job. A matching complete job with a
valid final WAV returns successfully without inference. Successful jobs retain
their manifest and chunk WAVs for audit and later reconstruction.

## CLI commands

### `infer synthesize`

Required inputs are a model bundle, explicit `zh`, `ja`, `en`, or `mixed`
language, exactly one of `--text` and `--text-file`, and a safe relative
`--output`. It accepts device, output root, pause, seed, decoding options,
optional reference override, and `--overwrite`.

Inline and TXT input use the existing `resolve_text_source`; TXT remains
UTF-8/UTF-8-SIG and `.txt` only.

### `infer batch`

`infer batch --config configs/infer.yaml` reads a strict YAML document:

```yaml
model: models/speaker_name
device: cuda:0
output_root: outputs
reference:
  audio: reference.wav
  text: "今日はいい天気ですね。"  # optional
  language: ja
defaults:
  language: mixed
  pause_ms: 10
  seed: 0
  top_k: 5
  top_p: 1.0
  temperature: 1.0
  repetition_penalty: 1.35
  noise_scale: 0.5
  speed: 1.0
jobs:
  - name: greeting
    text: "你好。"
    language: zh
  - name: article
    text_file: input.txt
```

Unknown fields are rejected. `reference` is optional and applies to the whole
batch. Defaults may be overridden per job for language, pause, seed, and
decoding values. Each job requires exactly one text source and resolves to
`<output_root>/<model_name>/<name>.wav`. Names are safe relative names without
a suffix; duplicate resolved names are rejected.

The session loads once. Jobs run sequentially and stop on the first failure.
Rerunning the batch resumes each job from its own manifest.

### `infer benchmark`

Benchmark accepts the same model, text source, language, reference override,
pause, seed, and decoding options as synthesis. Model and reference loading are
outside the timed region. Defaults are one warm-up and three measured runs;
warm-up must be nonnegative and runs must be positive.

Each run synthesizes the complete text, including chunking and pauses, with the
same base seed. The command prints generated audio duration, average and
fastest wall time, and real-time factor. It writes no WAV or manifest and does
not add concurrency, profiler, or VRAM reporting.

## Errors and observability

Expected path, config, model, reference, text, manifest, and inference failures
produce a concise `Error:` message and exit code 1. Successful synthesis prints
the final absolute WAV path and whether chunks were generated or resumed.
Batch prints job progress and a final count. Benchmark prints its four numeric
measurements. Tracebacks are not swallowed by reusable inference functions.

## Tests

Tests use a small fake `InferenceSession`; they do not load weights or perform
training. Coverage must include:

- inline/TXT mutual exclusion and language validation;
- target-person output namespacing and path traversal rejection;
- 10 ms default silence and parameter override;
- deterministic per-chunk seeds;
- atomic WAV/manifest creation;
- resume of valid chunks and regeneration of missing/damaged chunks;
- signature rejection and explicit overwrite;
- one-load batch behavior with per-job overrides;
- benchmark warm-up/run counts and no output artifacts;
- reference override validation;
- CLI help and controlled expected errors.

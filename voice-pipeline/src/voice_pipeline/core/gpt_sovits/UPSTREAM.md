# GPT-SoVITS Upstream Provenance

- Repository: `RVC-Boss/GPT-SoVITS`
- Pinned revision: `48b1a0169a28582a8984402f82cf438d3bfa6aca`
- License: MIT
- Target profile in this project: `v2ProPlus`

## Migration policy

Only code required by the v2ProPlus training and inference path is migrated. WebUI, API servers, downloaders, legacy-version compatibility branches, V3/V4-only backends, and unrelated utilities are excluded.

Algorithm-sensitive model, loss, phoneme/G2P, BERT-alignment, semantic-token, and reference-conditioning behavior must remain compatible with the pinned upstream revision. Engineering-only changes such as package paths, configuration plumbing, logging, state handling, and CLI integration are allowed and must be documented here as migration proceeds.

## Frontend compatibility slices

### ZH/JA/EN symbol vocabulary

Source: `GPT_SoVITS/text/symbols2.py`.

The internal `frontend/symbols.py` preserves the sorted shared Chinese/Japanese/English
symbol set and Japanese `[`/`]` pitch tokens. Korean and Cantonese-only symbols are
not copied in v0.1 because they are appended after the shared upstream prefix and
therefore do not change ZH/JA/EN phone IDs.

### Japanese normalization

Source: `GPT_SoVITS/text/japanese.py`.

The first slice preserves `post_replace_ph` and repeated ASCII punctuation
normalization. The pyopenjtalk G2P/prosody slice is now also migrated and covered by
dependency-gated parity tests, as documented below.

### Japanese pyopenjtalk G2P/prosody

Source: `GPT_SoVITS/text/japanese.py`.

The internal implementation now preserves the upstream full-context-label prosody
rules (`^`, `$`, `?`, `_`, `[`, `]`, `#`), Japanese mark splitting, percent-symbol
replacement, and final punctuation post-processing. The logic is exercised by pure
HTS-label parity tests so it remains testable without a local Open JTalk install.
When `pyopenjtalk` is installed, additional real-sentence compatibility tests run.

Upstream also ships a roughly 17 MB `ja_userdic/userdict.csv`. It is intentionally
not copied into the Python source package: it is a language resource rather than
source code. Integration of that optional dictionary into the project asset store is
deferred to the frontend-resource slice; until then the runtime uses pyopenjtalk's
base dictionary, matching upstream behavior when its optional user dictionary is not
available.

### Chinese pure frontend core

Source: `GPT_SoVITS/text/chinese2.py` and `GPT_SoVITS/text/opencpop-strict.txt`.

This slice preserves punctuation cleanup, consecutive-punctuation collapsing,
upstream erhua allow/deny behavior, OpenCPOP pinyin-to-phone mapping, and the
`word2ph` contract. The full 429-entry `opencpop-strict.txt` resource is copied
unchanged from the pinned revision. Heavy runtime dependencies (`TextNormalizer`,
`jieba_fast`, `pypinyin`, `ToneSandhi`, and G2PW) are intentionally not introduced
in this slice; their sentence-level orchestration will be migrated under separate
parity tests.

### Chinese G2PW/jieba orchestration contract

Source: `GPT_SoVITS/text/chinese2.py`.

The internal Chinese frontend now preserves the high-level sentence/G2PW orchestration
independently of the heavy runtime packages. Tests pin these upstream-sensitive rules:
ASCII letters are stripped before G2PW batching, empty stripped segments do not consume
a batch result, POS=`eng` advances the source-character cursor without emitting phones,
`correct_pronunciation` is applied before tone modification, erhua is applied after tone
modification, and final phone/`word2ph` mapping uses the same OpenCPOP path.

`jieba_fast`, `pypinyin` and G2PW remain runtime dependencies to wire in the production
frontend. A separate upstream-derived `frontend/tone_sandhi.py` now restores the pure
`不`/`一`/two-third-tone and merge rules under parity tests. Heavy jieba/pypinyin-dependent
continuous-third-tone and neutral-tone dictionary behavior is intentionally not claimed
complete in this baseline; that remains a later frontend integration Part.


## v2ProPlus route contract

Verified against the pinned WebUI router and `GPT_SoVITS/s2_train.py`:

- `v2ProPlus` belongs to the V2/Pro SoVITS family.
- S2 training entry is `s2_train.py`, not the V3/V4 CFM trainer.
- Generator is `SynthesizerTrn`; discriminator is `MultiPeriodDiscriminator`.
- Speaker embedding conditioning is enabled.
- Training uses adversarial G/D updates and `text_low_lr_rate=0.4`.
- V3/V4 CFM, LoRA and external-vocoder paths are excluded from the v2ProPlus profile.

These facts are encoded in `ModelProfile` and covered by route compatibility tests so
a future refactor cannot silently route v2ProPlus into the wrong architecture family.

## Optimizer/checkpoint compatibility restored in baseline

- S1 preserves the upstream optimizer gate: backward every mini-batch, optimizer step
  only when `batch_idx > 0 and batch_idx % 4 == 0`; the first step therefore contains
  mini-batches 0 through 4 and no loss division is introduced by the controller.
- S2 one-batch orchestration preserves discriminator-before-generator order and AMP
  `unscale -> clip -> step` boundaries; one completed D+G batch counts as one S2 step.
- SoVITS checkpoints use the upstream `06` two-byte header around the torch archive.

## S1 text-to-semantic core

Sources from the pinned revision:

- `GPT_SoVITS/AR/models/t2s_model.py`
- `GPT_SoVITS/AR/models/utils.py`
- `GPT_SoVITS/AR/modules/embedding.py`
- `GPT_SoVITS/AR/modules/transformer.py`
- `GPT_SoVITS/AR/modules/activation.py`
- `GPT_SoVITS/AR/modules/patched_mha_with_cache.py`
- `GPT_SoVITS/AR/modules/scaling.py`

The files are vendored under `core/gpt_sovits/s1/`. Algorithm and model
definitions are unchanged; the upstream top-level `AR.*` imports are converted
to package-relative imports, three trailing spaces are removed, and one inactive
commented-out `activation_relu_or_gelu` block is omitted.
`compatibility/s1_checkpoint.py` removes
the upstream Lightning wrapper's `model.` state-dict prefix and then performs a
strict load into `Text2SemanticDecoder`.

Optimizer, scheduler, dataset, trainer, ONNX, and V3/V4 code are intentionally
excluded from this migration slice.

## v2ProPlus S2 core

Sources from the pinned revision:

- `GPT_SoVITS/module/models.py`
- `GPT_SoVITS/module/{attentions,commons,modules,mrte_model}.py`
- `GPT_SoVITS/module/{quantize,core_vq,distrib,ddp_utils,transforms}.py`
- `GPT_SoVITS/module/{losses,mel_processing}.py`

The internal `s2_v2proplus/model.py` contains only the `TextEncoder`, residual
flow/posterior, waveform generator, multi-period discriminator, and
`SynthesizerTrn` ranges required by v2ProPlus. The upstream CFM, F5-TTS,
`SynthesizerTrnV3`, V3/V4 and external-vocoder code is not imported.

Package imports are relative. The V2 text embedding retains the official 732-row
checkpoint shape as `V2PROPLUS_SYMBOL_COUNT`; the shared ZH/JA/EN frontend still
emits the unchanged 324-ID prefix, while the unused Korean/Cantonese tail rows
remain loadable for strict checkpoint compatibility.

`compatibility/s2_checkpoint.py` constructs the profile-specific G or D model,
forces the `v2ProPlus` route, and strictly loads the official state dict.

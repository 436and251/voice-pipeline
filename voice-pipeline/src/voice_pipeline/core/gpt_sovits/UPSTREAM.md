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

### Japanese normalization and pyopenjtalk G2P/prosody

Source: `GPT_SoVITS/text/japanese.py`.

The internal implementation now preserves the upstream full-context-label prosody
rules (`^`, `$`, `?`, `_`, `[`, `]`, `#`), Japanese mark splitting, percent-symbol
replacement, repeated-punctuation normalization, and final punctuation
post-processing. Pure HTS-label parity tests and real pyopenjtalk sentence tests cover
the implementation. `JapaneseFrontend` normalizes once and returns `word2ph=None`.

Upstream also ships a roughly 17 MB `ja_userdic/userdict.csv`. It is intentionally
not copied into the Python source package: it is a language resource rather than
source code. The runtime uses pyopenjtalk's base dictionary, matching upstream
behavior when its optional user dictionary is unavailable.

### Chinese G2PW production frontend

Sources from the pinned revision:

- `GPT_SoVITS/text/chinese2.py`
- `GPT_SoVITS/text/opencpop-strict.txt`
- `GPT_SoVITS/text/tone_sandhi.py`
- `GPT_SoVITS/text/zh_normalization/`
- `GPT_SoVITS/text/g2pw/`

The internal frontend preserves TextNormalizer cleanup, jieba POS segmentation,
the complete ToneSandhi merge/modification rules, batched G2PW polyphone inference,
pronunciation correction, erhua, OpenCPOP mapping, and `word2ph`. The 429-entry
OpenCPOP map and G2PW polyphonic dictionaries/caches are copied from the pinned
revision. Real-model tests verify that `重庆。` produces `ch ong2 q ing4`.

The G2PW/PaddleSpeech-derived files retain their Apache-2.0 headers. Package imports
are relative, the tokenizer is local-only, and the upstream ModelScope ZIP downloader
has been removed. Construction validates every required local G2PW file and raises
`FileNotFoundError` instead of downloading. The G2PW ONNX directory and Chinese
RoBERTa model remain external model assets.

### English G2P and lexical resources

Sources: `GPT_SoVITS/text/english.py`, `GPT_SoVITS/text/en_normalization/expend.py`,
and the pinned `cmudict*.rep`, `engdict*`, and `namedict_cache.pickle` resources.

The adapter preserves GPT-SoVITS dictionary precedence, name lookup, homograph/POS
routing, number normalization, OOV prediction, ARPA filtering, and punctuation
mapping. All five lexical files are byte-identical to the pinned source. CMUdict's
copyright and redistribution terms remain embedded in `cmudict.rep`.
`EnglishFrontend` loads the current NLTK English perceptron tagger only from the
explicit asset directory. It neutralizes `g2p_en`'s obsolete import-time downloader
and does not require NLTK's duplicate CMU corpus because GPT-SoVITS supplies its own.

### Unified multilingual routing and BERT alignment

`MultilingualFrontend` exposes one ZH/JA/EN/mixed contract. Explicit text language
always selects its matching frontend; the speaker's Japanese training origin never
overrides text-language routing. Mixed spans use an explicit local `lid.176.bin`, are
restricted to ZH/JA/EN, processed independently, and concatenated in source order.
Only Chinese spans use local RoBERTa features expanded through `word2ph`; Japanese
and English spans contribute aligned zero BERT columns. Every result validates as
1024 by final phone count.


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

## v2ProPlus feature extractors

Sources from the pinned revision:

- `GPT_SoVITS/feature_extractor/cnhubert.py`
- `GPT_SoVITS/eres2net/ERes2NetV2.py`
- `GPT_SoVITS/eres2net/fusion.py`
- `GPT_SoVITS/eres2net/kaldi.py`
- `GPT_SoVITS/prepare_datasets/2-get-sv.py`
- `tools/my_utils.py`

`features/cnhubert.py` preserves the local-only Transformers HuBERT loading and
returns upstream content layout `(batch, 768, frames)`. `features/audio.py`
uses the same ffmpeg mono float32 decode and 32 kHz resampling contract without
the unrelated WebUI helpers from `tools/my_utils.py`.

`features/speaker.py` adapts only the ERes2NetV2/AFF layers required by the
official v2ProPlus `forward3()` speaker vector path. It strictly loads the
official checkpoint and preserves `32k -> 16k -> 80-bin Kaldi fbank ->
(batch, 20480)`. The copied Kaldi implementation is omitted because the
installed `torchaudio.compliance.kaldi.fbank` produced bit-identical output for
the same input and arguments.

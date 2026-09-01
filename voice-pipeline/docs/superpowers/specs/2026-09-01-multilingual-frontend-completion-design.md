# Multilingual Frontend Completion Design

## Status and purpose

Tasks 8 and 9 provide real S1 and v2ProPlus S2 model construction, and Task 7
now provides HuBERT and speaker features. Task 6, however, has only partial
Chinese and Japanese compatibility slices. Task 10 preprocessing cannot be
wired correctly until one production frontend produces phones, phone IDs,
`word2ph`, normalized text, and BERT features for every supported language.

This design completes that prerequisite without adding preprocessing, training,
inference, CLI, V3/V4, Cantonese, or Korean behavior.

## Supported contract

The public result is immutable and contains:

```python
FrontendResult(
    normalized_text: str,
    phones: list[str],
    phone_ids: list[int],
    word2ph: list[int],
    bert_features: torch.Tensor,
)
```

`MultilingualFrontend.process(text, language)` accepts `zh`, `ja`, `en`, and
`mixed`. All outputs use the existing v2ProPlus ZH/JA/EN symbol prefix. The BERT
tensor has shape `(1024, phone_count)`.

## External assets

The frontend receives paths explicitly; it does not download at runtime or
read the external GPT-SoVITS checkout.

- Chinese RoBERTa: `chinese-roberta-wwm-ext-large`
- Chinese polyphone model: `G2PWModel` 1.1 with its ONNX model and five metadata
  files
- English POS data: NLTK `averaged_perceptron_tagger_eng`
- Japanese runtime: installed `pyopenjtalk` and its base dictionary

The G2PW model and NLTK data are already present under the ignored project
model store. Existing RoBERTa weights may remain in the user's current
pretrained-model directory during compatibility tests; callers pass that path.

## Internal components

### Chinese

Complete the current pure Chinese primitives with the pinned upstream
normalizer, jieba POS segmentation, full ToneSandhi rules, G2PW ONNX adapter,
pronunciation corrections, erhua handling, and OpenCPOP phone mapping. G2PW is
mandatory for the production Chinese path; there is no silent pypinyin-only
fallback.

### Japanese

Use the migrated pyopenjtalk full-context-label prosody path. Complete
normalization and `word2ph` behavior needed by training, while retaining the
already tested pitch-token rules. The optional upstream custom Japanese user
dictionary remains outside this slice because the approved baseline uses the
installed base dictionary when that asset is absent.

### English

Migrate the pinned GPT-SoVITS English G2P behavior and its CMU, hot-word, name,
and cache resources. It uses `g2p_en` and the explicit NLTK data path. English
phones receive an all-zero `(1024, phone_count)` BERT tensor, matching upstream.

### Mixed language

Segment only ZH/JA/EN spans. Each span goes through its language frontend, then
phones, IDs, normalized text, and BERT columns are concatenated in source order.
Unsupported detected languages fail clearly instead of being routed through a
different language.

### BERT alignment

Load Chinese RoBERTa locally with Transformers. Use hidden layer `-3`, discard
special-token rows, and repeat each character feature by `word2ph`. Only Chinese
spans use model-derived BERT features; Japanese and English spans use zeros.
Assert that the final number of BERT columns equals the number of phone IDs.

## Dependency and import policy

Heavy libraries and assets are loaded when a production frontend instance is
constructed, not when the package is imported. Source code and small lexical
resources are vendored with pinned provenance. Model weights, NLTK data, and
generated caches remain ignored external assets. Dependency declarations remain
unchanged until the final environment-consolidation Part requested by the user.

## Errors

- Missing model or dictionary assets raise `FileNotFoundError` naming the path.
- Unsupported language codes raise `ValueError`.
- Unknown phone symbols and BERT/phone alignment mismatches fail immediately.
- No runtime downloader or network fallback is permitted.

## Verification

Development follows RED, GREEN, full regression:

1. Add fixed ZH/JA/EN/mixed contract and parity cases.
2. Run real G2PW inference on ambiguity-sensitive Chinese phrases.
3. Run real RoBERTa alignment and assert `(1024, phone_count)`.
4. Run real pyopenjtalk and `g2p_en` paths.
5. Prove the package imports without the external GPT-SoVITS checkout.
6. Run all compatibility tests with the real S1, S2, HuBERT, SV, BERT, G2PW,
   NLTK, and pyopenjtalk assets.
7. Run `compileall` and staged whitespace checks.

## Acceptance boundary

This Part ends when the production `MultilingualFrontend` contract is green for
ZH, JA, EN, and mixed text with real assets. It does not create preprocessing
files or commands. Task 10 begins only after this Part is reviewed.

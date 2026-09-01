# Multilingual Frontend Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Task 6 with one production ZH/JA/EN/mixed frontend whose language-specific phones and BERT columns are identical in training and inference.

**Architecture:** A single `MultilingualFrontend` facade dispatches to pinned GPT-SoVITS Chinese, Japanese, and English backends. Chinese alone uses G2PW and RoBERTa-derived BERT; Japanese and English use their native G2P backends and zero BERT; mixed text is segmented and concatenated after each span is fully aligned.

**Tech Stack:** Python 3.12, PyTorch, Transformers, ONNX Runtime, OpenCC, jieba, pypinyin, pyopenjtalk, g2p_en, NLTK, fast_langdetect, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-multilingual-frontend-completion-design.md`

## Global constraints

- Work directly on `main`; the user explicitly approved this repository workflow.
- Use only `D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe`.
- Use `--basetemp .pytest_cache\codex-temp` because the system pytest temp root is not readable.
- Do not add dependency declarations yet; environment consolidation is deferred by user request.
- Do not generate ZIP files.
- Runtime code must not import or read the external GPT-SoVITS checkout.
- Pin behavioral provenance to GPT-SoVITS commit `48b1a0169a28582a8984402f82cf438d3bfa6aca`.
- Support only `zh`, `ja`, `en`, and `mixed`; do not add Korean, Cantonese, V3, or V4 behavior.
- No runtime downloads. Every missing asset fails with its exact path.
- `bert_features.shape` must always equal `(1024, len(phone_ids))`.
- Pure Chinese results contain `word2ph`; Japanese, English, and mixed results contain `None`.

---

### Task 1: Add the frontend result contract and Chinese BERT aligner

**Files:**
- Create: `src/voice_pipeline/core/gpt_sovits/frontend/contract.py`
- Create: `src/voice_pipeline/core/gpt_sovits/frontend/bert.py`
- Create: `tests/compat/test_frontend_contract.py`

**Interfaces:**
- Produces: `FrontendResult(normalized_text, phones, phone_ids, word2ph, bert_features)`
- Produces: `BertAligner(model_path, device="cpu")`
- Produces: `BertAligner.extract(text, word2ph) -> torch.Tensor`
- Produces: `expand_character_features(features, word2ph) -> torch.Tensor`

- [ ] **Step 1: Write failing contract and alignment tests**

```python
def test_frontend_result_rejects_misaligned_bert_columns():
    with pytest.raises(ValueError, match="BERT columns"):
        FrontendResult("a", ["AA"], [5], None, torch.zeros(1024, 2))


def test_character_features_repeat_by_literal_word2ph():
    characters = torch.tensor([[1.0, 2.0], [10.0, 20.0]])
    assert torch.equal(
        expand_character_features(characters, [2, 1]),
        torch.tensor([[1.0, 1.0, 2.0], [10.0, 10.0, 20.0]]),
    )
```

Add a real-asset test using `VOICE_PIPELINE_TEST_BERT_DIR`, text `"你好"`, and
`word2ph=[2, 2]`; assert shape `(1024, 4)`, float32 dtype, finite values, and a
nonzero tensor.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest -q --basetemp .pytest_cache\codex-temp `
  tests\compat\test_frontend_contract.py
```

Expected: import failure because `contract.py` and `bert.py` do not exist.

- [ ] **Step 3: Implement the minimum contract and aligner**

```python
@dataclass(frozen=True, slots=True)
class FrontendResult:
    normalized_text: str
    phones: list[str]
    phone_ids: list[int]
    word2ph: list[int] | None
    bert_features: torch.Tensor

    def __post_init__(self) -> None:
        if len(self.phones) != len(self.phone_ids):
            raise ValueError("phone and phone-id lengths differ")
        if self.bert_features.shape != (1024, len(self.phone_ids)):
            raise ValueError("BERT columns must match phone IDs")
```

`BertAligner` must load tokenizer/model with `local_files_only=True`, use hidden
state `[-3]`, remove `[CLS]`/`[SEP]`, and call the independently tested repeat
helper. It must not mutate global Transformers logging.

- [ ] **Step 4: Run GREEN and existing symbol tests**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest -q --basetemp .pytest_cache\codex-temp `
  tests\compat\test_frontend_contract.py tests\compat\test_symbols_parity.py
```

- [ ] **Step 5: Commit**

```powershell
git add -- src/voice_pipeline/core/gpt_sovits/frontend/contract.py `
  src/voice_pipeline/core/gpt_sovits/frontend/bert.py `
  tests/compat/test_frontend_contract.py
git commit -m "feat: add frontend contract and bert alignment"
```

---

### Task 2: Complete the real Chinese G2PW frontend

**Files:**
- Modify: `src/voice_pipeline/core/gpt_sovits/frontend/chinese.py`
- Replace: `src/voice_pipeline/core/gpt_sovits/frontend/tone_sandhi.py`
- Create: `src/voice_pipeline/core/gpt_sovits/frontend/chinese_runtime.py`
- Create: `src/voice_pipeline/core/gpt_sovits/frontend/g2pw/`
- Create: `src/voice_pipeline/core/gpt_sovits/frontend/zh_normalization/`
- Test: `tests/compat/test_chinese_runtime.py`
- Modify: `tests/compat/test_tone_sandhi_parity.py`

**Interfaces:**
- Consumes: `BertAligner.extract(text, word2ph)` from Task 1 only at the facade layer.
- Produces: `ChineseFrontend(g2pw_model_path, tokenizer_path)`
- Produces: `ChineseFrontend.process(text) -> tuple[str, list[str], list[int]]`

- [ ] **Step 1: Record exact pinned source closure before copying**

Use these upstream sources only:

```text
GPT_SoVITS/text/chinese2.py
GPT_SoVITS/text/tone_sandhi.py
GPT_SoVITS/text/g2pw/{g2pw,onnx_api,dataset,utils}.py
GPT_SoVITS/text/g2pw/{polyphonic.rep,polyphonic-fix.rep,polyphonic.pickle,polyphonic.md5}
GPT_SoVITS/text/zh_normalization/**
GPT_SoVITS/text/opencpop-strict.txt
```

Remove the upstream downloader from `onnx_api.py`. Its constructor must validate
these exact model files: `config.py`, either `g2pW.onnx` or `g2pw.onnx`,
`POLYPHONIC_CHARS.txt`, `MONOPHONIC_CHARS.txt`,
`bopomofo_to_pinyin_wo_tune_dict.json`, and `char_bopomofo_dict.json`.

- [ ] **Step 2: Write failing real G2PW and no-download tests**

```python
def test_real_g2pw_disambiguates_chongqing(g2pw_dir, bert_dir):
    normalized, phones, word2ph = ChineseFrontend(g2pw_dir, bert_dir).process("重庆。")
    assert normalized == "重庆."
    assert phones[:4] == ["ch", "ong2", "q", "ing4"]
    assert sum(word2ph) == len(phones)
    assert len(word2ph) == len(normalized)


def test_missing_g2pw_directory_fails_without_downloading(tmp_path, bert_dir):
    missing = tmp_path / "G2PWModel"
    with pytest.raises(FileNotFoundError, match="G2PWModel"):
        ChineseFrontend(missing, bert_dir)
    assert not missing.exists()
```

Keep the existing injectable orchestration tests. Add full upstream ToneSandhi
cases for `不`, `一`, neutral tone, reduplication, three-tone sequences, and the
`merge_bu`, `merge_yi`, `merge_er`, and continuous-third-tone paths.

- [ ] **Step 3: Run RED**

```powershell
$env:VOICE_PIPELINE_TEST_G2PW_DIR='models\pretrained\v2proplus\g2pw\G2PWModel'
$env:VOICE_PIPELINE_TEST_BERT_DIR='D:\AI-Training\voice-clone\GPT-SoVITS\GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large'
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest -q --basetemp .pytest_cache\codex-temp `
  tests\compat\test_chinese_runtime.py tests\compat\test_tone_sandhi_parity.py
```

Expected: import/constructor failures for the missing production adapter and
failures for full ToneSandhi rules not present in the current partial slice.

- [ ] **Step 4: Adapt the source with package-relative imports**

`ChineseFrontend.process` must follow this order exactly:

```text
TextNormalizer.normalize
-> punctuation replacement/collapse
-> sentence split
-> jieba POS + ToneSandhi pre-merge
-> batched G2PW
-> pronunciation correction
-> tone modification + erhua
-> OpenCPOP phone mapping
```

Delete commented-out executable code while retaining license and algorithm
explanations. Do not import the external checkout and do not add a pypinyin-only
production fallback.

- [ ] **Step 5: Run GREEN plus all existing Chinese parity tests**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest -q --basetemp .pytest_cache\codex-temp `
  tests\compat\test_chinese_runtime.py `
  tests\compat\test_chinese_core_parity.py `
  tests\compat\test_chinese_g2p_orchestration.py `
  tests\compat\test_tone_sandhi_parity.py
```

- [ ] **Step 6: Commit**

```powershell
git add -- src/voice_pipeline/core/gpt_sovits/frontend/chinese.py `
  src/voice_pipeline/core/gpt_sovits/frontend/chinese_runtime.py `
  src/voice_pipeline/core/gpt_sovits/frontend/tone_sandhi.py `
  src/voice_pipeline/core/gpt_sovits/frontend/g2pw `
  src/voice_pipeline/core/gpt_sovits/frontend/zh_normalization `
  tests/compat/test_chinese_runtime.py `
  tests/compat/test_tone_sandhi_parity.py
git commit -m "feat: complete chinese g2pw frontend"
```

---

### Task 3: Complete the Japanese training frontend

**Files:**
- Modify: `src/voice_pipeline/core/gpt_sovits/frontend/japanese.py`
- Create: `src/voice_pipeline/core/gpt_sovits/frontend/japanese_runtime.py`
- Create: `tests/compat/test_japanese_runtime.py`

**Interfaces:**
- Produces: `JapaneseFrontend.process(text) -> tuple[str, list[str], None]`

- [ ] **Step 1: Write the failing real runtime test**

```python
def test_japanese_runtime_uses_pyopenjtalk_prosody():
    normalized, phones, word2ph = JapaneseFrontend().process("こんにちは。")
    assert normalized == "こんにちは。"
    assert phones == ["k", "o", "[", "N", "n", "i", "ch", "i", "w", "a", "."]
    assert word2ph is None
```

Add normalization cases for `%` replacement and repeated punctuation using
literals derived from the pinned upstream module. Do not add number or text
normalization rules absent from the pinned Japanese frontend.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest -q --basetemp .pytest_cache\codex-temp `
  tests\compat\test_japanese_runtime.py
```

Expected: import failure because `JapaneseFrontend` does not exist.

- [ ] **Step 3: Complete normalization and add the thin runtime adapter**

Reuse the existing `pyopenjtalk_g2p_prosody`; do not create another prosody
implementation. The adapter normalizes once, calls `g2p(..., with_prosody=True)`,
and returns `word2ph=None`.

- [ ] **Step 4: Run GREEN and all Japanese parity tests**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest -q --basetemp .pytest_cache\codex-temp `
  tests\compat\test_japanese_runtime.py `
  tests\compat\test_japanese_normalization_parity.py `
  tests\compat\test_japanese_prosody_parity.py
```

- [ ] **Step 5: Commit**

```powershell
git add -- src/voice_pipeline/core/gpt_sovits/frontend/japanese.py `
  src/voice_pipeline/core/gpt_sovits/frontend/japanese_runtime.py `
  tests/compat/test_japanese_runtime.py
git commit -m "feat: complete japanese frontend"
```

---

### Task 4: Migrate the pinned English G2P and lexical resources

**Files:**
- Create: `src/voice_pipeline/core/gpt_sovits/frontend/english.py`
- Copy: `src/voice_pipeline/core/gpt_sovits/frontend/resources/cmudict.rep`
- Copy: `src/voice_pipeline/core/gpt_sovits/frontend/resources/cmudict-fast.rep`
- Copy: `src/voice_pipeline/core/gpt_sovits/frontend/resources/engdict-hot.rep`
- Copy: `src/voice_pipeline/core/gpt_sovits/frontend/resources/engdict_cache.pickle`
- Copy: `src/voice_pipeline/core/gpt_sovits/frontend/resources/namedict_cache.pickle`
- Create: `tests/compat/test_english_runtime.py`

**Interfaces:**
- Produces: `EnglishFrontend(nltk_data_path)`
- Produces: `EnglishFrontend.process(text) -> tuple[str, list[str], None]`

- [ ] **Step 1: Write failing resource and real G2P tests**

```python
def test_english_runtime_uses_official_cmu_path(nltk_data_dir):
    normalized, phones, word2ph = EnglishFrontend(nltk_data_dir).process("Hello world.")
    assert normalized == "Hello world."
    assert phones == ["HH", "AH0", "L", "OW1", "W", "ER1", "L", "D", "."]
    assert word2ph is None


def test_english_runtime_requires_current_nltk_tagger(tmp_path):
    with pytest.raises(LookupError, match="averaged_perceptron_tagger_eng"):
        EnglishFrontend(tmp_path).process("Hello.")
```

- [ ] **Step 2: Run RED**

```powershell
$env:VOICE_PIPELINE_TEST_NLTK_DATA='models\pretrained\v2proplus\g2p\en\nltk_data'
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest -q --basetemp .pytest_cache\codex-temp `
  tests\compat\test_english_runtime.py
```

Expected: import failure because the internal English runtime does not exist.

- [ ] **Step 3: Adapt the pinned English module**

Keep GPT-SoVITS dictionary precedence, name handling, homograph POS routing,
number normalization, ARPA filtering, and punctuation mapping. Resolve lexical
resources relative to the internal `resources` directory. Prepend the explicit
NLTK data directory only for this runtime; do not invoke `nltk.download()`.

- [ ] **Step 4: Run GREEN**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest -q --basetemp .pytest_cache\codex-temp `
  tests\compat\test_english_runtime.py
```

- [ ] **Step 5: Commit**

```powershell
git add -- src/voice_pipeline/core/gpt_sovits/frontend/english.py `
  src/voice_pipeline/core/gpt_sovits/frontend/resources `
  tests/compat/test_english_runtime.py
git commit -m "feat: migrate english g2p frontend"
```

---

### Task 5: Add mixed segmentation and the unified production facade

**Files:**
- Create: `src/voice_pipeline/core/gpt_sovits/frontend/language_segmenter.py`
- Create: `src/voice_pipeline/core/gpt_sovits/frontend/multilingual.py`
- Modify: `src/voice_pipeline/core/gpt_sovits/frontend/__init__.py`
- Create: `tests/compat/test_multilingual_frontend.py`

**Interfaces:**
- Consumes: `FrontendResult`, `BertAligner`, and the three language backends.
- Produces: `LanguageSegmenter(model_dir).segment(text) -> list[tuple[str, str]]`
- Produces: `MultilingualFrontend(bert_path, g2pw_path, nltk_data_path, langdetect_path, device="cpu")`
- Produces: `MultilingualFrontend.process(text, language) -> FrontendResult`
- Internal: `MultilingualFrontend._process_span(text, language) -> FrontendResult`

- [ ] **Step 1: Write failing route and real mixed tests**

```python
def test_explicit_languages_never_route_through_japanese(frontend):
    zh = frontend.process("重庆。", "zh")
    ja = frontend.process("こんにちは。", "ja")
    en = frontend.process("Hello world.", "en")
    assert zh.phones[:2] == ["ch", "ong2"]
    assert ja.phones[:2] == ["k", "o"]
    assert en.phones[:2] == ["HH", "AH0"]
    assert zh.word2ph is not None
    assert ja.word2ph is None and en.word2ph is None


def test_mixed_result_concatenates_aligned_language_spans(frontend):
    result = frontend.process("你好。Hello world.", "mixed")
    assert result.word2ph is None
    assert result.bert_features.shape == (1024, len(result.phone_ids))
    assert "HH" in result.phones
    assert torch.count_nonzero(result.bert_features) > 0
```

Also test empty text, unsupported language `ko`, unknown detector output, and a
missing `lid.176.bin` path.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest -q --basetemp .pytest_cache\codex-temp `
  tests\compat\test_multilingual_frontend.py
```

Expected: import failure because the segmenter and facade do not exist.

- [ ] **Step 3: Implement explicit routing and concatenation**

```python
def process(self, text: str, language: str) -> FrontendResult:
    if language not in {"zh", "ja", "en", "mixed"}:
        raise ValueError(f"unsupported language: {language}")
    spans = [(language, text)] if language != "mixed" else self.segmenter.segment(text)
    results = [self._process_span(span_text, span_language) for span_language, span_text in spans]
    phones = [phone for result in results for phone in result.phones]
    bert = torch.cat([result.bert_features for result in results], dim=1)
    return FrontendResult(
        "".join(result.normalized_text for result in results),
        phones,
        phone_ids(phones),
        results[0].word2ph if language == "zh" else None,
        bert,
    )
```

Adapt the pinned LangSegmenter rules, but inject the exact fast_langdetect model
directory. Reject mapped languages outside ZH/JA/EN. Japanese training origin
must never override the explicit text language.

- [ ] **Step 4: Run GREEN and the complete frontend compatibility suite**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest -q --basetemp .pytest_cache\codex-temp tests\compat -k "frontend or chinese or japanese or english or symbols or tone"
```

- [ ] **Step 5: Commit**

```powershell
git add -- src/voice_pipeline/core/gpt_sovits/frontend `
  tests/compat/test_multilingual_frontend.py
git commit -m "feat: add unified multilingual frontend"
```

---

### Task 6: Record provenance and run real-asset regression

**Files:**
- Modify: `src/voice_pipeline/core/gpt_sovits/UPSTREAM.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `.sdd-progress.md`

**Interfaces:**
- Verifies the completed Task 6 contract; adds no production API.

- [ ] **Step 1: Record every copied/adapted path and license**

Document the GPT-SoVITS pinned paths, G2PW/PaddleSpeech-derived Apache notices,
English lexical resource origins, removed runtime download behavior, and the
explicitly external model assets.

- [ ] **Step 2: Run all real-asset tests**

```powershell
$env:VOICE_PIPELINE_TEST_S1_CHECKPOINT='D:\AI-Training\voice-clone\GPT-SoVITS\GPT_SoVITS\pretrained_models\s1v3.ckpt'
$env:VOICE_PIPELINE_TEST_S2_DIR='D:\AI-Training\voice-clone\GPT-SoVITS\GPT_SoVITS\pretrained_models\v2Pro'
$env:VOICE_PIPELINE_TEST_HUBERT_DIR='D:\AI-Training\voice-clone\GPT-SoVITS\GPT_SoVITS\pretrained_models\chinese-hubert-base'
$env:VOICE_PIPELINE_TEST_SV_CHECKPOINT='D:\AI-Training\voice-clone\GPT-SoVITS\GPT_SoVITS\pretrained_models\sv\pretrained_eres2netv2w24s4ep4.ckpt'
$env:VOICE_PIPELINE_TEST_BERT_DIR='D:\AI-Training\voice-clone\GPT-SoVITS\GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large'
$env:VOICE_PIPELINE_TEST_G2PW_DIR='models\pretrained\v2proplus\g2pw\G2PWModel'
$env:VOICE_PIPELINE_TEST_NLTK_DATA='models\pretrained\v2proplus\g2p\en\nltk_data'
$env:VOICE_PIPELINE_TEST_LANGDETECT_DIR='D:\AI-Training\voice-clone\GPT-SoVITS\GPT_SoVITS\pretrained_models\fast_langdetect'
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m pytest -q --basetemp .pytest_cache\codex-temp
```

Expected: every applicable test passes; no frontend test skips because all
required assets and language runtimes are present.

- [ ] **Step 3: Run compilation, coupling, and whitespace checks**

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\python.exe' `
  -m compileall -q src tests
rg -n "GPT_SoVITS|sys\.path" src/voice_pipeline/core/gpt_sovits/frontend
git diff --check
```

The `rg` command may find prose provenance only; production Python must contain
no external-checkout import or path.

- [ ] **Step 4: Commit the Part ledger and provenance**

```powershell
git add -- src/voice_pipeline/core/gpt_sovits/UPSTREAM.md `
  THIRD_PARTY_NOTICES.md .sdd-progress.md
git commit -m "docs: record multilingual frontend provenance"
```

- [ ] **Step 5: Stop for user review**

Report commits, exact test count, real assets exercised, remaining warnings,
and the Task 10 boundary. Do not start preprocessing and do not create a ZIP.

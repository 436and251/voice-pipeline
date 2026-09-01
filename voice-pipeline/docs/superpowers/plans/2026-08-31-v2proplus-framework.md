# GPT-SoVITS v2ProPlus Voice Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, CLI-first GPT-SoVITS v2ProPlus training, evaluation, model-selection, export, and inference framework that does not depend on an external GPT-SoVITS source checkout.

**Architecture:** The project uses a `src/voice_pipeline/` layout. GPT-SoVITS v2ProPlus model behavior is migrated into an internal compatibility-preserving core, while orchestration, profiles, training, evaluation, inference, state, and CLI remain separate. The first release implements only `v2ProPlus`, but all model-family decisions are routed through a GPT-SoVITS profile registry so later GPT-SoVITS versions can be added without rewriting the platform layer.

**Tech Stack:** Python 3.12, PyTorch, torchaudio, transformers, pytorch-lightning components where required by upstream S1 compatibility, PyYAML, Typer, pytest, librosa, soundfile, numpy, scipy, tqdm.

**Spec:** `GPT-SoVITS_v2ProPlus_Training_Inference_Framework_Design_v0.1.md`

## Global Constraints

- Repository layout must be `voice-pipeline/src/voice_pipeline/`; do not place the Python package directly at the repository root.
- Runtime must not depend on `D:/AI-Training/voice-clone/GPT-SoVITS/` or any other external GPT-SoVITS source checkout.
- First release implements GPT-SoVITS `v2ProPlus` only.
- Keep a GPT-SoVITS profile registry so later GPT-SoVITS versions such as V4 can be added without rewriting CLI, run-state, evaluation, artifact, and orchestration layers.
- Do not modify model architecture, forward behavior, official losses, phoneme vocabulary, G2P semantics, BERT alignment, semantic representation, or v2ProPlus reference-conditioning behavior.
- Do not add LoRA/PEFT or new preservation losses in the first release.
- Training input manifest format is `audio_path|speaker|language|text`.
- Training data may be Japanese-only, while validation targets must support Chinese, Japanese, and English.
- S1 uses explicit gradient accumulation and optimizer-step budgeting.
- S2 uses native v2ProPlus G/D training semantics and step budgeting.
- Preprocessing must include text/BERT, 32k audio, CN-HuBERT SSL, ERes2NetV2 SV embedding, and semantic-token extraction.
- `--text` and `--text-file` are mutually exclusive; `--text-file` supports TXT only with UTF-8/UTF-8-SIG.
- Evaluation is separate from training and does not backpropagate.
- Model selection uses hard constraints, ranking, and a human-listening shortlist.
- Models, runs, and outputs are not committed to Git.
- Upstream-derived code must retain the MIT notice and provenance.

---

## File Structure

```text
voice-pipeline/
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
│   ├── train.example.yaml
│   ├── infer.example.yaml
│   └── eval/
│       ├── zh.txt
│       ├── ja.txt
│       ├── en.txt
│       └── mixed.txt
├── models/
├── runs/
├── outputs/
├── src/
│   └── voice_pipeline/
│       ├── __init__.py
│       ├── cli/
│       ├── common/
│       ├── profiles/
│       ├── core/gpt_sovits/
│       ├── training/
│       ├── evaluation/
│       ├── inference/
│       └── pipeline/
└── tests/
```

---

### Task 1: Bootstrap the package, CLI entry point, and repository policy

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/voice_pipeline/__init__.py`
- Create: `src/voice_pipeline/cli/main.py`
- Create: `tests/test_cli.py`
- Create: `README.md`
- Create: `THIRD_PARTY_NOTICES.md`

**Interfaces:**
- Produces: console entry point `voice-pipeline`
- Produces: `voice_pipeline.__version__`

- [ ] **Step 1: Write failing CLI tests**

```python
from typer.testing import CliRunner
from voice_pipeline.cli.main import app

runner = CliRunner()

def test_root_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "voice-pipeline" in result.stdout.lower()

def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: import failure because package and CLI do not exist.

- [ ] **Step 3: Add minimal package and CLI**

```python
# src/voice_pipeline/__init__.py
__version__ = "0.1.0"
```

```python
# src/voice_pipeline/cli/main.py
import typer
from voice_pipeline import __version__

app = typer.Typer(help="GPT-SoVITS voice training and inference pipeline.")

@app.command()
def version() -> None:
    typer.echo(__version__)

if __name__ == "__main__":
    app()
```

Configure `pyproject.toml` with:

```toml
[project]
name = "voice-pipeline"
version = "0.1.0"
requires-python = ">=3.12"

[project.scripts]
voice-pipeline = "voice_pipeline.cli.main:app"

[tool.pytest.ini_options]
pythonpath = ["src"]
```

Add to `.gitignore`:

```gitignore
/models/
/runs/
/outputs/
__pycache__/
.pytest_cache/
*.pyc
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt .gitignore README.md THIRD_PARTY_NOTICES.md src tests/test_cli.py
git commit -m "chore: bootstrap voice pipeline package"
```

---

### Task 2: Add configuration models and v2ProPlus profile registry

**Files:**
- Create: `src/voice_pipeline/common/errors.py`
- Create: `src/voice_pipeline/common/paths.py`
- Create: `src/voice_pipeline/profiles/base.py`
- Create: `src/voice_pipeline/profiles/registry.py`
- Create: `src/voice_pipeline/profiles/v2proplus.py`
- Create: `tests/test_profile.py`

**Interfaces:**
- Produces: `ModelProfile`
- Produces: `V2ProPlusProfile`
- Produces: `ProfileRegistry.get(name: str) -> ModelProfile`
- Produces: `resolve_project_path(project_root: Path, relative: str) -> Path`

- [ ] **Step 1: Write profile tests**

```python
from pathlib import Path
import pytest
from voice_pipeline.profiles.registry import ProfileRegistry

def test_v2proplus_profile_paths():
    profile = ProfileRegistry.get("v2ProPlus")
    assert profile.name == "v2ProPlus"
    assert profile.sample_rate == 32000
    assert profile.semantic_frame_rate == "25hz"
    assert profile.requires_sv is True
    assert profile.s1_relative_path == "models/pretrained/v2proplus/s1/s1v3.ckpt"
    assert profile.s2g_relative_path.endswith("s2Gv2ProPlus.pth")
    assert profile.s2d_relative_path.endswith("s2Dv2ProPlus.pth")

def test_unknown_profile_rejected():
    with pytest.raises(KeyError):
        ProfileRegistry.get("v4")
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/test_profile.py -v
```

- [ ] **Step 3: Implement immutable profile model**

Implement `ModelProfile` as a frozen dataclass with:

```python
@dataclass(frozen=True)
class ModelProfile:
    name: str
    sample_rate: int
    semantic_frame_rate: str
    requires_sv: bool
    s1_relative_path: str
    s2g_relative_path: str
    s2d_relative_path: str
    bert_relative_path: str
    hubert_relative_path: str
    speaker_relative_path: str
```

Implement only `v2ProPlus` in registry.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_profile.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/voice_pipeline/common src/voice_pipeline/profiles tests/test_profile.py
git commit -m "feat: add v2proplus model profile"
```

---

### Task 3: Add asset verification and model-store layout

**Files:**
- Create: `src/voice_pipeline/common/assets.py`
- Create: `src/voice_pipeline/cli/models.py`
- Modify: `src/voice_pipeline/cli/main.py`
- Create: `tests/test_assets.py`

**Interfaces:**
- Produces: `AssetCheck`
- Produces: `verify_profile_assets(profile, project_root) -> list[AssetCheck]`
- Produces CLI: `voice-pipeline models verify`

- [ ] **Step 1: Write failing tests**

Test a temporary project root where only S1 exists and assert missing assets are returned deterministically.

- [ ] **Step 2: Run failing test**

```bash
pytest tests/test_assets.py -v
```

- [ ] **Step 3: Implement asset verification**

Required assets:

```text
models/pretrained/v2proplus/s1/s1v3.ckpt
models/pretrained/v2proplus/s2/s2Gv2ProPlus.pth
models/pretrained/v2proplus/s2/s2Dv2ProPlus.pth
models/pretrained/v2proplus/bert/chinese-roberta-wwm-ext-large/
models/pretrained/v2proplus/hubert/chinese-hubert-base/
models/pretrained/v2proplus/speaker/pretrained_eres2netv2w24s4ep4.ckpt
```

CLI must print one line per asset with `OK` or `MISSING` and exit non-zero if any required asset is missing.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_assets.py tests/test_cli.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/voice_pipeline/common/assets.py src/voice_pipeline/cli tests/test_assets.py
git commit -m "feat: add pretrained asset verification"
```

---

### Task 4: Add manifest parser and experiment/run state

**Files:**
- Create: `src/voice_pipeline/training/manifest.py`
- Create: `src/voice_pipeline/common/state.py`
- Create: `src/voice_pipeline/training/experiment.py`
- Create: `tests/test_manifest.py`
- Create: `tests/test_state.py`

**Interfaces:**
- Produces: `ManifestItem`
- Produces: `load_manifest(path: Path) -> list[ManifestItem]`
- Produces: `StageStatus`
- Produces: `StageState`
- Produces: `RunState`
- Produces: `Experiment.create(...)`

- [ ] **Step 1: Write manifest tests**

Cover valid JA entry, embedded spaces, malformed field count, missing audio file, empty text.

- [ ] **Step 2: Write state tests**

Cover:

```text
pending -> running -> completed
pending -> running -> failed
completed -> invalidated
```

and reject illegal transitions.

- [ ] **Step 3: Implement manifest parser**

Parse with `line.split("|", 3)` to preserve `|` only if absent from text contract; if `|` occurs in text, reject explicitly with a clear `ManifestError`.

- [ ] **Step 4: Implement state persistence**

Persist JSON atomically by writing `state.json.tmp` then `replace()`.

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_manifest.py tests/test_state.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/voice_pipeline/training src/voice_pipeline/common/state.py tests/test_manifest.py tests/test_state.py
git commit -m "feat: add manifest and run state"
```

---

### Task 5: Add stage contracts, signatures, and cache invalidation

**Files:**
- Create: `src/voice_pipeline/pipeline/stage.py`
- Create: `src/voice_pipeline/pipeline/signature.py`
- Create: `src/voice_pipeline/pipeline/graph.py`
- Create: `tests/test_stage_graph.py`

**Interfaces:**
- Produces: `StageContract`
- Produces: `compute_stage_signature(...) -> str`
- Produces: `StageGraph.invalidate_downstream(stage_name)`

- [ ] **Step 1: Write graph test**

Graph:

```text
manifest -> text
manifest -> wav32k
wav32k -> hubert
wav32k -> sv
hubert -> semantic
```

Assert invalidating `wav32k` invalidates `hubert`, `sv`, `semantic`, `s2`, and any downstream evaluation/export stages but not independent completed `text`.

- [ ] **Step 2: Run failure**

```bash
pytest tests/test_stage_graph.py -v
```

- [ ] **Step 3: Implement deterministic SHA-256 signatures**

Signature inputs include:

- relevant config subset serialized with sorted keys
- input file size + mtime + content hash for small manifests/configs
- model asset file size + mtime for large checkpoints
- profile name
- stage implementation version

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_stage_graph.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/voice_pipeline/pipeline tests/test_stage_graph.py
git commit -m "feat: add stage dependency and cache model"
```

---

### Task 6: Migrate upstream provenance, frontend, symbols, and text processing

**Files:**
- Create: `src/voice_pipeline/core/gpt_sovits/UPSTREAM.md`
- Create: `src/voice_pipeline/core/gpt_sovits/LICENSE`
- Create: `src/voice_pipeline/core/gpt_sovits/frontend/...`
- Create: `tests/compat/test_frontend_parity.py`

**Interfaces:**
- Produces: `FrontendResult`
- Produces: `MultilingualFrontend.process(text: str, language: str) -> FrontendResult`

- [ ] **Step 1: Record upstream revision and copied modules**

`UPSTREAM.md` must name the exact GPT-SoVITS commit used for migration and list each copied/adapted source path.

- [ ] **Step 2: Copy only the v2ProPlus-relevant frontend modules**

Preserve upstream logic for:

- cleaner
- symbols
- Chinese G2P
- Japanese G2P
- English G2P
- language segmentation
- text normalization
- phone-id mapping
- BERT alignment hooks

Refactor imports only.

- [ ] **Step 3: Create parity fixtures**

For fixed ZH/JA/EN/mixed examples, store expected normalized text and phone IDs derived from the selected upstream revision.

- [ ] **Step 4: Run parity tests**

```bash
pytest tests/compat/test_frontend_parity.py -v
```

Expected: exact equality of phone IDs and normalized text.

- [ ] **Step 5: Commit**

```bash
git add src/voice_pipeline/core/gpt_sovits tests/compat/test_frontend_parity.py THIRD_PARTY_NOTICES.md
git commit -m "feat: migrate gpt-sovits multilingual frontend"
```

---

### Task 7: Migrate CN-HuBERT and ERes2NetV2 feature extractors

**Files:**
- Create: `src/voice_pipeline/core/gpt_sovits/features/cnhubert.py`
- Create: `src/voice_pipeline/core/gpt_sovits/features/speaker.py`
- Create: `src/voice_pipeline/core/gpt_sovits/features/audio.py`
- Create: `tests/compat/test_feature_shapes.py`

**Interfaces:**
- Produces: `CNHubertExtractor.extract(wav_16k) -> Tensor`
- Produces: `SpeakerEncoder.extract(wav_32k) -> Tensor`
- Produces: `load_audio_32k(path) -> Tensor`

- [ ] **Step 1: Add tests around shape/dtype/device contracts**

SV expected dimensional behavior must match upstream `forward3()` output semantics.

- [ ] **Step 2: Migrate upstream algorithms without numerical changes**

Keep:

```text
32k -> 16k
Kaldi fbank
ERes2NetV2.forward3
```

for SV.

- [ ] **Step 3: Run**

```bash
pytest tests/compat/test_feature_shapes.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/voice_pipeline/core/gpt_sovits/features tests/compat/test_feature_shapes.py
git commit -m "feat: migrate v2proplus feature extractors"
```

---

### Task 8: Migrate S1 model core and checkpoint compatibility

**Files:**
- Create: `src/voice_pipeline/core/gpt_sovits/s1/model.py`
- Create: `src/voice_pipeline/core/gpt_sovits/s1/modules/...`
- Create: `src/voice_pipeline/core/gpt_sovits/s1/optimizer.py`
- Create: `src/voice_pipeline/core/gpt_sovits/s1/scheduler.py`
- Create: `src/voice_pipeline/core/gpt_sovits/compatibility/s1_checkpoint.py`
- Create: `tests/compat/test_s1_checkpoint.py`

**Interfaces:**
- Produces: `TextSemanticModel`
- Produces: `load_s1_checkpoint(path, device) -> TextSemanticModel`

- [ ] **Step 1: Write checkpoint load test**

Load official `s1v3.ckpt` if test assets are present; otherwise mark test with `pytest.skip`.

- [ ] **Step 2: Migrate upstream AR core**

Keep model and optimizer mathematics unchanged.

- [ ] **Step 3: Add compatibility adapter**

Hide official checkpoint dictionary layout behind a codec.

- [ ] **Step 4: Run**

```bash
pytest tests/compat/test_s1_checkpoint.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/voice_pipeline/core/gpt_sovits/s1 src/voice_pipeline/core/gpt_sovits/compatibility tests/compat/test_s1_checkpoint.py
git commit -m "feat: migrate s1 text semantic core"
```

---

### Task 9: Migrate v2ProPlus S2 generator, discriminator, losses, and checkpoint compatibility

**Files:**
- Create: `src/voice_pipeline/core/gpt_sovits/s2_v2proplus/model.py`
- Create: `src/voice_pipeline/core/gpt_sovits/s2_v2proplus/discriminator.py`
- Create: `src/voice_pipeline/core/gpt_sovits/s2_v2proplus/commons.py`
- Create: `src/voice_pipeline/core/gpt_sovits/s2_v2proplus/losses.py`
- Create: `src/voice_pipeline/core/gpt_sovits/s2_v2proplus/mel.py`
- Create: `src/voice_pipeline/core/gpt_sovits/compatibility/s2_checkpoint.py`
- Create: `tests/compat/test_s2_checkpoint.py`

**Interfaces:**
- Produces: `SynthesizerTrnV2ProPlus`
- Produces: `MultiPeriodDiscriminator`
- Produces: `load_s2_generator(...)`
- Produces: `load_s2_discriminator(...)`

- [ ] **Step 1: Write checkpoint compatibility tests**

Validate that official S2G/S2D state dicts load with the same missing/unexpected keys as upstream.

- [ ] **Step 2: Migrate exact v2ProPlus model/loss behavior**

Do not introduce V3/V4 branches.

- [ ] **Step 3: Run**

```bash
pytest tests/compat/test_s2_checkpoint.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/voice_pipeline/core/gpt_sovits/s2_v2proplus src/voice_pipeline/core/gpt_sovits/compatibility tests/compat/test_s2_checkpoint.py
git commit -m "feat: migrate v2proplus s2 core"
```

---

### Task 10: Implement preprocessing pipeline

**Files:**
- Create: `src/voice_pipeline/training/preprocess/base.py`
- Create: `src/voice_pipeline/training/preprocess/text_stage.py`
- Create: `src/voice_pipeline/training/preprocess/wav32k_stage.py`
- Create: `src/voice_pipeline/training/preprocess/hubert_stage.py`
- Create: `src/voice_pipeline/training/preprocess/sv_stage.py`
- Create: `src/voice_pipeline/training/preprocess/semantic_stage.py`
- Create: `src/voice_pipeline/training/preprocess/pipeline.py`
- Create: `src/voice_pipeline/cli/preprocess.py`
- Create: `tests/test_preprocess_pipeline.py`

**Interfaces:**
- Produces CLI:
  - `voice-pipeline preprocess all -c <config>`
  - `voice-pipeline preprocess stage <name> -c <config>`

- [ ] **Step 1: Write fake-stage orchestration test**

Use test doubles instead of GPU models to verify dependency order, resume, and invalidation.

- [ ] **Step 2: Implement stage runners**

Each writes only under:

```text
runs/<experiment>/preprocess/<stage>/
```

- [ ] **Step 3: Wire actual feature implementations**

Semantic stage must use base `s2Gv2ProPlus.pth`, never a fine-tuned checkpoint.

- [ ] **Step 4: Run**

```bash
pytest tests/test_preprocess_pipeline.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/voice_pipeline/training/preprocess src/voice_pipeline/cli/preprocess.py tests/test_preprocess_pipeline.py
git commit -m "feat: add v2proplus preprocessing pipeline"
```

---

### Task 11: Implement structured logging

**Files:**
- Create: `src/voice_pipeline/common/logging.py`
- Create: `tests/test_logging.py`

**Interfaces:**
- Produces: `PipelineLogger`
- Produces JSONL records with keys:
  - timestamp
  - stage
  - event
  - mini_step
  - optimizer_step
  - metrics

- [ ] **Step 1: Write JSONL test**
- [ ] **Step 2: Implement console + JSONL sink**
- [ ] **Step 3: Run**

```bash
pytest tests/test_logging.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/voice_pipeline/common/logging.py tests/test_logging.py
git commit -m "feat: add structured training logs"
```

---

### Task 12: Implement S2 trainer first

**Files:**
- Create: `src/voice_pipeline/training/s2/config.py`
- Create: `src/voice_pipeline/training/s2/trainer.py`
- Create: `src/voice_pipeline/training/s2/checkpoint.py`
- Create: `tests/test_s2_trainer.py`

**Interfaces:**
- Produces: `S2TrainConfig`
- Produces: `S2Trainer.train()`
- Produces checkpoints keyed by global S2 update step.

- [ ] **Step 1: Write one-step smoke test using tiny mocked modules**
- [ ] **Step 2: Implement optimizer groups**

Exactly preserve:

```text
base LR = 1e-4
text embedding LR = base * 0.4
encoder_text LR = base * 0.4
MRTE LR = base * 0.4
```

- [ ] **Step 3: Preserve official G/D update sequence**
- [ ] **Step 4: Implement checkpoint/resume**
- [ ] **Step 5: Run**

```bash
pytest tests/test_s2_trainer.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/voice_pipeline/training/s2 tests/test_s2_trainer.py
git commit -m "feat: add step-budgeted s2 trainer"
```

---

### Task 13: Implement S1 trainer with explicit gradient accumulation

**Files:**
- Create: `src/voice_pipeline/training/s1/config.py`
- Create: `src/voice_pipeline/training/s1/trainer.py`
- Create: `src/voice_pipeline/training/s1/checkpoint.py`
- Create: `tests/test_s1_trainer.py`

**Interfaces:**
- Produces: `S1TrainConfig`
- Produces: `S1Trainer.train()`
- Logs both mini-batch and optimizer steps.

- [ ] **Step 1: Write accumulation test**

For physical batch count 8 and accumulation 4, assert exactly 2 optimizer steps.

- [ ] **Step 2: Implement accumulation explicitly**

Do not preserve the upstream first-step 5-batch anomaly. Preserve the intended 4-mini-batch accumulation semantics and document this deliberate framework-level normalization in `UPSTREAM.md`.

- [ ] **Step 3: Keep S1 model/loss/optimizer/scheduler math unchanged**
- [ ] **Step 4: Implement checkpoint/resume**
- [ ] **Step 5: Run**

```bash
pytest tests/test_s1_trainer.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/voice_pipeline/training/s1 tests/test_s1_trainer.py
git commit -m "feat: add explicit-step s1 trainer"
```

---

### Task 14: Add training CLI and YAML configuration

**Files:**
- Create: `src/voice_pipeline/training/config.py`
- Create: `src/voice_pipeline/cli/train.py`
- Create: `configs/train.example.yaml`
- Create: `tests/test_training_config.py`

**Interfaces:**
- Produces:
  - `voice-pipeline train s1`
  - `voice-pipeline train s2`
  - `voice-pipeline train all`

- [ ] **Step 1: Write YAML validation tests**
- [ ] **Step 2: Implement typed config parsing**
- [ ] **Step 3: Reject unknown profile and invalid target-step values**
- [ ] **Step 4: Run**

```bash
pytest tests/test_training_config.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/voice_pipeline/training/config.py src/voice_pipeline/cli/train.py configs/train.example.yaml tests/test_training_config.py
git commit -m "feat: add training cli configuration"
```

---

### Task 15: Implement ModelBundle schema and export

**Files:**
- Create: `src/voice_pipeline/common/model_bundle.py`
- Create: `src/voice_pipeline/cli/export.py`
- Create: `tests/test_bundle.py`

**Interfaces:**
- Produces: `ModelBundle`
- Produces: `ModelBundle.validate()`
- Produces CLI: `voice-pipeline export --run ...`

- [ ] **Step 1: Write bundle roundtrip test**
- [ ] **Step 2: Implement schema version 1**
- [ ] **Step 3: Store relative paths only**
- [ ] **Step 4: Include reference audio metadata and validated languages**
- [ ] **Step 5: Run**

```bash
pytest tests/test_bundle.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/voice_pipeline/common/model_bundle.py src/voice_pipeline/cli/export.py tests/test_bundle.py
git commit -m "feat: add model bundle export"
```

---

### Task 16: Implement inference text sources and chunker

**Files:**
- Create: `src/voice_pipeline/inference/text_source.py`
- Create: `src/voice_pipeline/inference/text_chunker.py`
- Create: `tests/test_text_source.py`
- Create: `tests/test_text_chunker.py`

**Interfaces:**
- Produces: `resolve_text_source(text: str | None, text_file: Path | None) -> str`
- Produces: `TextChunker.chunk(text: str) -> list[str]`

- [ ] **Step 1: Test `--text` / `--text-file` exclusivity**
- [ ] **Step 2: Test UTF-8-SIG**
- [ ] **Step 3: Test empty TXT rejection**
- [ ] **Step 4: Test paragraph/sentence/hard-limit chunking**
- [ ] **Step 5: Run**

```bash
pytest tests/test_text_source.py tests/test_text_chunker.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/voice_pipeline/inference/text_source.py src/voice_pipeline/inference/text_chunker.py tests/test_text_source.py tests/test_text_chunker.py
git commit -m "feat: add inference text sources"
```

---

### Task 17: Implement v2ProPlus inference session

**Files:**
- Create: `src/voice_pipeline/inference/reference.py`
- Create: `src/voice_pipeline/inference/session.py`
- Create: `src/voice_pipeline/inference/semantic.py`
- Create: `src/voice_pipeline/inference/acoustic.py`
- Create: `src/voice_pipeline/inference/result.py`
- Create: `tests/test_inference_session.py`

**Interfaces:**
- Produces: `InferenceSession.load(bundle_path, device)`
- Produces: `InferenceSession.synthesize(...) -> InferenceResult`

- [ ] **Step 1: Write lifecycle test with mocked models**
- [ ] **Step 2: Preserve official reference conditioning**

Keep:

- reference semantic extraction
- reference spectrogram
- v2ProPlus speaker-conditioning audio path
- optional reference text semantics

- [ ] **Step 3: Cache loaded models and reference condition**
- [ ] **Step 4: Support fixed seed and decoding parameters**
- [ ] **Step 5: Run**

```bash
pytest tests/test_inference_session.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/voice_pipeline/inference tests/test_inference_session.py
git commit -m "feat: add v2proplus inference session"
```

---

### Task 18: Add inference CLI and long-text resume

**Files:**
- Create: `src/voice_pipeline/cli/infer.py`
- Create: `configs/infer.example.yaml`
- Create: `tests/test_infer_cli.py`

**Interfaces:**
- Produces:
  - `voice-pipeline infer synthesize`
  - `voice-pipeline infer batch`
  - `voice-pipeline infer benchmark`

- [ ] **Step 1: Write CLI mutual-exclusion test**
- [ ] **Step 2: Implement inline and TXT synthesis**
- [ ] **Step 3: Write chunk manifest**

Each chunk entry includes:

```json
{
  "index": 0,
  "text": "...",
  "status": "completed",
  "output": "chunks/000001.wav"
}
```

- [ ] **Step 4: Resume completed chunks**
- [ ] **Step 5: Run**

```bash
pytest tests/test_infer_cli.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/voice_pipeline/cli/infer.py configs/infer.example.yaml tests/test_infer_cli.py
git commit -m "feat: add inference cli and chunk resume"
```

---

### Task 19: Implement evaluation suite generation

**Files:**
- Create: `src/voice_pipeline/evaluation/suite.py`
- Create: `src/voice_pipeline/evaluation/runner.py`
- Create: `configs/eval/zh.txt`
- Create: `configs/eval/ja.txt`
- Create: `configs/eval/en.txt`
- Create: `configs/eval/mixed.txt`
- Create: `tests/test_eval_suite.py`

**Interfaces:**
- Produces: `EvaluationSuite`
- Produces: deterministic generation manifest for each candidate ModelBundle.

- [ ] **Step 1: Write suite parse test**
- [ ] **Step 2: Implement fixed reference/seed/decoding config**
- [ ] **Step 3: Generate candidate WAV tree**
- [ ] **Step 4: Run**

```bash
pytest tests/test_eval_suite.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/voice_pipeline/evaluation configs/eval tests/test_eval_suite.py
git commit -m "feat: add cross-language evaluation suites"
```

---

### Task 20: Implement evaluator metric interfaces

**Files:**
- Create: `src/voice_pipeline/evaluation/base.py`
- Create: `src/voice_pipeline/evaluation/speaker_similarity.py`
- Create: `src/voice_pipeline/evaluation/pronunciation.py`
- Create: `src/voice_pipeline/evaluation/language_consistency.py`
- Create: `src/voice_pipeline/evaluation/prosody.py`
- Create: `tests/test_evaluator_metrics.py`

**Interfaces:**
- Produces: `MetricResult`
- Produces: `EvaluatorMetric.evaluate(sample) -> MetricResult`

- [ ] **Step 1: Define common result schema**

```python
@dataclass
class MetricResult:
    name: str
    value: float | None
    details: dict[str, object]
    available: bool
```

- [ ] **Step 2: Implement prosody metrics locally**
- [ ] **Step 3: Implement speaker/ASR/lang-id as optional adapters**
- [ ] **Step 4: Ensure missing optional evaluator dependencies do not fail training/inference**
- [ ] **Step 5: Run**

```bash
pytest tests/test_evaluator_metrics.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/voice_pipeline/evaluation tests/test_evaluator_metrics.py
git commit -m "feat: add modular evaluation metrics"
```

---

### Task 21: Implement ranking, hard constraints, and anonymous listening shortlist

**Files:**
- Create: `src/voice_pipeline/evaluation/ranking.py`
- Create: `src/voice_pipeline/evaluation/report.py`
- Create: `src/voice_pipeline/cli/evaluate.py`
- Create: `src/voice_pipeline/cli/select.py`
- Create: `tests/test_ranking.py`

**Interfaces:**
- Produces:
  - `voice-pipeline evaluate --run ...`
  - `voice-pipeline select --run ...`

- [ ] **Step 1: Write ranking test**

Candidate with high JA similarity but failed EN minimum must be rejected before ranking.

- [ ] **Step 2: Implement hard constraints**
- [ ] **Step 3: Implement ranking without hiding raw metrics**
- [ ] **Step 4: Create anonymous A/B/C listening folders**
- [ ] **Step 5: Run**

```bash
pytest tests/test_ranking.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/voice_pipeline/evaluation src/voice_pipeline/cli/evaluate.py src/voice_pipeline/cli/select.py tests/test_ranking.py
git commit -m "feat: add checkpoint selection workflow"
```

---

### Task 22: Implement simple pipeline YAML orchestration

**Files:**
- Create: `src/voice_pipeline/pipeline/orchestrator.py`
- Create: `src/voice_pipeline/cli/run.py`
- Create: `configs/pipeline.example.yaml`
- Create: `tests/test_orchestrator.py`

**Interfaces:**
- Produces: `voice-pipeline run <pipeline.yaml>`

- [ ] **Step 1: Write ordered-stage test**
- [ ] **Step 2: Accept only known stage names**
- [ ] **Step 3: Stop on failed stage and preserve resume state**
- [ ] **Step 4: Run**

```bash
pytest tests/test_orchestrator.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/voice_pipeline/pipeline src/voice_pipeline/cli/run.py configs/pipeline.example.yaml tests/test_orchestrator.py
git commit -m "feat: add pipeline yaml orchestration"
```

---

### Task 23: Add end-to-end smoke fixture

**Files:**
- Create: `tests/integration/test_end_to_end_smoke.py`
- Create: `tests/fixtures/manifest.list`
- Create: `tests/fixtures/config.yaml`

**Interfaces:**
- Verifies framework-level flow without requiring long training.

- [ ] **Step 1: Use tiny/mock model adapters where real GPU execution would be excessive**
- [ ] **Step 2: Execute**

```text
manifest
→ preprocess state
→ mock S2 checkpoint
→ mock S1 checkpoint
→ bundle
→ inference
→ evaluation report
```

- [ ] **Step 3: Run**

```bash
pytest tests/integration/test_end_to_end_smoke.py -v
```

- [ ] **Step 4: Commit**

```bash
git add tests/integration tests/fixtures
git commit -m "test: add end to end pipeline smoke test"
```

---

### Task 24: Verify real v2ProPlus compatibility on the user's weight set

**Files:**
- Modify: compatibility tests only if needed
- Create: `docs/compatibility-v2proplus.md`

**Interfaces:**
- Real validation checkpoint before declaring training/inference ready.

- [ ] **Step 1: Put assets in the designed model store**
- [ ] **Step 2: Run**

```bash
voice-pipeline models verify
```

Expected: all six asset groups `OK`.

- [ ] **Step 3: Run frontend and model-load compatibility tests**

```bash
pytest tests/compat -v
```

- [ ] **Step 4: Run one real preprocessing sample**
- [ ] **Step 5: Run one real S2 update**
- [ ] **Step 6: Run one real S1 optimizer update**
- [ ] **Step 7: Run one real v2ProPlus inference**
- [ ] **Step 8: Record exact environment and results in `docs/compatibility-v2proplus.md`**
- [ ] **Step 9: Commit**

```bash
git add docs/compatibility-v2proplus.md
git commit -m "docs: record v2proplus compatibility validation"
```

---

### Task 25: Finish README and operator documentation

**Files:**
- Modify: `README.md`
- Create: `docs/architecture.md`
- Create: `docs/training.md`
- Create: `docs/inference.md`
- Create: `docs/evaluation.md`
- Create: `docs/troubleshooting.md`

**Interfaces:**
- Documentation must describe actual implemented commands only.

- [ ] **Step 1: Add 5-minute Quick Start**
- [ ] **Step 2: Add exact pretrained model layout**
- [ ] **Step 3: Add complete first-training walkthrough**
- [ ] **Step 4: Add complete first-inference walkthrough**
- [ ] **Step 5: Explain Text/BERT, HuBERT, SV, semantic, S1, S2, reference conditioning, evaluator, ModelBundle, cache/resume**
- [ ] **Step 6: Add cross-language objective section**
- [ ] **Step 7: Verify every README command with CLI `--help` and config parser**
- [ ] **Step 8: Commit**

```bash
git add README.md docs
git commit -m "docs: complete voice pipeline documentation"
```

---

### Task 26: Final verification and release ZIP

**Files:**
- No implementation changes unless verification finds defects.

**Interfaces:**
- Produces final ZIP excluding model weights, runs, outputs, and caches.

- [ ] **Step 1: Run full unit/integration suite**

```bash
pytest -v
```

Expected: all applicable tests pass; hardware/asset-dependent tests may skip only with explicit reasons.

- [ ] **Step 2: Run CLI checks**

```bash
voice-pipeline --help
voice-pipeline models verify
voice-pipeline preprocess --help
voice-pipeline train --help
voice-pipeline infer --help
voice-pipeline evaluate --help
```

- [ ] **Step 3: Run real smoke path on RTX 4060 environment**
- [ ] **Step 4: Check repository contains no weights or generated runs**
- [ ] **Step 5: Check MIT notices and `UPSTREAM.md`**
- [ ] **Step 6: Build ZIP from repository files only**
- [ ] **Step 7: Re-extract ZIP into a clean directory and run `uv pip install -e .` plus `voice-pipeline --help`**
- [ ] **Step 8: Release only after clean-directory verification succeeds**

---

## Self-review

### Spec coverage

Covered:

- `src/voice_pipeline` layout
- no external GPT-SoVITS source dependency
- v2ProPlus-only implementation with profile extension seam
- controlled upstream migration and MIT provenance
- standardized model store
- manifest contract
- all v2ProPlus preprocessing stages including SV
- S1 and S2 step-budget training
- no new loss
- cross-language evaluation
- hard-constraint selection
- human listening shortlist
- ModelBundle
- official reference-conditioning behavior
- inline and TXT inference
- long-text chunk resume
- single CLI
- simple pipeline YAML
- state/cache invalidation
- structured logs
- upstream compatibility tests
- README/operator documentation

### Important deliberate compatibility choice

The S1 upstream source performs its first optimizer update after five mini-batches because of `batch_idx > 0 and batch_idx % 4 == 0`. The approved design interprets this as intended approximately-4-batch accumulation rather than a model-quality-critical algorithm. The framework will use explicit accumulation of exactly four mini-batches. This deviation must be documented and benchmarked. All S1 model/loss/optimizer/scheduler mathematics remain unchanged.

### Scope

The plan intentionally excludes GUI and Dataset Builder integration. Those become separate follow-up plans after this CLI framework is verified.

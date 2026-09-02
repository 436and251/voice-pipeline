# GPT-SoVITS v2ProPlus 训推一体化框架设计文档

> 版本：v0.1 Design Draft  
> 目标：构建一个**独立于官方 GPT-SoVITS 仓库**、但尽可能保持官方 v2ProPlus 算法行为不变的训练 / 推理 / 评测框架。  
> 第一版仅实现 **GPT-SoVITS v2ProPlus**，但保留 GPT-SoVITS 家族内其他版本（如 v4）的扩展间隙。  
> 核心质量目标：**仅使用日语训练数据进行 speaker adaptation 后，在中文、日语、英文三种语言下都保持逼真的同一说话人音色，同时尽量保持各语言正确的音素、重音、节奏和语调。**

## 1. 设计目标

本项目不是官方 WebUI 的换皮，也不是简单把官方脚本重新包装成 subprocess 调用，而是重新组织出一套 CLI-first、训推一致、阶段可解耦、可恢复、可评测、可追踪、可组合、可继续向 GPT-SoVITS 家族其他版本扩展的工程框架。

最终完整 Voice Pipeline 的长期目标是：

```text
Source
  ↓
Dataset Builder
  ↓
GPT-SoVITS Preprocess
  ↓
Training
  ↓
Checkpoint Evaluation
  ↓
Model Selection
  ↓
ModelBundle Export
  ↓
Inference
```

但本版本只实现：

```text
GPT-SoVITS Preprocess
→ Training
→ Evaluation
→ Model Selection
→ Export
→ Inference
```

不接 Dataset Builder，不做 GUI。

## 2. 第一原则：Speaker Adaptation，而不是 Language Adaptation

项目最核心的训练目标不是“让日语训练集上的生成声音尽可能像目标 speaker”，而是：

> 使用日语单语训练数据适配 speaker identity，同时尽量保持 v2ProPlus 原有的中文 / 日语 / 英文跨语言发音、韵律与自然度能力。

因此需要区分三层能力：

```text
Speaker Identity
├── timbre / 声纹
├── resonance / 共振特征
└── voice texture / 声音质感

Linguistic Realization
├── phoneme correctness
├── language-specific articulation
└── code-switch pronunciation

Prosody
├── pitch contour
├── rhythm
├── stress / accent
└── intonation
```

训练过程中：

```text
Speaker Identity           强适配
Speaker-specific Prosody   适度适配
ZH / JA / EN Phoneme       尽量保护
ZH / JA / EN Prosody       尽量保护
```

因此：S2 是主要 speaker adaptation 层；S1 是受控 linguistic / prosody adaptation 层；不假设 S1 训练越久越好；不假设 latest checkpoint 就是 best checkpoint；checkpoint 必须通过跨语言评测后再选择。

## 3. 第一版明确不做的事情

为控制代码量和验证风险，第一版明确不做：GUI、Web API、Dataset Builder 整合、Qwen3-TTS / CosyVoice / FishSpeech 等其他模型家族、修改 GPT-SoVITS 原始 Loss、新增 preservation loss、LoRA / PEFT、V3 / V4 实际实现、Markdown / HTML 文档解析、模型自动下载器、自动超参搜索、复杂多模型 evaluator ensemble、完全自动化最终模型裁决。

第一版只解决：

```text
可稳定复现
+
训推一致
+
阶段可拆
+
checkpoint 可比较
+
三语能力可观察
```

## 4. 官方源码复用原则

### 4.1 不依赖外部 clone

新框架不能依赖：

```text
D:/AI-Training/voice-clone/GPT-SoVITS/
```

作为运行时源码。删除外部 GPT-SoVITS clone 后，只要模型权重已经放入本框架规定目录，项目仍可运行。

### 4.2 允许复制和工程化整理

GPT-SoVITS 当前仓库使用 MIT License。允许复制、修改、再分发、整理结构、改 import、改日志、改配置入口、改 CLI、改状态管理、改 checkpoint 生命周期、改外围训练调度。

但第一版不修改影响算法表现的核心行为：模型结构、forward、Loss、phoneme vocabulary、G2P、BERT 对齐、semantic representation、v2ProPlus S2 核心生成机制、reference conditioning 逻辑、官方训练参数含义。

### 4.3 保留 upstream provenance

项目内必须存在：

```text
voice_pipeline/core/gpt_sovits/UPSTREAM.md
```

至少记录 upstream repository、upstream commit SHA、导入 / 改造的模块、排除的模块、修改范围、License notice。这样未来可以可靠 backport 官方 bugfix。

## 5. 项目总体结构

建议项目路径：

```text
D:/AI-Training/voice-clone/voice-pipeline/
```

项目结构：

```text
voice-pipeline/
│
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── train.example.yaml
│   ├── infer.example.yaml
│   └── eval/
│       ├── zh.txt
│       ├── ja.txt
│       ├── en.txt
│       └── mixed.txt
│
├── models/
│   └── pretrained/
│       └── v2proplus/
│           ├── s1/
│           ├── s2/
│           ├── bert/
│           ├── hubert/
│           └── speaker/
│
├── runs/
├── outputs/
│
├── voice_pipeline/
│   ├── cli/
│   ├── common/
│   ├── profiles/
│   ├── core/gpt_sovits/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── pipeline/
│
└── tests/
```

推荐细分：

```text
voice_pipeline/
├── cli/
│   ├── main.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export.py
│   └── infer.py
│
├── common/
│   ├── paths.py
│   ├── device.py
│   ├── logging.py
│   ├── metadata.py
│   ├── checkpoint.py
│   ├── state.py
│   └── errors.py
│
├── profiles/
│   ├── base.py
│   ├── registry.py
│   └── v2proplus.py
│
├── core/gpt_sovits/
│   ├── UPSTREAM.md
│   ├── frontend/
│   ├── s1/
│   ├── s2_v2proplus/
│   ├── features/
│   └── compatibility/
│
├── training/
│   ├── config.py
│   ├── experiment.py
│   ├── artifacts.py
│   ├── preprocess/
│   ├── s1/
│   └── s2/
│
├── evaluation/
│   ├── runner.py
│   ├── suite.py
│   ├── speaker_similarity.py
│   ├── pronunciation.py
│   ├── language_consistency.py
│   ├── prosody.py
│   ├── ranking.py
│   └── report.py
│
├── inference/
│   ├── config.py
│   ├── session.py
│   ├── reference.py
│   ├── text_source.py
│   ├── text_chunker.py
│   ├── semantic.py
│   ├── acoustic.py
│   ├── synthesizer.py
│   └── result.py
│
└── pipeline/
    ├── orchestrator.py
    ├── stage.py
    └── graph.py
```

## 6. 模型 Profile 设计

第一版只实现 v2ProPlus，但框架使用 Profile Registry，而不是把版本判断散落在业务代码里。

```text
Profile Registry
├── v2ProPlus    ← 第一版完整实现
└── v4           ← 未来扩展
```

Profile 负责描述：S1 architecture、S2 architecture、pretrained assets、preprocessing requirements、sample rate、semantic frame rate、是否需要 SV embedding、checkpoint codec、inference backend、trainer adapter。

示意：

```python
class ModelProfile:
    name: str

    def validate_assets(self): ...
    def create_preprocessor(self): ...
    def create_s1_model(self): ...
    def create_s2_model(self): ...
    def create_s1_trainer(self): ...
    def create_s2_trainer(self): ...
    def create_inference_session(self): ...
    def validate_bundle(self): ...
```

第一版：

```python
ProfileRegistry.get("v2ProPlus")
```

未来支持 V4 时新增 `V4Profile`，而不改 CLI、Experiment、Run State、Evaluation、ModelBundle、TXT input、Logging、Pipeline orchestration。

## 7. 预训练模型目录

建议统一移动为：

```text
models/
└── pretrained/
    └── v2proplus/
        ├── s1/
        │   └── s1v3.ckpt
        ├── s2/
        │   ├── s2Gv2ProPlus.pth
        │   └── s2Dv2ProPlus.pth
        ├── bert/
        │   └── chinese-roberta-wwm-ext-large/
        ├── hubert/
        │   └── chinese-hubert-base/
        ├── speaker/
        │   └── pretrained_eres2netv2w24s4ep4.ckpt
        ├── g2pw/
        │   └── G2PWModel/
        ├── g2p/en/
        │   └── nltk_data/
        └── langdetect/
            └── lid.176.bin
```

源码不写绝对路径，只通过项目 root + profile relative path 解析。用户默认只配置：

```yaml
profile:
  name: v2ProPlus
```

## 8. Git 策略

以下目录默认不提交：

```gitignore
/models/
/runs/
/outputs/
```

Git 仓库只保存 source code、configs、README、tests、license notices。后续可提供：

```bash
voice-pipeline models verify
```

检查 S1、S2G、S2D、BERT、HuBERT、Speaker Encoder，但第一版不做下载器。

## 9. Dataset Manifest

训练框架接受统一 manifest：

```text
audio_path|speaker|language|text
```

例如：

```text
D:/dataset/clips/000001.wav|speaker|ja|こんにちは。
```

第一版假定上游已经完成切分、ASR 文本已经准备好、音频路径有效、language 标签存在。本框架不重复做 ASR 和音频切片。

## 10. v2ProPlus 完整预处理

v2ProPlus 预处理必须包含四个核心分支：

```text
Manifest
   │
   ├── Text Feature
   │     ├── normalize
   │     ├── phoneme
   │     └── BERT feature
   │
   ├── Audio Feature
   │     ├── wav32k
   │     └── CN-HuBERT SSL
   │
   ├── Speaker Feature
   │     └── ERes2NetV2 SV embedding
   │
   └── Semantic Feature
         └── pretrained S2G semantic token
```

### 10.1 Text

输出至少包含 normalized text、phoneme sequence、BERT feature、speaker、language。

### 10.2 32k Audio

统一为 v2ProPlus 所需 32k 音频表示。

### 10.3 CN-HuBERT

输出 SSL feature。

### 10.4 SV Embedding

v2ProPlus 必须提取说话人特征：

```text
32k wav
→ resample 16k
→ Kaldi fbank
→ ERes2NetV2
→ SV embedding
```

该部分保持官方行为。

### 10.5 Semantic Token

使用 matching pretrained S2G：

```text
s2Gv2ProPlus.pth
```

提取 semantic token。semantic preprocessing 使用 base pretrained S2G，而不是后续 fine-tuned S2G。

## 11. Preprocess 依赖图

```text
                 dataset.list
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
        text        wav32k      metadata
                      │
                 ┌────┴────┐
                 ↓         ↓
              hubert       sv
                 │
                 ↓
              semantic
```

S1 训练依赖 `text + semantic`；S2 训练依赖 `text + wav32k + hubert + sv`。

当前实现直接读取已有 `data.list`，不重复切片、ASR、改写 speaker 或统一成某个
语言 frontend。每条记录按自身 `language`（`zh`、`ja`、`en`、`mixed`）处理。
完整运行命令为：

```powershell
voice-pipeline preprocess all -c pipeline.yaml
voice-pipeline preprocess stage semantic -c pipeline.yaml
```

`stage semantic` 自动执行依赖闭包 `wav32k -> hubert -> semantic`；`stage text`
只要求 BERT、G2PW、NLTK 与本地语言识别资产。相对的模型路径和 output root
以启动命令时的项目根目录解析，相对音频路径则以 `data.list` 所在目录解析。

容错上限为 `min(5, ceil(data.list 非空记录数 * 20%))`。坏 manifest 行与任一
样本阶段失败共用同一个 quarantine 计数；未超限时命令退出码仍为 0，并打印
坏样本数、上限、有效样本数和 `quarantine.jsonl` 路径。任一阶段失败的样本会
从所有训练索引中排除；超过上限或没有有效样本时不发布正式训练视图。

## 12. Stage Contract

每个 Stage 必须定义：stage name、required inputs、optional inputs、generated outputs、config subset、dependency stages、cache signature、invalidation rule、resumability、failure semantics。

所有 Stage 使用统一状态：

```text
pending
running
completed
failed
invalidated
```

## 13. Run / Experiment 目录

每个实验完全隔离：

```text
runs/
└── speaker_001/
    ├── input/            # 后续 run 快照预留；原 data.list 不改写
    ├── preprocess/
    │   ├── assets.json
    │   ├── state.json
    │   ├── quarantine.jsonl
    │   ├── valid_samples.jsonl
    │   ├── 2-name2text.txt
    │   ├── 6-name2semantic.tsv
    │   ├── text/       # <sample>.json + <sample>.bert.pt + index.jsonl
    │   ├── wav32k/     # <sample>.wav + index.jsonl
    │   ├── hubert/     # <sample>.pt + index.jsonl
    │   ├── sv/         # <sample>.pt + index.jsonl
    │   └── semantic/   # <sample>.pt + index.jsonl
    ├── training/s1/      # 后续 S1 trainer 使用
    ├── training/s2/      # 后续 S2 trainer 使用
    ├── evaluation/       # 后续 evaluator 使用
    └── export/           # 后续 bundle export 使用
```

禁止中间产物散落在 TEMP、logs、weights、output 等项目根目录。

训练明确成功后，训练入口应调用 `cleanup_after_training(..., True)`：默认只删除
目标目录内的 `*.tmp` 和已 quarantine 且不在 valid 集合中的已知阶段产物，保留
所有正式预处理结果。训练失败或中断时传入 `False`，不删除任何缓存。实际训练
入口会在后续训练阶段接入该生命周期钩子。

## 14. State 与 Cache

每个 stage 写入：

```json
{
  "name": "sv",
  "status": "completed",
  "input_hash": "...",
  "config_hash": "...",
  "started_at": "...",
  "finished_at": "...",
  "outputs": ["preprocess/sv/..."]
}
```

缓存命中必须可解释。如果 dataset.list、profile、对应模型权重或 stage config 改变，则相关 stage 自动 invalidated，并递归使 downstream 失效。

## 15. S1 训练设计

S1 为：

```text
Text2SemanticDecoder + s1v3.ckpt
```

第一版不引入 LoRA。

S1 使用全模型可训练参数。官方当前行为约每 4 mini-batch 执行一次 optimizer step。第一版框架把这一语义显式配置：

```yaml
s1:
  batch_size: 2
  gradient_accumulation: 4
```

训练日志明确区分 mini_batch_step、optimizer_step、effective_batch。

第一版训练控制的核心不是 epoch，而是：

```text
target_optimizer_steps
```

例如：

```yaml
s1:
  target_optimizer_steps: 500
  checkpoint_every_steps: 100
```

## 16. S2 训练设计

S2 使用 v2ProPlus 官方原生训练行为：

```text
Generator + Discriminator
```

初始化：

```text
s2Gv2ProPlus.pth
s2Dv2ProPlus.pth
```

统一称：

> native S2 trainable-parameter fine-tuning

原因：quantizer frozen、SSL encoder 不参与训练、speaker encoder 只用于预处理、text/MRTE 使用低 LR、部分模块并非所有参数都以相同方式更新。

保持官方行为：

```text
learning_rate       1e-4
text_low_lr_rate    0.4
freeze_quantizer    true
fp16                 true
```

即 base params 1e-4；text_embedding / encoder_text / MRTE 4e-5。

S2 每个 batch 大体包含：

```text
D backward
→ D step
→ G backward
→ G step
```

因此：

```text
1 S2 step ≈ 1 G update + 1 D update
```

## 17. 第一版不修改 Loss

冻结原则：第一版不引入任何新的 Loss。

S1 保持官方 Text2SemanticDecoder loss；S2 保持官方 adversarial、mel、feature、KL 等 loss。

后续只有在 baseline 数据明确证明跨语言退化来自训练目标后，才考虑 L2-SP、teacher-student KL、selective regularization，这些不进入第一版。

## 18. 跨语言训练策略

训练数据：JA only。目标语言：ZH / JA / EN。

配置必须分开：

```yaml
objective:
  training_languages:
    - ja
  target_languages:
    - zh
    - ja
    - en
  cross_language_preservation: strict
```

S2 作为主要 speaker adaptation；S1 采用受控 adaptation。

第一轮实验采用分层搜索：先比较多个 S2 checkpoint，再在更合适的 S2 上比较 S1 checkpoint，并始终保留 `Base S1 + Fine-tuned S2` 作为正式候选。

## 19. Baseline Candidate Matrix

至少支持：

```text
A: Base S1 + Base S2
B: Base S1 + Fine-tuned S2
C: Fine-tuned S1(light) + Fine-tuned S2
D: Fine-tuned S1(stronger) + Fine-tuned S2
```

用于判断相似度提升来自哪里、跨语言退化发生在哪里、是否 S1 根本不需要 fine-tune、S1 适配强度的最佳范围。

## 20. Evaluation 与 Training 完全分离

Validation 指标不参与反向传播。Evaluator 只消费：

```text
generated wav + target text + language + reference speaker audio
```

不感知内部训练算法。

## 21. Evaluation Suite

固定测试集：ZH、JA、EN、Mixed。ZH 覆盖普通陈述、长句、数字/英文混合、特征音素；JA 覆盖普通陈述、长句、情绪句、训练域外句；EN 覆盖普通陈述、stress、consonant clusters、长句；Mixed 覆盖 ZH+EN、JA+EN、ZH+JA。

配置：

```text
configs/eval/
├── zh.txt
├── ja.txt
├── en.txt
└── mixed.txt
```

## 22. 自动评测维度

第一版实现 Speaker Similarity、Pronunciation、Language Consistency、Basic Prosody。

### 22.1 Speaker Similarity

使用独立 speaker encoder evaluator，避免与训练用 ERes2NetV2 完全同源。推荐使用 speaker centroid：多个 reference embedding 求平均。输出 `Sim_ZH / Sim_JA / Sim_EN / WorstSim`。

### 22.2 Pronunciation

```text
generated wav
→ ASR
→ compare target text
```

指标：ZH CER、JA CER、EN WER。

### 22.3 Language Consistency

对输出做语言识别，检测中文/英文/日语是否发生明显语言漂移。

### 22.4 Basic Prosody

第一版只做基础统计：F0 median、F0 range、F0 variance、voiced/unvoiced ratio、speaking rate、pause ratio、energy statistics。不做逐帧 F0 MSE。

## 23. Model Selection

不简单使用 highest average score，而使用：

```text
hard constraints
→ ranking
→ human shortlist
```

例如：

```yaml
selection:
  constraints:
    min_speaker_similarity:
      zh: 0.80
      ja: 0.85
      en: 0.80
    max_cer:
      zh: 0.10
      ja: 0.10
    max_wer:
      en: 0.15
```

特别关注：

```text
WorstSim = min(Sim_ZH, Sim_JA, Sim_EN)
```

避免 JA 很高但 EN 很差却被平均分掩盖。

## 24. Human Listening

最终仍保留人耳裁决，但只试听自动筛选后的 3~5 个候选。

```text
evaluation/listening/
├── candidate_A/
├── candidate_B/
├── candidate_C/
└── manifest.json
```

候选名称不暴露 step，减少心理预期偏差。人工建议评分：Speaker Similarity、Pronunciation、Prosody、Naturalness、Artifact。

## 25. ModelBundle

训练与推理之间唯一正式接口：

```text
ModelBundle
```

目录：

```text
models/
└── speaker_name/
    ├── model.yaml
    ├── metadata.json
    ├── weights/
    │   ├── s1.ckpt
    │   └── s2.pth
    └── reference/
        ├── default.wav
        └── default.json
```

`model.yaml` 示例：

```yaml
schema_version: 1
profile: v2ProPlus
weights:
  s1: weights/s1.ckpt
  s2: weights/s2.pth
reference:
  audio: reference/default.wav
  text: "今日はいい天気ですね。"
  language: ja
languages:
  trained:
    - ja
  validated:
    - zh
    - ja
    - en
```

推理只认 ModelBundle，不直接依赖 training run。

## 26. Reference Audio

保持官方 GPT-SoVITS v2ProPlus reference-conditioned 推理机制。reference audio 不是 prompt engineering，而是模型条件输入的一部分。

底层完整推理支持 reference audio、reference text、reference language、target text、target language。第一版 CLI 可显式传，GUI 阶段再考虑隐藏和简化。reference text 保持 optional，行为与官方兼容。

## 27. InferenceSession

标准生命周期：

```text
ModelBundle
   ↓
validate
   ↓
load profile
   ↓
load BERT
   ↓
load HuBERT
   ↓
load S1
   ↓
load S2
   ↓
load reference
   ↓
cache prompt semantic / reference condition
   ↓
READY
```

同一 session 后续多次推理不重复加载模型。

## 28. 训练 / 推理共享 Frontend

训练与推理只允许使用同一份 multilingual frontend。统一处理：

```text
text normalization
→ language segmentation
→ G2P
→ phoneme IDs
→ BERT alignment
```

禁止训练和推理分别复制一份不同实现。这是跨语言一致性的硬约束。

## 29. 推理文本输入

第一版只支持 Inline Text 和 TXT。

Inline：

```bash
voice-pipeline infer synthesize \
  --model models/speaker_name \
  --text "今天天气很好。" \
  --lang zh
```

TXT：

```bash
voice-pipeline infer synthesize \
  --model models/speaker_name \
  --text-file input.txt \
  --lang zh
```

规则：`--text` 与 `--text-file` 互斥；`.txt` only；UTF-8 / UTF-8-SIG；空文件报错；不支持 Markdown、HTML、富文本。

## 30. 长文本处理

```text
TextFileSource
→ TextChunker
→ Shared Frontend
→ Inference
```

TextChunker 顺序：paragraph → sentence → hard max fallback。

输出：

```text
outputs/article/
├── chunks/
│   ├── 000001.wav
│   ├── 000002.wav
│   └── ...
├── manifest.json
└── article.wav
```

每个 chunk 维护状态，支持 resume。

## 31. CLI 统一入口

唯一入口：

```text
voice-pipeline
```

第一版命令树：

```text
voice-pipeline
├── models verify
├── inspect
├── preprocess all|stage
├── train s1|s2|all
├── evaluate
├── select
├── export
├── infer synthesize|batch|benchmark
└── run
```

## 32. CLI 示例

```bash
voice-pipeline preprocess all -c configs/train.yaml
voice-pipeline preprocess stage sv -c configs/train.yaml
voice-pipeline train s2 -c configs/train.yaml
voice-pipeline train s1 -c configs/train.yaml
voice-pipeline evaluate --run runs/speaker_001
voice-pipeline select --run runs/speaker_001
voice-pipeline export --run runs/speaker_001
```

推理：

```bash
voice-pipeline infer synthesize \
  --model models/speaker_name \
  --reference ref.wav \
  --reference-text "今日はいい天気ですね。" \
  --reference-lang ja \
  --text "今天天气很好。" \
  --lang zh \
  --output outputs/test.wav
```

TXT：

```bash
voice-pipeline infer synthesize \
  --model models/speaker_name \
  --reference ref.wav \
  --reference-text "今日はいい天気ですね。" \
  --reference-lang ja \
  --text-file input.txt \
  --lang zh \
  --output outputs/article.wav
```

## 33. Pipeline YAML

第一版保持简单：

```yaml
pipeline:
  stages:
    - preprocess
    - s2
    - s1
    - evaluate
    - select
    - export
```

运行：

```bash
voice-pipeline run pipeline.yaml
```

Pipeline Orchestrator 只负责编排 Stage，不引入复杂 DAG DSL、插件图或动态 workflow DSL。

## 34. Training YAML 示例

```yaml
profile:
  name: v2ProPlus

experiment:
  name: speaker_001
  output_root: runs

device:
  device: cuda:0
  precision: fp16

dataset:
  manifest: D:/dataset/train.list

objective:
  training_languages:
    - ja
  target_languages:
    - zh
    - ja
    - en
  cross_language_preservation: strict

preprocess:
  resume: true

s2:
  enabled: true
  batch_size: 2
  target_steps: 800
  checkpoint_every_steps: 200
  learning_rate: 0.0001
  text_low_lr_rate: 0.4
  freeze_quantizer: true
  grad_ckpt: false

s1:
  enabled: true
  batch_size: 2
  gradient_accumulation: 4
  target_optimizer_steps: 500
  checkpoint_every_steps: 100

evaluation:
  enabled: true
  reference:
    audio: D:/dataset/reference.wav
    text: "..."
    language: ja
  suites:
    zh: configs/eval/zh.txt
    ja: configs/eval/ja.txt
    en: configs/eval/en.txt
    mixed: configs/eval/mixed.txt
```

## 35. 日志设计

终端日志必须可观察。

S1：

```text
[22:14:31] [S1] epoch=3 mini_step=101 optimizer_step=25
[22:14:31] [S1] loss=2.841 acc=0.742 lr=...
[22:14:31] [S1] effective_batch=8
```

S2：

```text
[22:18:03] [S2] step=201
[22:18:03] [S2] loss_g=...
[22:18:03] [S2] loss_d=...
[22:18:03] [S2] mel=...
[22:18:03] [S2] kl=...
```

同时写 `logs/*.jsonl`，为未来 GUI 提供机器可读日志。

## 36. 错误处理

统一异常分类：ConfigurationError、AssetMissingError、ManifestError、StageDependencyError、StageExecutionError、CheckpointError、ModelCompatibilityError、InferenceError、EvaluationError。

CLI 输出失败 stage、失败原因、是否可 resume、需要修复的输入、对应日志路径。默认不吞 traceback；debug 模式允许完整 traceback。

## 37. Resume

Resume 分三层：

- Stage-level：已完成且 signature 未变化则 skip。
- Training-level：恢复 model、optimizer、scheduler、scaler、optimizer step、epoch、RNG state（可行时）。
- Long-text inference：已成功 chunk 跳过，从未完成 chunk 继续。

## 38. Checkpoint 兼容性

项目不依赖官方源码，但必须兼容：

```text
s1v3.ckpt
s2Gv2ProPlus.pth
s2Dv2ProPlus.pth
```

`core/gpt_sovits/compatibility/` 专门负责 official checkpoint → internal model，业务代码不直接依赖官方 checkpoint 字段细节。

## 39. Upstream Compatibility Test

第一版必须存在高价值兼容测试：official pretrained weights + same input + same seed，比较 frontend phoneme sequence、BERT shape、SV shape、semantic shape、S1 key tensor behavior、S2 key tensor behavior、inference output basic behavior。

目的：先证明 Core 搬运正确，再开始 fine-tuning。

## 40. 测试策略

Unit：config parse、profile registry、manifest parse、text source、stage state、cache invalidation、ModelBundle schema。

Integration：preprocess small fixture、S1 one-step smoke、S2 one-step smoke、bundle export、inference smoke、evaluator smoke。

Compatibility：upstream parity。

CLI：`--help`、invalid config、missing model、text/text-file mutual exclusion、resume behavior。

## 41. README 结构

README 必须先解决“怎么跑”，再解释架构：

```text
1. 项目解决什么问题
2. 5 分钟 Quick Start
3. 环境要求
4. 权重目录
5. 第一次训练
6. 第一次推理
7. 项目架构
8. Preprocess 原理
9. S1 原理
10. S2 原理
11. SV embedding 原理
12. 跨语言保护原则
13. Evaluator 原理
14. ModelBundle
15. CLI Reference
16. YAML Reference
17. Output Directory
18. Resume / Cache
19. Troubleshooting
20. Upstream / License
```

README 中的命令必须与真实 CLI 一致。

## 42. 从 Dataset Builder 吸取的工程教训

1. CLI 只有一个入口：`voice-pipeline`。
2. 配置就是事实：不允许大量隐藏 `os.environ` 改变业务行为；底层必须依赖 env 时，只允许 adapter 临时转换。
3. 每个 Stage 必须有明确输入输出：统一 `StageContract`。
4. Cache 必须可解释：为什么命中、为什么失效、downstream 为什么 invalidated。
5. README 与真实运行同步：优先保证 Quick Start 能跑通。

## 43. 第一版 v2ProPlus 核心代码搬运范围

预计需要迁移 / 改造：

Frontend：multilingual text normalization、language segmentation、phoneme/G2P、symbol mapping、BERT alignment。

S1：Text2SemanticDecoder、相关 AR modules、optimizer、scheduler、dataset/collate、checkpoint loader。

S2 v2ProPlus：SynthesizerTrn、MultiPeriodDiscriminator、commons、losses、mel processing、v2ProPlus dataset/collate、checkpoint save/load。

Features：CN-HuBERT、ERes2NetV2 SV、semantic extraction。

Inference：prompt semantic extraction、reference spec、v2Pro speaker conditioning、semantic generation、acoustic synthesis、text segmentation。

明确不搬：Gradio WebUI、API server、model downloader、V1/V2/V3/V4 compatibility branches、unrelated tools、old scripts、GUI utilities、unused model backends。

## 44. 未来支持 GPT-SoVITS V4

第一版不实现 V4，但保留 ProfileRegistry。未来新增：

```text
profiles/v4.py
core/gpt_sovits/s2_v4/
training/s2_v4/
inference/v4/
```

可复用 CLI、pipeline、run/state、evaluator、ModelBundle outer schema、logging、text source、artifact manager、S1 中可共享部分、frontend 中可共享部分。

V4 专属变化主要集中：S2 architecture、dataset、training strategy、checkpoint codec、vocoder、acoustic inference backend。因此不需要全套重写。

## 45. 设计冻结项

当前已经确认冻结：

1. 第一版只实现 GPT-SoVITS v2ProPlus。
2. 不依赖外部 GPT-SoVITS clone。
3. 必要官方核心代码复制进本项目。
4. 复制只做工程化整理，不改变模型表现相关算法行为。
5. 训练和推理分离实现，但共享 Profile / Frontend / ModelBundle。
6. v2ProPlus preprocessing 包含 SV embedding。
7. S1 使用全模型训练逻辑。
8. S1 显式 gradient accumulation。
9. S1/S2 使用 step budget，而不是仅以 epoch 作为主训练尺度。
10. S2 不称为严格全参 FT。
11. 第一版不修改 Loss。
12. 跨语言 ZH / JA / EN 是最高级质量约束。
13. Evaluator 与 Training 完全分离。
14. checkpoint selection 采用 hard constraints + ranking + human shortlist。
15. reference audio 保持官方机制。
16. 推理支持 `--text` 和 `--text-file`。
17. `--text-file` 第一版只支持 TXT。
18. CLI 只有统一入口。
19. Pipeline YAML 第一版保持简单。
20. GUI 最后做。

## 46. 第一版成功标准

第一版完成后，应该能够从一份已有：

```text
audio_path|speaker|language|text
```

manifest 开始，通过 CLI 完整完成：

```text
preprocess
→ S2
→ S1
→ evaluate
→ select
→ export
→ infer
```

并且：不需要官方 GPT-SoVITS 源码仓库；能直接加载官方 v2ProPlus pretrained weights；能输出兼容的 fine-tuned checkpoint；能生成 ZH/JA/EN 音频；能自动比较 checkpoint；能保留人工最终选择入口；每个 stage 都能独立执行和 resume；全部配置可追踪；没有隐藏 WebUI 行为；没有散落的临时目录；后续可被更高层 end-to-end pipeline 调用；后续可增加 GUI；后续可增加 GPT-SoVITS V4 profile。

## 47. 实现顺序建议

```text
Phase 1
Project skeleton + config + profile + paths + state + CLI

Phase 2
v2ProPlus core migration + frontend + S1 + S2 + features

Phase 3
upstream compatibility tests

Phase 4
preprocess pipeline

Phase 5
S2 trainer

Phase 6
S1 trainer

Phase 7
ModelBundle

Phase 8
InferenceSession + inline text + TXT

Phase 9
Evaluator

Phase 10
Model selection + export

Phase 11
pipeline.yaml orchestrator

Phase 12
README + full integration tests
```

这里故意把 compatibility test 放在正式训练之前：先证明 Core 搬运正确，再开始训练。

## 48. 最终架构图

```text
                         voice-pipeline
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
      Pipeline             Evaluation           Inference
          │                    │                    │
          │                    │                    │
      Training ─────────── ModelBundle ─────────────┘
          │
          │
    ┌─────┴─────────────┐
    │                   │
Preprocess           Trainer
    │              ┌────┴────┐
    │              │         │
    │             S1         S2
    │
    └────────── Shared Profile / Frontend
                       │
                       ↓
            Internal GPT-SoVITS Core
                       │
                       ↓
                  v2ProPlus
```

长期：

```text
Profile Registry
├── v2ProPlus   ← v1 implementation
└── v4          ← future
```

但第一版代码严格控制在 v2ProPlus 所需范围内。

## 49. 核心设计结论

本框架的核心不是“把 GPT-SoVITS WebUI 换成 CLI”，而是：

> 将经过验证的 GPT-SoVITS v2ProPlus 算法核心，重组为一套可独立版本管理、可阶段编排、可训练、可推理、可评测、可恢复、可长期维护的工程系统。

同时将“跨语言 speaker identity 保持”提升为一等质量目标：

```text
JA-only speaker data
        ↓
speaker adaptation
        ↓
same identity in
ZH / JA / EN
```

训练过程不以单语言 loss 或最后一个 checkpoint 为最终标准，而以跨语言综合评测和人耳最终裁决为准。

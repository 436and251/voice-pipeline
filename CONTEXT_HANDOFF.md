# Voice Pipeline / GPT-SoVITS v2ProPlus Context Handoff

> 用途：在新的 ChatGPT 对话中恢复本项目上下文。
>
> 新对话使用方式：上传本文件和最新完整 milestone ZIP，然后说：
>
> **“按 CONTEXT_HANDOFF.md 恢复项目上下文。先解压最新 milestone ZIP 并跑全量测试确认真实基线，然后继续已经审核通过的下一 Part。严格 TDD，每个 Part 完成后停下来让我审核。”**

---

## 1. 项目目标

开发一个 **self-contained GPT-SoVITS v2ProPlus 训练 + 推理 CLI 框架**。

硬约束：

- V1 **只实现 v2ProPlus**
- 仅为未来 v4 保留 Profile 扩展位，不实现 V3/V4/CFM/LoRA
- 最终运行时不依赖外部 GPT-SoVITS clone
- 必要 upstream 代码迁入项目内部
- 模型权重作为 external assets
- CLI 优先，无 GUI
- 严格 TDD：RED → GREEN → 全量 regression
- 用户最新要求：**每一个 Part 完成后停下来审核，审核通过后才进入下一 Part**
- 不允许用 mock/unit test 数量掩盖真实 model core 尚未迁移
- 最终必须区分 unit/compat、integration、real-weight、GPU smoke test

---

## 2. 官方 GPT-SoVITS 基线

```text
Repository: RVC-Boss/GPT-SoVITS
Pinned commit: 48b1a0169a28582a8984402f82cf438d3bfa6aca
```

所有版本路由、训练、推理兼容性判断都以该 commit 为准。

---

## 3. 最重要的 v2ProPlus 路由结论

### 正确路由

```text
v2ProPlus
→ GPT_SoVITS/s2_train.py
→ GPT_SoVITS/configs/s2v2ProPlus.json
→ SynthesizerTrn
→ MultiPeriodDiscriminator
→ SV conditioning
→ G/D adversarial training
```

### V1 禁止进入

```text
s2_train_v3.py
s2_train_v3_lora.py
SynthesizerTrnV3
CFM-only trainer
V3/V4 LoRA
V3/V4 external vocoder / BigVGAN path
```

### 已确认的严重历史错误

曾错误推断：

```text
v2ProPlus = s2_train_v3.py = CFM-only / no discriminator
```

**该结论已判定为错误，禁止恢复。**

官方 WebUI router 明确：

```python
if version in ["v1", "v2", "v2Pro", "v2ProPlus"]:
    # GPT_SoVITS/s2_train.py
else:
    # V3/V4 branch
```

---

## 4. v2ProPlus S2 官方训练语义

### 模型

```text
Generator:     SynthesizerTrn
Discriminator: MultiPeriodDiscriminator
```

### v2ProPlus batch

额外包含：

```text
sv_emb
```

Generator forward：

```text
net_g(
    ssl,
    spec,
    spec_lengths,
    text,
    text_lengths,
    sv_emb
)
```

### 一个 batch 的训练顺序

```text
G forward
  ↓
y_hat
  ├─ y_hat.detach()
  │      ↓
  │    D forward
  │      ↓
  │ discriminator_loss
  │      ↓
  │    D step
  │
  └─ y_hat
         ↓
       D forward
         ↓
       adversarial generator loss
       + feature matching
       + mel loss
       + kl_ssl
       + KL loss
         ↓
       G step
```

### loss adapter

官方：

```python
discriminator_loss(...)
```

返回：

```text
(total_loss, real_branch_losses, fake_branch_losses)
```

官方：

```python
generator_loss(...)
```

返回：

```text
(total_loss, branch_losses)
```

不能直接将 tuple 送入 backward。

### AMP / gradient 顺序

必须保留：

```text
backward
→ scaler.unscale_(optimizer)
→ clip_grad_value_
→ scaler.step(optimizer)
```

### optimizer 参数组

v2ProPlus：

```text
learning_rate = 1e-4
text_low_lr_rate = 0.4
```

分组：

```text
base parameters        1.0 × LR
enc_p.text_embedding   0.4 × LR
enc_p.encoder_text     0.4 × LR
enc_p.mrte             0.4 × LR
```

要求：
- 无重复参数
- 不漏掉其他 trainable parameters

### S2 step budget

项目定义：

```text
1 个 S2 optimizer step
= 同一个 batch 完成一次 D update + 一次 G update
```

checkpoint / stop 在 G step 完成后计数。

---

## 5. S1 官方 optimizer 语义

必须精确保留：

```python
self.manual_backward(loss)

if batch_idx > 0 and batch_idx % 4 == 0:
    opt.step()
    opt.zero_grad()
    scheduler.step()
```

因此：

```text
batch_idx 0,1,2,3 → 不 step
batch_idx 4       → 第一次 step
```

第一次更新累计：

```text
mini-batch 0..4
= 5 个 mini-batch
```

之后例如：

```text
5..8
= 4 个 mini-batch
```

硬约束：

- loss 不除以 4
- 不改成标准“每 4 mini-batch”
- epoch 末不 flush residual gradients
- batch_idx 每 epoch 从 0 重置
- target_optimizer_steps 统计真实 `opt.step()` 次数
- checkpoint 按真实 optimizer step

---

## 6. v2ProPlus 推理路由

v2ProPlus：

```text
SynthesizerTrn
use_vocoder = False
is_v2pro = True
```

不是：

```text
SynthesizerTrnV3
+ V3/V4 external vocoder
```

### Reference conditioning

参考音频：

```text
3 ~ 10 sec
```

prompt semantic：

```text
reference audio → 16k
+ zero padding
→ raw CN-HuBERT model
→ vits_model.extract_latent()
→ prompt_semantic
```

Pinned upstream 一个不直观但必须保留的行为：

```text
zero padding length
= 32000 × 0.3
= 9600 samples
```

虽然拼接对象是 16k reference，V1 不自行“修正”为 4800。

### phones 路由

S1 semantic prediction：

```text
prompt phones/BERT + target phones/BERT
```

S2 decode：

```text
target phones only
+ predicted semantic
+ reference spectrogram
+ v2ProPlus SV embedding
```

不能把 prompt phones 传给 S2 decode。

---

## 7. 预处理 DAG

官方兼容产物：

```text
dataset.list
│
├─ 2-name2text.txt
├─ 3-bert/
├─ 4-cnhubert/
├─ 5-wav32k/
├─ 6-name2semantic.tsv
└─ 7-sv_cn/
```

### wav32k / HuBERT

官方从**原始音频**分别产生：

```text
32k saved wav
16k HuBERT input
```

即使我们的 stage 拆开，HuBERT stage 也必须重新从原始音频生成 16k 输入。

禁止：

```text
saved int16 wav32
→ resample to 16k
→ HuBERT
```

避免额外量化漂移。

### semantic

必须使用：

```text
base pretrained S2G
```

不能使用正在 fine-tune 的 S2 checkpoint。

### SV

```text
32k wav
→ 16k
→ Kaldi fbank (80 bins)
→ ERes2NetV2 forward3
→ speaker embedding
```

---

## 8. Frontend

训练与推理共用唯一 frontend。

建议输出：

```python
FrontendOutput(
    normalized_text=...,
    phones=...,
    phone_ids=...,
    word2ph=...,
    bert_features=...,
)
```

### 中文链路

```text
jieba/POS
→ ToneSandhi
→ G2PW / pypinyin
→ correct_pronunciation
→ erhua
→ OpenCPOP phone mapping
→ word2ph
```

G2PW 必须是显式 external asset。

### ToneSandhi

来源：

```text
GPT_SoVITS/text/tone_sandhi.py
blob SHA:
4ed737811a54456adbd5178c6398deb3ff6a12ab
```

策略：

- 不把手工复制版本冒充 byte-exact vendor
- 允许明确标记为 adapted upstream-derived
- heavy deps 可改成 lazy import
- 行为必须 parity test

至少覆盖：

```text
不
一
双三声
轻声
merge_bu
merge_yi
merge_er
重叠词
continuous third-tone merge
```

---

## 9. Asset Layout

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
        └── g2pw/
            └── G2PWModel/
```

CLI：

```text
voice-pipeline models verify
```

---

## 10. Profile

V1：

```python
PROFILE_REGISTRY = {
    "v2ProPlus": V2ProPlusProfile,
}
```

业务层通过：

```python
profile = registry.get(config.profile)
```

Profile 应锁定：

```text
s2_family = v2/pro
generator = SynthesizerTrn
discriminator = MultiPeriodDiscriminator
uses_sv_embedding = true
uses_gan_training = true
uses_text_low_lr_groups = true
uses_cfm_training = false
uses_external_vocoder = false
```

未来 v4 只保留扩展位：

```text
profiles/v4.py
core/gpt_sovits/s2_v4/
training/s2_v4/
inference/v4/
```

当前不实现。

---

## 11. ModelBundle

```text
models/speaker_name/
├── model.yaml
├── metadata.json
├── weights/
│   ├── s1.ckpt
│   └── s2.pth
└── reference/
    ├── default.wav
    └── default.json
```

注意：

```text
S2D
```

只用于训练，不进入 inference bundle。

BERT/HuBERT/SV/G2PW 属于共享 profile assets。

---

## 12. Evaluation

不能只看 training loss。

至少：

```text
zh
ja
en
mixed
```

核心指标：

```text
speaker similarity
ASR / pronunciation
language drift
prosody
```

选择必须 worst-language aware：

```text
MinSim = min(Sim_ZH, Sim_JA, Sim_EN)
WorstPronunciation
WorstLanguageDrift
```

---

## 13. Pipeline

```text
preprocess
→ s2
→ s1
→ evaluate
→ select
→ export
```

YAML：

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

约束：

- 只允许已知 stage
- 不允许重复 stage
- 失败立即停
- resume 跳过 completed
- cache signature 包括 input/config/profile/implementation/assets

---

## 14. Vendor Strategy

最终要求：

```text
self-contained implementation
+
external model assets
```

运行时不能依赖外部官方 clone。

官方 clone 只允许作为一次性 migration source：

```text
source GPT-SoVITS root
→ pinned manifest
→ Git blob SHA verification
→ copy exact required files
→ internal vendor tree
```

### S2 models.py 注意

官方 `module/models.py` 顶层包含 V3/V4 相关：

```text
f5_tts.model.DiT
```

但 v2ProPlus 的 `SynthesizerTrn` 不应因此拖入 CFM/F5-TTS。

需要：

```text
minimal closure
or
adapted/lazy import boundary
```

---

## 15. 下一 Part：S1 Core Minimal Vendoring + Real Model Construction

这是用户已经审核通过的下一 Part。

范围：

```text
AR/models/t2s_model.py
AR/models/utils.py
AR/modules/embedding.py
AR/modules/transformer.py
AR/modules/activation.py
AR/modules/patched_mha_with_cache.py
AR/modules/scaling.py
```

训练后续可能需要：

```text
AR/modules/optim.py
AR/modules/lr_schedulers.py
```

但这个 Part 暂时不接 optimizer/data/CLI。

测试目标：

```text
S1 core 能独立 import
不依赖外部 GPT-SoVITS clone
不引入 ONNX/V3/V4 路径
最小 config 能构造真实 S1 model
embedding/vocab shape 与官方 config 对齐
真实 S1 checkpoint 可以 load
```

完成后：

```text
模块测试
→ 全量 pytest
→ compileall
→ git commit
→ 完整 milestone ZIP
→ 停下来给用户审核
```

---

## 16. 开发协作方式

用户最新要求：

> 每一个 Part 完成后都让我审核，一步步来。

固定流程：

```text
Part N
→ 说明目标 / 边界 / upstream 依据
→ RED
→ 确认 RED 原因正确
→ 最小实现
→ GREEN
→ 全量 regression
→ compileall
→ 阶段总结
→ git commit
→ milestone ZIP
→ 用户审核
→ 才进入下一 Part
```

---

## 17. 持久化要求

沙箱曾多次重置，因此每个审核通过的 Part 都必须：

```text
git commit
+
完整 milestone ZIP
```

不能只依赖聊天记录或 ledger。

新对话如果工作区缺失：

1. 恢复用户上传的**最新完整 milestone ZIP**
2. 重新跑全量测试
3. 以真实绿色结果为基线
4. 不要把历史 `pytest passed` 数字当作当前状态

---

## 18. 禁止恢复的错误结论

### 错误 1

```text
v2ProPlus → s2_train_v3.py
```

正确：

```text
v2ProPlus → s2_train.py
```

### 错误 2

```text
v2ProPlus = CFM-only / no D
```

正确：

```text
SynthesizerTrn + MultiPeriodDiscriminator + G/D
```

### 错误 3

```text
text_low_lr_rate=0.4 是无效旧配置
```

错误。它是 active optimizer parameter grouping。

### 错误 4

```text
S1 = 标准每 4 mini-batch accumulation
```

错误。首次 optimizer step 在 batch_idx=4，累计 0..4 共 5 个 mini-batch。

### 错误 5

```text
HuBERT 可以从保存后的 wav32 int16 再降采样
```

错误。应从原始音频重新生成官方 16k 输入。

---

## 19. 新对话推荐首句

```text
按 CONTEXT_HANDOFF.md 恢复项目上下文。
先解压我上传的最新完整 milestone ZIP 并运行全量测试确认真实基线。

继续已经审核通过的下一 Part：
S1 Core Minimal Vendoring + Real Model Construction。

严格 RED→GREEN→全量 regression。
完成这一 Part 后停下来给我 code review，不要自动进入下一 Part。
V1 只实现 v2ProPlus，禁止引入 V3/V4/CFM。
```

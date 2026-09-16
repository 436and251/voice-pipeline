# Voice Pipeline 中文使用指南

本文说明如何从官方格式 `data.list` 开始完成 GPT-SoVITS v2ProPlus 预处理、S2/S1
增训，以及如何使用正式 `ModelBundle` 推理。

## 1. 进入项目并启用环境

始终从下面这个项目根目录运行命令。公共 BERT、HuBERT、G2P 和 speaker 模型都按
这个目录解析：

```powershell
Set-Location 'D:\AI-Training\voice-clone\voice-pipeline\voice-pipeline'
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\Activate.ps1'
```

首次使用时把当前项目注册到已有 uv 虚拟环境。`--no-deps` 不会重复安装 Torch 等
现有依赖：

```powershell
uv pip install -e . --no-deps
voice-pipeline version
ffmpeg -version
```

后续只需进入项目目录、激活同一个虚拟环境，不必重复注册。

## 2. 检查模型目录

公共 v2ProPlus 资源应位于：

```text
models/pretrained/v2proplus/
├── bert/chinese-roberta-wwm-ext-large/
├── g2p/en/nltk_data/
├── g2pw/G2PWModel/
├── hubert/chinese-hubert-base/
├── langdetect/lid.176.bin
├── s1/s1v3.ckpt
├── s2/s2Gv2ProPlus.pth
├── s2/s2Dv2ProPlus.pth
└── speaker/pretrained_eres2netv2w24s4ep4.ckpt
```

执行一次完整检查：

```powershell
voice-pipeline models verify --project-root . --profile v2ProPlus
```

所有项目都显示 `OK` 后再开始预处理。

## 3. 从 data.list 开始训练

### 3.1 准备数据

本项目直接读取 GPT-SoVITS 官方四字段格式，不负责切片、ASR 或改写文本：

```text
音频路径|说话人名称|语言|文本
```

例如：

```text
wavs/001.wav|speaker_001|ja|今日はいい天気ですね。
wavs/002.wav|speaker_001|ja|明日もよろしくお願いします。
wavs/003.wav|speaker_001|en|Hello, nice to meet you.
wavs/004.wav|speaker_001|zh|你好，很高兴见到你。
wavs/005.wav|speaker_001|mixed|今日は sunny day ですね。
```

保存为 UTF-8 或 UTF-8-SIG。相对音频路径以 `data.list` 所在目录为基准，也可以使用
绝对路径。每条记录必须在一行内，正文不能包含 `|`。`language` 必须明确为 `zh`、
`ja`、`en` 或 `mixed`，并按该条音频实际内容填写；数据主要来自日语说话人，不代表
所有记录都强制走日语 frontend。项目不做多说话人校验。

S2 的硬性训练范围是音频时长严格大于 0.6 秒且严格小于 54 秒。实际 few-shot 数据
建议提前切成约 3～10 秒、无长静音、文本与发音一致的单声道片段；采样率和声道数可
不同，预处理会通过 FFmpeg 转成 32 kHz 单声道 PCM16。

### 3.2 创建训练配置

复制已经可同时用于预处理和训练的示例：

```powershell
Copy-Item configs/train.example.yaml configs/train.local.yaml
```

至少修改 `experiment.name` 和 `dataset.manifest`：

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
  manifest: D:/dataset/data.list

objective:
  training_languages: [ja]
  target_languages: [zh, ja, en]
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
  resume_from: null

s1:
  enabled: true
  batch_size: 2
  gradient_accumulation: 4
  target_optimizer_steps: 500
  checkpoint_every_steps: 100
  resume_from: null
```

`experiment.name` 决定 `runs/<name>/`，建议只使用 ASCII 字母、数字、下划线和连字符；
后续 shortlist 中的 `model_name` 才决定正式模型目录名。Windows YAML 路径优先使用
正斜杠。`objective` 是训练目标记录，不会把所有样本统一成某种语言；每条数据仍以
`data.list` 的 `language` 为准。

训练按 step 控制，不按 epoch 控制。上面的 800/500 是可直接开始的 few-shot 基线，
不是自动最佳值。S1 的 `gradient_accumulation` 固定为 4；`batch_size: 2` 时一个 S1
optimizer step 对应 8 条 mini-batch 样本。显存不足时可先把 S1 或 S2 的
`batch_size` 降为 1，不要修改 S1 的累积值。

### 3.3 推荐：统一运行到人工候选

训练配置确认后，复制统一编排示例：

```powershell
Copy-Item configs/pipeline.example.yaml configs/pipeline.local.yaml
voice-pipeline run configs/pipeline.local.yaml --project-root .
```

`pipeline.local.yaml` 中的 `config` 从 `--project-root` 解析；默认依次执行
`preprocess → s2 → s1 → evaluate`。状态写入
`runs/<experiment.name>/pipeline-state.json`。再次执行同一命令时，`completed` 阶段直接
跳过，`failed` 或中断时留下的 `running` 阶段会重试。

训练恢复不会自动猜测 checkpoint。需要恢复 S1/S2 时，先在
`configs/train.local.yaml` 的对应阶段填写明确的 `resume_from`，再重跑统一命令。

`voice-pipeline run` 在自动评测、候选 ModelBundle 转换和三语试听生成完成后停止，不执行
强清理；此时 `preprocess/`、`training/` 和原始 S1/S2 恢复 checkpoint 全部保留，供你
试听、比较或继续训练。只有人工确定最终候选并成功执行
`voice-pipeline export --select candidate_X` 后，才验证持久产物并执行强清理。

### 3.4 完整预处理

只运行下面这一条，才能发布 S1/S2 都需要的正式训练索引：

```powershell
voice-pipeline preprocess all -c configs/train.local.yaml
```

流程依次生成 Text/BERT、32 kHz WAV、HuBERT、speaker embedding 和 semantic token。
`preprocess.resume: true` 会复用签名仍有效的缓存；修改音频、文本、语言或相关模型后，
对应样本及下游阶段会自动重算。

`preprocess stage semantic` 等单阶段命令只用于排查，不会发布完整
`valid_samples.jsonl` 和训练索引，不能代替 `preprocess all`。

预处理完成后检查：

```powershell
Get-Content runs/speaker_001/preprocess/quarantine.jsonl
(Get-Content runs/speaker_001/preprocess/valid_samples.jsonl).Count
```

坏 manifest 行、解码/特征失败，以及不满足 S1/S2 训练条件的样本共用同一个容错
上限：`min(5, ceil(非空记录数 × 20%))`。未超过上限时坏样本进入
`quarantine.jsonl`，其余样本继续；超过上限或没有有效样本时预处理失败。不要直接
编辑生成的 `valid_samples.jsonl` 或各阶段 `index.jsonl`，修正原始数据后重新执行
`preprocess all`。

### 3.5 训练 S2 和 S1

首次真实训练建议分开运行，便于分别观察显存和日志：

```powershell
voice-pipeline train s2 -c configs/train.local.yaml --project-root .
voice-pipeline train s1 -c configs/train.local.yaml --project-root .
```

也可以按固定的 S2→S1 顺序连续运行：

```powershell
voice-pipeline train all -c configs/train.local.yaml --project-root .
```

S2 是 v2ProPlus 官方 GAN 路线，包含 Generator 与 Discriminator 更新，不是 CFM。
quantizer 保持冻结，text/MRTE 使用 `0.4` 倍学习率。CUDA FP16 使用动态
GradScaler；早期溢出会自动降 scale，不要手工固定为 1。S1 是完整
Text2SemanticDecoder 增训，严格每 4 个成功 mini-batch 执行一次 optimizer update。

运行产物位于：

```text
runs/speaker_001/
├── preprocess/
├── training/
│   ├── s2/
│   │   ├── events.jsonl
│   │   └── checkpoints/step-00000800.pt
│   └── s1/
│       ├── events.jsonl
│       └── checkpoints/step-00000500.pt
├── evaluation/
└── export/
```

这些 `.pt` 是包含模型、优化器、scheduler、GradScaler、随机数状态和精确 batch cursor
的内部恢复 checkpoint，不是可以直接交给推理器的最终权重。

#### 磁盘容量与 15 组评测工作包

评测落盘的中间组合数不等于最终人耳候选数。设 S2、S1 分别保存 `N2`、`N1` 个训练
checkpoint，第一阶段保留 `K = s2_keep` 个 S2，则唯一评测组合数为
`(1 + N2) + K × N1`。默认基线保存 4 个 S2 和 5 个 S1，且 `s2_keep: 2`，因此会生成
`5 + 2 × 5 = 15` 组临时工作包；自动评测结束后仍只导出 `shortlist_size: 3` 个匿名
候选给人试听。

Acane 首次真实运行的 16.36 GiB 占用拆分如下：

| 目录 | 占用 | 原因 |
|---|---:|---|
| `training/s2/` | 7.21 GiB | 4 个约 1.80 GiB 的完整恢复 checkpoint |
| `training/s1/` | 4.34 GiB | 5 个约 0.87 GiB 的完整恢复 checkpoint |
| `evaluation/work/` | 4.59 GiB | 15 组、每组约 0.31 GiB 的临时 S1/S2 推理权重 |
| `evaluation/generated/` | 0.13 GiB | 自动评测 WAV 与可恢复 chunk |
| `preprocess/` | 0.09 GiB | 正式训练特征 |

`evaluation/work/` 在评测完整成功、候选转换和试听生成完成后自动删除；评测失败或中断
时会保留，以便原样重跑并复用已有权重与 WAV。人工选择前不要手工删除这些目录。本基线
至少准备 20 GiB 可用空间，建议准备 25 GiB；`evaluation.models.cache_dir` 指向的
Faster-Whisper/WavLM 缓存还需在其所在磁盘另行预留空间。

如果磁盘不足，可增大 S1/S2 的 `checkpoint_every_steps` 来减少 checkpoint 数量，但会
同时减少恢复点和自动评测可比较的候选；也可降低 `evaluation.pairing.s2_keep` 来减少
第二阶段组合。`shortlist_size` 只控制最后导出给人耳的数量，不会减少前面的自动评测。

### 3.6 中断后恢复训练

恢复时在 `configs/train.local.yaml` 中填写对应阶段的内部 checkpoint，并把目标 step
保持为不小于 checkpoint 中已有的 step：

```yaml
s2:
  target_steps: 1200
  resume_from: runs/speaker_001/training/s2/checkpoints/step-00000800.pt

s1:
  target_optimizer_steps: 800
  resume_from: runs/speaker_001/training/s1/checkpoints/step-00000500.pt
```

然后只运行需要恢复的阶段：

```powershell
voice-pipeline train s2 -c configs/train.local.yaml --project-root .
voice-pipeline train s1 -c configs/train.local.yaml --project-root .
```

不要把 S1 checkpoint 填给 S2，反之亦然，也不要使用导出后的推理权重恢复训练。
checkpoint 文件名中的 step 必须与内部 cursor 一致，框架会在加载前严格校验结构。

### 3.7 自动评测与人工选择

训练完成后运行：

```powershell
voice-pipeline evaluate -c configs/train.local.yaml --project-root .
```

评测器先用官方 Base S1 对 Base S2 和全部 S2 checkpoint 做第一阶段筛选，再组合保留的
S2 与全部 S1 checkpoint。它使用 Faster-Whisper 计算中文/日文 CER、英文 WER 和语言一致性，
使用独立 WavLM speaker encoder 计算跨语言音色相似度，并统计基础韵律。ASR 和 WavLM
只参与离线评测，不会进入最终推理链路。

`evaluation.reference.audio` 和 `speaker_references` 可以位于项目目录之外。评测开始时会把
用于推理条件的 reference 原子快照到 `runs/<目标人>/evaluation/reference/<SHA256>.wav`，
shortlist 和候选包引用这份不可变快照；多个 speaker reference 仍直接用于计算评测 centroid。

结果位于 `runs/<目标人>/evaluation/`：

```text
stage1-report.md / stage1-results.json  # S2 初筛
report.md / results.json                # S1+S2 全部组合
shortlist.yaml                          # 通过硬约束的匿名候选
listening/
├── candidate_A/{zh,ja,en}.wav
├── candidate_B/{zh,ja,en}.wav
└── manifest.json
```

必须试听每个匿名候选的三种语言，再人工选择最终候选：

```powershell
voice-pipeline export --run runs/<目标人> --project-root . --select candidate_A
```

不要按 checkpoint step、单一 loss 或自动总分直接决定最终模型。阈值说明、模型缓存和
调参方法见 `docs/evaluation.md`。

### 3.8 清理规则

单独运行预处理或训练命令时，只清除阶段自身的 `*.tmp` 和已 quarantine 样本的孤立
产物，保留正式预处理结果与训练 checkpoint；异常或中断时保留现场以便排查和恢复。

`voice-pipeline run` 和单独的 `evaluate` 都不会执行强清理。每个评测候选已经提前转换
为可独立推理的 ModelBundle，保存在 `runs/<目标人>/export/candidates/`；预处理结果与
原始 checkpoint 会一直保留到人工确定最终候选。

试听后由人显式晋升最终候选：

```powershell
voice-pipeline export --run runs/<目标人> --project-root . --select candidate_A
```

候选成功晋升到正式模型目录后，系统才验证评分报告、shortlist、全部候选包和三语试听
哈希，并不可逆删除 `preprocess/`、`training/`（包括原始恢复 checkpoint）、
`evaluation/generated/`、`work/` 及其他 run 内杂项。保留状态、评测证据、试听音频、
全部候选包和正式模型。晋升失败时绝不清理；极少数清理验证失败时，已晋升的正式模型和
原始训练资源都会保留，并返回明确错误。

## 4. 使用正式模型推理

推理与训练解耦，不依赖原始 `data.list` 或完整 `runs/` 目录；只需要正式
`ModelBundle`、公共预训练资源和输入文字。

要推理的目标人必须是已经导出并人工选定的正式 ModelBundle：

```text
models/<目标人名称>/
├── model.yaml
├── metadata.json
├── weights/
│   ├── s1.ckpt
│   └── s2.pth
└── reference/
    ├── default.wav
    └── default.json
```

只有官方预训练权重、训练 checkpoint 或候选目录还不等于正式 ModelBundle。应先完成
候选导出和人工选择，例如：

```powershell
voice-pipeline export --run runs/speaker_001 --project-root .
voice-pipeline export --run runs/speaker_001 --project-root . --select candidate_B
```

第二条命令会把人工选中的候选晋升到 `models/<目标人名称>/`。

### 4.1 最简单的文字转语音

以下命令使用 ModelBundle 内置参考音频：

```powershell
voice-pipeline infer synthesize `
  --model models/speaker_001 `
  --text '今天天气很好。' `
  --lang zh `
  --output hello.wav `
  --device cuda:0
```

`--output` 不是任意磁盘路径，而是目标人输出目录下的安全相对路径。上例实际生成：

```text
outputs/speaker_001/
├── hello.wav
└── hello.infer/
    ├── manifest.json
    └── chunks/000001.wav
```

这样不同目标人的音频不会混在一起。禁止给 `--output` 传绝对路径或包含 `..` 的
路径。需要改变整个输出根目录时可传 `--output-root D:\somewhere\outputs`，其下面
仍会保留 `<目标人名称>/` 这一层。

支持的语言值：

- `zh`：中文；
- `ja`：日文；
- `en`：英文；
- `mixed`：混合语言，按内容选择各自 frontend。

语言必须明确传入，不支持 `auto`。

### 4.2 使用 TXT 长文本

TXT 必须是 `.txt`，编码为 UTF-8 或 UTF-8-SIG：

```powershell
voice-pipeline infer synthesize `
  --model models/speaker_001 `
  --text-file inputs/article.txt `
  --lang mixed `
  --output articles/article.wav `
  --device cuda:0
```

`--text` 和 `--text-file` 必须二选一。长文本按段落、强标点、弱标点递归切分，最后
才按长度硬切。中文默认每段不超过 100 个 Unicode 码位；日文、英文和 mixed 默认
不超过 500。可用 `--max-chars` 覆盖。

音频 chunk 之间默认插入 10 ms 静音：

```powershell
voice-pipeline infer synthesize `
  --model models/speaker_001 `
  --text-file inputs/article.txt `
  --lang zh `
  --output article.wav `
  --pause-ms 30
```

设为 `--pause-ms 0` 即不额外插入静音。

### 4.3 临时覆盖参考音频

不传覆盖参数时使用 ModelBundle 内置参考条件。要临时更换参考音频，必须同时明确
参考语言。参考音频必须为 3～10 秒：

```powershell
voice-pipeline infer synthesize `
  --model models/speaker_001 `
  --reference D:\references\speaker_001_ja.wav `
  --reference-text '今日はいい天気ですね。' `
  --reference-lang ja `
  --text 'Hello, nice to meet you.' `
  --lang en `
  --output reference-tests/en.wav `
  --device cuda:0
```

`--reference-text` 可以省略；省略时 S1 使用 ref-free 路径，但 S2 仍使用参考频谱和
speaker embedding。不能在没有 `--reference` 时单独传参考文本或参考语言。

### 4.4 断点恢复与覆盖

每个长文本任务都会保存 manifest 和独立 chunk WAV。命令中断后，原样重新执行即可
跳过哈希仍然有效的 chunk：

```powershell
# 中断后再次执行相同命令
voice-pipeline infer synthesize `
  --model models/speaker_001 `
  --text-file inputs/article.txt `
  --lang zh `
  --output article.wav
```

模型、参考音频、文字、语言、seed、停顿或任一解码参数变化后，旧 manifest 不会被
误用。此时请换一个输出名称，或明确允许重建：

```powershell
voice-pipeline infer synthesize `
  --model models/speaker_001 `
  --text-file inputs/article.txt `
  --lang zh `
  --output article.wav `
  --overwrite
```

`--overwrite` 只删除该输出对应的 WAV 和 `.infer` 工作目录，不影响模型、训练结果或
其他目标人的输出。

### 4.5 批量推理

复制并编辑示例：

```powershell
Copy-Item configs/infer.example.yaml configs/infer.local.yaml
```

最小配置：

```yaml
model: models/speaker_001
device: cuda:0
output_root: outputs

defaults:
  language: mixed
  pause_ms: 10
  seed: 0

jobs:
  - name: demos/zh
    text: "你好，很高兴见到你。"
    language: zh
  - name: articles/ja
    text_file: inputs/japanese.txt
    language: ja
```

执行：

```powershell
voice-pipeline infer batch --config configs/infer.local.yaml
```

整批只加载一次模型，job 按顺序运行；失败时停止。再次运行会按各 job 的 manifest
恢复。`name: demos/zh` 对应 `outputs/speaker_001/demos/zh.wav`。

如需覆盖参考条件，只能在 batch 顶层定义一套：

```yaml
reference:
  audio: D:/references/speaker_001.wav
  text: "今日はいい天気ですね。"  # 可删除这一行以使用 ref-free S1
  language: ja
```

单个 job 可以覆盖 `language`、`pause_ms`、`max_chars`、`seed`、`top_k`、
`top_p`、`temperature`、`repetition_penalty`、`noise_scale` 和 `speed`，但不能
单独覆盖参考音频。

### 4.6 性能测试

benchmark 在模型和参考条件加载完成后开始计时，不写 WAV 或 manifest：

```powershell
voice-pipeline infer benchmark `
  --model models/speaker_001 `
  --text 'Hello from the desktop assistant.' `
  --lang en `
  --device cuda:0
```

默认预热 1 次、正式运行 3 次。可调整：

```powershell
voice-pipeline infer benchmark `
  --model models/speaker_001 `
  --text-file inputs/article.txt `
  --lang mixed `
  --warmup 2 `
  --runs 5
```

输出包括生成语音时长、平均耗时、最快耗时和 RTF。推理结果末尾固定追加的 300 ms
保护静音不计入语音时长；RTF 小于 1 表示平均生成速度快于实时播放速度。

### 4.7 在桌面助手或后台进程中调用

不需要通过 CLI，也不需要 WAV 文件：

```python
from voice_pipeline.inference import InferenceSession, synthesize_text

session = InferenceSession.load(
    "models/speaker_001",
    "cuda:0",
)

result = synthesize_text(
    session,
    "你好，我是你的桌面助手。",
    "zh",
    pause_ms=10,
    seed=123,
)

print(result.sample_rate)     # 32000
print(result.waveform.shape)  # 一维 float32 NumPy 数组
```

完整输出末尾会按官方 GPT-SoVITS 行为自动追加 300 ms 静音，避免播放器吞掉最后一个字；
`pause_ms` 只调整长文本 chunk 之间的停顿。

覆盖参考音频：

```python
session = InferenceSession.load(
    "models/speaker_001",
    "cuda:0",
    reference_audio="D:/references/speaker_001.wav",
    reference_text=None,
    reference_language="ja",
)
```

建议在后台进程启动时创建并长期复用 session，不要每句话都重新加载权重。同一个
session 已提供线程安全的串行推理；需要并发吞吐时使用多个工作进程或模型副本。
HTTP、鉴权和流式输出尚未加入，但后续服务层可以直接包装这里的内存接口。

### 4.8 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--pause-ms` | `10` | chunk 之间额外静音，必须为非负整数 |
| `--seed` | `0` | 基础随机种子；每个 chunk 使用派生 seed |
| `--top-k` | `5` | S1 采样候选数量 |
| `--top-p` | `1.0` | S1 nucleus sampling 上限，范围 `(0, 1]` |
| `--temperature` | `1.0` | S1 采样温度，必须大于 0 |
| `--repetition-penalty` | `1.35` | S1 重复惩罚，必须大于 0 |
| `--noise-scale` | `0.5` | S2 随机噪声强度，必须不小于 0 |
| `--speed` | `1.0` | 语速倍率，必须大于 0 |

建议先保持默认值，只调整 `pause-ms`、`seed` 和 `speed`。不同参数会生成不同的
manifest 签名，因此不会错误复用之前的 chunk。

### 4.9 常见问题

### 找不到公共模型

确认当前目录是：

```text
D:\AI-Training\voice-clone\voice-pipeline\voice-pipeline
```

不要从其父目录运行。公共资源必须与第 2 节路径完全一致。

### `voice-pipeline` 命令不存在

确认已激活指定 uv 环境，然后重新注册项目：

```powershell
uv pip install -e . --no-deps
```

### CUDA 不可用或显存不足

先用 CPU 验证目录和输入是否正确：

```powershell
voice-pipeline infer synthesize `
  --model models/speaker_001 `
  --text '测试。' `
  --lang zh `
  --output cpu-test.wav `
  --device cpu
```

CPU 使用 FP32，速度会明显慢于 CUDA；CUDA 默认使用 FP16。

### 提示 manifest 不匹配

说明同名任务的模型、文字、参考条件或参数发生了变化。换一个 `--output` 名称，或在
确认不需要旧结果后传 `--overwrite`。

## 5. AudioClone Studio 模块协议

本节面向 GUI/宿主开发者。普通用户仍可继续使用第 3、4 节的人类 CLI；训练模块可以
完全独立安装和运行，不依赖 Audio Miner 或 AudioClone Studio 前端。

### 5.1 发现模块

安装项目后执行：

```powershell
voice-pipeline module describe --json
```

该命令只向 stdout 输出一个 JSON 对象，不加载 Torch 或模型，也不创建运行目录。返回
`protocol_version: 1`、稳定的 `module_id`、框架能力，以及可由宿主直接渲染的中英日
三语参数标签、默认值和约束。宿主应读取描述符，不能在 GUI 中复制一套参数定义。

### 5.2 创建 job.json

每个任务使用独立目录，目录名必须等于 `job_id`：

```text
<project_root>/jobs/job-001/
└── job.json
```

完整示例：

```json
{
  "protocol_version": 1,
  "job_id": "job-001",
  "project_name": "Acane",
  "project_root": "D:/AudioClone/workspaces/Acane",
  "output_root": "D:/AudioClone/workspaces/Acane/runs",
  "dataset_list": "D:/AudioClone/workspaces/Acane/dataset/data.list",
  "dataset_sha256": "填写 data.list 的 64 位小写 SHA-256",
  "framework": "v2ProPlus",
  "stages": ["preprocess", "s2", "s1", "evaluate"],
  "device": "cuda:0",
  "precision": "fp16",
  "parameters": {
    "preprocess.resume": true,
    "s2.batch_size": 2,
    "s2.target_steps": 800,
    "s2.learning_rate": 0.0001,
    "s1.batch_size": 2,
    "s1.target_optimizer_steps": 500,
    "evaluation.shortlist_size": 3
  },
  "reference": {
    "audio": "D:/AudioClone/workspaces/Acane/reference/reference.wav",
    "text": "今日はいい天気ですね。",
    "language": "ja"
  },
  "job_dir": "D:/AudioClone/workspaces/Acane/jobs/job-001"
}
```

路径字段必须是绝对路径，并位于 `project_root` 内；`job_dir` 必须是 `job.json` 的父目录。
`stages` 只能按 `preprocess → s2 → s1 → evaluate` 的相对顺序填写，不能重复。启用
`evaluate` 时必须提供 `reference`，否则可写 `null`。`parameters` 可省略单个参数以使用
描述符默认值，但不接受描述符未声明的字段。

PowerShell 计算数据清单哈希：

```powershell
(Get-FileHash -Algorithm SHA256 'D:\dataset\data.list').Hash.ToLowerInvariant()
```

### 5.3 运行与读取事件

```powershell
voice-pipeline module run `
  --job D:/AudioClone/workspaces/Acane/jobs/job-001/job.json `
  --events-jsonl
```

stdout 只输出 UTF-8 JSONL，stderr 才是给人看的诊断。每行事件也会同步追加到
`<job_dir>/events.jsonl`，宿主可在进程重启后恢复进度。典型事件：

```json
{"protocol_version":1,"job_id":"job-001","type":"stage_progress","timestamp":"2026-09-17T08:00:00+00:00","stage":"s2","message_key":"training.step","message_args":{"step":200,"total":800},"current":200,"total":800}
{"protocol_version":1,"job_id":"job-001","type":"artifact","timestamp":"2026-09-17T08:20:00+00:00","artifacts":[{"type":"listening_manifest","path":"D:/AudioClone/workspaces/Acane/runs/Acane/evaluation/listening/manifest.json"}]}
```

宿主根据 `type`、`stage`、`message_key` 和 `message_args` 自行本地化显示，不解析英文
日志文本。运行期间可能出现 `job_started`、`stage_started`、`stage_progress`、
`checkpoint`、`stage_completed`、`stage_cache_hit`、`pipeline_completed`、`artifact`、
`job_completed`、`stage_failed`、`job_failed`、`stage_cancelled` 和 `job_cancelled`。

完成评测的标准产物类型为：

| 类型 | 内容 |
|---|---|
| `pipeline_state` | 可恢复阶段状态 |
| `evaluation_report` | 综合评测报告 |
| `listening_manifest` | 候选及三语试听音频索引 |
| `candidate_bundle` | 每个自动入围候选的可推理 ModelBundle |

### 5.4 安全取消

请求取消时创建标记文件，不要直接结束进程：

```powershell
New-Item -ItemType File `
  'D:\AudioClone\workspaces\Acane\jobs\job-001\cancel.requested'
```

pipeline 会在安全边界检测标记，发出 `stage_cancelled` / `job_cancelled` 并保留可恢复
状态。重新运行前由宿主删除这个明确的标记文件；不要删除 job 目录、checkpoint 或
`pipeline-state.json`。

### 5.5 人工晋升

宿主先读取 `listening_manifest`，为每个候选播放中文、日语、英文试听，再让人选择。
界面可以显示 `A/B/C`，但必须把对应的内部 ID 传给模块：

```powershell
voice-pipeline module promote `
  --job D:/AudioClone/workspaces/Acane/jobs/job-001/job.json `
  --selection candidate_A `
  --events-jsonl
```

`A`、`B` 或任意不在试听 manifest 中的 ID 都会被拒绝。模块会再次验证评测报告、候选
bundle、三语试听文件及 SHA-256；然后复制选中 bundle，成功后才清理临时训练产物。
成功事件包含：

```json
{"protocol_version":1,"job_id":"job-001","type":"artifact","timestamp":"2026-09-17T08:30:00+00:00","artifacts":[{"type":"promoted_model","path":"D:/AudioClone/workspaces/Acane/models/Acane"}]}
```

若校验或候选复制失败，不会执行清理，候选试听和原始 checkpoint 会保留。最终模型位于
`<project_root>/models/<project_name>/`，之后可以由第 4 节的独立推理接口加载。

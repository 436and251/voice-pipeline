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

含 `evaluate` 的流水线全部成功后会执行强清理：先验证评分报告、shortlist、所有候选
ModelBundle、中文/日文/英文试听文件及 SHA-256，然后不可逆删除 `preprocess/`、
`training/`（包括原始 S1/S2 恢复 checkpoint）、`evaluation/generated/`、`work/` 和
其他 run 内临时杂项。保留 `pipeline-state.json`、评测报告、shortlist、三语试听音频和
`export/candidates/`。失败、中断或未声明 `evaluate` 时不执行强清理。

因此，首次调试或希望暂时保留原始 checkpoint 时，请使用下面的分阶段命令；确认要完整
跑通并接受成功后清理时，再使用 `voice-pipeline run`。

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

评测器先用官方 Base S1 对全部 S2 checkpoint 做第一阶段筛选，再组合保留的 S2 与
全部 S1 checkpoint。它使用 Faster-Whisper 计算中文/日文 CER、英文 WER 和语言一致性，
使用独立 WavLM speaker encoder 计算跨语言音色相似度，并统计基础韵律。ASR 和 WavLM
只参与离线评测，不会进入最终推理链路。

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

只有 `voice-pipeline run` 声明了 `evaluate`、所有声明阶段完成，并且持久评测产物全部
通过验证后，才执行强清理。此时原始恢复 checkpoint 会被删除，不能再用于续训；每个
评测候选已经提前转换为可独立推理的 ModelBundle，保存在
`runs/<目标人>/export/candidates/`。试听后仍由人决定最终候选：

```powershell
voice-pipeline export --run runs/<目标人> --project-root . --select candidate_A
```

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

输出包括生成音频时长、平均耗时、最快耗时和 RTF。RTF 小于 1 表示平均生成速度快于
实时播放速度。

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

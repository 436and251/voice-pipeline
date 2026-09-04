# Voice Pipeline 中文使用指南

本文说明如何直接使用当前项目完成 GPT-SoVITS v2ProPlus 推理。推理不依赖训练
`runs/` 目录；只需要正式 `ModelBundle`、公共预训练资源和输入文字。

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

## 3. 最简单的文字转语音

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

## 4. 使用 TXT 长文本

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

## 5. 临时覆盖参考音频

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

## 6. 断点恢复与覆盖

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

## 7. 批量推理

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

## 8. 性能测试

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

## 9. 在桌面助手或后台进程中调用

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

## 10. 常用参数

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

## 11. 常见问题

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

# 推理指南

## 单条或长文本

正式推理只接受人工晋升后的 ModelBundle：

```powershell
voice-pipeline infer synthesize `
  --model models/speaker_001 `
  --text '你好，很高兴见到你。' `
  --lang zh `
  --output greeting.wav `
  --device cuda:0
```

`--lang` 必须明确为 `zh`、`ja`、`en` 或 `mixed`。也可用 `--text-file UTF8.txt`；它与
`--text` 二选一。中文默认每段最多 100 个 Unicode 码位，其他语言和 mixed 为 500；
先按强标点、再按弱标点递归切分，最后才硬切。chunk 间默认增加 10 ms 静音。

输出固定归类到 `outputs/<ModelBundle名称>/`。`--output` 必须是安全相对路径；更换整个
根目录使用 `--output-root`。

## 参考条件

默认使用 ModelBundle 内置参考音频、文本和语言。临时覆盖时必须同时提供参考音频与语言：

```powershell
voice-pipeline infer synthesize `
  --model models/speaker_001 `
  --reference D:/references/speaker.wav `
  --reference-text '今日はいい天気ですね。' `
  --reference-lang ja `
  --text 'Hello.' --lang en --output en.wav
```

`--reference-text` 可省略；此时 S1 使用 ref-free 路径，S2 仍使用参考频谱和 speaker
embedding。参考音频要求 3～10 秒。

## 批量与恢复

```powershell
Copy-Item configs/infer.example.yaml configs/infer.local.yaml
voice-pipeline infer batch --config configs/infer.local.yaml
```

单次加载模型并顺序处理 jobs。每个输出都有 `.infer/manifest.json` 和独立 chunk，原命令
重跑会复用有效 chunk；文字、模型、参考条件或参数改变时需换输出名或显式 `--overwrite`。

## 内存接口

```python
from voice_pipeline.inference import InferenceSession, synthesize_text

session = InferenceSession.load("models/speaker_001", "cuda:0")
result = synthesize_text(session, "你好。", "zh", pause_ms=10, seed=0)
```

`result.waveform` 是一维 float32 NumPy 数组，采样率为 32 kHz。后台应用应长期复用
session；当前核心没有 HTTP、鉴权或流式协议，这些可由后续服务适配层包装。

性能检查：

```powershell
voice-pipeline infer benchmark --model models/speaker_001 --text 'Hello.' --lang en
```

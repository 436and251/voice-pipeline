# voice-pipeline 中文使用指南

本文从已有训练数据开始，说明如何完成 GPT-SoVITS v2ProPlus 预处理、S2/S1 增训、自动评测、候选试听、人工晋升和独立推理。

## 1. 先理解三个目录

- **源码根目录**：本仓包含 `pyproject.toml`、`configs/`、`models/` 的目录。
- **训练数据目录**：保存音频与 `data.list`；可以位于源码根目录之外。
- **结果目录**：默认在源码根目录的 `runs/<目标人>`、`models/<目标人>` 与 `outputs/<目标人>`。

`dataset.manifest` 决定输入位置，`experiment.output_root` 决定训练工作目录。它们不是同一个概念。

## 2. 环境与命令

### 2.1 推荐环境

训练强依赖与显卡匹配的 PyTorch/CUDA，以及 GPT-SoVITS 的音频、文本和评测依赖。推荐在已经能够运行 GPT-SoVITS 的 Python 3.12 uv 环境中安装本项目本身：

```powershell
cd D:\path\to\voice-pipeline
uv pip install -e . --no-deps
```

安装后即使激活虚拟环境，也只有注册过 console script 才能直接使用 `voice-pipeline`。验证：

```powershell
voice-pipeline version
```

若不希望注册或命令暂时找不到，使用等价入口：

```powershell
python -m voice_pipeline version
```

本文后续使用较短的 `voice-pipeline` 写法。

### 2.2 验证权重

按根 README 的模型布局放置权重后运行：

```powershell
voice-pipeline models verify --project-root . --profile v2ProPlus
```

此命令应在开始耗时预处理前通过。它不会下载权重。

## 3. 准备 data.list

每行格式固定为：

```text
音频路径|speaker|language|文本
```

示例：

```text
D:/datasets/acane/wavs/0001.wav|Acane|ja|今日はいい天気ですね。
wavs/0002.wav|Acane|zh|你好，很高兴再次听到你的声音。
wavs/0003.wav|Acane|en|It is wonderful to hear your voice again.
```

要求：

- 文件保存为 UTF-8 或 UTF-8-SIG；
- 相对音频路径以 `data.list` 所在目录为基准；
- `language` 使用 `zh`、`ja`、`en` 或 `mixed`；
- 单条数据始终按自身 `language` 处理，日语来源的数据集不会把所有条目强制送入日语 frontend；
- 本流程按单说话人设计，不额外执行多说话人校验；
- 少量坏数据会进入 quarantine；超过配置允许的容错边界才终止预处理。

## 4. 配置训练

复制模板：

```powershell
Copy-Item .\configs\train.example.yaml .\configs\train.local.yaml
Copy-Item .\configs\pipeline.example.yaml .\configs\pipeline.local.yaml
```

最少要修改 `train.local.yaml` 中这些字段：

```yaml
experiment:
  name: Acane
  output_root: runs

device:
  device: cuda:0
  precision: fp16

dataset:
  manifest: D:/datasets/acane/data.list

objective:
  training_languages: [ja]
  target_languages: [zh, ja, en]
  cross_language_preservation: strict
```

`training_languages` 描述主要训练语种，不能覆盖逐条 `language`。`target_languages` 描述希望评测和推理保持的语言范围。

### 4.1 S2/S1 建议起点

模板使用保守的 few-shot 设置：

- S2：batch size 2，约 800 steps，定期保存 checkpoint；
- S1：batch size 2，gradient accumulation 4，约 500 optimizer steps；
- FP16 由 GradScaler 动态调整，不把 scale 人为固定为 1；
- 单元测试只验证能运行，真实训练步数由配置决定。

数据量、音质和显存差异很大；先使用模板，试听候选后再决定是否延长。

### 4.2 评测配置

`evaluation.reference` 与 `speaker_references` 是评测和试听使用的说话人参考，不是训练必填输入。独立 CLI 使用时请指向数据集中的一条干净音频及其原文；AudioClone Studio 会从训练数据中选择有效参考，无需 GUI 再填一次。

```yaml
evaluation:
  reference:
    audio: D:/datasets/acane/wavs/0001.wav
    text: 今日はいい天気ですね。
    language: ja
  models:
    asr: mobiuslabsgmbh/faster-whisper-large-v3-turbo
    speaker: microsoft/wavlm-base-plus-sv
    cache_dir: D:/AI-cache/huggingface/hub
```

评测模型用于 ASR 可懂度和说话人相似度打分，不参与最终语音生成。优先让 `cache_dir` 指向已有 Hugging Face 缓存；不要复制一套模型到项目内。

最后确认 `pipeline.local.yaml` 指向本地训练配置：

```yaml
schema_version: 1
config: configs/train.local.yaml
stages: [preprocess, s2, s1, evaluate]
```

## 5. 一键运行与断点继续

```powershell
voice-pipeline run .\configs\pipeline.local.yaml --project-root .
```

状态写入：

```text
runs/<目标人>/pipeline-state.json
```

再次执行同一命令时：

- `completed` 阶段跳过；
- `failed` 或中断时标记为 `running` 的阶段重试；
- 不会无条件从预处理重头开始；
- 若要从某个训练 checkpoint 续训，设置对应的 `resume_from`。

AudioClone Studio 会把“继续失败阶段”封装成 GUI 按钮；进度和异常进入三个训练页面共用的 Activity 区域。

## 6. 分阶段执行

排错或只重跑某阶段时可单独执行。

### 6.1 预处理

```powershell
voice-pipeline preprocess all -c .\configs\train.local.yaml
```

也可以运行单个 stage：

```powershell
voice-pipeline preprocess stage semantic -c .\configs\train.local.yaml
```

预处理每完成 5 条数据上报一次总进度。坏数据的孤立中间产物会进入 quarantine；成功训练并人工晋升后再清理临时缓存。

### 6.2 训练

```powershell
voice-pipeline train s2 -c .\configs\train.local.yaml --project-root .
voice-pipeline train s1 -c .\configs\train.local.yaml --project-root .
```

或连续执行：

```powershell
voice-pipeline train all -c .\configs\train.local.yaml --project-root .
```

### 6.3 自动评测

```powershell
voice-pipeline evaluate -c .\configs\train.local.yaml --project-root .
```

当前默认从 S2 保留 2 个、从 S1 训练中后段保留 3 个，形成 6 组配对，再综合 ASR、说话人相似度等指标选出最多 3 个 shortlist。评测不是 5×5 的 25 组，也不会自动晋升。

## 7. 候选试听与人工晋升

评测成功后查看：

```text
runs/<目标人>/evaluation/listening/
├── candidate_A/
├── candidate_B/
└── candidate_C/
```

每个候选都包含中文、日文、英文短试听。自动分数负责缩小范围，人耳负责最终判断。

选择后执行：

```powershell
voice-pipeline export `
  --run .\runs\Acane `
  --project-root . `
  --select candidate_A
```

正式模型写入：

```text
models/Acane/
```

候选 checkpoint 转换后的推理 bundle 明显小于训练 checkpoint，主要因为去掉了优化器、GradScaler 等续训状态。评测阶段会暂存多份完整候选，所以磁盘会短时增长；人工晋升前不能删除这些候选。

晋升成功后可清理原始训练 checkpoint、预处理缓存和非必要临时文件，但保留：

- 正式模型；
- evaluation 报告；
- shortlist 与试听音频；
- 用于人工复核的候选 bundle。

## 8. 独立推理

推理只依赖正式模型、文本和必要参考信息，不依赖原始 `data.list` 或完整训练目录。

### 8.1 输入文本

```powershell
voice-pipeline infer synthesize `
  --model .\models\Acane `
  --text "ふーん、面白い。乗ってみませんか？" `
  --lang ja `
  --output trial.wav `
  --output-root .\outputs\Acane
```

### 8.2 输入文本文件

```powershell
voice-pipeline infer synthesize `
  --model .\models\Acane `
  --text-file D:\texts\trial.txt `
  --lang ja `
  --output trial.wav `
  --output-root .\outputs\Acane
```

`--text` 与 `--text-file` 必须二选一，不能同时提供。

### 8.3 参考音频

当正式 bundle 没有可直接使用的默认参考，或希望指定语气时：

```powershell
voice-pipeline infer synthesize `
  --model .\models\Acane `
  --text "你好。" `
  --lang zh `
  --reference D:\refs\acane.wav `
  --reference-text "今日はいい天気ですね。" `
  --reference-lang ja `
  --output hello.wav `
  --output-root .\outputs\Acane
```

### 8.4 长文本与标点

推理按语言明确切分：中文默认上限 100，日文、英文与 `mixed` 默认上限 500。先按句号等强边界递归切，再按逗号、分号等弱边界切；两层仍超长时才硬切。标点会保留停顿，不应造成后半句丢失。

chunk 之间默认额外插入 10 ms 静音：

```powershell
--pause-ms 10
```

可用 `--max-chars` 覆盖单次限制。该参数只控制离线分段，不等于流式大模型对接；真正低时延服务需要增量文本缓冲、流式推理和音频输出协议。

推理结果位于 `--output-root`，命令成功时也会返回实际文件路径。应用播放后应及时释放 WAV 文件句柄，避免 Windows 下无法删除。

## 9. AudioClone Studio 模块模式

AudioMiner 绑定本仓后会调用模块协议，而不是把两个虚拟环境合并。可先手工检查：

```powershell
voice-pipeline module describe --json
```

GUI 中：

1. 素材挖掘生成位于外部工作区的训练数据；
2. 训练配置选择同一目标人和训练数据；
3. 后台进程逐阶段上报进度；
4. 失败后可继续失败阶段；
5. 候选试听页面保持 A/B/C 三栏；
6. 晋升后的模型会出现在推理实验的本地历史列表。

### 9.1 jobs 生命周期

`jobs/` 只保存模块通信状态和事件，不是永久档案。AudioMiner 启动和任务结束时执行清理：

- 未完成任务保留；
- 等待人工晋升的评测任务保留；
- 每个目标人只保留最近一次、且年龄不超过 48 小时的失败任务；
- 推理完成、已经晋升、取消以及其他已终结任务删除；
- 缺少身份或无法确认安全边界的目录保留并告警。

Activity 只显示一次汇总，不刷出逐文件删除日志。独立 CLI 不使用 `jobs/`。

## 10. 存储位置与清理边界

| 目录 | 内容 | 清理策略 |
|---|---|---|
| `runs/<目标人>/preprocess` | 正式预处理结果与训练输入 | 晋升成功前保留 |
| `runs/<目标人>/checkpoints` | S1/S2 续训 checkpoint | 晋升成功前保留 |
| `runs/<目标人>/evaluation` | 分数、shortlist、试听和候选 bundle | 保留人工复核所需内容 |
| `models/<目标人>` | 正式晋升模型 | 永久保留，除非用户明确删除 |
| `outputs/<目标人>` | 推理 WAV 与 manifest | 用户管理 |
| `jobs/` | GUI 模块临时任务 | 按生命周期自动清理 |
| Hugging Face cache | ASR/说话人评测模型 | 复用共享缓存，不复制到每个任务 |

源码仓的 Git 忽略规则应覆盖 `runs/`、`outputs/`、`jobs/`、本地模型、虚拟环境、IDE 状态和本地私有 `docs/`。

## 11. 常见问题

### `voice-pipeline` 无法识别

激活环境只会修改 PATH，不会自动注册本项目命令。执行：

```powershell
uv pip install -e . --no-deps
```

或使用 `python -m voice_pipeline`。

### 评测尝试下载已有 Whisper

检查 `evaluation.models.cache_dir` 是否指向真实的 Hugging Face hub 缓存，而不是另一个空目录。AudioClone Studio 应复用素材挖掘配置的缓存。Windows 非开发者模式下创建 symlink 可能触发 `WinError 1314`；正确复用已有 snapshot 可避免无意义的重复物化。

### 重启后是否从头训练

不会。相同 `runs/<目标人>/pipeline-state.json` 下，完成阶段跳过，失败阶段重试。不要删除或改名该目录。

### 评测为什么慢

每个 S1/S2 配对都要完成多语种推理、ASR 和说话人相似度计算。默认 2×3 共 6 组，是质量、磁盘和耗时之间的折中；它不是实时推理速度测试。

### 为什么推理还不是实时流式

当前命令会递归切分长文本并顺序合成，适合离线 CLI 和 GUI 试验。与持续输出的大模型低时延对接需要独立服务层；仅放宽 `max-chars` 不能替代流式架构。

## 12. 验证清单

部署后依次执行：

```powershell
voice-pipeline version
voice-pipeline models verify --project-root . --profile v2ProPlus
voice-pipeline module describe --json
```

开发验证：

```powershell
pytest -q
```

遇到问题时请保留：命令、对应 `train.local.yaml`（隐去隐私路径亦可）、`pipeline-state.json`、错误阶段末尾日志，以及显卡/驱动/PyTorch 版本。

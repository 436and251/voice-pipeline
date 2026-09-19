# voice-pipeline

`voice-pipeline` 是面向单说话人的 GPT-SoVITS v2ProPlus 增训与推理模块。它从官方格式的 `data.list` 开始，完成：

`预处理 → S2/S1 训练 → 自动评测 → A/B/C 候选试听 → 人工晋升 → 独立推理`

它既可以独立通过 CLI 使用，也可以作为训练模块接入 AudioMiner，组成 AudioClone Studio。素材采集并不是本仓职责；训练数据可以位于项目目录之外。

## 当前边界

- 训练框架：GPT-SoVITS v2ProPlus。
- 数据：单说话人；每条记录仍按自身 `language` 使用对应 frontend。
- 训练：S2 全量微调，S1 短程微调，支持 FP16 与断点续跑。
- 评测：自动打分只生成 shortlist，不自动决定最终模型。
- 推理：与训练解耦，晋升后的模型目录可独立使用。
- 实时服务：当前 CLI 是离线合成；流式 HTTP/后台服务不在本仓当前范围内。

## 目录约定

```text
voice-pipeline/
├── configs/                         # 配置模板
├── models/
│   ├── pretrained/v2proplus/        # 训练与推理基础权重
│   ├── evaluators/                  # 可选的评测模型缓存
│   └── <目标人>/                    # 人工晋升后的正式模型
├── runs/<目标人>/                   # 训练、评测、候选试听
├── outputs/<目标人>/                # 推理结果
└── jobs/                            # AudioClone Studio 临时任务；不进入 Git
```

`runs/`、`outputs/`、`jobs/`、模型权重和本地 `docs/` 都不应提交到源码仓库。

## 快速开始

推荐复用已安装 PyTorch/CUDA 与 GPT-SoVITS 运行依赖的 Python 3.12 环境，再注册本项目命令：

```powershell
cd D:\path\to\voice-pipeline
uv pip install -e . --no-deps
voice-pipeline version
```

若不注册命令，所有示例也可把 `voice-pipeline` 替换为：

```powershell
python -m voice_pipeline
```

先复制配置模板：

```powershell
Copy-Item .\configs\train.example.yaml .\configs\train.local.yaml
Copy-Item .\configs\pipeline.example.yaml .\configs\pipeline.local.yaml
```

修改 `train.local.yaml` 中的目标人、`dataset.manifest`、设备、训练步数与评测配置，并让 `pipeline.local.yaml` 指向它。随后：

```powershell
voice-pipeline models verify --project-root . --profile v2ProPlus
voice-pipeline run .\configs\pipeline.local.yaml --project-root .
```

评测结束后试听：

```text
runs/<目标人>/evaluation/listening/
```

人工选择 A、B 或 C，再晋升，例如：

```powershell
voice-pipeline export --run .\runs\<目标人> --project-root . --select candidate_A
```

正式模型位于 `models/<目标人>/`，可以直接推理：

```powershell
voice-pipeline infer synthesize `
  --model .\models\<目标人> `
  --text "你好，很高兴再次听到你的声音。" `
  --lang zh `
  --output hello.wav `
  --output-root .\outputs\<目标人>
```

## AudioClone Studio 接入

AudioMiner 与本仓使用两个独立虚拟环境，避免 PySide6、Torch、CUDA 等依赖互相污染。AudioMiner 设置页只需要绑定本仓根目录并通过连接检测；之后 GUI 会以模块协议启动训练、读取进度、恢复失败阶段、展示候选并执行人工晋升。

AudioMiner 作为宿主时，还会按生命周期清理 `jobs/`：

- 保留未完成任务；
- 每个目标人仅保留最近一次且不超过 48 小时的失败任务；
- 保留等待人工晋升的评测任务；
- 删除其余已终结任务；
- 无法安全识别的目录不自动删除，并在 Activity 中汇总提示。

独立 CLI 不创建 `jobs/`，其断点状态保存在 `runs/<目标人>/pipeline-state.json`。

## 模型布局

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

评测模型的缓存目录由 `evaluation.models.cache_dir` 指定。接入 AudioMiner 时，优先复用素材挖掘已经配置的 Hugging Face 缓存，避免重复下载和 Windows 符号链接权限问题。

## 文档

完整的数据格式、配置、分阶段命令、恢复、评测、晋升、推理与存储说明见 [README_使用指南.md](README_使用指南.md)。第三方来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 常用检查

```powershell
voice-pipeline --help
voice-pipeline module describe --json
voice-pipeline models verify --project-root . --profile v2ProPlus
pytest -q
```

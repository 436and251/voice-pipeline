# 训练指南

## 1. 输入

`data.list` 使用官方四字段 UTF-8 格式：

```text
D:/dataset/clips/001.wav|speaker_001|ja|今日はいい天気ですね。
```

语言必须逐条填写 `zh`、`ja`、`en` 或 `mixed`。相对音频路径从 `data.list` 所在目录
解析；项目假设数据来自一个说话人，不做多说话人校验。

## 2. 配置

```powershell
Copy-Item configs/train.example.yaml configs/train.local.yaml
```

至少修改：

- `experiment.name`；
- `dataset.manifest`；
- `evaluation.reference` 和 `speaker_references`；
- 四个 `evaluation.suites`；
- evaluator 的本地缓存/模型路径。

日语 few-shot 基线可从示例的 S2 800 steps、S1 500 optimizer steps、batch size 2 开始。
这不是自动最优值；最终看跨语言评测和人耳试听。

## 3. 完整运行

```powershell
Copy-Item configs/pipeline.example.yaml configs/pipeline.local.yaml
voice-pipeline run configs/pipeline.local.yaml --project-root .
```

也可分阶段排查：

```powershell
voice-pipeline preprocess all -c configs/train.local.yaml
voice-pipeline train s2 -c configs/train.local.yaml --project-root .
voice-pipeline train s1 -c configs/train.local.yaml --project-root .
voice-pipeline evaluate -c configs/train.local.yaml --project-root .
```

预处理允许的坏数据数量为 `min(5, ceil(有效输入行数 × 20%))`；未超过上限的坏样本写入
`quarantine.jsonl`，其余样本继续。

## 4. 恢复

`run` 只恢复阶段状态，不猜 checkpoint。训练中断后，在训练 YAML 明确填写：

```yaml
s2:
  resume_from: runs/speaker_001/training/s2/checkpoints/step-00000800.pt
s1:
  resume_from: runs/speaker_001/training/s1/checkpoints/step-00000500.pt
```

然后重跑原命令。不要交叉使用 S1/S2 checkpoint，也不要用导出的推理权重恢复训练。

## 5. 人工选择

自动评测会把候选和三语试听写到：

```text
runs/<目标人>/evaluation/
runs/<目标人>/export/candidates/
```

试听后只晋升一个：

```powershell
voice-pipeline export --run runs/<目标人> --project-root . --select candidate_A
```

晋升成功后才强清理原始 checkpoint。若还需要续训，不要提前执行人工晋升命令。

完整字段和产物说明见 [中文使用指南](../README_使用指南.md)及
[评测说明](evaluation.md)。

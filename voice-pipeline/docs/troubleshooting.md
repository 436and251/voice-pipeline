# 故障排查

## 命令不存在

确认激活指定 uv 环境并从项目根目录注册：

```powershell
& 'D:\Python_program_codes\TTS-Inference\.venv-gpt-sovits\Scripts\Activate.ps1'
uv pip install -e . --no-deps
voice-pipeline version
```

开发态也可临时设置 `$env:PYTHONPATH='src'` 后使用
`python -m voice_pipeline.cli.main`。

## 模型缺失或路径错误

```powershell
voice-pipeline models verify --project-root . --profile v2ProPlus
```

必须看到 S1、S2G、S2D、BERT、HuBERT、speaker、G2PW、NLTK 和 langdetect 九组 `OK`。
模型目录的精确结构见 [中文使用指南](../README_使用指南.md)。

## 预处理 quarantine

查看：

```powershell
Get-Content runs/<目标人>/preprocess/quarantine.jsonl
```

常见原因是音频不存在/无法解码、文本语言填写错误、音频不在训练时长范围、特征非有限或
S1/S2 对齐失败。修正原始 `data.list` 或音频后重跑；不要手改生成的索引。

## CUDA OOM

- 关闭占用显存的其他程序；
- 先把 S1/S2 `batch_size` 从 2 降为 1；
- 不要修改 S1 固定的 `gradient_accumulation: 4`；
- 评测仍不足时可改 `device: cpu`、`precision: fp32`，但会明显变慢。

FP16 第一次梯度溢出可能只是动态 GradScaler 探测范围；只要后续 scale 自动下降且 loss
恢复有限，不要手工把 scale 固定为 1。

## 中断恢复失败

`pipeline-state.json` 只决定重跑哪个阶段；S1/S2 模型恢复必须在 YAML 中显式配置正确的
`resume_from`。checkpoint 必须属于同一阶段和 profile，且目标 step 不小于已有 step。

## 没有评测候选

查看 `evaluation/stage1-report.md` 和 `evaluation/report.md`，判断失败来自音色相似度、
CER/WER 还是语言一致性。优先检查训练数据和测试句，不要一次性放宽全部硬阈值。

## 人工选择后清理失败

这种情况通常表示报告、候选包或试听音频被移动/修改。正式晋升模型会保留，原始训练资源
不会因验证失败被删除。恢复缺失文件或重新评测后，再用同一候选和必要的 `--overwrite`
重新执行晋升命令。

## 推理 manifest 不匹配

说明模型、文本、参考条件或推理参数发生变化。换一个 `--output`，或确认不需要旧输出后
加 `--overwrite`。它只重建对应 WAV 和 `.infer` 目录，不影响模型或训练结果。

# 自动评测与人工选择

自动评测的职责是从训练产生的多个 S1/S2 checkpoint 中筛出少量可靠候选；最终模型仍由
人耳裁决。Faster-Whisper 与 WavLM 只在这个离线阶段使用，不会增加正式推理的显存、延迟
或部署依赖。

## 运行前配置

在训练配置的 `evaluation` 段填写同一说话人的参考音频、四套测试句和 evaluator 模型：

```yaml
evaluation:
  enabled: true
  reference:
    audio: D:/dataset/reference.wav
    text: 今日はいい天気ですね。
    language: ja
  speaker_references:
    - D:/dataset/reference.wav
  suites:
    zh: configs/eval/zh.txt
    ja: configs/eval/ja.txt
    en: configs/eval/en.txt
    mixed: configs/eval/mixed.txt
  models:
    asr: mobiuslabsgmbh/faster-whisper-large-v3-turbo
    speaker: microsoft/wavlm-base-plus-sv
    cache_dir: D:/AI-Training/AI-cache/huggingface/hub
  pairing:
    s2_keep: 2
    shortlist_size: 3
```

每个 suite 文件每行一句，不能有空行；语言由 suite 所在字段明确指定。建议使用训练集中
没有出现过、长度适中的自然句。`mixed.txt` 使用确实需要切换语言的句子。

`reference.audio` 和 `speaker_references` 均可使用项目目录外的绝对路径。评测开始时会把
推理 reference 原子快照到 `runs/<目标人>/evaluation/reference/<SHA256>.wav`，之后生成的
工作包、shortlist 和候选包都引用该不可变副本；`speaker_references` 仍按原路径读取，
用于计算说话人 centroid。外部 reference 内容变化时会创建新的哈希文件，不会覆盖已有
候选所引用的旧快照。

模型字段既可写 Hugging Face model ID，也可写已下载模型的绝对目录。Windows 未启用
Developer Mode 时，推荐把 WavLM 下载到普通目录并将 `speaker` 直接设为该目录，例如：

```yaml
speaker: D:/AI-Training/AI-cache/huggingface/hub/wavlm-base-plus-sv
```

通过 AudioClone Studio 启动时，训练模块会复用 AudioMiner“模型存储根目录”下的
`huggingface/hub`，不会另建项目内缓存；独立 CLI 未设置 `HF_HOME` 时仍使用训练配置中的
`evaluation.models.cache_dir`。

评测时 ASR、WavLM 和候选推理会使用配置中的同一设备。建议 CUDA 显存至少 8 GB，并在
运行前关闭其他占用显存的程序；显存不足时可改为 `device: cpu`、`precision: fp32`，但
完整评测会明显变慢。评测按候选顺序执行，不会并行常驻多组 S1/S2 权重。

## 执行与产物

```powershell
voice-pipeline evaluate -c configs/train.local.yaml --project-root .
```

流程分两段：先用官方 Base S1 对全部 S2 checkpoint 做初筛，保留 `s2_keep` 个；再评测
全部 S1 与这些 S2 的组合。通过硬约束的前 `shortlist_size` 个组合会以匿名 ID 导出为
CandidateBundle，并分别生成中文、日文、英文试听音频。

Base S2 也参加第一阶段。设训练产生 `N2` 个 S2 checkpoint、`N1` 个 S1 checkpoint，
保留 `K = s2_keep` 个 S2，实际生成的唯一评测工作包数量为
`(1 + N2) + K × N1`。例如默认 4 个 S2、5 个 S1、`s2_keep: 2` 会产生 15 组
（`5 + 2 × 5 = 15`）`evaluation/work/<pair>/bundle`；这 15 组用于机器评测，不是 15 个
人耳候选。只有排名前
`shortlist_size: 3` 的组合才会进入 `export/candidates/` 和 `evaluation/listening/`。

重点查看：

- `evaluation/stage1-report.md`：S2 初筛的人类可读报告。
- `evaluation/report.md`：最终组合的分项得分与淘汰原因。
- `evaluation/results.json`：供后续 GUI 使用的结构化结果。
- `evaluation/shortlist.yaml`：匿名候选与真实 S1/S2 checkpoint 的绑定。
- `evaluation/listening/candidate_*/{zh,ja,en}.wav`：人工试听样本。

评测中断时可重新运行；已生成且校验一致的中间语音会复用。评测成功后会清理
`evaluation/work/`，正式报告、CandidateBundle 和试听音频会保留。

磁盘峰值可能出现在最终候选已经导出、`evaluation/work/` 尚未删除的瞬间。Acane 基线
实测中，4 个 S2 完整恢复 checkpoint 占 7.21 GiB，5 个 S1 占 4.34 GiB，15 组临时
推理权重占 4.59 GiB，评测 WAV 仅占 0.13 GiB。当前 800/500 step 基线至少预留
20 GiB，建议预留 25 GiB；配置的 Hugging Face `cache_dir` 位于 run 目录之外，需要
另行计算。失败或中断时不会自动删除 `work/`，这是为了保留可恢复现场。

## 指标和阈值

- `min_speaker_similarity`：WavLM cosine similarity 下限，分别约束中/日/英，防止平均值
  掩盖某个目标语言的音色崩坏。
- `max_cer`：中文、日文字符错误率上限。
- `max_wer`：英文词错误率上限。
- `min_language_consistency`：Whisper 识别为目标语言的平均置信度下限。
- `ranking`：通过所有硬约束后才计算的排序权重，四项必须非负且总和为 1。

示例配置中的阈值是初始基线，不是跨数据集的绝对标准。第一次真实训练后应结合报告和
试听校准；若没有候选通过，应先定位失败的是音色、发音还是语言漂移，不要直接整体放宽。

## 最终选择

完整试听所有匿名候选后，只晋升人工选中的一个：

```powershell
voice-pipeline export --run runs/<目标人> --project-root . --select candidate_A
```

该命令把选中的 CandidateBundle 原子晋升为 `models/<目标人>/` 下的正式 ModelBundle，
后续桌面应用、批处理或服务端推理只需要这个正式模型目录。晋升成功后才会强清理
`preprocess/`、原始 S1/S2 checkpoint 和可重建评测缓存；未选择、晋升失败或清理验证
失败时不删除原始训练资源。

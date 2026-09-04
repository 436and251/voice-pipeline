# GPT-SoVITS v2ProPlus 自动评测设计

日期：2026-09-05
范围：Task 19–21（评测集、客观指标、分层筛选、匿名试听 shortlist）

## 1. 目标与边界

本模块在训练结束后，对 S1/S2 checkpoint 组合进行跨语言客观评测，自动筛出 3～5 个候选供人工试听。它不参与反向传播，也不得自动晋升正式模型。

完整边界为：

```text
training checkpoints
→ deterministic evaluation audio
→ objective metrics
→ hard constraints
→ ranking
→ anonymous shortlist
→ CandidateBundle conversion
→ ZH/JA/EN listening previews
→ human final choice
→ existing export --select promotion
```

ASR 和 speaker encoder 都只是评测器。正常 TTS 推理仍是文字到语音，不加载评测模型，不因评测依赖缺失而受影响。

官方 GPT-SoVITS 的 ASR 主要用于制作训练标注，S2 `evaluate()` 主要输出验证重建结果；本设计新增的跨语言 checkpoint 排名不冒充官方既有功能。

## 2. 核心原则

1. 音色保持优先，尤其关注中文、日文、英文中的最低说话人相似度 `WorstSim`。
2. S2 是主要音色适配层，先筛 S2；S1 是语义自回归层，再在少量 S2 上比较 S1。
3. 先执行硬门槛，再计算综合排名，不能用平均分掩盖单语种严重退化。
4. 所有原始指标、失败原因、配置和模型来源都必须可追踪。
5. shortlist 只缩小人工试听范围，不替代人耳最终裁决。
6. 评测运行可恢复；输入未变化时复用已完成样本，输入变化时只失效受影响结果。

## 3. 评测模型

### 3.1 ASR

统一使用 `mobiuslabsgmbh/faster-whisper-large-v3-turbo`：

- 当前机器已有完整 Hugging Face 缓存，可离线复用；
- 中文、日文计算规范化 CER；
- 英文计算规范化 WER；
- 单语样本记录识别语种和置信信息；
- Mixed 样本评估转写内容与各语种片段保留情况，不要求输出单一语种标签。

自动评测不调用官方 ASR 脚本的文件夹式 CLI，而是通过本项目中的窄适配器直接调用 `faster-whisper`，返回结构化转写结果。这样不会依赖官方源码目录，也不会产生中间 `.list` 文件。

### 3.2 Speaker evaluator

使用 `microsoft/wavlm-base-plus-sv`，通过现有 `torch` 与 `transformers` 加载。该模型与训练使用的 ERes2NetV2 架构独立，避免同源评测偏差。

参考音频先分别提取 embedding，再归一化并求 centroid；生成音频 embedding 与 centroid 计算余弦相似度。输出每条样本相似度、`Sim_ZH`、`Sim_JA`、`Sim_EN` 和三者最小值 `WorstSim`。

评测配置可显式提供多条 `speaker_references`。未提供时只使用 `evaluation.reference.audio`，不会擅自从训练集抽样。

### 3.3 Prosody

使用现有 `librosa`/NumPy 本地计算：

- F0 median、range、variance；
- voiced/unvoiced ratio；
- speaking rate；
- pause ratio；
- RMS energy 统计；
- 静音占比和削波占比。

第一版不做逐帧 F0 MSE，也不引入额外韵律神经网络。原始统计用于识别明显异常，韵律只占较低排序权重。

## 4. Evaluation Suite

固定文件位于：

```text
configs/eval/zh.txt
configs/eval/ja.txt
configs/eval/en.txt
configs/eval/mixed.txt
```

每个非空、非 `#` 行是一条测试句。文件所属语种就是明确传给推理前端的 language；不会根据训练集来源猜测语言。Mixed 明确传 `mixed`。

覆盖范围：

- ZH：普通陈述、长句、数字/英文混合、易错音；
- JA：普通陈述、长句、情绪句、训练域外句；
- EN：普通陈述、重音、辅音连缀、长句；
- Mixed：ZH+EN、JA+EN、ZH+JA。

每条样本拥有稳定 ID，生成使用固定 seed 和固定推理参数。manifest 记录文本、语言、seed、推理参数、checkpoint 哈希、参考音频哈希和输出 WAV 哈希。

## 5. 分层 checkpoint 配对

### 5.1 第一阶段：筛选 S2

候选为：

```text
Base S1 × (Base S2 + every valid trained S2 checkpoint)
```

所有 checkpoint 都按内部 envelope 验证，通过嵌入的 optimizer step 确定顺序，不只依赖文件名。第一阶段主要依据跨语言 speaker similarity，随后考虑发音和异常音频指标；仅保留通过硬门槛的前 `s2_keep` 个 S2。

### 5.2 第二阶段：比较 S1

候选为：

```text
(Base S1 + every valid trained S1 checkpoint) × retained S2
```

第一阶段已经生成的 `Base S1 + retained S2` 结果直接复用。不会对所有 S1×所有 S2 做完整笛卡尔积。

Base 权重可以进入最终 shortlist：如果微调模型没有稳定优于 base 组合，评测结果必须如实保留该事实。

## 6. 指标模型

统一接口：

```text
EvaluatorMetric.evaluate(sample) -> MetricResult
```

`MetricResult` 至少包含 `name`、`value`、`details`、`available` 和失败信息。单条样本输入包含生成 WAV、目标文本、明确语言和 reference centroid。

### 6.1 发音正确性

文本比较前执行可复现的语言特定规范化：

- Unicode 规范化；
- 统一空白与标点处理；
- 中文和日文按字符计算编辑距离；
- 英文小写化并按词计算编辑距离；
- 同时保留原始 ASR 文本和规范化文本，便于人工复核。

### 6.2 语言一致性

- 单语样本：记录 ASR 的目标语言一致性；
- Mixed 样本：结合目标文本与转写文本的脚本覆盖和内容编辑距离检查明显语言丢失；
- Mixed 不因 ASR 只能给出一个主语言标签而被错误淘汰。

### 6.3 聚合

每个指标先保留逐句结果，再聚合为逐语言结果。speaker similarity 使用最低语言值作为关键保护指标；CER/WER 分语言保留，不合成一个会掩盖短板的错误率。

## 7. 硬门槛与排序

永远拒绝：

- WAV 无法读取、为空或包含非有限数；
- ASR 完全无输出；
- 推理失败或评测模型失败；
- 明显全静音或严重削波；
- 任一配置的逐语言硬门槛不满足。

相似度、CER/WER 等数值门槛全部由配置声明。示例值只是首次运行起点，不写死在指标实现中；真实首轮报告用于后续校准。

通过硬门槛后计算可解释的综合分。默认优先级为 speaker similarity（并强调 `WorstSim`），其次 pronunciation，再其次 language consistency 与 basic prosody。每个分项、归一化方式和权重都写入报告。无论综合分多高，硬门槛失败的候选都不能进入 shortlist。

若没有足够候选通过，不放宽门槛、不伪造 shortlist；保留报告并明确提示需要人工调整配置或检查训练结果。

## 8. 产物与匿名化

运行目录：

```text
<run>/evaluation/
├── manifest.json
├── results.json
├── report.md
├── shortlist.yaml
├── listening/
│   ├── candidate_A/
│   │   ├── zh.wav
│   │   ├── ja.wav
│   │   └── en.wav
│   ├── candidate_B/
│   └── manifest.json
├── generated/              # 正式评测音频与逐句记录
└── work/                   # 可恢复的临时转换区
```

`results.json` 和 `report.md` 保存完整 checkpoint 映射与原始指标。试听目录只暴露 `candidate_A` 一类稳定 ID，不在文件名中暴露 step。

最终 shortlist 写入既有严格 schema，每个候选显式绑定一份 S1 与一份 S2。生成 shortlist 后调用既有导出能力，把所有 shortlist 项转换成 CandidateBundle；然后使用这些永久 Bundle 分别重新生成 ZH/JA/EN 短试听音频，证明实际导出物可推理。

人耳选定后继续使用既有命令：

```powershell
voice-pipeline export --run <run> --project-root . --select candidate_A
```

评测模块不增加“自动选择并晋升最佳模型”的接口。

## 9. 配置与 CLI

扩展现有 `evaluation` 配置，至少包含：

```yaml
evaluation:
  enabled: true
  reference:
    audio: data/reference.wav
    text: "今日はいい天気ですね。"
    language: ja
  speaker_references:
    - data/reference.wav
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
  constraints:
    # 逐语言 similarity / CER / WER 与音频安全门槛
  ranking:
    # speaker 权重最高；其余权重显式可见
```

主入口：

```powershell
voice-pipeline evaluate --config configs/train.yaml --project-root .
```

不新增容易与人工最终选择混淆的 evaluator `select` 命令。人工晋升仍只由 `export --select` 完成。

模型加载必须 lazy：只有 `evaluate` 命令导入 `faster-whisper`、`transformers` 和音频分析依赖。评测资源缺失时评测明确失败，但训练和独立推理保持可用。

## 10. 恢复、失败与清理

- 对配置、suite、reference、checkpoint 和推理参数计算指纹；指纹一致才复用结果。
- 单个候选或样本失败时记录失败原因并继续评测其他候选，最终由硬门槛淘汰失败候选。
- ASR 或 speaker evaluator 整体不可用时不降级成伪完整评分，也不生成 shortlist。
- 写入 JSON、YAML、WAV 和 Bundle 时沿用原子发布策略。
- 成功后删除临时转换 Bundle、未入选的临时试听产物和孤立文件；保留正式报告、评测结果、shortlist、shortlist CandidateBundle 及其试听 WAV。
- 失败时保留可恢复的有效缓存和诊断信息，下一次运行只补未完成部分。

## 11. 测试策略

### 单元测试

- suite 解析、稳定 ID 与非法输入；
- 中/日 CER、英文 WER 和 Mixed 规范化；
- speaker centroid 与余弦聚合；
- prosody/静音/削波边界；
- hard constraints、权重排序和稳定匿名 ID；
- 分层配对不会退化为完整笛卡尔积；
- 缺失评测依赖不会影响训练/推理 import。

### 集成测试

- 使用写死的小型 WAV/假 evaluator 完成评测 runner、恢复和 shortlist 生成；
- 验证每个 shortlist 项可导出为合法 CandidateBundle；
- 验证每个候选产生 ZH/JA/EN 试听文件；
- 验证无合格候选时不生成伪 shortlist。

### 本机真实资源 smoke

- 从指定 Hugging Face cache 离线加载现有 Faster-Whisper；
- 下载并缓存缺失的 WavLM speaker evaluator 后验证真实 embedding；
- 用真实 v2ProPlus 权重生成少量音频并跑完整指标闭环；
- 最后运行全量测试并清理测试缓存、`__pycache__` 和额外测试权重。

## 12. 实施顺序

1. Task19：suite、候选发现、临时 Bundle、确定性生成与 manifest/resume。
2. Task20：ASR、独立 speaker similarity、language consistency、prosody 指标适配器。
3. Task21：硬门槛、综合排序、匿名 shortlist、全部候选导出与试听生成。
4. 更新示例配置、中文使用指南和架构 README。
5. 完成单元、集成、真实资源 smoke 与全量回归。

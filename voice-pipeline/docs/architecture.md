# 架构说明

## 边界

本项目独立于官方 GPT-SoVITS 仓库运行，只支持 v2ProPlus。它复用并固定官方算法行为，
但重新提供配置、状态、训练、评测、ModelBundle 和推理接口。数据切片、ASR 建集、GUI、
HTTP 服务和流式协议不在当前版本内。

```text
data.list
  → multilingual frontend / BERT
  → 32 kHz WAV / HuBERT / SV / semantic
  → S2 GAN speaker adaptation
  → S1 Text2Semantic adaptation
  → ASR + WavLM cross-language evaluation
  → anonymous CandidateBundles + ZH/JA/EN previews
  → human selection
  → promoted ModelBundle
  → standalone inference
```

## 组件职责

- Text/BERT：每条记录按自己的 `language` 进入中文、日文、英文或 mixed frontend；只有
  中文使用 BERT 字符特征，最终都对齐到 phone 序列。
- HuBERT：提取与说话内容相关的 SSL 表征，供 semantic 和 S2 使用。
- SV：官方 ERes2NetV2 生成 20,480 维 speaker embedding，用于 S2 条件输入。
- Semantic：使用官方 Base S2G quantizer 生成离散语义 token。
- S2：v2ProPlus GAN Generator + Discriminator 全量增训，是主要音色适配层。
- S1：Text2SemanticDecoder 短增训，适度学习说话人的节奏和语言实现。
- Evaluator：Faster-Whisper 衡量发音/语言一致性，独立 WavLM 衡量跨语言声纹；只用于
  离线选候选，不参与正式推理。
- ModelBundle：训练与推理的唯一正式边界，包含 S1/S2 推理权重、参考条件和可审计元数据。

## 状态与清理

`voice-pipeline run` 只按 `preprocess → s2 → s1 → evaluate` 的声明子序列执行。状态原子
写入 `runs/<目标人>/pipeline-state.json`；完成阶段跳过，失败或中断阶段重试。S1/S2
checkpoint 恢复只认训练 YAML 的显式 `resume_from`。

自动评测后保留全部 checkpoint，等待人工试听。只有显式 `export --select` 成功晋升最终
ModelBundle 后，才验证持久评测产物并强清理 run 内可重建内容。

## 推理解耦

`InferenceSession` 只依赖正式 ModelBundle 和公共模型，不依赖 `data.list`、训练状态或
evaluator。CLI、批处理、桌面助手及未来 HTTP 服务可以复用同一个内存接口。

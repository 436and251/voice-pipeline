# v2ProPlus 本地兼容性报告

验证日期：2026-09-07
代码版本：`3e994cb1250c259a588d622f90b06597bc763bdc`

## 结论

当前框架与本机 GPT-SoVITS v2ProPlus 正式权重兼容。模型仓库、中文/日文/英文前端、
Text/BERT、HuBERT、speaker encoder、semantic 提取、S1、S2 GAN 训练恢复和
reference-conditioned 推理均已使用真实资源通过验证。

本报告证明工程和权重兼容，不代表特定 few-shot 数据集已经达到目标音色质量；训练后的
跨语言质量仍以自动评测候选和人工三语试听为最终标准。

## 环境

```text
OS: Windows 11 10.0.26200
Python: 3.12.13
PyTorch: 2.7.0+cu128
CUDA runtime: 12.8
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
VRAM: 8,585,216,000 bytes
环境: D:/Python_program_codes/TTS-Inference/.venv-gpt-sovits
```

## 核心权重

| 权重 | 字节数 | SHA-256 |
|---|---:|---|
| `s1/s1v3.ckpt` | 155,284,856 | `87133414860EA14FF6620C483A3DB5ED07B44BE42E2C3FCDAD65523A729A745A` |
| `s2/s2Gv2ProPlus.pth` | 200,125,741 | `D42A22BBBF65FB2BBDD45AD6A66841156977DB45C7AABE0A6992FF378D9C7D3B` |
| `s2/s2Dv2ProPlus.pth` | 126,432,639 | `635CD84BF6F7F9B8D41C88C7106F81D782C794C61F931845214EA037B0C5BEF2` |

## 验证结果

### 模型仓库

`models verify` 检查 9 组资源，全部为 `OK`：S1、S2 Generator、S2 Discriminator、
BERT、HuBERT、speaker encoder、G2PW、NLTK 和语言识别模型。

### 真实兼容测试

设置本地资源环境变量后运行完整 `tests/compat`：

```text
75 passed, 0 skipped, 5 warnings in 53.21s
```

覆盖内容包括：

- G2PW 对“重庆”的真实消歧；
- 英文 CMU/NLTK 与日文 pyopenjtalk；
- 中文 BERT 对齐、Mixed 分段与三语独立 frontend；
- HuBERT `(1, 768, T)` 与 SV `(1, 20480)` 特征契约；
- 官方 S1 v3 权重严格加载，embedding hidden size 为 512；
- 官方 v2ProPlus S2 Generator/Discriminator 严格加载；
- S2 为官方八分支 GAN 路线，并使用 `06` checkpoint codec；
- semantic 提取与直接调用官方 S2G 路径一致。

### 真实预处理

使用一条日语音频执行完整预处理：

```text
1 passed in 38.43s
```

Text/BERT、32 kHz WAV、HuBERT、SV、semantic token 和 S1/S2 训练索引全部生成且形状
对齐，无 quarantine。

### 真实训练更新

```text
S2: 1 CUDA FP16 GAN step + checkpoint restore passed in 28.26s
S1: 4 CUDA FP16 mini-batches / 1 optimizer update + checkpoint restore passed in 30.17s
```

两项测试均使用受版本控制的五条固定预处理 fixture。临时 checkpoint 只写入 pytest
缓存目录。

### 真实推理

使用官方 Base S1/S2、Acane 日语参考音频和中文目标文本执行真实 CUDA 推理：

```text
sample_rate=32000
samples=81920
wav_bytes=163884
finite=true
```

临时 CandidateBundle、推理工作目录和 WAV 已在验证结束后删除。

## 已知非阻塞警告

验证中仅出现既有上游警告：`weight_norm`、旧版 `torch.cuda.amp.autocast`、
`return_complex=False` STFT 的弃用提示，以及 PyTorch `ComplexHalf` 实验性提示。未出现
权重键缺失、形状不匹配、非有限 loss、CUDA OOM 或推理失败。

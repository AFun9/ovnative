# 项目来源、模型血缘与致谢

## 1. 这个项目是什么

`OVNative` 是一个**独立的 native 推理引擎**，目标是在不链接 ONNX Runtime、不链接
libtorch 的前提下运行 OmniVoice 语音模型：

- 语言模型：Qwen3-0.6B backbone + 8 路 audio head，迭代 unmask + CFG 采样
- 音频 tokenizer：AT encoder（voice clone 参考音频）+ acoustic decoder（codes → PCM）
- 文本前端：OmniVoice 的多语种 tokenizer / 文本模板 / 时长估计
- 后端：C++17 CPU（AVX2/FMA/AVX-VNNI）与 CUDA（cuBLAS + 自定义 kernel）

它不是 OmniVoice 官方实现，也不隶属于 k2-fsa、Boson AI、阿里云或 Qwen 团队。
`OVNative` 是本工程的独立名字；模型权重不随本仓库分发。

## 2. 模型是谁的

### 2.1 OmniVoice

- 上游：<https://github.com/k2-fsa/OmniVoice>
- 角色：完整的 TTS 模型、训练/推理代码、tokenizer、模型权重
- 上游声明：代码 Apache-2.0；预训练模型 **CC-BY-NC**（受训练数据约束，例如 Emilia）
- 本仓库只包含**针对该模型重写的 native 推理代码**，不包含权重

### 2.2 Qwen3-0.6B

- 上游：<https://github.com/QwenLM/Qwen3>
- 角色：OmniVoice 的语言模型 backbone
- 权重不在本仓库；使用时遵循 Qwen 的模型许可证

### 2.3 Boson Higgs Audio 2

- 上游：<https://github.com/boson-ai/boson-multimodal>
- 角色：OmniVoice 的音频 tokenizer（encoder/decoder）来源
- 许可证：**Boson Higgs Audio 2 Community License Agreement**，并包含
  Meta Llama 3 Community License 的条款
- 上游 checkpoint 中附带的 `audio_tokenizer/LICENSE` 是完整许可证文本
- 本仓库不分发 Higgs Materials（模型权重），只包含独立编写的推理代码；
  如再分发权重，必须同时遵守该许可证的 notice / 使用限制

> 重要：模型权重（`model.safetensors`、`audio_tokenizer/model.safetensors`）以及
> tokenizer JSON **不在本仓库**。请从上游获取，并自行确认许可证是否满足你的使用场景
> （尤其是 OmniVoice 模型的 CC-BY-NC 与 Higgs Audio 2 的条款）。

## 3. 代码是怎么来的

时间线大致如下：

1. **PyTorch 参考实现**：用上游 OmniVoice 权重跑通 LM / AT，产生 logits、attention、
   encoder/decoder 的数值参考。
2. **Python/ONNX 工具链**：在 `AFun9/Omnivoice-onnx` 中完成导出、INT8/FP16 量化、
   ONNX Runtime 数值校验和端到端推理；这套工具链仍然只用于离线参考，不是运行时。
3. **C++ CPU 引擎**：自定义 `OVW1` 二进制格式；手写 AVX2/FMA GEMM、AVX-VNNI
   `vpdpbusd` int8 GEMM、attention、声码器卷积；逐步对齐 PyTorch/ONNX。
4. **CUDA 引擎**：LM 常驻显存，cuBLAS + fused/batched attention；随后把 AT
   encoder/decoder 也搬到 GPU，形成 auto/clone 的纯 GPU 路径。
5. **性能工程**：per-32k 激活量化、ragged CFG、4 行 query attention、sliced
   qkv/gate-up、fp16/TF32 GEMM、im2col + cuBLAS conv、warp softmax 等；每一项都在
   [perf-review.md](perf-review.md) 里有 A/B 和数值门禁。

原始仓库（ONNX 工具链 / PyTorch 参考）：
<https://github.com/AFun9/Omnivoice-onnx>

## 4. 为什么要做 native

- **部署**：不想在目标机器上装 PyTorch / ONNX Runtime / CUDA Python 栈
- **启动**：`mmap` OVW1 权重，避免 Python/ONNX 初始化
- **可控**：CPU/GPU kernel、量化口径、内存布局都可以针对这台机器调
- **可嵌入**：C ABI + 静态库，方便 Unity/C#/Rust/Go 等宿主调用

## 5. 致谢

这个项目建立在大量公开工作之上：

- **k2-fsa / OmniVoice**：模型、训练代码、tokenizer、推理语义和公开讨论
- **Boson AI / Higgs Audio 2**：音频 tokenizer 架构与权重
- **Qwen / 阿里云**：Qwen3-0.6B backbone
- **ONNX Runtime、PyTorch、NumPy、safetensors**：导出、量化、数值对齐和参考实现
- **CUDA / cuBLAS / NVIDIA**：GPU 运行时与 GEMM
- **开源社区**：论文、issue、kernel 讨论、性能博客，以及所有贡献者

没有这些公开的模型、代码、数据和讨论，`OVNative` 不可能存在。感谢。

## 6. 名字与商标

`OVNative` 是本工程的独立名称。`OmniVoice`、`Higgs Audio 2`、`Qwen` 等名称、
商标和模型归各自上游作者所有；本仓库不使用它们作为官方背书。

本项目不提供法律意见；模型和权重的使用条件以各自上游许可证为准。

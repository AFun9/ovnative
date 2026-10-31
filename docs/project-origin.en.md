# Project Origin, Model Lineage and Acknowledgements

[中文](project-origin.md) | English

## 1. What this project is

`OVNative` is an **independent native inference engine** for the OmniVoice
speech model. Its goal is to run the model without linking ONNX Runtime or
libtorch:

- **Language model**: Qwen3-0.6B backbone with an 8-way audio head, iterative
  unmasking with classifier-free guidance (CFG)
- **Audio tokenizer**: AT encoder (for voice-clone reference audio) and acoustic
  decoder (codes → PCM)
- **Text front-end**: OmniVoice's multilingual tokenizer, prompt templates and
  duration estimation
- **Backends**: C++17 CPU (AVX2/FMA/AVX-VNNI) and CUDA (cuBLAS + custom kernels)

It is not the official OmniVoice implementation and is not affiliated with
k2-fsa, Boson AI, Alibaba Cloud, or the Qwen team. `OVNative` is this project's
independent name. Model weights are not distributed in this repository.

## 2. Whose model is it?

### 2.1 OmniVoice

- Upstream: <https://github.com/k2-fsa/OmniVoice>
- Role: the complete TTS model, training/inference code, tokenizer and weights
- Upstream terms: code Apache-2.0; the pre-trained model is **CC-BY-NC**
  (constrained by its training data, e.g. Emilia)
- This repository contains only an independent native inference implementation;
  it does not contain the weights

### 2.2 Qwen3-0.6B

- Upstream: <https://github.com/QwenLM/Qwen3>
- Role: the OmniVoice language-model backbone
- Weights are not included; follow the Qwen model license when using them

### 2.3 Boson Higgs Audio 2

- Upstream: <https://github.com/boson-ai/boson-multimodal>
- Role: the source of OmniVoice's audio tokenizer (encoder/decoder)
- License: **Boson Higgs Audio 2 Community License Agreement**, which
  incorporates the Meta Llama 3 Community License terms
- The complete license text ships with the upstream checkpoint at
  `audio_tokenizer/LICENSE`
- This repository does not distribute Higgs Materials (model weights). It
  contains independently written inference code. If you redistribute weights,
  you must also comply with the upstream notice and use restrictions.

> Important: the model weights (`model.safetensors`,
> `audio_tokenizer/model.safetensors`) and tokenizer JSON are **not** in this
> repository. Obtain them from upstream and check the licenses for your use case,
> especially the OmniVoice CC-BY-NC model and the Higgs Audio 2 terms.

## 3. How the code came about

The rough timeline:

1. **PyTorch reference**: run the upstream OmniVoice weights and capture
   numerical references for LM logits, attention, encoder and decoder.
2. **Python/ONNX toolchain**: in `AFun9/Omnivoice-onnx`, export, quantize
   (INT8/FP16), validate against ONNX Runtime, and run end-to-end inference.
   This toolchain remains an offline reference, not the runtime.
3. **C++ CPU engine**: define the `OVW1` binary format; hand-write AVX2/FMA
   GEMM, AVX-VNNI `vpdpbusd` int8 GEMM, attention and vocoder convolutions;
   align step by step with PyTorch/ONNX.
4. **CUDA engine**: keep LM weights resident on the device; use cuBLAS plus
   fused/batched attention; move the AT encoder/decoder to the GPU as well,
   producing a fully GPU auto/clone path.
5. **Performance work**: per-32k activation quantization, ragged CFG, 4-row
   query attention, sliced qkv/gate-up, FP16/TF32 GEMM, im2col + cuBLAS
   convolutions, warp-level softmax, and more. Every accepted change has an A/B
   measurement and a numerical gate in
   [perf-review.md](perf-review.md).

Original repository (ONNX toolchain / PyTorch references):
<https://github.com/AFun9/Omnivoice-onnx>

## 4. Why native?

- **Deployment**: no PyTorch / ONNX Runtime / Python CUDA stack on the target
- **Startup**: `mmap` the OVW1 weights; no Python or ONNX initialization
- **Control**: tune kernels, quantization and memory layout for the target machine
- **Embedding**: C ABI + static library for Unity/C#/Rust/Go hosts

## 5. Acknowledgements

This project builds on a large body of public work:

- **k2-fsa / OmniVoice**: the model, training code, tokenizer, inference
  semantics and public discussions
- **Boson AI / Higgs Audio 2**: the audio-tokenizer architecture and weights
- **Qwen / Alibaba Cloud**: the Qwen3-0.6B backbone
- **ONNX Runtime, PyTorch, NumPy, safetensors**: export, quantization,
  numerical alignment and reference implementations
- **CUDA / cuBLAS / NVIDIA**: GPU runtime and GEMM
- **The open-source community**: papers, issues, kernel discussions,
  performance write-ups and every contributor

Without these public models, code, data and discussions, `OVNative` would not
exist. Thank you.

## 6. Names and trademarks

`OVNative` is this project's independent name. `OmniVoice`, `Higgs Audio 2`,
`Qwen` and other names, trademarks and models belong to their respective
upstream owners; this repository does not claim official endorsement.

This project does not provide legal advice. The terms for using models and
weights are governed by their respective upstream licenses.

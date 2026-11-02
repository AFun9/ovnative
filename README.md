# OVNative

[English](README_EN.md) | 中文

**OVNative** 是 OmniVoice 语音模型的本地推理引擎：C++17 从零实现，CPU 走 AVX2 /
FMA / AVX-VNNI，GPU 走 CUDA + cuBLAS。运行时不链接 ONNX Runtime，也不链接
libtorch；权重使用自定义的 `OVW1` 二进制格式，`mmap` 即加载。

> 这是一个独立的推理工程，不是 OmniVoice 官方仓库，也不隶属于 k2-fsa、Boson AI
> 或阿里云。模型、算法和商标归各自上游作者所有；详见
[docs/project-origin.md](docs/project-origin.md) 与 [NOTICE](NOTICE)。

## 现在能做到什么

| 路径 | 依赖 | 32 步端到端 RTF |
|---|---|---:|
| CPU（AVX-VNNI，`int8-nk`） | C++17 + OpenMP | **0.65–0.67** |
| CUDA auto（LM + sampler + decoder） | CUDA 12 + cuBLAS | **0.05–0.06** |
| CUDA voice-clone（encoder + LM + decoder） | CUDA 12 + cuBLAS | **0.09–0.11** |
| 参考：PyTorch + ONNX Runtime | Python | ~0.8–1.5 |

数字是同一台机器上 3.44 s 英文 / 3.16 s 中文 clone 的实测；CPU 为
i9-14900KF + 32 线程，GPU 为 RTX 4090。完整剖析和每一轮优化记录在
[docs/perf-review.md](docs/perf-review.md)。

能力：

- 文本 → 语音（auto / voice-design / voice-clone）
- 600+ 语种 tokenizer / 文本前端
- CPU AVX2/FMA + AVX-VNNI int8 GEMM（per-32k 激活量化）
- CUDA：int8 → fp16/TF32 GEMM、fused/batched attention、GPU encoder/decoder
- C ABI（[cpp/include/ovnative/ovnative.h](cpp/include/ovnative/ovnative.h)），方便
  Unity / C# / Rust / Go 通过 P/Invoke 嵌入

## 模型是谁的，代码怎么来的

**模型**：OmniVoice，由 [k2-fsa](https://github.com/k2-fsa/OmniVoice) 发布，语言模型
backbone 基于 Qwen3-0.6B，音频 tokenizer 基于 Boson AI 的 Higgs Audio 2。本仓库
**不包含模型权重**；权重需要从上游获取，并遵守上游许可证。

**native 代码**：从 [AFun9/Omnivoice-onnx](https://github.com/AFun9/Omnivoice-onnx)
的 Python/ONNX 工具链和 PyTorch 参考实现出发，重写为不依赖 ORT/PyTorch 的
C++/CUDA 引擎。ONNX Runtime 与 PyTorch 只用于离线导出、数值对齐和性能对照，不在
运行时出现。

**名字**：`OVNative` 是本工程的独立名字；“OmniVoice”“Higgs Audio 2”“Qwen”等
名称和商标归各自上游作者所有。模型权重不随本仓库分发。

## 目录

```
.
├── cpp/                       C++17 / CUDA 引擎
│   ├── include/ovnative/      公共头（C ABI + 内部类型）
│   ├── src/core/              OVW1 loader（mmap + CRC）
│   ├── src/backend/cpu/       AVX2/FMA + AVX-VNNI kernel
│   ├── src/backend/cuda/      CUDA LM / encoder / decoder
│   ├── src/model/             LM、AT encoder/decoder、TTS pipeline
│   ├── tools/                 ovtts / ovdecode / ovdump
│   └── tests/                 GEMM / LM / AT / tokenizer / C API
├── tools/                     权重转换、tokenizer 编译、参考数据与性能探针
├── docs/                      架构、权重格式、性能评审、项目来源
├── assets/                    voice-clone 参考音频
└── output/ref/                数值测试参考数据（不含模型权重）
```

## 构建

```bash
# CPU
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j

# CPU + CUDA
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DOV_ENABLE_CUDA=ON
cmake --build cpp/build -j
```

要求：CMake ≥ 3.16、C++17；CUDA 构建需要 CUDA Toolkit（cuBLAS）。CPU 构建不依赖
CUDA；GPU 构建也不依赖 PyTorch/ONNX Runtime。

### Windows（MSVC / Visual Studio 2022）

```powershell
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64 `
      -DOV_ENABLE_CUDA=ON
cmake --build cpp/build --config Release -j
ctest --test-dir cpp/build -C Release --output-on-failure
```

- 需要 Visual Studio 2022（MSVC）和 CUDA Toolkit 12.x；CUDA 构建会自动找 cuBLAS。
- MSVC 下 CPU kernel 以 `/arch:AVX2` 编译，**不启用 AVX-VNNI**；CPU 推理仍可用
  `int8-nk`，CUDA 用 `int8`（`[K,N]`）文件。
- 命令行文本和文件路径按 UTF-8 处理（`GetCommandLineW` + `CreateFileW/_wfopen`），
  中文文本、中文路径不需要额外转码。
- 没有 CUDA 时去掉 `-DOV_ENABLE_CUDA=ON`，只构建 CPU。

## 权重（不在本仓库）

先按上游说明获取 OmniVoice 权重，得到一个目录：

```
/path/to/OmniVoice/
├── model.safetensors
├── tokenizer.json
└── audio_tokenizer/model.safetensors
```

然后生成 OVW1：

```bash
# fp32 + int8（[K,N] 布局，CUDA / 旧 CPU 路径可用）
python tools/convert_weights.py \
  --model-dir /path/to/OmniVoice --mode both

# CPU AVX-VNNI 推荐再加一份 [N,K] int8
python tools/convert_weights.py \
  --model-dir /path/to/OmniVoice --mode int8 --int8-layout nk --only lm

python tools/convert_tokenizer.py \
  --tokenizer /path/to/OmniVoice/tokenizer.json
```

产物在 `output/weights/`。**不要**把权重提交到 git；模型许可证见
[docs/project-origin.md](docs/project-origin.md)。

## 运行

```bash
# CPU（VNNI 布局）
./cpp/build/ovtts output/weights/omnivoice-lm-int8-nk.ovw \
  output/weights/omnivoice-at-fp32.ovw output/weights/tokenizer.ovt \
  "Hello, this is OVNative." out.wav --lang English --seed 1234 --steps 32

# CUDA（需要旧 [K,N] int8 文件）
./cpp/build/ovtts output/weights/omnivoice-lm-int8.ovw \
  output/weights/omnivoice-at-fp32.ovw output/weights/tokenizer.ovt \
  "Hello, this is OVNative." out.wav --lang English --seed 1234 --steps 32 \
  --backend cuda

# voice clone
./cpp/build/ovtts output/weights/omnivoice-lm-int8.ovw \
  output/weights/omnivoice-at-fp32.ovw output/weights/tokenizer.ovt \
  "你好，这是 OVNative 的语音克隆。" out.wav --lang Chinese \
  --ref-audio assets/zero_shot_prompt.wav \
  --ref-text "希望你以后过的比我还好哟！" --backend cuda
```

> 输出默认整理静音：开头保留 60 ms、结尾保留 80 ms；超过 300 ms 的内部停顿
> 压到约 180 ms，避免短句被时长估计“撑”出一大段前后静音或把一句话听成两段。
> 需要原始输出时加 `--no-trim-silence`；也可用 `--max-pause-ms` /
> `--keep-pause-ms` 调整。静音整理只作用于最终 WAV，不改变 codes 和模型计算。

## 测试

```bash
# 需要上面生成的权重；output/ref/ 已包含参考数据
ctest --test-dir cpp/build --output-on-failure
```

测试覆盖：int8/fp32 GEMM、VNNI tile、attention 边界、LM 数值门禁、AT 解码/编码、
tokenizer、C API、CUDA LM / encoder / decoder。当前 **10/10**。

## 文档

- [README_EN.md](README_EN.md) — English README
- [docs/architecture.en.md](docs/architecture.en.md) — English architecture
- [docs/weight-format.en.md](docs/weight-format.en.md) — English OVW1 spec
- [docs/performance.en.md](docs/performance.en.md) — English performance summary
- [docs/project-origin.en.md](docs/project-origin.en.md) — English provenance
- [docs/cpp-engine.md](docs/cpp-engine.md) — 引擎架构、kernel、移植说明
- [docs/weight-format.md](docs/weight-format.md) — OVW1 二进制格式
- [docs/perf-review.md](docs/perf-review.md) — 逐轮性能剖析与优化记录
- [docs/project-origin.md](docs/project-origin.md) — 项目来源、模型血缘、许可证
- [cpp/src/backend/cuda/README.md](cpp/src/backend/cuda/README.md) — CUDA 后端

## 许可证与致谢

本仓库的 **native 引擎代码**采用 Apache License 2.0（见 [LICENSE](LICENSE)）。
**模型权重不在本仓库**，其许可证与使用条件由上游决定：

- OmniVoice 代码 Apache-2.0；预训练模型 **CC-BY-NC**（上游 model card）
- 音频 tokenizer 来自 **Boson Higgs Audio 2**，适用 Boson Higgs Audio 2
  Community License（含 Meta Llama 3 条款）
- 语言模型 backbone 为 **Qwen3-0.6B**，遵循 Qwen 的模型许可证

完整第三方声明见 [NOTICE](NOTICE)。使用模型前请自行阅读并遵守上游许可证；本仓库
不提供法律意见，也不对模型输出或第三方权利作任何担保。

感谢 **k2-fsa OmniVoice**、**Boson AI Higgs Audio 2**、**Qwen**、
**ONNX Runtime**、**PyTorch** 以及所有开源贡献者：没有这些公开的模型、论文、
工具和讨论，这个 native 引擎不可能存在。

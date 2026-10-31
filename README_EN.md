# OVNative

**OVNative** is a local inference engine for the OmniVoice speech model, written
from scratch in C++17. The CPU path uses AVX2 / FMA / AVX-VNNI; the optional GPU
path uses CUDA + cuBLAS. The runtime does **not** link ONNX Runtime and does
**not** link libtorch. Weights use a small self-describing binary format called
`OVW1`, loaded with `mmap`.

> This is an independent inference project. It is not the official OmniVoice
> repository and is not affiliated with k2-fsa, Boson AI, Alibaba Cloud, or the
> Qwen project. Models, algorithms and trademarks belong to their respective
> owners. See [docs/project-origin.en.md](docs/project-origin.en.md) and
> [NOTICE](NOTICE).

[中文](README.md) | English

## What it can do today

| Path | Runtime dependencies | End-to-end RTF, 32 steps |
|---|---|---:|
| CPU (AVX-VNNI, `int8-nk`) | C++17 + OpenMP | **0.65–0.67** |
| CUDA auto (LM + sampler + decoder) | CUDA 12 + cuBLAS | **0.05–0.06** |
| CUDA voice-clone (encoder + LM + decoder) | CUDA 12 + cuBLAS | **0.09–0.11** |
| Reference: PyTorch + ONNX Runtime | Python | ~0.8–1.5 |

Numbers are measured on the same machine: 3.44 s English / 3.16 s Chinese clone,
i9-14900KF (32 threads) and RTX 4090. The full breakdown, rejected experiments
and per-round optimization log are in [docs/perf-review.md](docs/perf-review.md)
(Chinese; an English summary is in [docs/performance.en.md](docs/performance.en.md)).

Capabilities:

- Text-to-speech: auto, voice-design, and voice-clone
- 600+ language tokenizer / text front-end
- CPU AVX2/FMA and AVX-VNNI int8 GEMM (per-32k activation quantization)
- CUDA: int8 → FP16/TF32 GEMM, fused and batched attention, GPU encoder/decoder
- C ABI ([cpp/include/ovnative/ovnative.h](cpp/include/ovnative/ovnative.h))
  for Unity / C# / Rust / Go embedders

## Model and code provenance

**Model.** OmniVoice is published by [k2-fsa](https://github.com/k2-fsa/OmniVoice).
Its language-model backbone is Qwen3-0.6B and its audio tokenizer is based on
Boson AI's Higgs Audio 2. **Model weights are not included in this repository.**
You must obtain them from the upstream projects and comply with their licenses.

**Native code.** This engine started from the Python/ONNX toolchain and PyTorch
reference implementation in
[AFun9/Omnivoice-onnx](https://github.com/AFun9/Omnivoice-onnx), and was then
rewritten as a C++/CUDA runtime that does not depend on ORT or PyTorch. ONNX
Runtime and PyTorch are used only offline for export, numerical validation and
performance comparison; they are not part of the runtime.

**Name.** `OVNative` is this project's independent name. "OmniVoice",
"Higgs Audio 2", "Qwen" and other names/marks belong to their upstream owners.
Model weights are not redistributed here.

## Repository layout

```
.
├── cpp/                       C++17 / CUDA engine
│   ├── include/ovnative/      public headers (C ABI + internal types)
│   ├── src/core/              OVW1 loader (mmap + CRC)
│   ├── src/backend/cpu/       AVX2/FMA + AVX-VNNI kernels
│   ├── src/backend/cuda/      CUDA LM / encoder / decoder
│   ├── src/model/             LM, AT encoder/decoder, TTS pipeline
│   ├── tools/                 ovtts / ovdecode / ovdump
│   └── tests/                 GEMM / LM / AT / tokenizer / C API
├── tools/                     weight conversion, tokenizer, reference data, probes
├── docs/                      architecture, weight format, performance, provenance
├── assets/                    voice-clone reference audio
└── output/ref/                numerical test references (no model weights)
```

## Build

```bash
# CPU
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j

# CPU + CUDA
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DOV_ENABLE_CUDA=ON
cmake --build cpp/build -j
```

Requirements: CMake ≥ 3.16 and a C++17 compiler. The CUDA build additionally
needs the CUDA Toolkit (cuBLAS). Neither build needs PyTorch or ONNX Runtime.

## Weights (not in this repository)

Follow the upstream instructions to obtain an OmniVoice checkpoint:

```
/path/to/OmniVoice/
├── model.safetensors
├── tokenizer.json
└── audio_tokenizer/model.safetensors
```

Then generate OVW1:

```bash
# FP32 + INT8 ([K,N] layout; usable by CUDA and the legacy CPU path)
python tools/convert_weights.py \
  --model-dir /path/to/OmniVoice --mode both

# Recommended extra file for CPU AVX-VNNI ([N,K] layout)
python tools/convert_weights.py \
  --model-dir /path/to/OmniVoice --mode int8 --int8-layout nk --only lm

python tools/convert_tokenizer.py \
  --tokenizer /path/to/OmniVoice/tokenizer.json
```

Outputs go to `output/weights/`. **Do not commit weights to git.** See
[docs/project-origin.en.md](docs/project-origin.en.md) for model licenses.

## Run

```bash
# CPU (VNNI layout)
./cpp/build/ovtts output/weights/omnivoice-lm-int8-nk.ovw \
  output/weights/omnivoice-at-fp32.ovw output/weights/tokenizer.ovt \
  "Hello, this is OVNative." out.wav --lang English --seed 1234 --steps 32

# CUDA (needs the legacy [K,N] int8 file)
./cpp/build/ovtts output/weights/omnivoice-lm-int8.ovw \
  output/weights/omnivoice-at-fp32.ovw output/weights/tokenizer.ovt \
  "Hello, this is OVNative." out.wav --lang English --seed 1234 --steps 32 \
  --backend cuda

# Voice clone
./cpp/build/ovtts output/weights/omnivoice-lm-int8.ovw \
  output/weights/omnivoice-at-fp32.ovw output/weights/tokenizer.ovt \
  "Hello, this is OVNative voice cloning." out.wav --lang English \
  --ref-audio assets/zero_shot_prompt.wav \
  --ref-text "希望你以后过的比我还好哟！" --backend cuda
```

## Tests

```bash
# Requires the generated weights; output/ref/ already contains the references.
ctest --test-dir cpp/build --output-on-failure
```

Coverage: int8/fp32 GEMM, VNNI tiles, attention boundaries, LM numerical gates,
AT decode/encode, tokenizer, C API, CUDA LM / encoder / decoder. The current
suite is **10/10** when weights are present.

## Documentation

- [docs/architecture.en.md](docs/architecture.en.md) — engine architecture and kernels
- [docs/weight-format.en.md](docs/weight-format.en.md) — OVW1 binary format
- [docs/performance.en.md](docs/performance.en.md) — benchmark summary and optimizations
- [docs/project-origin.en.md](docs/project-origin.en.md) — provenance, model lineage, licenses
- [cpp/src/backend/cuda/README.md](cpp/src/backend/cuda/README.md) — CUDA backend

The detailed Chinese documents are [docs/cpp-engine.md](docs/cpp-engine.md),
[docs/weight-format.md](docs/weight-format.md), and
[docs/perf-review.md](docs/perf-review.md).

## License and acknowledgements

The **native engine code** in this repository is released under the Apache
License 2.0; see [LICENSE](LICENSE). **Model weights are not part of this
repository**, and their terms come from upstream:

- OmniVoice: code Apache-2.0; pre-trained model **CC-BY-NC** (upstream model card)
- Audio tokenizer: **Boson Higgs Audio 2**, under the Boson Higgs Audio 2
  Community License (which incorporates Meta Llama 3 terms)
- Language-model backbone: **Qwen3-0.6B**, under the Qwen model license

See [NOTICE](NOTICE) for the complete third-party notice. Read and follow the
upstream licenses before using the model. This project does not provide legal
advice and makes no warranty regarding model outputs or third-party rights.

Thanks to **k2-fsa OmniVoice**, **Boson AI Higgs Audio 2**, **Qwen**,
**ONNX Runtime**, **PyTorch**, and all open-source contributors: without their
public models, papers, tools and discussions, this native engine would not exist.

# Performance Summary

[中文完整版](perf-review.md) | English

Machine: **i9-14900KF** (8 P-cores + 16 E-cores, 32 threads, AVX2/FMA/AVX-VNNI)
and **RTX 4090** (sm_89, 24 GB), Release build (`-O3`). All numbers below are
measured on this machine unless marked otherwise.

## End-to-end

| Path | RTF | Notes |
|---|---:|---|
| CPU AVX-VNNI (`int8-nk`) | **0.65–0.67** | 3.44 s output, 32 steps |
| CUDA auto | **0.05–0.06** | LM + sampler + decoder on GPU |
| CUDA voice-clone | **0.09–0.11** | encoder + LM + decoder on GPU |
| PyTorch + ONNX Runtime reference | ~0.8–1.5 | not the target runtime |

Recent `ovtts` runs:

```text
auto : load=0.36 s  sample=0.19–0.20 s  decode=0.01–0.02 s  RTF=0.05–0.06
clone: load=0.47–0.52 s  sample=0.27–0.34 s  decode=0.01–0.02 s  RTF=0.09–0.11
```

## How the CPU got here

| Round | Change | RTF |
|---|---|---:|
| baseline | scalar / naive loops | 3.55 |
| 1 | AVX2/FMA GEMM, OpenMP, fused ops | 2.07 |
| 2 | real AVX-VNNI `vpdpbusd`, per-32k activation quantization | 1.15 |
| 3 | attention query tiling, activation pooling, host CFG parallelism | 0.89–0.91 |
| 4 | ragged CFG, fused pipeline, 4-row attention, sliced qkv/gate-up | 0.65–0.67 |

The main lesson: the math was never the bottleneck; memory movement, scalar
loops and scheduling were.

## How the CUDA path got here

| Step | Change | Result |
|---|---|---:|
| initial | int8 LM with per-token dequant, CPU encoder/decoder | RTF ~0.15 |
| 1 | TF32 + merged qkv/gate-up + FP16 GEMMs with FP32 accumulation | RTF 0.10–0.11 |
| 2 | GPU decoder (RVQ/conv/ConvT/Snake on device) | auto RTF 0.05–0.06 |
| 3 | GPU encoder (resample/HuBERT/semantic/acoustic/RVQ) | clone RTF 0.20 |
| 4 | encoder conv → im2col + cuBLAS | encoder 193 → 8–9 ms |
| 5 | RVQ argmin → block-per-frame reduction | 17 → 0.4 ms |
| 6 | batched TF32 attention + warp softmax for long sequences | clone attention 4.5 → 1.44 ms |
| 7 | fused per-head RMSNorm + RoPE, GPU conv-weight prep | qk prep 0.4 → 0.31 ms; clone load 0.70 → 0.49 s |

## Per-forward stage breakdown

### auto (S=109, int8 LM)

| Stage | ms |
|---|---:|
| qkv | 1.00–1.02 |
| gate/up | 1.17–1.30 |
| attention (fused) | 1.37 |
| down | 0.75–0.77 |
| o_proj | 0.69–0.70 |
| qk norm+RoPE (fused) | 0.31 |
| SiLU×up | 0.16 |
| final norm + head | 0.10 |
| **total** | **~5.8** |

### voice-clone (S=187, batched attention)

| Stage | ms |
|---|---:|
| gate/up | ~2.0 |
| attention (batched TF32) | 1.44–1.9 |
| qkv | ~1.2 |
| down / o_proj | ~0.75 each |
| qk norm+RoPE | ~0.5 |
| SiLU×up | ~0.19 |
| final norm + head | ~0.11 |

The clone encoder is a one-time ~8–9 ms; the 32-step LM loop dominates the
remaining time.

## Key optimizations

### CPU

- **AVX-VNNI dot product**: `vpdpbusd` with 4×8 tiles; activations quantized per
  32 k-elements, offset corrected in the epilogue. int8 GEMM 2.5–3× faster than
  AVX2.
- **Per-32k activation quantization**: one scale per block, no per-row rescale.
- **Ragged CFG**: cond/uncond sequences packed without padding.
- **4-row query attention**: 4 query rows per block, 32 threads per row.
- **Sliced qkv/gate-up**: avoid large-N GEMM shapes that deoptimize.

### CUDA

- **TF32 / FP16 GEMM**: int8 weights are dequantized once; fp32 accumulation
  keeps accuracy. fp32 checkpoints use exact SGEMM.
- **Merged weights, sliced GEMMs**: qkv/gate-up are merged for memory locality,
  but the large-N GEMM is split where cuBLAS picks a bad algorithm.
- **Fused attention**: shared-memory query/key/value tiles, online softmax,
  float4 value loads, 32 threads per query row.
- **Batched attention** for `seq >= 160`: GQA K/V expansion + two strided
  batched TF32 GEMMs + warp-per-row bias/softmax.
- **GPU encoder**: im2col + cuBLAS conv, device-side weight transpose, block
  RVQ argmin.
- **GPU decoder**: all stages on device; only codes in and PCM out.

## Numerical gates

- fp32 CUDA LM: near-bit-exact vs PyTorch.
- int8 LM: argmax **92.38%**, mean **4.920e-2**, cos **0.99999964**.
- AT decoder vs CPU: `max|Δ| = 1.7e-6 / 2.3e-6 / 3.4e-6` for T=25/50/87.
- AT encoder: RVQ codes **400/400** identical.
- WAV: ≤ 1 LSB vs CPU decoder.
- `ctest`: **10/10** with weights present.

## Reproduce

```bash
# auto
OMP_NUM_THREADS=32 ./cpp/build/ovtts \
  output/weights/omnivoice-lm-int8.ovw \
  output/weights/omnivoice-at-fp32.ovw \
  output/weights/tokenizer.ovt \
  "Hello, this is OVNative." /tmp/auto.wav \
  --lang English --seed 1234 --steps 32 --backend cuda

# voice clone
OMP_NUM_THREADS=32 ./cpp/build/ovtts \
  output/weights/omnivoice-lm-int8.ovw \
  output/weights/omnivoice-at-fp32.ovw \
  output/weights/tokenizer.ovt \
  "Hello, this is OVNative voice cloning." /tmp/clone.wav \
  --lang English --ref-audio assets/zero_shot_prompt.wav \
  --ref-text "希望你以后过的比我还好哟！" \
  --backend cuda --steps 32
```

Generated WAVs are trimmed at the output layer by default (60 ms lead, 80 ms
trail, interior pauses > 300 ms shortened to 180 ms); use `--no-trim-silence`
to measure raw model output.

Useful environment switches: `OV_CUDA_PROFILE=1` (per-stage timing),
`OV_ENC_PROFILE=1` (encoder stages), `OV_ATTN_FUSED=1`,
`OV_ATTN_BATCHED=1`, `OV_CUDA_FP16=0`, `OV_CUDA_TF32=0`.

The full Chinese review — rejected experiments, per-kernel analysis, and the
optimization backlog — is in [perf-review.md](perf-review.md).

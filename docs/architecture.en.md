# Architecture

[中文](cpp-engine.md) | English

Goal: **zero inference dependencies** — no ONNX Runtime, no libtorch — with
weights in the `OVW1` binary format and clean separation between CPU and CUDA
backends.

## Layers

```
include/ovnative/      public headers (C ABI + internal types)
src/core/              OVW1 loader (mmap), allocator helpers
src/model/             LM, AT encoder/decoder, TTS pipeline
src/backend/cpu/       GEMM, ops, conv, scratch buffers
src/backend/cuda/      LM, encoder, decoder, attention, device workspace
src/api/               C ABI (ovnative_c)
tools/                 ovtts / ovdecode / ovdump
tests/                 numerical and boundary tests
```

`ovcore` is backend-neutral. The CUDA backend is a separate static library,
`ovcuda`, only added when `-DOV_ENABLE_CUDA=ON`. Model code installs optional
device function pointers, so `ovcore` never includes a CUDA header.

## Backend contract

The CPU backend and the CUDA runners expose the same logical operations:

```cpp
void linear_f32(int64_t M, int64_t N, int64_t K,
                const float* A, const float* W, float* C);

void linear_s8(int64_t M, int64_t N, int64_t K,
               const float* A, const int8_t* W,
               const float* col_scale, float* C);
```

The LM / encoder / decoder forward is written once; the runner hook decides
whether each stage executes on CPU or GPU. The GPU decoder/encoder are installed
by `ovtts` and fall back to CPU if creation fails.

## LM data flow

```
token IDs ──embedding──► hidden
                         │
        ┌────────────────┴────────────────┐
        │ 28 × Qwen3 block:               │
        │   RMSNorm → qkv (int8/fp16)     │
        │   q/k RMSNorm + RoPE            │
        │   GQA attention (fused/batched) │
        │   o_proj → residual             │
        │   RMSNorm → gate/up → SiLU×up   │
        │   down → residual               │
        └────────────────┬────────────────┘
                         │ final RMSNorm
                         ▼
                   audio head (8 × vocab)
                         │
                   argmax / CFG sampling
```

The int8 checkpoint uses:

- per-output-channel symmetric int8 weights;
- TF32 or FP16 cuBLAS GEMMs with FP32 accumulation on CUDA;
- per-32k activation quantization on AVX-VNNI;
- merged qkv / gate-up weight buffers where profitable, sliced GEMMs where the
  large-N shape selects a bad cuBLAS algorithm.

The fp32 checkpoint keeps an exact SGEMM path for numerical validation.

## Attention

Two implementations coexist:

- **Fused kernel** (short sequences): query rows in shared memory, 32 threads
  per row, keys tiled by 32, online softmax, float4 value loads.
- **Batched TF32 path** (`seq >= 160`): expand K/V per query head, one strided
  batched GEMM for QK and one for PV, then a warp-per-row bias+softmax kernel
  over the `[B, NH, S, S]` score matrix.

`OV_ATTN_FUSED=1` / `OV_ATTN_BATCHED=1` force either path for A/B testing.
`OV_ATTN_CUBLAS=1` selects the older non-batched cuBLAS fallback.

## Audio tokenizer — decoder

The decoder is fully device-resident on CUDA:

- RVQ gather + `project_out`
- `fc2`, direct conv1d, ConvTranspose as per-tap GEMMs
- Snake activation, residual units
- PCM copy back to host

Decoder output is validated against the CPU decoder at
`max|Δ| ≤ 3.4e-6` (T=25/50/87), with a 1-LSB difference after WAV quantization.

## Audio tokenizer — encoder

Voice-clone encoding runs entirely on the GPU when CUDA is enabled:

1. 24 kHz → 16 kHz polyphase resample
2. 7-layer conv stem + GroupNorm/GELU
3. projection + positional conv (`groups=16, k=128`) + LayerNorm
4. 12 post-norm HuBERT transformer layers
5. semantic conv stack
6. DAC acoustic encoder
7. concat + fc + 8-stage RVQ encode

Conv1d uses im2col + cuBLAS. Weights are uploaded raw and transposed on the
device at load time; this removed a large host-side preparation loop and cut
clone startup from ~0.70 s to ~0.49 s. RVQ argmin uses one block per frame with
a shared-memory reduction.

The encoder produces **exactly** the same RVQ codes as the CPU encoder on the
validation reference (400/400).

## TTS pipeline

```
text ──tokenizer/template──► prompt IDs
reference audio ──AT encoder──► reference codes (voice clone)
prompt IDs + ref codes ──LM sampling (32 steps, CFG)──► target codes
target codes ──AT decoder──► 24 kHz PCM ──WAV──► file
```

The Python/ONNX toolchain is used only to generate reference tensors for tests;
the C++ binary is self-contained at runtime.

## Numerical gates

- fp32 CUDA LM: near-bit-exact against the PyTorch reference.
- int8 LM: `cos > 0.99999`, `mean|Δ| < 0.15`, argmax agreement ≥ 80%
  (current: **92.38%**, mean **4.920e-2**, cos **0.99999964**).
- AT decoder: `max|Δ| ≤ 3.4e-6`.
- AT encoder: RVQ codes 400/400 identical.
- WAV output: ≤ 1 LSB difference vs the CPU decoder.

## Porting notes

- The CPU path is little-endian and assumes x86 AVX2/FMA; AVX-VNNI is optional
  with an AVX2 fallback.
- OVW1 stores linear weights pre-transposed (`layout=1`) for `[K,N]` GEMMs, or
  as `[N,K]` with `layout=0` for embedding/VNNI.
- CUDA needs a device with enough shared memory for the fused attention tile;
  the batched path has no such constraint.
- The C ABI is versioned (`OV_ABI_VERSION`) and does not expose C++ types.

For the complete Chinese design notes, kernel inventory and porting checklist,
see [cpp-engine.md](cpp-engine.md).

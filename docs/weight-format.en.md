# OVW1 — OVNative Binary Weight Format

[中文](weight-format.md) | English

Goal: **zero dependencies** — no ONNX, PyTorch, protobuf or JSON runtime. One
self-describing file, `mmap`-ed directly; a tensor pointer is a weight pointer,
so loading a 3.2 GB checkpoint costs roughly one `mmap`.

## File layout

```
┌──────────────┬─────────────┬──────────────┬───────────────┬──────────────┐
│ Header 128 B │ Index n×64B │ Names blob   │ Tensor data   │ Aux (scales) │
└──────────────┴─────────────┴──────────────┴───────────────┴──────────────┘
                                                  each block 64 B aligned
```

All integers are little-endian (x86/ARM64), `u32/u64` unsigned.

## Header (128 bytes)

| off | type | field | meaning |
|----:|------|-------|---------|
| 0 | char[8] | `magic` | `"OMNIVOIC"` |
| 8 | u32 | `version` | 1 |
| 12 | u32 | `dtype_mode` | 0 = all FP32, 1 = contains INT8 tensors |
| 16 | u32 | `n_tensors` | number of tensors |
| 20 | u32 | `header_bytes` | 128 |
| 24 | u64 | `index_off` | absolute index offset |
| 32 | u64 | `names_off` | absolute names offset |
| 40 | u64 | `data_off` | absolute data offset |
| 48 | u64 | `file_size` | total bytes |
| 56 | u64 | `flags` | bit0: transposed weights; bit1: INT8 |
| 64 | u64 | `n_params` | original parameter count (sanity check) |
| 72 | u8[56] | reserved | 0 |

`magic` keeps the historical upstream bytes `"OMNIVOIC"`; the format name is
OVW1 and the current project is OVNative.

## TensorDesc (64 bytes × n)

| off | type | field | meaning |
|----:|------|-------|---------|
| 0 | u32 | `name_off` | relative to names_off |
| 4 | u32 | `name_len` | excluding trailing NUL |
| 8 | u8 | `dtype` | 0=FP32, 1=INT8, 2=FP16, 3=INT64 |
| 9 | u8 | `ndim` | 0..4 |
| 10 | u8 | `layout` | 0=row-major as-is; 1=transposed (2-D linear weight) |
| 11 | u8 | `quant` | 0=none; 1=per-output-channel symmetric INT8 |
| 12 | u32[4] | `dims` | shape in stored layout |
| 28 | u32 | `aux_n` | number of scales (= output channels for INT8) |
| 32 | u64 | `hash` | CRC32 of payload (low 32 bits) |
| 40 | u64 | `data_off` | absolute payload offset |
| 48 | u64 | `nbytes` | payload bytes |
| 56 | u64 | `aux_off` | absolute scale offset (0 if none) |

## Key conventions

### 1. Linear weights are pre-transposed by default; INT8 also has an `[N][K]` variant

PyTorch's `nn.Linear.weight` is `[out, in]`. ONNX needs a `transB` on every
MatMul, while the C++ side wants the output dimension contiguous for SIMD.
Therefore the default (`--int8-layout kn`) stores all 2-D linear weights as
`[in, out]` (`layout=1`, `dims=[in, out]`), and the runtime GEMM is:

```
C[M, N] = A[M, K] · Wt[K, N]          (Wt contiguous, vectorize over N)
```

Exceptions (`layout=0`, row-major):

- `llm.embed_tokens.weight`, `audio_embeddings.weight` — used by Gather
- all 1-D norms/biases and 3-D convolution kernels

#### 1a. Output-major INT8 for AVX-VNNI

`tools/convert_weights.py --int8-layout nk` (the default remains `kn`; old
files keep working; output goes to `omnivoice-<model>-int8-nk.ovw`) stores
quantized linear layers as:

| field | value |
|---|---|
| `dtype` | `1` (INT8) |
| `layout` | `0` (row-major) |
| `quant` | `1` |
| `dims` | `[out, in]` (same as PyTorch) |
| `aux_n` | `out` (one scale per output channel, on rows) |
| payload | `q[out, in]` (not transposed) |

This is the same encoding embeddings already use (`layout=0 + quant=1 +
per-row scale`), but consumed by `make_linear` (`Mat`) instead of
`Emb::gather`. Readers dispatch on `layout`; old `layout=1` files still work.

**Why:** AVX-VNNI `vpdpbusd` needs four consecutive k values for one output
column in a dword. `[N][K]` keeps each output's K contiguous and is repacked
into 4×8 tiles at load time. `[K][N]` cannot feed `vpdpbusd` efficiently.

**CUDA note:** the CUDA runner only understands the `layout=1` dequant path;
`-int8-nk.ovw` is **CPU/VNNI-only**. `ovtts --backend cuda` and
`ov_test_lm --cuda` reject it with a clear error instead of silently producing
wrong results.

### 2. INT8 is per-output-channel symmetric

For a row of the original `[out, in]` weight:
`scale = max|row| / 127`, `q = round(row / scale)`, clamped to `[-127, 127]`
(not -128, to preserve symmetry). Stored as:

- `layout=1` (default `kn`): payload `q.T = [in, out]`, aux `scale[out]`, one
  per column;
- `layout=0` (`nk` or embedding): payload `q = [out, in]`, aux `scale[out]`,
  one per row.

Runtime dequantization is free: accumulate in int32/fp32, then multiply by the
per-column/per-row scale:

```
layout=1:  C[m, n] = scale[n] · Σ_k A[m, k] · q[k, n]
layout=0:  C[m, n] = scale[n] · Σ_k A[m, k] · q[n, k]
```

No fp32 weight copy is materialized; the memory footprint is the int8 payload.

The VNNI path additionally quantizes **activations** to u8, without changing
OVW1 semantics:

```
per 32 k-elements: ascale = max|A_block| / 127
                   a_u8   = clamp(round(A / ascale), -127, 127) + 128
per block:         C[m,n] += (Σ a_u8·q[n,k] - 128·Σ q[n,k]) · ascale · wscale[n]
```

The `+128` shifts signed activations into the unsigned domain required by
`vpdpbusd`; the epilogue subtracts the precomputed per-block weight sum exactly.

`audio_heads` (the LM output head) is protected from quantization by default:
it directly produces sampling logits, and quantization noise maps one-to-one to
argmax flips.

### 3. Why a CRC

Each tensor carries a CRC32 of its payload. `ovdump --verify` proves, without
Python, that "the bytes C++ reads == the bytes the converter wrote", and gives
cache invalidation / incremental updates a concrete check.

## Relationship to safetensors

`tools/convert_weights.py` depends only on NumPy: it parses the safetensors
8-byte length + JSON header by hand, `mmap`s the file and creates `np.ndarray`
views from `data_offsets` — no torch. The whole chain
(safetensors → OVW1 → C++) has no framework dependency.

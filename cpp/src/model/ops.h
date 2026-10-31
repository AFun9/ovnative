// Backend-agnostic operator contracts used by the model code.
// CPU implementations live in src/backend/cpu/ops_cpu.cpp; the CUDA backend
// will provide the same symbols (see src/backend/cuda/README.md).
#pragma once

#include <cstdint>

namespace ov {

// Row-wise RMSNorm, in place: x[i] = x[i] / sqrt(mean(x²)+eps) * w[i % cols]
void rmsnorm_f32(float* x, const float* w, int64_t rows, int64_t cols, float eps);

// Same, reading from `src` and writing to `dst` (dst may alias src).  Lets the
// caller skip a full copy of the activation before normalising it.
void rmsnorm_into_f32(float* dst, const float* src, const float* w, int64_t rows, int64_t cols,
                      float eps);

// dst[i] += src[i] (residual add), SIMD when available.
void add_inplace_f32(float* dst, const float* src, int64_t n);

// SwiGLU activation: out[i] = silu(gate[i]) * up[i]
void silu_mul_f32(float* out, const float* gate, const float* up, int64_t n);

// In-place RoPE.  x is [rows, n_heads * head_dim]; each row uses pos[row].
void rope_f32(float* x, int64_t rows, int n_heads, int head_dim, const int64_t* pos, float theta);

// Fused per-head RMSNorm + RoPE for Q and K.  q/k are [rows, heads*head_dim];
// qw/kw are the per-head norm weights; positions are shared.  Equivalent to
// rmsnorm_f32(q,...) + rmsnorm_f32(k,...) + rope_f32(q,...) + rope_f32(k,...),
// but one pass over q/k and one OpenMP region instead of four serial passes.
void qk_norm_rope_f32(float* q, float* k, const float* qw, const float* kw, int64_t rows,
                      int n_heads, int n_kv_heads, int head_dim, const int64_t* pos,
                      float theta, float eps);

// Row-wise softmax over `cols`, with an optional additive bias of length `cols`
// (values of -1e30 act as "masked out").  Handles fully-masked rows by writing 0.
void softmax_bias_f32(float* scores, const float* bias, int64_t rows, int64_t cols);

// Fused single-head attention: scores = Q·Kᵀ·scale + bias, row-wise softmax,
// out = P·V.  K and V are copied once into contiguous L1/L2-resident buffers,
// and each output row is produced completely (score -> softmax -> PV) so the
// S x S score matrix never travels through global memory.
//   q: [S, q_stride]   k/v: [S, kv_stride] (head slice already offset)
//   out: [S, out_stride]
void attention_head_f32(const float* q, int64_t q_stride, const float* k, const float* v,
                        int64_t kv_stride, const float* bias, int S, int D, float scale,
                        float* out, int64_t out_stride);

// Enable flush-to-zero + denormals-are-zero for the calling thread.
// Masked softmax rows produce exp(-1e30) ~ 1e-38 (subnormal); x86 runs
// subnormal arithmetic through a microcode assist that is ~20-100x slower, and
// those values then poison every downstream GEMM.  Must be called once per
// thread (MXCSR is thread state); idempotent.
void enable_flush_to_zero();

// LayerNorm over the last dim: y = (x-mean)/sqrt(var+eps)*w + b
void layer_norm_f32(float* x, const float* w, const float* b, int64_t rows, int64_t cols, float eps);

// ELU (alpha=1), in place.
void elu_f32(float* x, int64_t n);

// Exact (erf-based) GELU, in place.
void gelu_f32(float* x, int64_t n);

// GroupNorm with num_groups == channels: each channel is normalised over time
// (per batch item), then scaled/shifted.  This is what HuBERT's conv stem uses.
void group_norm_channels_f32(float* x, const float* w, const float* b, int channels, int t,
                             float eps);

// Generic 1-D convolution (grouped), time-major [T, C].
struct Conv1dSpec {
  int cin = 0, cout = 0, k = 0, stride = 1, pad = 0, dil = 1, groups = 1;
  const float* w = nullptr;   // [cout, cin/groups, k] or null (caller may pass it explicitly)
  const float* b = nullptr;   // [cout] or null
};
void conv1d_generic_f32(const Conv1dSpec& c, const float* w, const float* x, int t_in, float* y,
                        int t_out);

}  // namespace ov

#include "backend/cuda/cuda_lm.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <map>
#include <string>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define OV_CUDA_CHECK(x)                                                          \
  do {                                                                            \
    const cudaError_t e_ = (x);                                                   \
    if (e_ != cudaSuccess) {                                                      \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    }                                                                             \
  } while (0)

#define OV_CUBLAS_CHECK(x)                                                        \
  do {                                                                            \
    const cublasStatus_t s_ = (x);                                                \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                            \
      throw std::runtime_error("cuBLAS error " + std::to_string(static_cast<int>(s_)) + \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    }                                                                             \
  } while (0)

namespace ov {
namespace cuda {
namespace {

// ------------------------------------------------------------------ kernels

__global__ void embed_kernel(const int64_t* ids, const uint8_t* mask, const float* tok_emb,
                             const float* aud_emb, int batch, int seq, int codebooks,
                             int audio_vocab, int hidden, float* out) {
  const int row = blockIdx.x;  // b * seq + s
  const int b = row / seq;
  const int s = row % seq;
  float* o = out + static_cast<size_t>(row) * hidden;
  if (mask[row]) {
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) o[h] = 0.f;
    __syncthreads();
    for (int c = 0; c < codebooks; ++c) {
      const int64_t id = ids[(static_cast<size_t>(b) * codebooks + c) * seq + s] + c * audio_vocab;
      const float* e = aud_emb + id * hidden;
      for (int h = threadIdx.x; h < hidden; h += blockDim.x) o[h] += e[h];
      __syncthreads();
    }
  } else {
    const int64_t id = ids[static_cast<size_t>(b) * codebooks * seq + s];
    const float* e = tok_emb + id * hidden;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) o[h] = e[h];
  }
}

__global__ void rmsnorm_kernel(const float* x, const float* w, int rows, int cols, float eps,
                               float* out) {
  const int row = blockIdx.x;
  const float* xr = x + static_cast<size_t>(row) * cols;
  float* orow = out + static_cast<size_t>(row) * cols;
  float acc = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) acc += xr[i] * xr[i];
  __shared__ float red[256];
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(red[0] / static_cast<float>(cols) + eps);
  for (int i = threadIdx.x; i < cols; i += blockDim.x) orow[i] = xr[i] * scale * w[i];
}

// cos/sin table, built once per forward (positions are identical for every
// layer, and the old kernel recomputed powf/cosf/sinf per element per layer --
// T*heads*half transcendentals, 28 times over).
__global__ void rope_table_kernel(float* tab, const int64_t* pos, int rows, int half, int head_dim,
                                  float theta) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(rows) * half) return;
  const int r = static_cast<int>(i / half);
  const int d = static_cast<int>(i % half);
  const float freq = __powf(theta, -2.f * d / static_cast<float>(head_dim));
  const float angle = static_cast<float>(pos[r]) * freq;
  tab[i] = __cosf(angle);
  tab[static_cast<int64_t>(rows) * half + i] = __sinf(angle);
}

__global__ void rope_apply_kernel(float* x, const float* tab, int rows, int heads, int half,
                                 int head_dim) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(rows) * heads * half) return;
  const int r = static_cast<int>(i / (heads * half));
  const int rem = static_cast<int>(i % (heads * half));
  const int h = rem / half;
  const int d = rem % half;
  const float c = tab[static_cast<int64_t>(r) * half + d];
  const float s = tab[static_cast<int64_t>(rows) * half + static_cast<int64_t>(r) * half + d];
  float* v = x + (static_cast<int64_t>(r) * heads + h) * head_dim;
  const float a = v[d];
  const float b = v[d + half];
  v[d] = a * c - b * s;
  v[d + half] = b * c + a * s;
}

// Fused per-head RMSNorm + RoPE: one block per (row, head), D=128 threads.
__global__ void norm_rope_heads_kernel(float* __restrict__ x, const float* __restrict__ w,
                                       const float* __restrict__ tab, int rows, int heads,
                                       int D, float eps) {
  const int b = blockIdx.x;
  const int r = b / heads;
  const int h = b - r * heads;
  float* v = x + (static_cast<int64_t>(r) * heads + h) * D;
  const int half = D / 2;
  float acc = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) acc += v[i] * v[i];
  __shared__ float red[128];
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
    __syncthreads();
  }
  const float scale = rsqrtf(red[0] / static_cast<float>(D) + eps);
  __shared__ float tmp[128];
  for (int i = threadIdx.x; i < D; i += blockDim.x) tmp[i] = (v[i] * scale) * w[i];
  __syncthreads();
  if (threadIdx.x < half) {
    const int d = threadIdx.x;
    const float c = tab[static_cast<int64_t>(r) * half + d];
    const float s = tab[static_cast<int64_t>(rows) * half + static_cast<int64_t>(r) * half + d];
    const float a = tmp[d];
    const float b2 = tmp[d + half];
    v[d] = a * c - b2 * s;
    v[d + half] = b2 * c + a * s;
  }
}

__global__ void silu_mul_kernel(float* gate, const float* up, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float g = gate[i];
    gate[i] = (g / (1.f + __expf(-g))) * up[i];
  }
}

// gate/up are two halves of a [rows, 2*cols] combined GEMM result.
__global__ void silu_mul_combined_kernel(const float* __restrict__ gu, int rows, int cols,
                                         float* __restrict__ out) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) return;
  for (int r = blockIdx.y; r < rows; r += gridDim.y) {
    const int64_t off = static_cast<int64_t>(r) * 2 * cols;
    const float g = gu[off + col];
    out[static_cast<int64_t>(r) * cols + col] = (g / (1.f + __expf(-g))) * gu[off + cols + col];
  }
}

__global__ void rmsnorm_half_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                    int rows, int cols, float eps, __half* __restrict__ out) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const float* row = x + static_cast<int64_t>(r) * cols;
  float acc = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) acc += row[i] * row[i];
  __shared__ float red[256];
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(red[0] / static_cast<float>(cols) + eps);
  __half* dst = out + static_cast<int64_t>(r) * cols;
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    dst[i] = __float2half(row[i] * scale * w[i]);
}

__global__ void float_to_half_kernel(const float* __restrict__ in, __half* __restrict__ out,
                                     int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i]);
}

__global__ void silu_mul_half_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                    __half* __restrict__ out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float g = gate[i];
    out[i] = __float2half((g / (1.f + __expf(-g))) * up[i]);
  }
}


// Split the merged qkv GEMM result [T, QD+2*KVD] into contiguous q/k/v slices.
__global__ void split_qkv_kernel(const float* __restrict__ qkv, int rows, int qd, int kvd,
                                 float* __restrict__ q, float* __restrict__ k,
                                 float* __restrict__ v) {
  const int stride = qd + 2 * kvd;
  const int64_t n = static_cast<int64_t>(rows) * stride;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int r = static_cast<int>(i / stride);
    const int c = static_cast<int>(i - static_cast<int64_t>(r) * stride);
    const float x = qkv[i];
    if (c < qd)
      q[static_cast<int64_t>(r) * qd + c] = x;
    else if (c < qd + kvd)
      k[static_cast<int64_t>(r) * kvd + (c - qd)] = x;
    else
      v[static_cast<int64_t>(r) * kvd + (c - qd - kvd)] = x;
  }
}

// Weight-only int8 -> fp32 on the device.  The host path used to expand every
// quantised tensor into a 2.45 GB std::vector before uploading it: 0.87 s of
// load time and 2.45 GB of host RSS, for data the device can expand itself.
__global__ void dequant_col_kernel(const int8_t* q, const float* scale, int64_t K, int64_t N,
                                   float* out) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < K * N) out[i] = static_cast<float>(q[i]) * scale[i % N];
}

__global__ void dequant_row_kernel(const int8_t* q, const float* scale, int64_t rows, int64_t cols,
                                   float* out) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < rows * cols) out[i] = static_cast<float>(q[i]) * scale[i / cols];
}

__global__ void mask_to_bias_kernel(const uint8_t* mask, float* bias, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) bias[i] = mask[i] ? 0.f : -1e30f;
}


// Row-wise masked softmax over the score plane, folded with the additive bias.
// One block per (batch, head, query row); the row is a contiguous `seq` slice.
__global__ void softmax_bias_rows_kernel(float* scr, const float* bias, int seq) {
  const int row = blockIdx.x;    // query index s
  float* r = scr + static_cast<size_t>(row) * seq;
  const float* bp = bias ? bias + static_cast<size_t>(row) * seq : nullptr;

  float m = -INFINITY;
  for (int j = threadIdx.x; j < seq; j += blockDim.x) {
    const float v = bp ? r[j] + bp[j] : r[j];
    r[j] = v;
    m = fmaxf(m, v);
  }
  __shared__ float red[256];
  red[threadIdx.x] = m;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + st]);
    __syncthreads();
  }
  m = red[0];
  __syncthreads();

  float sum = 0.f;
  if (m < -1e29f) {  // fully masked row
    for (int j = threadIdx.x; j < seq; j += blockDim.x) r[j] = 0.f;
    return;
  }
  for (int j = threadIdx.x; j < seq; j += blockDim.x) {
    const float e = __expf(r[j] - m);
    r[j] = e;
    sum += e;
  }
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
    __syncthreads();
  }
  const float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < seq; j += blockDim.x) r[j] *= inv;
}

// Expand the GQA K/V heads to one [S,D] slice per query head, so the two big
// batched GEMMs below can run NH independent matrices per batch.
__global__ void expand_kv_kernel(const float* __restrict__ k, const float* __restrict__ v,
                                 int B, int S, int NH, int KVD, int D, float* __restrict__ ke,
                                 float* __restrict__ ve) {
  const int64_t n = static_cast<int64_t>(B) * NH * S * D;
  const int nkv = KVD / D;
  const int g = NH / nkv;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int d = static_cast<int>(i % D);
    int64_t r = i / D;
    const int s = static_cast<int>(r % S);
    r /= S;
    const int h = static_cast<int>(r % NH);
    const int b = static_cast<int>(r / NH);
    const int kv = h / g;
    const int64_t src = (static_cast<int64_t>(b) * S + s) * KVD + static_cast<int64_t>(kv) * D + d;
    ke[i] = k[src];
    ve[i] = v[src];
  }
}

__global__ void softmax_bias_batched_kernel(float* __restrict__ scores, const float* __restrict__ bias,
                                            int B, int NH, int S) {
  const int64_t row = blockIdx.x;
  const int64_t nrows = static_cast<int64_t>(B) * NH * S;
  if (row >= nrows) return;
  const int i = static_cast<int>(row % S);
  int64_t r = row / S;
  const int b = static_cast<int>(r / NH);
  float* srow = scores + row * S;
  const float* brow = bias + (static_cast<int64_t>(b) * S + i) * S;
  __shared__ float red[256];
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < S; j += blockDim.x) {
    const float v = srow[j] + brow[j];
    srow[j] = v;
    mx = fmaxf(mx, v);
  }
  red[threadIdx.x] = mx;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + st]);
    __syncthreads();
  }
  mx = red[0];
  __syncthreads();
  if (mx < -1e29f) {
    for (int j = threadIdx.x; j < S; j += blockDim.x) srow[j] = 0.f;
    return;
  }
  float sum = 0.f;
  for (int j = threadIdx.x; j < S; j += blockDim.x) {
    const float e = __expf(srow[j] - mx);
    srow[j] = e;
    sum += e;
  }
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
    __syncthreads();
  }
  const float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < S; j += blockDim.x) srow[j] *= inv;
}

__global__ void softmax_bias_batched_warp_kernel(float* __restrict__ scores,
                                                 const float* __restrict__ bias, int B, int NH,
                                                 int S) {
  constexpr int kWarps = 4;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + warp;
  const int64_t nrows = static_cast<int64_t>(B) * NH * S;
  if (row >= nrows) return;
  const int i = static_cast<int>(row % S);
  const int b = static_cast<int>((row / S) / NH);
  float* srow = scores + row * S;
  const float* brow = bias + (static_cast<int64_t>(b) * S + i) * S;
  float mx = -INFINITY;
  for (int j = lane; j < S; j += 32) {
    const float v = srow[j] + brow[j];
    srow[j] = v;
    mx = fmaxf(mx, v);
  }
  for (int st = 16; st > 0; st >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, st));
  mx = __shfl_sync(0xffffffff, mx, 0);
  if (mx < -1e29f) {
    for (int j = lane; j < S; j += 32) srow[j] = 0.f;
    return;
  }
  float sum = 0.f;
  for (int j = lane; j < S; j += 32) {
    const float e = __expf(srow[j] - mx);
    srow[j] = e;
    sum += e;
  }
  for (int st = 16; st > 0; st >>= 1) sum += __shfl_down_sync(0xffffffff, sum, st);
  const float inv = 1.f / __shfl_sync(0xffffffff, sum, 0);
  for (int j = lane; j < S; j += 32) srow[j] *= inv;
}

// Batched attention: QK and PV are two cuBLAS strided-batched calls per batch;
// the score matrix is [NH,S,S] in the scratch buffer.  This uses tensor-core
// SGEMM (TF32) on the 4090 instead of the scalar fused kernel.
void attention_batched_fast(cublasHandle_t blas, const float* dq, const float* dk,
                            const float* dv, const float* dbias, int B, int S, int D, int QD,
                            int KVD, int NH, int NKV, float scale, float* dout, float* ke,
                            float* ve, float* scr) {
  const int nkv = NKV;
  const int g = NH / nkv;
  const int64_t n = static_cast<int64_t>(B) * NH * S * D;
  expand_kv_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(dk, dv, B, S, NH, KVD, D, ke,
                                                                     ve);
  const float one = 1.f, zero = 0.f;
  for (int b = 0; b < B; ++b) {
    const float* qb = dq + static_cast<int64_t>(b) * S * QD;
    float* sb = scr + static_cast<int64_t>(b) * NH * S * S;
    OV_CUBLAS_CHECK(cublasSgemmStridedBatched(
        blas, CUBLAS_OP_T, CUBLAS_OP_N, S, S, D, &scale, ke + static_cast<int64_t>(b) * NH * S * D,
        D, static_cast<int64_t>(S) * D, qb, QD, D, &zero, sb, S, static_cast<int64_t>(S) * S, NH));
    softmax_bias_batched_warp_kernel<<<static_cast<unsigned>((NH * S + 3) / 4), 128>>>(
        sb, dbias + static_cast<int64_t>(b) * S * S, 1, NH, S);
    OV_CUDA_CHECK(cudaGetLastError());
    float* ob = dout + static_cast<int64_t>(b) * S * QD;
    OV_CUBLAS_CHECK(cublasSgemmStridedBatched(
        blas, CUBLAS_OP_N, CUBLAS_OP_N, D, S, S, &one,
        ve + static_cast<int64_t>(b) * NH * S * D, D, static_cast<int64_t>(S) * D, sb, S,
        static_cast<int64_t>(S) * S, &zero, ob, QD, D, NH));
  }
}

// ---------------------------------------------------------------- fused attention
//
// One launch per layer.  The cuBLAS path below needs 48 calls per layer, and the
// per-call overhead of those (host-side + launch) is what kept attention at
// ~10 ms/forward even after the kernel itself got 20x faster.
//
// One block per (query tile of 32 rows, batch*head): K/V tiles are staged in
// shared memory, the row softmax is online (flash style) so the S x S score
// matrix never reaches global memory, and the 128-wide output row is accumulated
// in registers.  head_dim is fixed at 128 (the only value this model uses); the
// caller falls back to the cuBLAS path for anything else.
constexpr int kAttnBM = 32;            // query rows per block
constexpr int kAttnBK = 32;            // keys per shared-memory tile
constexpr int kAttnD = 128;            // head_dim handled by this kernel
constexpr int kAttnTPR = 32;            // threads per query row in the PV phase
constexpr int kAttnDSL = kAttnD / kAttnTPR;   // output dims owned by one thread
// Odd stride for the scalar q/k tiles (no shared-memory bank conflicts), 16B
// aligned stride for the float4 v tile.
constexpr int kAttnQS = kAttnD + 1;
constexpr int kAttnVS = kAttnD + 4;
constexpr int kAttnSmem =
    (kAttnBM * kAttnQS + kAttnBK * kAttnQS + kAttnBK * kAttnVS + kAttnBM * kAttnBK +
     4 * kAttnBM) * static_cast<int>(sizeof(float));

__global__ void attention_fused_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                       const float* __restrict__ v,
                                       const float* __restrict__ bias, int seq, int QD, int KVD,
                                       int NH, int group, float scale, float* __restrict__ out) {
  extern __shared__ float smem[];
  float* qs = smem;                            // [kAttnBM][kAttnQS]
  float* ks = qs + kAttnBM * kAttnQS;          // [kAttnBK][kAttnQS]
  float* vs = ks + kAttnBK * kAttnQS;          // [kAttnBK][kAttnVS]
  float* sc = vs + kAttnBK * kAttnVS;          // [kAttnBM][kAttnBK]
  float* rmax = sc + kAttnBM * kAttnBK;        // [kAttnBM]
  float* rsum = rmax + kAttnBM;                // [kAttnBM]
  float* rsc = rsum + kAttnBM;                 // [kAttnBM]
  float* tmax = rsc + kAttnBM;                 // [kAttnBM]

  const int q0 = blockIdx.x * kAttnBM;
  const int bh = blockIdx.y;
  const int b = bh / NH;
  const int h = bh - b * NH;
  const int kv = h / group;
  const int bm = min(kAttnBM, seq - q0);
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const float* qbase = q + static_cast<size_t>(b * seq + q0) * QD + static_cast<size_t>(h) * kAttnD;
  const float* kbase = k + static_cast<size_t>(b * seq) * KVD + static_cast<size_t>(kv) * kAttnD;
  const float* vbase = v + static_cast<size_t>(b * seq) * KVD + static_cast<size_t>(kv) * kAttnD;
  float* obase = out + static_cast<size_t>(b * seq + q0) * QD + static_cast<size_t>(h) * kAttnD;

  for (int i = tid; i < bm * kAttnD; i += nthreads) {
    const int r = i / kAttnD, d = i - r * kAttnD;
    qs[r * kAttnQS + d] = qbase[static_cast<size_t>(r) * QD + d];
  }
  if (tid < kAttnBM) {
    rmax[tid] = -INFINITY;
    rsum[tid] = 0.f;
    tmax[tid] = -INFINITY;
  }
  __syncthreads();

  // Register accumulator: thread tid owns query row tid/4 and 32 of its 128 dims.
  const int orow = tid / kAttnTPR;
  const int od = (tid % kAttnTPR) * kAttnDSL;
  float4 acc[kAttnDSL / 4];
#pragma unroll
  for (int i = 0; i < kAttnDSL / 4; ++i) acc[i] = make_float4(0.f, 0.f, 0.f, 0.f);

  for (int t0 = 0; t0 < seq; t0 += kAttnBK) {
    const int bk = min(kAttnBK, seq - t0);
    __syncthreads();
    for (int i = tid; i < bk * kAttnD; i += nthreads) {
      const int r = i / kAttnD, d = i - r * kAttnD;
      const size_t src = static_cast<size_t>(t0 + r) * KVD + d;
      ks[r * kAttnQS + d] = kbase[src];
      vs[r * kAttnVS + d] = vbase[src];
    }
    __syncthreads();

    for (int idx = tid; idx < bm * bk; idx += nthreads) {
      const int r = idx / bk, j = idx - r * bk;
      const float* qr = qs + r * kAttnQS;
      const float* kr = ks + j * kAttnQS;
      // Four independent accumulator chains: a single chain stalls on the 4-cycle
      // FMA latency and the block only has four warps to hide it.
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll 8
      for (int d = 0; d < kAttnD; d += 4) {
        a0 = fmaf(qr[d + 0], kr[d + 0], a0);
        a1 = fmaf(qr[d + 1], kr[d + 1], a1);
        a2 = fmaf(qr[d + 2], kr[d + 2], a2);
        a3 = fmaf(qr[d + 3], kr[d + 3], a3);
      }
      float val = ((a0 + a1) + (a2 + a3)) * scale;
      if (bias) val += bias[static_cast<size_t>(b * seq + q0 + r) * seq + t0 + j];
      sc[r * kAttnBK + j] = val;
    }
    __syncthreads();

    if (tid < bm) {
      float mx = -INFINITY;
      for (int j = 0; j < bk; ++j) mx = fmaxf(mx, sc[tid * kAttnBK + j]);
      if (mx > tmax[tid]) tmax[tid] = mx;
    }
    __syncthreads();
    if (tid < bm) {
      const float m_old = rmax[tid], m_new = tmax[tid];
      rsc[tid] = (m_old <= -1e29f) ? 0.f : __expf(m_old - m_new);
      rmax[tid] = m_new;
    }
    __syncthreads();

    const float rs = (orow < bm) ? rsc[orow] : 1.f;
#pragma unroll
    for (int i = 0; i < kAttnDSL / 4; ++i) {
      acc[i].x *= rs; acc[i].y *= rs; acc[i].z *= rs; acc[i].w *= rs;
    }
    if (tid < bm) {
      const float m_new = rmax[tid];
      float sum = 0.f;
      float* row = sc + tid * kAttnBK;
      for (int j = 0; j < bk; ++j) {
        const float e = (m_new <= -1e29f) ? 0.f : __expf(row[j] - m_new);
        row[j] = e;
        sum += e;
      }
      rsum[tid] = rsum[tid] * rsc[tid] + sum;
    }
    __syncthreads();

    if (orow < bm) {
      const float* prow = sc + orow * kAttnBK;
      const float* vrow = vs + od;
      for (int j = 0; j < bk; ++j) {
        const float p = prow[j];
        const float4* vv = reinterpret_cast<const float4*>(vrow + j * kAttnVS);
#pragma unroll
        for (int i = 0; i < kAttnDSL / 4; ++i) {
          acc[i].x = fmaf(p, vv[i].x, acc[i].x);
          acc[i].y = fmaf(p, vv[i].y, acc[i].y);
          acc[i].z = fmaf(p, vv[i].z, acc[i].z);
          acc[i].w = fmaf(p, vv[i].w, acc[i].w);
        }
      }
    }
  }

  if (orow < bm) {
    const float l = rsum[orow];
    const float inv = (l > 0.f) ? 1.f / l : 0.f;
    float4* o = reinterpret_cast<float4*>(obase + static_cast<size_t>(orow) * QD + od);
#pragma unroll
    for (int i = 0; i < kAttnDSL / 4; ++i) {
      o[i].x = acc[i].x * inv;
      o[i].y = acc[i].y * inv;
      o[i].z = acc[i].z * inv;
      o[i].w = acc[i].w * inv;
    }
  }
}

// Returns false (and leaves `dout` untouched) when this kernel cannot run.
bool attention_fused(const float* dq, const float* dk, const float* dv, const float* dbias,
                     int batch, int seq, int D, int QD, int KVD, int NH, int NKV, float scale,
                     float* dout) {
  static const bool disabled = std::getenv("OV_ATTN_CUBLAS") != nullptr;
  if (disabled || D != kAttnD || seq <= 0) return false;
  static bool smem_ready = false;
  if (!smem_ready) {
    if (cudaFuncSetAttribute(attention_fused_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kAttnSmem) !=
        cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    smem_ready = true;
  }
  const int gx = (seq + kAttnBM - 1) / kAttnBM;
  attention_fused_kernel<<<dim3(gx, batch * NH), kAttnBM * kAttnTPR, kAttnSmem>>>(
      dq, dk, dv, dbias, seq, QD, KVD, NH, NH / NKV, scale, dout);
  return cudaGetLastError() == cudaSuccess;
}

// GQA attention as three batched cuBLAS calls per (batch, head-in-group):
//
//   scores[s][t] = q[s] . k[t]        (m=seq, n=seq, k=D, opA=T on k, opB=N on q)
//   P            = softmax(scores + bias)
//   out[s][d]    = sum_t P[s][t] v[t] (m=D, n=seq, k=seq, both operands op=N)
//
// The batch dimension is the KV head, so all 8 KV heads of one query head are
// computed by a single strided-batched call.  The previous one-block-per-head
// kernel is 6x (seq=109) to 22x (seq=256) slower: 32 blocks on 128 SMs, fully
// uncoalesced K reads (stride KVD between adjacent threads) and a PV pass that
// left 3/4 of each block idle.
void attention_batched(cublasHandle_t blas, const float* dq, const float* dk, const float* dv,
                       const float* dbias, int batch, int seq, int D, int QD, int KVD, int NH,
                       int NKV, float scale, float* dout, float* dscr) {
  if (attention_fused(dq, dk, dv, dbias, batch, seq, D, QD, KVD, NH, NKV, scale, dout)) return;
  const int G = NH / NKV;
  const float one = 1.f, zero = 0.f;
  for (int b = 0; b < batch; ++b) {
    const float* bias_b = dbias ? dbias + static_cast<size_t>(b) * seq * seq : nullptr;
    for (int g = 0; g < G; ++g) {
      for (int kv = 0; kv < NKV; ++kv) {
        const int h = kv * G + g;
        const float* qg = dq + static_cast<size_t>(b) * seq * QD + static_cast<size_t>(h) * D;
        const float* kg = dk + static_cast<size_t>(b) * seq * KVD + static_cast<size_t>(kv) * D;
        const float* vg = dv + static_cast<size_t>(b) * seq * KVD + static_cast<size_t>(kv) * D;
        float* sc = dscr + (static_cast<size_t>(b) * NH + h) * seq * seq;
        // scores[key][query] in memory, read back as [query][key] rows.
        OV_CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, seq, seq, D, &scale, kg, KVD,
                                    qg, QD, &zero, sc, seq));
        const int threads = seq < 128 ? 64 : 256;
        softmax_bias_rows_kernel<<<seq, threads>>>(sc, bias_b, seq);
        float* og = dout + static_cast<size_t>(b) * seq * QD + static_cast<size_t>(h) * D;
        OV_CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, D, seq, seq, &one, vg, KVD, sc,
                                    seq, &zero, og, QD));
      }
    }
  }
}


// ----------------------------------------------------------------- guided sampling
//
// The per-step "log_softmax(cond) + guidance + log_softmax(mix) + argmax" pass used
// to run on the host over the full [B, C, S, V] logits: a 21 MB device-to-host copy
// at seq=325, a host permute, and ~9M scalar exp() calls per step.  Here it runs on
// the device and only C*T values come back.
__device__ __forceinline__ float block_reduce_max(float v, float* red, int tid, int n) {
  red[tid] = v;
  __syncthreads();
  for (int st = n / 2; st > 0; st >>= 1) {
    if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
    __syncthreads();
  }
  const float r = red[0];
  __syncthreads();
  return r;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* red, int tid, int n) {
  red[tid] = v;
  __syncthreads();
  for (int st = n / 2; st > 0; st >>= 1) {
    if (tid < st) red[tid] += red[tid + st];
    __syncthreads();
  }
  const float r = red[0];
  __syncthreads();
  return r;
}

// One block per (codebook, target frame).
__global__ void guided_sample_kernel(const float* __restrict__ cond, int cond_seq,
                                     const float* __restrict__ uncond, int uncond_seq,
                                     int codebooks, int vocab, int target_len, float guidance,
                                     float layer_penalty, int mask_id, int32_t* __restrict__ pred,
                                     float* __restrict__ scores) {
  const int idx = blockIdx.x;
  const int c = idx / target_len;
  const int t = idx - c * target_len;
  const int CV = codebooks * vocab;
  // Both planes are token-major [S, C*V]: cond at its last `target_len` positions,
  // uncond at its first `target_len` positions (the CFG batch runs as two forwards).
  const float* lc =
      cond + static_cast<size_t>(cond_seq - target_len + t) * CV + static_cast<size_t>(c) * vocab;
  const float* lu = uncond + static_cast<size_t>(t) * CV + static_cast<size_t>(c) * vocab;
  (void)uncond_seq;

  extern __shared__ float sm[];
  float* lp = sm;                  // [vocab] running log-prob / mix
  float* red = lp + vocab;         // [blockDim]
  const int tid = threadIdx.x;
  const int n = blockDim.x;

  float m = -INFINITY;
  for (int j = tid; j < vocab; j += n) m = fmaxf(m, lc[j]);
  m = block_reduce_max(m, red, tid, n);
  float sum = 0.f;
  for (int j = tid; j < vocab; j += n) sum += __expf(lc[j] - m);
  sum = block_reduce_sum(sum, red, tid, n);
  const float lse_c = __logf(sum) + m;

  m = -INFINITY;
  for (int j = tid; j < vocab; j += n) m = fmaxf(m, lu[j]);
  m = block_reduce_max(m, red, tid, n);
  sum = 0.f;
  for (int j = tid; j < vocab; j += n) sum += __expf(lu[j] - m);
  sum = block_reduce_sum(sum, red, tid, n);
  const float lse_u = __logf(sum) + m;

  float m2 = -INFINITY;
  for (int j = tid; j < vocab; j += n) {
    const float cj = lc[j] - lse_c;
    const float v = cj + guidance * (cj - (lu[j] - lse_u));
    lp[j] = v;
    m2 = fmaxf(m2, v);
  }
  m2 = block_reduce_max(m2, red, tid, n);
  float s2 = 0.f;
  for (int j = tid; j < vocab; j += n) s2 += __expf(lp[j] - m2);
  s2 = block_reduce_sum(s2, red, tid, n);
  const float lse2 = __logf(s2) + m2;

  // argmax over everything except the mask token (the reference sets
  // lp_mix[mask] = -inf after the log-softmax, so it is excluded here, not from lse2)
  float best = -INFINITY;
  int bestj = 0;
  for (int j = tid; j < vocab; j += n) {
    if (j == mask_id) continue;
    const float v = lp[j] - lse2;
    if (v > best) { best = v; bestj = j; }
  }
  // reduce (value, index) pairs
  __shared__ float sval[2][256];
  __shared__ int sidx[2][256];
  int cur = 0;
  sval[cur][tid] = best;
  sidx[cur][tid] = bestj;
  __syncthreads();
  for (int st = n / 2; st > 0; st >>= 1) {
    if (tid < st) {
      const bool take = sval[cur][tid + st] > sval[cur][tid];
      sval[cur][tid] = take ? sval[cur][tid + st] : sval[cur][tid];
      sidx[cur][tid] = take ? sidx[cur][tid + st] : sidx[cur][tid];
    }
    __syncthreads();
  }
  if (tid == 0) {
    pred[idx] = sidx[cur][0];
    scores[idx] = sval[cur][0] - static_cast<float>(c) * layer_penalty;
  }
}

}  // namespace

// Grow-only device arena.  forward() used to do 15 cudaMalloc / cudaFree pairs per
// call, and cudaMalloc implicitly synchronises the device (1.5-2 ms per forward,
// i.e. ~60 ms over a 32-step decode).
class Arena {
 public:
  ~Arena() { if (base_) cudaFree(base_); }
  void reserve(size_t bytes) {
    if (bytes <= cap_) return;
    if (base_) OV_CUDA_CHECK(cudaFree(base_));
    const size_t want = bytes + bytes / 4 + (1u << 20);
    OV_CUDA_CHECK(cudaMalloc(&base_, want));
    cap_ = want;
    used_ = 0;
  }
  void reset() { used_ = 0; }
  template <typename T>
  T* get(size_t n) {
    const size_t off = (used_ + 255) & ~static_cast<size_t>(255);
    used_ = off + n * sizeof(T);
    if (used_ > cap_) throw std::runtime_error("cuda arena overflow");
    return reinterpret_cast<T*>(static_cast<char*>(base_) + off);
  }

 private:
  void* base_ = nullptr;
  size_t cap_ = 0, used_ = 0;
};

class LmRunner {
 public:
  explicit LmRunner(const WeightFile& wf, const LMConfig& cfg) : cfg_(cfg) {
    int dev = 0;
    OV_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    OV_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::fprintf(stderr, "[cuda] device %d: %s (%.1f GB, %d SMs)\n", dev, prop.name,
                 prop.totalGlobalMem / 1e9, prop.multiProcessorCount);
    OV_CUBLAS_CHECK(cublasCreate(&blas_));
    // TF32 tensor cores for the int8-dequantised checkpoint.  The fp32
    // checkpoint keeps the exact SGEMM path (its reference test is near-bit-exact);
    // int8 is already approximate and is judged by the int8 gate.
    combine_mlp_ = (wf.dtype_mode() == 1) && std::getenv("OV_CUDA_NO_MERGE") == nullptr;
    const char* fp16_env = std::getenv("OV_CUDA_FP16");
    use_half_ = combine_mlp_ && (!fp16_env || (fp16_env[0] && fp16_env[0] != '0'));
    const char* tf32_env = std::getenv("OV_CUDA_TF32");
    const bool tf32 = tf32_env ? (tf32_env[0] && tf32_env[0] != '0') : (wf.dtype_mode() == 1);
    if (tf32) {
      OV_CUBLAS_CHECK(cublasSetMathMode(blas_, CUBLAS_TF32_TENSOR_OP_MATH));
      std::fprintf(stderr, "[cuda] TF32 tensor-core math enabled\n");
    }
    load_weights(wf);
  }

  ~LmRunner() {
    for (auto& L : layers_) {
      cudaFree(const_cast<float*>(L.qkv));
      cudaFree(const_cast<float*>(L.gate_up));
      cudaFree(const_cast<__half*>(L.qkv_h));
      cudaFree(const_cast<__half*>(L.gate_up_h));
      cudaFree(const_cast<__half*>(L.o_h));
      cudaFree(const_cast<__half*>(L.down_h));
    }
    for (auto& kv : cache_) cudaFree(kv.second);
    if (blas_) cublasDestroy(blas_);
  }

  const LMConfig& cfg() const { return cfg_; }
  cublasHandle_t blas() const { return blas_; }

  const __half* to_half(const float* src, int64_t n) {
    __half* h = nullptr;
    OV_CUDA_CHECK(cudaMalloc(&h, sizeof(__half) * n));
    float_to_half_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(src, h, n);
    OV_CUDA_CHECK(cudaGetLastError());
    return h;
  }

  // Dequantises (if needed) and uploads a weight once; returns device pointer.
  const float* weight(const Tensor& t) {
    auto it = cache_.find(t.data);
    if (it != cache_.end()) return it->second;
    const int64_t n = t.shape.numel();
    float* d = nullptr;
    OV_CUDA_CHECK(cudaMalloc(&d, sizeof(float) * n));
    if (t.dtype == DType::F32) {
      OV_CUDA_CHECK(cudaMemcpy(d, t.data, sizeof(float) * n, cudaMemcpyHostToDevice));
    } else if (t.dtype == DType::I8) {
      // Upload the int8 payload as-is and expand it on the device (4x less PCIe).
      const int8_t* q = t.i8();
      int8_t* d_q = nullptr;
      float* d_s = nullptr;
      OV_CUDA_CHECK(cudaMalloc(&d_q, static_cast<size_t>(n)));
      OV_CUDA_CHECK(cudaMalloc(&d_s, sizeof(float) * static_cast<size_t>(t.n_scales)));
      OV_CUDA_CHECK(cudaMemcpy(d_q, q, static_cast<size_t>(n), cudaMemcpyHostToDevice));
      OV_CUDA_CHECK(cudaMemcpy(d_s, t.scales, sizeof(float) * static_cast<size_t>(t.n_scales),
                               cudaMemcpyHostToDevice));
      const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
      if (t.transposed) {
        // [K, N] with one scale per column
        dequant_col_kernel<<<blocks, 256>>>(d_q, d_s, t.shape[0], t.shape[1], d);
      } else {
        dequant_row_kernel<<<blocks, 256>>>(d_q, d_s, t.shape[0], t.shape[1], d);
      }
      OV_CUDA_CHECK(cudaDeviceSynchronize());
      OV_CUDA_CHECK(cudaFree(d_q));
      OV_CUDA_CHECK(cudaFree(d_s));
    } else {
      throw std::runtime_error("unsupported weight dtype on CUDA");
    }
    cache_.emplace(t.data, d);
    return d;
  }

  Arena& arena() { return ws_; }

  const float* last_logits() const { return last_logits_; }
  int last_batch() const { return last_batch_; }
  int last_seq() const { return last_seq_; }
  void set_skip_host_logits(bool v) { skip_host_logits_ = v; }

  void forward(int batch, int seq, const int64_t* input_ids, const uint8_t* audio_mask,
               const uint8_t* attn_mask, const int64_t* position_ids, float* logits);

 private:
  void load_weights(const WeightFile& wf);

  struct Layer {
    const float* q = nullptr;
    const float* k = nullptr;
    const float* v = nullptr;
    const float* qkv = nullptr;  // combined [H, QD+2*KVD] (int8 checkpoints)
    const __half* qkv_h = nullptr;
    const __half* o_h = nullptr;
    const __half* down_h = nullptr;
    const __half* gate_up_h = nullptr;
    const float* o = nullptr;
    const float* gate = nullptr;
    const float* up = nullptr;
    const float* gate_up = nullptr;  // combined [H, 2*INTER] (int8 checkpoints)
    const float* down = nullptr;
    const float* ln1 = nullptr;
    const float* ln2 = nullptr;
    const float* qn = nullptr;
    const float* kn = nullptr;
  };

  LMConfig cfg_;
  cublasHandle_t blas_ = nullptr;
  std::unordered_map<const void*, float*> cache_;
  Arena ws_;
  const float* last_logits_ = nullptr;
  int last_batch_ = 0;
  int last_seq_ = 0;
  bool skip_host_logits_ = false;
  bool combine_mlp_ = false;
  bool use_half_ = false;

  const float* tok_emb_ = nullptr;
  const float* aud_emb_ = nullptr;
  const float* audio_heads_ = nullptr;
  const float* final_norm_ = nullptr;
  std::vector<Layer> layers_;
};

void LmRunner::load_weights(const WeightFile& wf) {
  tok_emb_ = weight(wf.Get("llm.embed_tokens.weight"));
  aud_emb_ = weight(wf.Get("audio_embeddings.weight"));
  audio_heads_ = weight(wf.Get("audio_heads.weight"));
  final_norm_ = weight(wf.Get("llm.norm.weight"));
  layers_.resize(static_cast<size_t>(cfg_.n_layers));
  for (int i = 0; i < cfg_.n_layers; ++i) {
    const std::string p = "llm.layers." + std::to_string(i) + ".";
    Layer& L = layers_[static_cast<size_t>(i)];
    L.q = weight(wf.Get(p + "self_attn.q_proj.weight"));
    L.k = weight(wf.Get(p + "self_attn.k_proj.weight"));
    L.v = weight(wf.Get(p + "self_attn.v_proj.weight"));
    if (combine_mlp_) {
      const int64_t qd = static_cast<int64_t>(cfg_.n_heads) * cfg_.head_dim;
      const int64_t kvd = static_cast<int64_t>(cfg_.n_kv_heads) * cfg_.head_dim;
      const int64_t stride = qd + 2 * kvd;
      float* qkv = nullptr;
      OV_CUDA_CHECK(cudaMalloc(&qkv, sizeof(float) * cfg_.hidden * stride));
      OV_CUDA_CHECK(cudaMemcpy2D(qkv, sizeof(float) * stride, L.q, sizeof(float) * qd,
                                 sizeof(float) * qd, cfg_.hidden, cudaMemcpyDeviceToDevice));
      OV_CUDA_CHECK(cudaMemcpy2D(qkv + qd, sizeof(float) * stride, L.k, sizeof(float) * kvd,
                                 sizeof(float) * kvd, cfg_.hidden, cudaMemcpyDeviceToDevice));
      OV_CUDA_CHECK(cudaMemcpy2D(qkv + qd + kvd, sizeof(float) * stride, L.v, sizeof(float) * kvd,
                                 sizeof(float) * kvd, cfg_.hidden, cudaMemcpyDeviceToDevice));
      L.qkv = qkv;
      if (use_half_) L.qkv_h = to_half(qkv, cfg_.hidden * stride);
    }
    L.o = weight(wf.Get(p + "self_attn.o_proj.weight"));
    if (use_half_)
      L.o_h = to_half(L.o, static_cast<int64_t>(cfg_.hidden) * static_cast<int64_t>(cfg_.n_heads) *
                               cfg_.head_dim);
    L.gate = weight(wf.Get(p + "mlp.gate_proj.weight"));
    L.up = weight(wf.Get(p + "mlp.up_proj.weight"));
    if (combine_mlp_) {
      const int64_t inter = cfg_.inter;
      float* gu = nullptr;
      OV_CUDA_CHECK(cudaMalloc(&gu, sizeof(float) * cfg_.hidden * 2 * inter));
      OV_CUDA_CHECK(cudaMemcpy2D(gu, sizeof(float) * 2 * inter, L.gate, sizeof(float) * inter,
                                 sizeof(float) * inter, cfg_.hidden, cudaMemcpyDeviceToDevice));
      OV_CUDA_CHECK(cudaMemcpy2D(gu + inter, sizeof(float) * 2 * inter, L.up,
                                 sizeof(float) * inter, sizeof(float) * inter, cfg_.hidden,
                                 cudaMemcpyDeviceToDevice));
      L.gate_up = gu;
      if (use_half_) L.gate_up_h = to_half(gu, cfg_.hidden * 2 * inter);
    }
    L.down = weight(wf.Get(p + "mlp.down_proj.weight"));
    if (use_half_)
      L.down_h = to_half(L.down, static_cast<int64_t>(cfg_.inter) * cfg_.hidden);
    L.ln1 = weight(wf.Get(p + "input_layernorm.weight"));
    L.ln2 = weight(wf.Get(p + "post_attention_layernorm.weight"));
    L.qn = weight(wf.Get(p + "self_attn.q_norm.weight"));
    L.kn = weight(wf.Get(p + "self_attn.k_norm.weight"));
  }
}

namespace {

// Row-major C[M,N] = A[M,K] · W[K,N]  (W already row-major [K,N]).
inline void gemm(cublasHandle_t h, int M, int N, int K, const float* A, const float* W, float* C,
                 float beta = 0.f) {
  const float alpha = 1.f;
  OV_CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, W, N, A, K, &beta, C,
                              N));
}

}  // namespace

namespace {
struct CudaStageProf {
  bool on = false;
  std::vector<cudaEvent_t> ev;
  std::vector<const char*> lab;
  void mark(const char* l) {
    if (!on) return;
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e);
    ev.push_back(e);
    lab.push_back(l);
  }
  void finish() {
    if (!on || ev.empty()) return;
    cudaEventSynchronize(ev.back());
    std::map<std::string, double> sum;
    for (size_t i = 1; i < ev.size(); ++i) {
      float ms = 0;
      cudaEventElapsedTime(&ms, ev[i - 1], ev[i]);
      sum[lab[i]] += ms;
    }
    std::fprintf(stderr, "[cuda-stage]");
    for (auto& kv : sum) std::fprintf(stderr, " %s=%.2f", kv.first.c_str(), kv.second);
    std::fprintf(stderr, "\n");
    for (auto e : ev) cudaEventDestroy(e);
    ev.clear();
    lab.clear();
  }
};
inline void gemm_half(cublasHandle_t h, int M, int N, int K, const __half* A, const __half* W,
                      float* C, float beta = 0.f, int ldw = -1) {
  const float alpha = 1.f;
  const int lw = ldw > 0 ? ldw : N;
  OV_CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, lw, A,
                               CUDA_R_16F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

}  // namespace

void LmRunner::forward(int batch, int seq, const int64_t* input_ids, const uint8_t* audio_mask,
                       const uint8_t* attn_mask, const int64_t* position_ids, float* logits) {
  static const bool prof = std::getenv("OV_CUDA_PROFILE") != nullptr;
  CudaStageProf sp;
  sp.on = prof;
  sp.mark("start");
  const int H = cfg_.hidden, NH = cfg_.n_heads, NKV = cfg_.n_kv_heads, D = cfg_.head_dim;
  const int C = cfg_.n_codebooks, V = cfg_.audio_vocab, INTER = cfg_.inter;
  const int64_t T = static_cast<int64_t>(batch) * seq;
  const int QD = NH * D, KVD = NKV * D;
  const int half = D / 2;

  // One arena for the whole forward: sized from (batch, seq), reused across calls.
  const size_t n_scr = static_cast<size_t>(batch) * NH * seq * seq;
  const size_t n_kv = static_cast<size_t>(batch) * NH * seq * D;
  ws_.reserve(sizeof(float) * (4 * T * H + T * (QD + 2 * KVD) + 2 * T * QD + 2 * T * KVD +
                               3 * T * INTER + T * C * V +
                               2 * T * seq + 2 * T * half + n_scr + 2 * n_kv) +
              (use_half_ ? sizeof(__half) * T * (H + QD + INTER) : 0) +
              sizeof(int64_t) * (T * C + T) + sizeof(uint8_t) * (T + T * seq));
  ws_.reset();

  int64_t* d_ids = ws_.get<int64_t>(static_cast<size_t>(T) * C);
  uint8_t* d_mask = ws_.get<uint8_t>(static_cast<size_t>(T));
  uint8_t* d_amask_raw = ws_.get<uint8_t>(static_cast<size_t>(T) * seq);
  float* d_bias = ws_.get<float>(static_cast<size_t>(T) * seq);
  int64_t* d_pos = ws_.get<int64_t>(static_cast<size_t>(T));
  float* d_x = ws_.get<float>(static_cast<size_t>(T) * H);
  float* d_xn = ws_.get<float>(static_cast<size_t>(T) * H);
  __half* d_xn_h = use_half_ ? ws_.get<__half>(static_cast<size_t>(T) * H) : nullptr;
  const int64_t QKV = QD + 2 * KVD;
  float* d_qkv = combine_mlp_ ? ws_.get<float>(static_cast<size_t>(T) * QKV) : nullptr;
  float* d_q = ws_.get<float>(static_cast<size_t>(T) * QD);
  float* d_k = ws_.get<float>(static_cast<size_t>(T) * KVD);
  float* d_v = ws_.get<float>(static_cast<size_t>(T) * KVD);
  float* d_attn = ws_.get<float>(static_cast<size_t>(T) * QD);
  __half* d_attn_h = use_half_ ? ws_.get<__half>(static_cast<size_t>(T) * QD) : nullptr;
  float* d_gate = ws_.get<float>(static_cast<size_t>(T) * INTER);
  __half* d_gate_h = use_half_ ? ws_.get<__half>(static_cast<size_t>(T) * INTER) : nullptr;
  float* d_up = (combine_mlp_ && !use_half_) ? nullptr : ws_.get<float>(static_cast<size_t>(T) * INTER);
  float* d_gu = (combine_mlp_ && !use_half_) ? ws_.get<float>(static_cast<size_t>(T) * 2 * INTER) : nullptr;
  float* d_logits = ws_.get<float>(static_cast<size_t>(T) * C * V);
  float* d_scr = ws_.get<float>(n_scr);
  float* d_ke = ws_.get<float>(n_kv);
  float* d_ve = ws_.get<float>(n_kv);
  float* d_rope = ws_.get<float>(static_cast<size_t>(T) * half * 2);

  OV_CUDA_CHECK(cudaMemcpy(d_ids, input_ids, sizeof(int64_t) * T * C, cudaMemcpyHostToDevice));
  OV_CUDA_CHECK(cudaMemcpy(d_mask, audio_mask, sizeof(uint8_t) * T, cudaMemcpyHostToDevice));
  OV_CUDA_CHECK(cudaMemcpy(d_pos, position_ids, sizeof(int64_t) * T, cudaMemcpyHostToDevice));
  OV_CUDA_CHECK(cudaMemcpy(d_amask_raw, attn_mask, sizeof(uint8_t) * T * seq, cudaMemcpyHostToDevice));
  {
    const int64_t n = T * seq;
    mask_to_bias_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(d_amask_raw, d_bias, n);
  }
  {
    const int64_t n = T * half;
    rope_table_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(d_rope, d_pos,
                                                                       static_cast<int>(T), half, D,
                                                                       cfg_.rope_theta);
  }

  embed_kernel<<<static_cast<unsigned>(T), 256>>>(d_ids, d_mask, tok_emb_, aud_emb_, batch, seq, C,
                                                  V, H, d_x);
  sp.mark("prep+embed");

  const float scale = 1.f / std::sqrt(static_cast<float>(D));
  for (int li = 0; li < cfg_.n_layers; ++li) {
    const Layer& L = layers_[static_cast<size_t>(li)];
    if (use_half_) {
      rmsnorm_half_kernel<<<static_cast<unsigned>(T), 256>>>(d_x, L.ln1, static_cast<int>(T), H,
                                                            cfg_.rms_eps, d_xn_h);
    } else {
      rmsnorm_kernel<<<static_cast<unsigned>(T), 256>>>(d_x, L.ln1, static_cast<int>(T), H,
                                                        cfg_.rms_eps, d_xn);
    }
    if (use_half_) {
      const int lw = static_cast<int>(QKV);
      gemm_half(blas_, static_cast<int>(T), QD, H, d_xn_h, L.qkv_h, d_q, 0.f, lw);
      gemm_half(blas_, static_cast<int>(T), KVD, H, d_xn_h, L.qkv_h + QD, d_k, 0.f, lw);
      gemm_half(blas_, static_cast<int>(T), KVD, H, d_xn_h, L.qkv_h + QD + KVD, d_v, 0.f, lw);
    } else if (combine_mlp_) {
      gemm(blas_, static_cast<int>(T), static_cast<int>(QKV), H, d_xn, L.qkv, d_qkv);
      const int64_t n = T * QKV;
      split_qkv_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(
          d_qkv, static_cast<int>(T), static_cast<int>(QD), static_cast<int>(KVD), d_q, d_k, d_v);
    } else {
      gemm(blas_, static_cast<int>(T), QD, H, d_xn, L.q, d_q);
      gemm(blas_, static_cast<int>(T), KVD, H, d_xn, L.k, d_k);
      gemm(blas_, static_cast<int>(T), KVD, H, d_xn, L.v, d_v);
    }
    sp.mark("qkv");

    // per-head RMSNorm + RoPE fused in one pass.
    norm_rope_heads_kernel<<<static_cast<unsigned>(T * NH), 128>>>(
        d_q, L.qn, d_rope, static_cast<int>(T), NH, D, cfg_.rms_eps);
    norm_rope_heads_kernel<<<static_cast<unsigned>(T * NKV), 128>>>(
        d_k, L.kn, d_rope, static_cast<int>(T), NKV, D, cfg_.rms_eps);
    sp.mark("qk_norm_rope");

    // Small S: the fused scalar kernel wins (fewer launches, no score round trip).
    // Large S: the strided-batched TF32 GEMMs win.
    const bool force_fused = std::getenv("OV_ATTN_FUSED") != nullptr;
    const bool force_batched = std::getenv("OV_ATTN_BATCHED") != nullptr;
    if (!force_fused && (force_batched || seq >= 160)) {
      attention_batched_fast(blas_, d_q, d_k, d_v, d_bias, batch, seq, D, QD, KVD, NH, NKV, scale,
                             d_attn, d_ke, d_ve, d_scr);
    } else {
      attention_batched(blas_, d_q, d_k, d_v, d_bias, batch, seq, D, QD, KVD, NH, NKV, scale,
                        d_attn, d_scr);
    }
    sp.mark("attn");

    // x += o_proj(attn) with beta=1: one call and one pass over x instead of a
    // separate GEMM buffer plus a cublasSaxpy.
    if (use_half_) {
      const int64_t n = T * QD;
      float_to_half_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(d_attn, d_attn_h, n);
      gemm_half(blas_, static_cast<int>(T), H, QD, d_attn_h, L.o_h, d_x, /*beta=*/1.f);
    } else {
      gemm(blas_, static_cast<int>(T), H, QD, d_attn, L.o, d_x, /*beta=*/1.f);
    }
    sp.mark("o_proj");

    if (use_half_) {
      rmsnorm_half_kernel<<<static_cast<unsigned>(T), 256>>>(d_x, L.ln2, static_cast<int>(T), H,
                                                            cfg_.rms_eps, d_xn_h);
    } else {
      rmsnorm_kernel<<<static_cast<unsigned>(T), 256>>>(d_x, L.ln2, static_cast<int>(T), H,
                                                        cfg_.rms_eps, d_xn);
    }
    if (use_half_) {
      const int lw = 2 * INTER;
      gemm_half(blas_, static_cast<int>(T), INTER, H, d_xn_h, L.gate_up_h, d_gate, 0.f, lw);
      gemm_half(blas_, static_cast<int>(T), INTER, H, d_xn_h, L.gate_up_h + INTER, d_up, 0.f, lw);
      sp.mark("gate_up");
      silu_mul_half_kernel<<<static_cast<unsigned>((T * INTER + 255) / 256), 256>>>(
          d_gate, d_up, d_gate_h, T * INTER);
    } else if (combine_mlp_) {
      gemm(blas_, static_cast<int>(T), 2 * INTER, H, d_xn, L.gate_up, d_gu);
      sp.mark("gate_up");
      const dim3 sgrid(static_cast<unsigned>((INTER + 255) / 256), static_cast<unsigned>(T));
      silu_mul_combined_kernel<<<sgrid, 256>>>(d_gu, static_cast<int>(T), INTER, d_gate);
    } else {
      gemm(blas_, static_cast<int>(T), INTER, H, d_xn, L.gate, d_gate);
      gemm(blas_, static_cast<int>(T), INTER, H, d_xn, L.up, d_up);
      sp.mark("gate_up");
      silu_mul_kernel<<<static_cast<unsigned>((T * INTER + 255) / 256), 256>>>(d_gate, d_up,
                                                                              T * INTER);
    }
    sp.mark("silu");
    if (use_half_) {
      gemm_half(blas_, static_cast<int>(T), H, INTER, d_gate_h, L.down_h, d_x, /*beta=*/1.f);
    } else {
      gemm(blas_, static_cast<int>(T), H, INTER, d_gate, L.down, d_x, /*beta=*/1.f);
    }
    sp.mark("down");
  }

  rmsnorm_kernel<<<static_cast<unsigned>(T == 0 ? 1 : T), 256>>>(d_x, final_norm_,
                                                                 static_cast<int>(T), H,
                                                                 cfg_.rms_eps, d_xn);
  gemm(blas_, static_cast<int>(T), C * V, H, d_xn, audio_heads_, d_logits);
  sp.mark("final_norm+head");
  last_logits_ = d_logits;
  last_batch_ = batch;
  last_seq_ = seq;
  if (skip_host_logits_) {
    sp.finish();
    return;
  }

  // The kernel writes [T, C*V] (token-major); the public contract is
  // [B, C, S, V], so permute during the copy back.
  std::vector<float> flat(static_cast<size_t>(T) * C * V);
  OV_CUDA_CHECK(cudaMemcpy(flat.data(), d_logits, sizeof(float) * flat.size(),
                           cudaMemcpyDeviceToHost));
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < C; ++c) {
      for (int t = 0; t < seq; ++t) {
        const float* src = flat.data() + (static_cast<size_t>(b) * seq + t) * (C * V) + static_cast<size_t>(c) * V;
        float* dst = logits + ((static_cast<size_t>(b) * C + c) * seq + t) * V;
        std::memcpy(dst, src, sizeof(float) * V);
      }
    }
  }
  sp.finish();
}

// Runs the guided-sampling kernel on the logits of the last forward().
const void* LmRunnerLastLogitsThunk(void* ctx) {
  return static_cast<LmRunner*>(ctx)->last_logits();
}

void LmRunnerSampleThunk(void* ctx, const void* cond, int cond_seq, const void* uncond,
                         int uncond_seq, int target_len, float guidance, float layer_penalty,
                         int32_t* pred, float* scores) {
  auto* r = static_cast<LmRunner*>(ctx);
  const int C = r->cfg().n_codebooks, V = r->cfg().audio_vocab;
  const int n = C * target_len;
  int32_t* d_pred = r->arena().get<int32_t>(static_cast<size_t>(n));
  float* d_scores = r->arena().get<float>(static_cast<size_t>(n));
  // The arena was reset by forward(); re-reserve so these two buffers land after it.
  const int threads = 128;
  const size_t smem = (static_cast<size_t>(V) + threads) * sizeof(float);
  static bool smem_ready = false;
  if (!smem_ready) {
    if (cudaFuncSetAttribute(guided_sample_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem)) != cudaSuccess) {
      (void)cudaGetLastError();
      std::memset(pred, 0, sizeof(int32_t) * static_cast<size_t>(n));
      std::memset(scores, 0, sizeof(float) * static_cast<size_t>(n));
      return;
    }
    smem_ready = true;
  }
  guided_sample_kernel<<<n, threads, smem>>>(static_cast<const float*>(cond), cond_seq,
                                            static_cast<const float*>(uncond), uncond_seq, C, V,
                                            target_len, guidance, layer_penalty, 1024, d_pred,
                                            d_scores);
  if (cudaGetLastError() != cudaSuccess) {
    std::memset(pred, 0, sizeof(int32_t) * static_cast<size_t>(n));
    std::memset(scores, 0, sizeof(float) * static_cast<size_t>(n));
    return;
  }
  OV_CUDA_CHECK(cudaMemcpy(pred, d_pred, sizeof(int32_t) * static_cast<size_t>(n),
                           cudaMemcpyDeviceToHost));
  OV_CUDA_CHECK(cudaMemcpy(scores, d_scores, sizeof(float) * static_cast<size_t>(n),
                           cudaMemcpyDeviceToHost));
}

void LmRunnerDelete(LmRunner* runner) { delete runner; }

void LmRunnerSetDeviceSampling(LmRunner& runner, bool on) { runner.set_skip_host_logits(on); }

LmRunnerPtr LmRunnerCreate(const WeightFile& wf, const LMConfig& cfg) {
  try {
    return LmRunnerPtr(new LmRunner(wf, cfg), &LmRunnerDelete);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[cuda] unavailable: %s\n", e.what());
    return LmRunnerPtr(nullptr, &LmRunnerDelete);
  }
}

void LmRunnerForwardThunk(void* ctx, int batch, int seq, const int64_t* input_ids,
                          const uint8_t* audio_mask, const uint8_t* attn_mask,
                          const int64_t* position_ids, float* logits) {
  static_cast<LmRunner*>(ctx)->forward(batch, seq, input_ids, audio_mask, attn_mask,
                                       position_ids, logits);
}

void LmRunnerForward(LmRunner& runner, int batch, int seq, const int64_t* input_ids,
                     const uint8_t* audio_mask, const uint8_t* attn_mask,
                     const int64_t* position_ids, float* logits) {
  runner.forward(batch, seq, input_ids, audio_mask, attn_mask, position_ids, logits);
}

bool Available() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
  return true;
}

}  // namespace cuda
}  // namespace ov

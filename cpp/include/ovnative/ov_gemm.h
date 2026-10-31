// GEMM entry points shared by every backend.
//
// The whole engine only ever needs ONE matmul shape:
//
//     C[M, N] = A[M, K] · W[K, N]        (W row-major, contiguous along N)
//
// because OVW1 stores linear weights pre-transposed ("layout=1").  Keeping a
// single signature means the CUDA backend and the CPU backend are drop-in
// interchangeable behind it.
#pragma once

#include <cstdint>

namespace ov {

// FP32 weights.
void linear_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* W, float* C);

// Same product with an explicit leading dimension for C, optionally accumulated:
//   accumulate ?  C[m,n] += A[m,:]·W[:,n]  :  C[m,n] = A[m,:]·W[:,n]
// `ldc >= N`.  Used by the convolution kernels, which scatter one GEMM result per
// tap into a strided slice of the output instead of materialising an im2col copy.
void linear_f32_ld(int64_t M, int64_t N, int64_t K, const float* A, const float* W, float* C,
                   int64_t ldc, bool accumulate);

// INT8 weights, symmetric per-output-column scale: C[m,n] = scale[n] · Σ A[m,k]·W[k,n].
// Dequantisation happens after accumulation, so no fp32 copy of W is ever built.
void linear_s8(int64_t M, int64_t N, int64_t K, const float* A, const int8_t* W,
               const float* col_scale, float* C);

// INT8 weights stored output-major ([N, K], OVW1 layout=0) with one symmetric
// scale per output row: C[m,n] = row_scale[n] · Σ A[m,k]·W[n,k].
//
// This is the layout AVX-VNNI needs (all K of one output column are contiguous).
// It is a separate entry point because the inner loop is a different machine:
// `linear_s8` dequantises 8 int8 weights into one ymm and runs 8 FMAs, whereas
// the VNNI path consumes 4 k across 8 output columns from a packed tile.
void linear_s8_nk(int64_t M, int64_t N, int64_t K, const float* A, const int8_t* W,
                  const float* row_scale, float* C);

// --- AVX-VNNI preparation for the [N, K] layout -----------------------------
//
// `vpdpbusd` multiplies 4 unsigned bytes of A by 4 signed bytes of W and adds
// into an int32 per 32-bit lane.  One ymm therefore holds 8 output columns and
// 4 consecutive k.  The raw [N, K] rows are repacked once at load into tiles:
//
//   for n0 in 0..N step 8, for g in 0..ceil(K/4)-1:
//     tile[32] = { W[n0..n0+7][4g+l] }  with l (byte in dword) major,
//                 i.e. byte (j*4 + l) = W[n0+j][4g+l].
//
// Activations are quantised to u8 (q + 128, so a zero activation maps to 128)
// in blocks of 32 k values, each block with its own scale.  `wsum` holds the
// per-block weight sums used to remove the +128 zero point:
//
//   wsum[(k/32) * align8(N) + n] = Σ_{k in block} W[n, k]
//
// The epilogue per block is C += (acc - 128*wsum_block) * (ascale_block *
// wscale_n), which keeps the 8k quantisation noise local instead of scaling it
// by the row maximum.  `wsum` must have s8_nk_wsum_elems(N, K) int32.
bool cpu_vnni_available();
int64_t s8_nk_packed_bytes(int64_t N, int64_t K);
int64_t s8_nk_wsum_elems(int64_t N, int64_t K);
void s8_nk_pack(int64_t N, int64_t K, const int8_t* W, int8_t* packed, int32_t* wsum);

// Pre-packed form.  Requires cpu_vnni_available(); Mat only calls it when the
// tiles were built.
void linear_s8_nk_packed(int64_t M, int64_t N, int64_t K, const float* A,
                         const int8_t* packed, const int32_t* wsum,
                         const float* row_scale, float* C);

// C += A·W for the pre-packed VNNI form.  Keeps the residual add inside the
// epilogue instead of a separate pass over the [M,N] result and C.
void linear_s8_nk_packed_add(int64_t M, int64_t N, int64_t K, const float* A,
                             const int8_t* packed, const int32_t* wsum,
                             const float* row_scale, float* C);

// Pre-quantised variant: caller already has aq/asc (e.g. from a fused activation).
void linear_s8_nk_packed_q_add(int64_t M, int64_t N, int64_t K, const uint8_t* aq,
                               const float* asc, const int8_t* packed, const int32_t* wsum,
                               const float* row_scale, float* C);

// out = silu(gate) * up, quantised in the same per-32-k layout as
// s8_nk_quantize(M, K, out, ...) but without materialising `out`.
void silu_mul_quantize_f32(int64_t M, int64_t K, const float* gate, const float* up,
                           uint8_t* aq, float* asc);

// --- Hoisted activation quantisation ---------------------------------------
// Several projections in one layer read the *same* A (q/k/v share the normed
// input, gate/up share the post-norm input), and quantising it costs ~83 us for
// M=218/K=1024 -- about 20% of the GEMM time for the narrow projections.  Split
// it out so the caller quantises once and reuses it.
//
//   aq  : s8_nk_quant_bytes(M, K)    bytes      (u8, 4x8-tile compatible, 32-k blocks)
//   asc : s8_nk_quant_scales(M, K)   floats     (one scale per 32-k block)
int64_t s8_nk_quant_bytes(int64_t M, int64_t K);
int64_t s8_nk_quant_scales(int64_t M, int64_t K);
void s8_nk_quantize(int64_t M, int64_t K, const float* A, uint8_t* aq, float* asc);
void linear_s8_nk_packed_q(int64_t M, int64_t N, int64_t K, const uint8_t* aq, const float* asc,
                           const int8_t* packed, const int32_t* wsum, const float* row_scale,
                           float* C);

// Slice of a combined [N][K] weight group: the packed/wsum/scale arrays are
// concatenations of the slices in order (wsum is block-major with the combined
// Npad), and each slice receives its own C buffer.  N must be a multiple of 64.
struct S8NkSlice {
  int64_t N = 0;
  float* C = nullptr;
};
void linear_s8_nk_packed_q_sliced(int64_t M, int64_t K, const uint8_t* aq, const float* asc,
                                  const int8_t* packed, const int32_t* wsum,
                                  const float* scale, const S8NkSlice* slices, int n_slices);

// Fused RMSNorm + the same per-32-k activation quantisation, so the VNNI path
// never has to materialise the normalised row:
//   xn[r,k] = (x[r,k] * 1/sqrt(mean(x[r,:]^2)+eps)) * w[k]
//   aq/asc  = quantise(xn)                 (bit-identical to rmsnorm + s8_nk_quantize)
// `aq` needs s8_nk_quant_bytes(rows, cols) and `asc` s8_nk_quant_scales(rows, cols).
void rmsnorm_quantize_f32(int64_t rows, int64_t cols, const float* x, const float* w, float eps,
                          uint8_t* aq, float* asc);

// Human-readable kernel that would be used on this CPU ("avx2+fma", "scalar").
const char* cpu_kernel_name();

// While a ScopedSerialGemm is alive on this thread, `linear_*` will not open an
// OpenMP region.  Use it when the *caller* already parallelised over a
// dimension that yields one gemm per thread (e.g. attention heads): spawning a
// 32-thread team for a 4 MFLOP gemm costs more than the gemm itself.
class ScopedSerialGemm {
 public:
  ScopedSerialGemm();
  ~ScopedSerialGemm();
  ScopedSerialGemm(const ScopedSerialGemm&) = delete;
  ScopedSerialGemm& operator=(const ScopedSerialGemm&) = delete;
};

}  // namespace ov

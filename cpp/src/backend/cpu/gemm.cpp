#include "ovnative/ov_gemm.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define OV_X86 1
#endif

namespace ov {

namespace {
thread_local bool t_serial_gemm = false;
}  // namespace

ScopedSerialGemm::ScopedSerialGemm() { t_serial_gemm = true; }
ScopedSerialGemm::~ScopedSerialGemm() { t_serial_gemm = false; }

namespace {

// ---------------------------------------------------------------- reference

void linear_s8_ref(int64_t M, int64_t N, int64_t K, const float* A, const int8_t* W,
                   const float* scale, float* C) {
  for (int64_t i = 0; i < M; ++i) {
    const float* a = A + i * K;
    float* c = C + i * N;
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.f;
      for (int64_t k = 0; k < K; ++k) acc += a[k] * static_cast<float>(W[k * N + j]);
      c[j] = acc * scale[j];
    }
  }
}

// ---------------------------------------------------------------- AVX2/FMA
//
// Two structural choices matter on this workload (M ~ 200-500, N/K ~ 1024-3072):
//
//   * column-block-outer loop order, so a K×64 panel of W (256 KB) stays in L2
//     while every row block streams past it;
//   * 8×8 micro-kernel, so each loaded W vector feeds 8 rows.  With Mr=4 the
//     W panel had to be re-read once per 4 rows and the kernel became L3
//     bandwidth bound: 190 GFLOPS on down_proj vs ~380 after the change.
#ifdef OV_X86
#ifdef _MSC_VER
// MSVC has no function-level target attributes. gemm.cpp / ops_cpu.cpp are
// compiled with /arch:AVX2, and the VNNI block below is disabled on MSVC.
#define OV_AVX2
#define OV_VNNI
#else
#define OV_AVX2 __attribute__((target("avx2,fma")))
#define OV_VNNI __attribute__((target("avx2,fma,avxvnni")))
#endif

constexpr int64_t kJb = 64;  // columns per W panel
constexpr int64_t kMr = 8;   // rows per micro-kernel

OV_AVX2 inline void micro_f32_8x8(int64_t j0, int64_t jj, int64_t i0, int rows, int64_t K, int64_t N,
                                  int64_t ldc, bool accumulate, const float* A, const float* W,
                                  float* C) {
  // `rows` < 8 is the M tail.  The missing rows alias row 0 so the k loop stays
  // branch-free and every load is in bounds; their accumulators are simply not
  // stored.  (The old code zero-padded A and copied C back, which cost ~18% on
  // M=218 because every LM projection goes through it.)
  const int64_t r1 = (rows > 1) ? i0 + 1 : i0;
  const int64_t r2 = (rows > 2) ? i0 + 2 : i0;
  const int64_t r3 = (rows > 3) ? i0 + 3 : i0;
  const int64_t r4 = (rows > 4) ? i0 + 4 : i0;
  const int64_t r5 = (rows > 5) ? i0 + 5 : i0;
  const int64_t r6 = (rows > 6) ? i0 + 6 : i0;
  const int64_t r7 = (rows > 7) ? i0 + 7 : i0;

  const float* a0 = A + (i0 + 0) * K;
  const float* a1 = A + r1 * K;
  const float* a2 = A + r2 * K;
  const float* a3 = A + r3 * K;
  const float* a4 = A + r4 * K;
  const float* a5 = A + r5 * K;
  const float* a6 = A + r6 * K;
  const float* a7 = A + r7 * K;

  __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
  __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
  __m256 c4 = _mm256_setzero_ps(), c5 = _mm256_setzero_ps();
  __m256 c6 = _mm256_setzero_ps(), c7 = _mm256_setzero_ps();

  const float* w = W + j0 + jj;
  for (int64_t k = 0; k < K; ++k, w += N) {
    const __m256 b = _mm256_loadu_ps(w);
    c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0 + k), b, c0);
    c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(a1 + k), b, c1);
    c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(a2 + k), b, c2);
    c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(a3 + k), b, c3);
    c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(a4 + k), b, c4);
    c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(a5 + k), b, c5);
    c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(a6 + k), b, c6);
    c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(a7 + k), b, c7);
  }
  // Unrolled so the eight accumulators stay in registers; `rows` only masks the
  // stores (M tail), and `accumulate` is loop-invariant.
  const int64_t o = j0 + jj;
  float* c = C + (i0 + 0) * ldc + o;
  if (accumulate) c0 = _mm256_add_ps(c0, _mm256_loadu_ps(c));
  _mm256_storeu_ps(c, c0);
  if (rows > 1) {
    c = C + (i0 + 1) * ldc + o;
    if (accumulate) c1 = _mm256_add_ps(c1, _mm256_loadu_ps(c));
    _mm256_storeu_ps(c, c1);
  }
  if (rows > 2) {
    c = C + (i0 + 2) * ldc + o;
    if (accumulate) c2 = _mm256_add_ps(c2, _mm256_loadu_ps(c));
    _mm256_storeu_ps(c, c2);
  }
  if (rows > 3) {
    c = C + (i0 + 3) * ldc + o;
    if (accumulate) c3 = _mm256_add_ps(c3, _mm256_loadu_ps(c));
    _mm256_storeu_ps(c, c3);
  }
  if (rows > 4) {
    c = C + (i0 + 4) * ldc + o;
    if (accumulate) c4 = _mm256_add_ps(c4, _mm256_loadu_ps(c));
    _mm256_storeu_ps(c, c4);
  }
  if (rows > 5) {
    c = C + (i0 + 5) * ldc + o;
    if (accumulate) c5 = _mm256_add_ps(c5, _mm256_loadu_ps(c));
    _mm256_storeu_ps(c, c5);
  }
  if (rows > 6) {
    c = C + (i0 + 6) * ldc + o;
    if (accumulate) c6 = _mm256_add_ps(c6, _mm256_loadu_ps(c));
    _mm256_storeu_ps(c, c6);
  }
  if (rows > 7) {
    c = C + (i0 + 7) * ldc + o;
    if (accumulate) c7 = _mm256_add_ps(c7, _mm256_loadu_ps(c));
    _mm256_storeu_ps(c, c7);
  }
}

OV_AVX2 inline void micro_s8_8x8(int64_t j0, int64_t jj, int64_t i0, int rows, int64_t K, int64_t N,
                                 const float* A, const int8_t* W, const float* scale,
                                 float* C) {
  const int64_t r1 = (rows > 1) ? i0 + 1 : i0;
  const int64_t r2 = (rows > 2) ? i0 + 2 : i0;
  const int64_t r3 = (rows > 3) ? i0 + 3 : i0;
  const int64_t r4 = (rows > 4) ? i0 + 4 : i0;
  const int64_t r5 = (rows > 5) ? i0 + 5 : i0;
  const int64_t r6 = (rows > 6) ? i0 + 6 : i0;
  const int64_t r7 = (rows > 7) ? i0 + 7 : i0;

  const float* a0 = A + (i0 + 0) * K;
  const float* a1 = A + r1 * K;
  const float* a2 = A + r2 * K;
  const float* a3 = A + r3 * K;
  const float* a4 = A + r4 * K;
  const float* a5 = A + r5 * K;
  const float* a6 = A + r6 * K;
  const float* a7 = A + r7 * K;

  __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
  __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
  __m256 c4 = _mm256_setzero_ps(), c5 = _mm256_setzero_ps();
  __m256 c6 = _mm256_setzero_ps(), c7 = _mm256_setzero_ps();

  const int8_t* w = W + j0 + jj;
  for (int64_t k = 0; k < K; ++k, w += N) {
    const __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w));
    const __m256 b = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed));
    c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0 + k), b, c0);
    c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(a1 + k), b, c1);
    c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(a2 + k), b, c2);
    c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(a3 + k), b, c3);
    c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(a4 + k), b, c4);
    c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(a5 + k), b, c5);
    c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(a6 + k), b, c6);
    c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(a7 + k), b, c7);
  }
  const int64_t o = j0 + jj;
  const __m256 s = _mm256_loadu_ps(scale + o);
  _mm256_storeu_ps(C + (i0 + 0) * N + o, _mm256_mul_ps(c0, s));
  if (rows > 1) _mm256_storeu_ps(C + (i0 + 1) * N + o, _mm256_mul_ps(c1, s));
  if (rows > 2) _mm256_storeu_ps(C + (i0 + 2) * N + o, _mm256_mul_ps(c2, s));
  if (rows > 3) _mm256_storeu_ps(C + (i0 + 3) * N + o, _mm256_mul_ps(c3, s));
  if (rows > 4) _mm256_storeu_ps(C + (i0 + 4) * N + o, _mm256_mul_ps(c4, s));
  if (rows > 5) _mm256_storeu_ps(C + (i0 + 5) * N + o, _mm256_mul_ps(c5, s));
  if (rows > 6) _mm256_storeu_ps(C + (i0 + 6) * N + o, _mm256_mul_ps(c6, s));
  if (rows > 7) _mm256_storeu_ps(C + (i0 + 7) * N + o, _mm256_mul_ps(c7, s));
}

OV_AVX2 inline __m256 load_s8x8_as_ps(const int8_t* p) {
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)p)));
}

OV_AVX2 inline float hsum256_ps(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}

// Fallback for the [N, K] layout on CPUs without AVX-VNNI (and the path used
// when OV_CPU_NO_VNNI=1).  Eight output columns at a time: one activation
// vector feeds eight int8->fp32 converts and eight FMAs, so the load/FMA ratio
// matches the old [K, N] kernel even though W is now row-major.
OV_AVX2 void linear_s8_nk_row_tile(int64_t i, int64_t jb, int64_t N, int64_t K, const float* A,
                                   const int8_t* W, const float* scale, float* C) {
  const int64_t j0 = jb * 8;
  const int64_t jn = (N - j0 < 8) ? (N - j0) : 8;
  const float* a = A + i * K;
  if (jn < 8) {  // column tail: at most 7 columns, keep it scalar
    for (int64_t jj = 0; jj < jn; ++jj) {
      const int8_t* w = W + (j0 + jj) * K;
      float acc = 0.f;
      for (int64_t k = 0; k < K; ++k) acc += a[k] * static_cast<float>(w[k]);
      C[i * N + j0 + jj] = acc * scale[j0 + jj];
    }
    return;
  }
  __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
  __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
  __m256 c4 = _mm256_setzero_ps(), c5 = _mm256_setzero_ps();
  __m256 c6 = _mm256_setzero_ps(), c7 = _mm256_setzero_ps();
  const int8_t* w0 = W + (j0 + 0) * K;
  const int8_t* w1 = W + (j0 + 1) * K;
  const int8_t* w2 = W + (j0 + 2) * K;
  const int8_t* w3 = W + (j0 + 3) * K;
  const int8_t* w4 = W + (j0 + 4) * K;
  const int8_t* w5 = W + (j0 + 5) * K;
  const int8_t* w6 = W + (j0 + 6) * K;
  const int8_t* w7 = W + (j0 + 7) * K;
  int64_t k = 0;
  for (; k + 8 <= K; k += 8) {
    const __m256 av = _mm256_loadu_ps(a + k);
    c0 = _mm256_fmadd_ps(av, load_s8x8_as_ps(w0 + k), c0);
    c1 = _mm256_fmadd_ps(av, load_s8x8_as_ps(w1 + k), c1);
    c2 = _mm256_fmadd_ps(av, load_s8x8_as_ps(w2 + k), c2);
    c3 = _mm256_fmadd_ps(av, load_s8x8_as_ps(w3 + k), c3);
    c4 = _mm256_fmadd_ps(av, load_s8x8_as_ps(w4 + k), c4);
    c5 = _mm256_fmadd_ps(av, load_s8x8_as_ps(w5 + k), c5);
    c6 = _mm256_fmadd_ps(av, load_s8x8_as_ps(w6 + k), c6);
    c7 = _mm256_fmadd_ps(av, load_s8x8_as_ps(w7 + k), c7);
  }
  float s[8] = {hsum256_ps(c0), hsum256_ps(c1), hsum256_ps(c2), hsum256_ps(c3),
                hsum256_ps(c4), hsum256_ps(c5), hsum256_ps(c6), hsum256_ps(c7)};
  for (; k < K; ++k) {
    const float av = a[k];
    s[0] += av * static_cast<float>(w0[k]);
    s[1] += av * static_cast<float>(w1[k]);
    s[2] += av * static_cast<float>(w2[k]);
    s[3] += av * static_cast<float>(w3[k]);
    s[4] += av * static_cast<float>(w4[k]);
    s[5] += av * static_cast<float>(w5[k]);
    s[6] += av * static_cast<float>(w6[k]);
    s[7] += av * static_cast<float>(w7[k]);
  }
  for (int jj = 0; jj < 8; ++jj) C[i * N + j0 + jj] = s[jj] * scale[j0 + jj];
}

OV_AVX2 void linear_s8_nk_avx2(int64_t M, int64_t N, int64_t K, const float* A, const int8_t* W,
                               const float* scale, float* C) {
  const int64_t jblocks = (N + 7) / 8;
  if (t_serial_gemm || M * jblocks < 8) {
    for (int64_t i = 0; i < M; ++i)
      for (int64_t jb = 0; jb < jblocks; ++jb)
        linear_s8_nk_row_tile(i, jb, N, K, A, W, scale, C);
    return;
  }
#pragma omp parallel for collapse(2) schedule(dynamic,4)
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t jb = 0; jb < jblocks; ++jb)
      linear_s8_nk_row_tile(i, jb, N, K, A, W, scale, C);
  }
}

OV_AVX2 void linear_f32_ld_avx2(int64_t M, int64_t N, int64_t K, const float* A, const float* W,
                                float* C, int64_t ldc, bool accumulate) {
  const int64_t jblocks = (N + kJb - 1) / kJb;
  const int64_t iblocks = (M + kMr - 1) / kMr;

  auto tile = [&](int64_t jb, int64_t ib) {
    const int64_t j0 = jb * kJb;
    const int64_t jn = (N - j0 < kJb) ? (N - j0) : kJb;
    const int64_t i0 = ib * kMr;
    const int rows = static_cast<int>((M - i0 < kMr) ? (M - i0) : kMr);
    int64_t jj = 0;
    for (; jj + 8 <= jn; jj += 8)
      micro_f32_8x8(j0, jj, i0, rows, K, N, ldc, accumulate, A, W, C);
    for (; jj < jn; ++jj) {  // column tail (N not a multiple of 8)
      const int64_t j = j0 + jj;
      for (int r = 0; r < rows; ++r) {
        const float* a = A + (i0 + r) * K;
        float acc = 0.f;
        for (int64_t k = 0; k < K; ++k) acc += a[k] * W[k * N + j];
        float* c = C + (i0 + r) * ldc + j;
        *c = accumulate ? (*c + acc) : acc;
      }
    }
  };
  // `dynamic` matters on hybrid CPUs: with `static` the E-core threads get the
  // same number of tiles as the P-cores and the whole region waits for the
  // slowest one (measured 2.33 s -> 1.83 s for an 8-step run on 8P+16E).
  // A serial call must not touch the OpenMP runtime at all: entering a
  // 1-thread parallel region costs ~10 us and attention makes 1792 such calls
  // per forward.
  if (t_serial_gemm || iblocks * jblocks < 8) {
    for (int64_t jb = 0; jb < jblocks; ++jb)
      for (int64_t ib = 0; ib < iblocks; ++ib) tile(jb, ib);
    return;
  }
#pragma omp parallel for collapse(2) schedule(dynamic,4)
  for (int64_t jb = 0; jb < jblocks; ++jb) {
    for (int64_t ib = 0; ib < iblocks; ++ib) tile(jb, ib);
  }
}

OV_AVX2 void linear_s8_avx2(int64_t M, int64_t N, int64_t K, const float* A, const int8_t* W,
                            const float* scale, float* C) {
  const int64_t jblocks = (N + kJb - 1) / kJb;
  const int64_t iblocks = (M + kMr - 1) / kMr;

  auto tile = [&](int64_t jb, int64_t ib) {
    const int64_t j0 = jb * kJb;
    const int64_t jn = (N - j0 < kJb) ? (N - j0) : kJb;
    const int64_t i0 = ib * kMr;
    const int rows = static_cast<int>((M - i0 < kMr) ? (M - i0) : kMr);
    int64_t jj = 0;
    for (; jj + 8 <= jn; jj += 8) micro_s8_8x8(j0, jj, i0, rows, K, N, A, W, scale, C);
    for (; jj < jn; ++jj) {
      const int64_t j = j0 + jj;
      for (int r = 0; r < rows; ++r) {
        const float* a = A + (i0 + r) * K;
        float acc = 0.f;
        for (int64_t k = 0; k < K; ++k) acc += a[k] * static_cast<float>(W[k * N + j]);
        C[(i0 + r) * N + j] = acc * scale[j];
      }
    }
  };
  if (t_serial_gemm || iblocks * jblocks < 8) {
    for (int64_t jb = 0; jb < jblocks; ++jb)
      for (int64_t ib = 0; ib < iblocks; ++ib) tile(jb, ib);
    return;
  }
#pragma omp parallel for collapse(2) schedule(dynamic,4)
  for (int64_t jb = 0; jb < jblocks; ++jb) {
    for (int64_t ib = 0; ib < iblocks; ++ib) tile(jb, ib);
  }
}

// ---------------------------------------------------------------- AVX-VNNI
//
// Weight tiles are built once at load (s8_nk_pack) and the activation is
// quantised per row / 32-k block just before the GEMM.  Inner loop per 4-k
// group of one 8-column tile:
//
//   for r in 0..3:  acc[r] = vpdpbusd(acc[r], bcast4(Aq[r][4g]), tile[g])
//
// i.e. one tile load + 4 broadcasts + 4 dpbusd produce 4 rows x 8 cols x 4 k
// = 128 MAC.  The int32 accumulators are flushed into float once per 32-k
// block (where the zero-point correction and block activation scale live);
// the per-column scale is applied once, at the end of the K loop.  A 4-row
// block is deliberate: with 8 rows the 8 int32 + 8 float accumulators spill,
// and the extra C read-modify-write per block measured ~35% slower.

#if !defined(_MSC_VER)
OV_VNNI inline __m256i bcast4_u8(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return _mm256_set1_epi32(static_cast<int>(v));
}

OV_VNNI void quantize_a_u8(int64_t M, int64_t K, int64_t Kb, int64_t Kp, const float* A,
                           uint8_t* aq, float* asc) {
  // Rows are independent; a single row of 1024 floats is ~300 ns, so the whole
  // 218-row block is 83 us serial and ~6 us across 32 threads.
  const __m256 sign = _mm256_set1_ps(-0.0f);
  const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
  const __m256i zp = _mm256_set1_epi32(128);
  const bool serial = t_serial_gemm || M < 32;
#pragma omp parallel for schedule(static) if (!serial)
  for (int64_t i = 0; i < M; ++i) {
    const float* a = A + i * K;
    uint8_t* q = aq + i * Kp;
    float* sc = asc + i * Kb;
    for (int64_t b = 0; b < Kb; ++b) {
      const int64_t k0 = b * 32;
      const int64_t k1 = (K - k0 < 32) ? (K - k0) : 32;
      // One scale per 32-k block: find its absmax (4 vectors).
      __m256 vmax = _mm256_setzero_ps();
      int64_t k = 0;
      for (; k + 8 <= k1; k += 8) {
        const __m256 v = _mm256_loadu_ps(a + k0 + k);
        vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign, v));
      }
      float lanes[8];
      _mm256_storeu_ps(lanes, vmax);
      float amax = 0.f;
      for (float v : lanes) amax = std::max(amax, v);
      for (; k < k1; ++k) amax = std::max(amax, std::fabs(a[k0 + k]));
      const float scale = amax / 127.0f;
      const float inv = amax > 0.f ? 127.0f / amax : 0.f;
      sc[b] = scale;
      k = 0;
      for (; k + 8 <= k1; k += 8) {
        __m256i qi = _mm256_cvtps_epi32(
            _mm256_mul_ps(_mm256_loadu_ps(a + k0 + k), _mm256_set1_ps(inv)));
        qi = _mm256_min_epi32(_mm256_max_epi32(qi, lo), hi);
        qi = _mm256_add_epi32(qi, zp);
        const __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(qi),
                                            _mm256_extracti128_si256(qi, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(q + k0 + k), _mm_packus_epi16(p16, p16));
      }
      for (; k < k1; ++k) {
        int v = static_cast<int>(std::nearbyint(a[k0 + k] * inv));
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        q[k0 + k] = static_cast<uint8_t>(v + 128);
      }
      // Pad the block to 32 bytes.  The matching weight tile bytes are zero,
      // so these values never contribute (nor does their zero point).
      if (k1 < 32) std::memset(q + k0 + k1, 0, static_cast<size_t>(32 - k1));
    }
  }
}

OV_VNNI void linear_s8_nk_vnni_tile_at(int64_t i0, int rows, int64_t n0_global,
                                       int64_t n0_local, int64_t N_out, float* C,
                                       const uint8_t* aq, const float* asc, const int8_t* packed,
                                       const int32_t* wsum, const float* scale, int64_t Kb,
                                       int64_t Npad_total, int64_t Kp, bool accumulate) {

  const int64_t jn = (N_out - n0_local < 64) ? (N_out - n0_local) : 64;
  const int64_t gb = Kb * 8;
  const int64_t o0 = (i0 + 0) * Kp;
  const int64_t o1 = (i0 + (rows > 1 ? 1 : 0)) * Kp;
  const int64_t o2 = (i0 + (rows > 2 ? 2 : 0)) * Kp;
  const int64_t o3 = (i0 + (rows > 3 ? 3 : 0)) * Kp;

  for (int64_t jj = 0; jj < jn; jj += 8) {
    const int64_t nc_global = n0_global + jj;
    const int64_t nc_local = n0_local + jj;
    const int nvalid = static_cast<int>((N_out - nc_local < 8) ? (N_out - nc_local) : 8);
    const int8_t* wp = packed + (nc_global / 8) * gb * 32;
    const __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(nvalid),
                                            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    const __m256 wsc = _mm256_maskload_ps(scale + nc_global, mask);
    __m256 f0 = _mm256_setzero_ps(), f1 = _mm256_setzero_ps();
    __m256 f2 = _mm256_setzero_ps(), f3 = _mm256_setzero_ps();
    for (int64_t b = 0; b < Kb; ++b) {
      __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
      __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
      for (int gg = 0; gg < 8; ++gg) {
        const int64_t g = b * 8 + gg;
        const __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + g * 32));
        const int64_t ko = 4 * g;
        c0 = _mm256_dpbusd_epi32(c0, bcast4_u8(aq + o0 + ko), bv);
        c1 = _mm256_dpbusd_epi32(c1, bcast4_u8(aq + o1 + ko), bv);
        c2 = _mm256_dpbusd_epi32(c2, bcast4_u8(aq + o2 + ko), bv);
        c3 = _mm256_dpbusd_epi32(c3, bcast4_u8(aq + o3 + ko), bv);
      }
      // Accumulate the block into float; the column scale is applied once, at
      // the end of the K loop.
      const __m256i corr = _mm256_slli_epi32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wsum + b * Npad_total + nc_global)), 7);
#define OV_VNNI_BLOCK_ROW(R, acc, facc)                                                 \
  if (rows > R) {                                                                       \
    const __m256 v = _mm256_cvtepi32_ps(_mm256_sub_epi32(acc, corr));                    \
    facc = _mm256_fmadd_ps(v, _mm256_set1_ps(asc[(i0 + R) * Kb + b]), facc);            \
  }
      OV_VNNI_BLOCK_ROW(0, c0, f0)
      OV_VNNI_BLOCK_ROW(1, c1, f1)
      OV_VNNI_BLOCK_ROW(2, c2, f2)
      OV_VNNI_BLOCK_ROW(3, c3, f3)
#undef OV_VNNI_BLOCK_ROW
    }
#define OV_VNNI_STORE_ROW(R, facc)                                                     \
  if (rows > R) {                                                                      \
    __m256 o = _mm256_mul_ps(facc, wsc);                                               \
    float* cp = C + (i0 + R) * N_out + nc_local;                                                 \
    if (accumulate) {                                                                  \
      asm volatile("" : "+x"(o));  /* keep mul/add separate: bit-match C += P */       \
      if (nvalid == 8)                                                                 \
        o = _mm256_add_ps(o, _mm256_loadu_ps(cp));                                     \
      else                                                                             \
        o = _mm256_add_ps(o, _mm256_maskload_ps(cp, mask));                            \
    }                                                                                  \
    if (nvalid == 8)                                                                   \
      _mm256_storeu_ps(cp, o);                                                         \
    else                                                                               \
      _mm256_maskstore_ps(cp, mask, o);                                                \
  }
    OV_VNNI_STORE_ROW(0, f0)
    OV_VNNI_STORE_ROW(1, f1)
    OV_VNNI_STORE_ROW(2, f2)
    OV_VNNI_STORE_ROW(3, f3)
#undef OV_VNNI_STORE_ROW
  }
}

// Single-matrix wrapper: a 64-column panel at n0, wsum stride Npad.
OV_VNNI void linear_s8_nk_vnni_tile(int64_t i0, int rows, int64_t jb, int64_t N,
                                    const uint8_t* aq, const float* asc, const int8_t* packed,
                                    const int32_t* wsum, const float* scale, float* C,
                                    int64_t Kb, int64_t Npad, int64_t Kp, bool accumulate) {
  const int64_t n0 = jb * 64;
  linear_s8_nk_vnni_tile_at(i0, rows, n0, n0, N, C, aq, asc, packed, wsum, scale, Kb, Npad, Kp,
                            accumulate);
}

OV_VNNI void linear_s8_nk_vnni(int64_t M, int64_t N, int64_t K, int64_t Kb, int64_t Npad,
                               const uint8_t* aq, const float* asc, const int8_t* packed,
                               const int32_t* wsum, const float* scale, float* C,
                               bool accumulate) {
  const int64_t Kp = Kb * 32;
  const int64_t jblocks = (N + 63) / 64;
  const int64_t iblocks = (M + 3) / 4;
  auto run = [&](int64_t jb, int64_t ib) {
    const int64_t i0 = ib * 4;
    const int rows = static_cast<int>((M - i0 < 4) ? (M - i0) : 4);
    linear_s8_nk_vnni_tile(i0, rows, jb, N, aq, asc, packed, wsum, scale, C, Kb, Npad, Kp,
                           accumulate);
  };
  if (t_serial_gemm || jblocks * iblocks < 8) {
    for (int64_t jb = 0; jb < jblocks; ++jb)
      for (int64_t ib = 0; ib < iblocks; ++ib) run(jb, ib);
    return;
  }
#pragma omp parallel for collapse(2) schedule(dynamic,4)
  for (int64_t jb = 0; jb < jblocks; ++jb) {
    for (int64_t ib = 0; ib < iblocks; ++ib) run(jb, ib);
  }
}
#endif  // !_MSC_VER (AVX-VNNI block)

#endif  // OV_X86

bool avx2_available() {
#if defined(OV_X86) && defined(_MSC_VER)
  return true;  // this translation unit is compiled with /arch:AVX2
#elif defined(OV_X86)
  static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  return ok;
#else
  return false;
#endif
}

}  // namespace

bool cpu_vnni_available() {
#ifdef _MSC_VER
  return false;  // AVX-VNNI is disabled in the MSVC build
#elif defined(OV_X86)
  static const bool ok = [] {
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma") ||
        !__builtin_cpu_supports("avxvnni")) {
      return false;
    }
    const char* e = std::getenv("OV_CPU_NO_VNNI");
    return !(e && e[0] && e[0] != '0');
  }();
  return ok;
#else
  return false;
#endif
}

int64_t s8_nk_packed_bytes(int64_t N, int64_t K) {
  if (N <= 0 || K <= 0) return 0;
  return ((N + 7) / 8) * (((K + 31) / 32) * 8) * 32;
}

int64_t s8_nk_wsum_elems(int64_t N, int64_t K) {
  if (N <= 0 || K <= 0) return 0;
  return ((K + 31) / 32) * ((N + 7) / 8) * 8;
}

void s8_nk_pack(int64_t N, int64_t K, const int8_t* W, int8_t* packed, int32_t* wsum) {
  const int64_t nb = (N + 7) / 8;
  const int64_t Npad = nb * 8;
  const int64_t Kb = (K + 31) / 32;
  const int64_t gb = Kb * 8;
  // The tile buffer only needs zeroing where the tiling leaves holes (N not a
  // multiple of 8, or K not a multiple of 32); otherwise every byte is written
  // below and the memset was a second full pass over 440 MB of weights.
  if (Npad != N || Kb * 32 != K) {
    std::memset(packed, 0, static_cast<size_t>(nb * gb * 32));
  }
  std::memset(wsum, 0, static_cast<size_t>(Kb * Npad) * sizeof(int32_t));
  // k-blocks are independent (disjoint tile slots and disjoint wsum rows), so the
  // repack parallelises cleanly: 0.43 s -> ~0.1 s on 32 threads.
#pragma omp parallel for schedule(dynamic,4)
  for (int64_t b = 0; b < Kb; ++b) {
    const int64_t k0 = b * 32;
    const int64_t kn = (K - k0 < 32) ? (K - k0) : 32;
    for (int64_t n = 0; n < N; ++n) {
      const int8_t* row = W + n * K + k0;
      int32_t sum = 0;
      for (int64_t k = 0; k < kn; ++k) sum += row[k];
      wsum[b * Npad + n] = sum;
    }
    for (int64_t g = b * 8; g < b * 8 + 8; ++g) {
      const int64_t kb = 4 * g;
      if (kb >= K) break;
      for (int64_t nn = 0; nn < nb; ++nn) {
        const int64_t n0 = nn * 8;
        int8_t* tile = packed + nn * gb * 32 + g * 32;
        for (int64_t j = 0; j < 8 && n0 + j < N; ++j) {
          const int8_t* row = W + (n0 + j) * K;
          for (int64_t l = 0; l < 4 && kb + l < K; ++l) tile[j * 4 + l] = row[kb + l];
        }
      }
    }
  }
}

int64_t s8_nk_quant_bytes(int64_t M, int64_t K) {
  const int64_t Kb = (K + 31) / 32;
  return M * Kb * 32;
}

int64_t s8_nk_quant_scales(int64_t M, int64_t K) {
  const int64_t Kb = (K + 31) / 32;
  return M * Kb;
}

// Fused RMSNorm + per-32-k u8 quantisation.  This must produce exactly the same
// aq/asc bytes as rmsnorm_into_f32() followed by s8_nk_quantize(): the norm math
// below mirrors rmsnorm_avx2() operation for operation, and the block
// quantiser mirrors quantize_a_u8().  The point is to avoid materialising the
// normalised row in global memory and to use one OpenMP region instead of two.
OV_AVX2 void rmsnorm_quantize_avx2(int64_t rows, int64_t cols, const float* x, const float* w,
                                  float eps, uint8_t* aq, float* asc) {
  const __m256 sign = _mm256_set1_ps(-0.0f);
  const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
  const __m256i zp = _mm256_set1_epi32(128);
  const int64_t Kb = (cols + 31) / 32;
  const int64_t Kp = Kb * 32;
  const bool serial = t_serial_gemm || rows < 32;
#pragma omp parallel for schedule(static) if (!serial)
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = x + r * cols;
    uint8_t* q = aq + r * Kp;
    float* sc = asc + r * Kb;

    // Exact copy of rmsnorm_avx2()'s reduction.
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= cols; i += 8) {
      const __m256 v = _mm256_loadu_ps(row + i);
      acc = _mm256_fmadd_ps(v, v, acc);
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc);
    float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] +
                lanes[7];
    for (; i < cols; ++i) sum += row[i] * row[i];
    const float scale = 1.f / std::sqrt(sum / static_cast<float>(cols) + eps);
    const __m256 vs = _mm256_set1_ps(scale);

    for (int64_t b = 0; b < Kb; ++b) {
      const int64_t k0 = b * 32;
      const int64_t kn = (cols - k0 < 32) ? (cols - k0) : 32;
      alignas(32) float v[32];
      i = 0;
      for (; i + 8 <= kn; i += 8) {
        const __m256 xv = _mm256_loadu_ps(row + k0 + i);
        const __m256 wv = _mm256_loadu_ps(w + k0 + i);
        _mm256_store_ps(v + i, _mm256_mul_ps(_mm256_mul_ps(xv, vs), wv));
      }
      for (; i < kn; ++i) v[i] = row[k0 + i] * scale * w[k0 + i];

      __m256 vmax = _mm256_setzero_ps();
      i = 0;
      for (; i + 8 <= kn; i += 8) vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign,
                                                                             _mm256_load_ps(v + i)));
      _mm256_store_ps(lanes, vmax);
      float amax = 0.f;
      for (float z : lanes) amax = std::max(amax, z);
      for (; i < kn; ++i) amax = std::max(amax, std::fabs(v[i]));
      const float bs = amax / 127.0f;
      const float inv = amax > 0.f ? 127.0f / amax : 0.f;
      sc[b] = bs;
      i = 0;
      for (; i + 8 <= kn; i += 8) {
        __m256i qi = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_load_ps(v + i), _mm256_set1_ps(inv)));
        qi = _mm256_min_epi32(_mm256_max_epi32(qi, lo), hi);
        qi = _mm256_add_epi32(qi, zp);
        const __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(qi),
                                            _mm256_extracti128_si256(qi, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(q + k0 + i), _mm_packus_epi16(p16, p16));
      }
      for (; i < kn; ++i) {
        int z = static_cast<int>(std::nearbyint(v[i] * inv));
        z = z < -127 ? -127 : (z > 127 ? 127 : z);
        q[k0 + i] = static_cast<uint8_t>(z + 128);
      }
      if (kn < 32) std::memset(q + k0 + kn, 0, static_cast<size_t>(32 - kn));
    }
  }
}

void s8_nk_quantize(int64_t M, int64_t K, const float* A, uint8_t* aq, float* asc) {
  if (M <= 0 || K <= 0) return;
#if defined(OV_X86) && !defined(_MSC_VER)
  if (!cpu_vnni_available()) throw std::runtime_error("s8_nk_quantize: AVX-VNNI is not available");
  const int64_t Kb = (K + 31) / 32;
  quantize_a_u8(M, K, Kb, Kb * 32, A, aq, asc);
  return;
#else
  (void)A; (void)aq; (void)asc;
  throw std::runtime_error("s8_nk_quantize: AVX-VNNI is not available");
#endif
}

void rmsnorm_quantize_f32(int64_t rows, int64_t cols, const float* x, const float* w, float eps,
                          uint8_t* aq, float* asc) {
  if (rows <= 0 || cols <= 0) return;
#ifdef OV_X86
  if (avx2_available()) {
    rmsnorm_quantize_avx2(rows, cols, x, w, eps, aq, asc);
    return;
  }
#endif
  // Scalar fallback (never used on the VNNI path in practice).
  const int64_t Kb = (cols + 31) / 32;
  const int64_t Kp = Kb * 32;
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = x + r * cols;
    float sum = 0.f;
    for (int64_t i = 0; i < cols; ++i) sum += row[i] * row[i];
    const float scale = 1.f / std::sqrt(sum / static_cast<float>(cols) + eps);
    uint8_t* q = aq + r * Kp;
    float* sc = asc + r * Kb;
    for (int64_t b = 0; b < Kb; ++b) {
      const int64_t k0 = b * 32, kn = std::min<int64_t>(32, cols - k0);
      float amax = 0.f;
      for (int64_t i = 0; i < kn; ++i)
        amax = std::max(amax, std::fabs(row[k0 + i] * scale * w[k0 + i]));
      const float bs = amax / 127.f, inv = amax > 0.f ? 127.f / amax : 0.f;
      sc[b] = bs;
      for (int64_t i = 0; i < kn; ++i) {
        int z = static_cast<int>(std::nearbyint(row[k0 + i] * scale * w[k0 + i] * inv));
        z = z < -127 ? -127 : (z > 127 ? 127 : z);
        q[k0 + i] = static_cast<uint8_t>(z + 128);
      }
      for (int64_t i = kn; i < 32; ++i) q[k0 + i] = 0;
    }
  }
}

void linear_s8_nk_packed_q(int64_t M, int64_t N, int64_t K, const uint8_t* aq, const float* asc,
                           const int8_t* packed, const int32_t* wsum, const float* row_scale,
                           float* C) {
  if (M <= 0 || N <= 0 || K <= 0) return;
#if defined(OV_X86) && !defined(_MSC_VER)
  if (!cpu_vnni_available()) {
    throw std::runtime_error("linear_s8_nk_packed_q: AVX-VNNI is not available");
  }
  const int64_t Kb = (K + 31) / 32;
  const int64_t Npad = ((N + 7) / 8) * 8;
  linear_s8_nk_vnni(M, N, K, Kb, Npad, aq, asc, packed, wsum, row_scale, C, false);
  return;
#else
  (void)aq; (void)asc; (void)packed; (void)wsum; (void)row_scale; (void)C;
  throw std::runtime_error("linear_s8_nk_packed_q: AVX-VNNI is not available");
#endif
}

void linear_s8_nk_packed_q_sliced(int64_t M, int64_t K, const uint8_t* aq, const float* asc,
                                  const int8_t* packed, const int32_t* wsum,
                                  const float* scale, const S8NkSlice* slices, int n_slices) {
  if (M <= 0 || K <= 0 || n_slices <= 0) return;
#if defined(OV_X86) && !defined(_MSC_VER)
  if (!cpu_vnni_available()) {
    throw std::runtime_error("linear_s8_nk_packed_q_sliced: AVX-VNNI is not available");
  }
  int64_t total_n = 0;
  for (int s = 0; s < n_slices; ++s) {
    if (slices[s].N <= 0 || slices[s].N % 64 != 0 || !slices[s].C) {
      throw std::invalid_argument("sliced GEMM: each slice N must be a positive multiple of 64");
    }
    total_n += slices[s].N;
  }
  const int64_t Kb = (K + 31) / 32;
  const int64_t Kp = Kb * 32;
  const int64_t Npad_total = ((total_n + 7) / 8) * 8;
  const int64_t panels = total_n / 64;
  const int64_t iblocks = (M + 3) / 4;
  std::vector<int> slice_of(static_cast<size_t>(panels));
  std::vector<int64_t> n0_local(static_cast<size_t>(panels));
  int64_t p = 0;
  for (int s = 0; s < n_slices; ++s) {
    for (int64_t j = 0; j < slices[s].N / 64; ++j) {
      slice_of[static_cast<size_t>(p)] = s;
      n0_local[static_cast<size_t>(p)] = j * 64;
      ++p;
    }
  }
  auto run = [&](int64_t jb, int64_t ib) {
    const int s = slice_of[static_cast<size_t>(jb)];
    const int64_t i0 = ib * 4;
    const int rows = static_cast<int>((M - i0 < 4) ? (M - i0) : 4);
    linear_s8_nk_vnni_tile_at(i0, rows, jb * 64, n0_local[static_cast<size_t>(jb)], slices[s].N,
                              slices[s].C, aq, asc, packed, wsum, scale, Kb, Npad_total, Kp,
                              false);
  };
  if (t_serial_gemm || panels * iblocks < 8) {
    for (int64_t jb = 0; jb < panels; ++jb)
      for (int64_t ib = 0; ib < iblocks; ++ib) run(jb, ib);
    return;
  }
#pragma omp parallel for collapse(2) schedule(dynamic,4)
  for (int64_t jb = 0; jb < panels; ++jb) {
    for (int64_t ib = 0; ib < iblocks; ++ib) run(jb, ib);
  }
#else
  (void)aq; (void)asc; (void)packed; (void)wsum; (void)scale; (void)slices; (void)n_slices;
  throw std::runtime_error("linear_s8_nk_packed_q_sliced: AVX-VNNI is not available");
#endif
}

void linear_s8_nk_packed(int64_t M, int64_t N, int64_t K, const float* A,
                         const int8_t* packed, const int32_t* wsum, const float* row_scale,
                         float* C) {
  if (M <= 0 || N <= 0 || K <= 0) return;
#if defined(OV_X86) && !defined(_MSC_VER)
  if (cpu_vnni_available()) {
    static thread_local std::vector<uint8_t> aq;
    static thread_local std::vector<float> asc;
    aq.resize(static_cast<size_t>(s8_nk_quant_bytes(M, K)));
    asc.resize(static_cast<size_t>(s8_nk_quant_scales(M, K)));
    s8_nk_quantize(M, K, A, aq.data(), asc.data());
    linear_s8_nk_packed_q(M, N, K, aq.data(), asc.data(), packed, wsum, row_scale, C);
    return;
  }
#endif
  throw std::runtime_error("linear_s8_nk_packed: AVX-VNNI is not available");
}

void linear_s8_nk_packed_q_add(int64_t M, int64_t N, int64_t K, const uint8_t* aq,
                               const float* asc, const int8_t* packed, const int32_t* wsum,
                               const float* row_scale, float* C) {
  if (M <= 0 || N <= 0 || K <= 0) return;
#if defined(OV_X86) && !defined(_MSC_VER)
  if (!cpu_vnni_available()) {
    throw std::runtime_error("linear_s8_nk_packed_q_add: AVX-VNNI is not available");
  }
  const int64_t Kb = (K + 31) / 32;
  const int64_t Npad = ((N + 7) / 8) * 8;
  linear_s8_nk_vnni(M, N, K, Kb, Npad, aq, asc, packed, wsum, row_scale, C, true);
#else
  (void)aq; (void)asc; (void)packed; (void)wsum; (void)row_scale; (void)C;
  throw std::runtime_error("linear_s8_nk_packed_q_add: AVX-VNNI is not available");
#endif
}

void linear_s8_nk_packed_add(int64_t M, int64_t N, int64_t K, const float* A,
                             const int8_t* packed, const int32_t* wsum, const float* row_scale,
                             float* C) {
  if (M <= 0 || N <= 0 || K <= 0) return;
#if defined(OV_X86) && !defined(_MSC_VER)
  if (cpu_vnni_available()) {
    static thread_local std::vector<uint8_t> aq;
    static thread_local std::vector<float> asc;
    aq.resize(static_cast<size_t>(s8_nk_quant_bytes(M, K)));
    asc.resize(static_cast<size_t>(s8_nk_quant_scales(M, K)));
    s8_nk_quantize(M, K, A, aq.data(), asc.data());
    const int64_t Kb = (K + 31) / 32;
    const int64_t Npad = ((N + 7) / 8) * 8;
    linear_s8_nk_vnni(M, N, K, Kb, Npad, aq.data(), asc.data(), packed, wsum, row_scale, C, true);
    return;
  }
#endif
  throw std::runtime_error("linear_s8_nk_packed_add: AVX-VNNI is not available");
}

void linear_f32_ld(int64_t M, int64_t N, int64_t K, const float* A, const float* W, float* C,
                   int64_t ldc, bool accumulate) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  if (ldc < N) throw std::invalid_argument("linear_f32_ld: ldc < N");
#ifdef OV_X86
  if (avx2_available()) {
    // The M tail is handled inside the micro-kernel (masked stores), so there is
    // no pad-and-copy path any more: that copy cost ~18% of every projection at
    // M=218 because T = batch*seq is rarely a multiple of 8.
    linear_f32_ld_avx2(M, N, K, A, W, C, ldc, accumulate);
    return;
  }
#endif
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.f;
      for (int64_t k = 0; k < K; ++k) acc += A[i * K + k] * W[k * N + j];
      float* c = C + i * ldc + j;
      *c = accumulate ? (*c + acc) : acc;
    }
  }
}

void linear_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* W, float* C) {
  linear_f32_ld(M, N, K, A, W, C, N, false);
}

void linear_s8(int64_t M, int64_t N, int64_t K, const float* A, const int8_t* W,
               const float* col_scale, float* C) {
  if (M <= 0 || N <= 0 || K <= 0) return;
#ifdef OV_X86
  if (avx2_available()) {
    linear_s8_avx2(M, N, K, A, W, col_scale, C);
    return;
  }
#endif
  linear_s8_ref(M, N, K, A, W, col_scale, C);
}

void linear_s8_nk(int64_t M, int64_t N, int64_t K, const float* A, const int8_t* W,
                  const float* row_scale, float* C) {
  if (M <= 0 || N <= 0 || K <= 0) return;
#ifdef OV_X86
  if (avx2_available()) {
    linear_s8_nk_avx2(M, N, K, A, W, row_scale, C);
    return;
  }
#endif
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      const int8_t* w = W + j * K;
      float acc = 0.f;
      for (int64_t k = 0; k < K; ++k) acc += A[i * K + k] * static_cast<float>(w[k]);
      C[i * N + j] = acc * row_scale[j];
    }
  }
}

const char* cpu_kernel_name() {
  if (cpu_vnni_available()) return "avx2+fma+vnni";
  return avx2_available() ? "avx2+fma" : "scalar";
}

}  // namespace ov

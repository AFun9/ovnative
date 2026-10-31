#include "model/ops.h"

#include "backend/cpu/scratch.h"

#include <stdexcept>

#include "ovnative/ov_gemm.h"

#include <vector>

#include <cmath>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define OV_X86 1
#endif

namespace ov {
namespace {

inline float silu(float x) { return x / (1.f + std::exp(-x)); }

#ifdef OV_X86
#define OV_AVX2 __attribute__((target("avx2,fma")))

OV_AVX2 void rmsnorm_avx2(float* dst, const float* src, const float* w, int64_t rows, int64_t cols,
                          float eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = src + r * cols;
    float* out = dst + r * cols;
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= cols; i += 8) {
      const __m256 v = _mm256_loadu_ps(row + i);
      acc = _mm256_fmadd_ps(v, v, acc);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < cols; ++i) sum += row[i] * row[i];
    const float scale = 1.f / std::sqrt(sum / static_cast<float>(cols) + eps);
    const __m256 vs = _mm256_set1_ps(scale);
    for (i = 0; i + 8 <= cols; i += 8) {
      const __m256 v = _mm256_loadu_ps(row + i);
      const __m256 wv = _mm256_loadu_ps(w + i);
      _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(v, vs), wv));
    }
    for (; i < cols; ++i) out[i] = row[i] * scale * w[i];
  }
}

OV_AVX2 void add_inplace_avx2(float* dst, const float* src, int64_t n) {
  int64_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(dst + i,
                     _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
  for (; i < n; ++i) dst[i] += src[i];
}

// exp(x) via round-to-nearest range reduction + degree-6 Horner on
// r in [-ln2/2, ln2/2]; max relative error ~1e-7, i.e. far below the fp32
// noise of the surrounding matmuls.
OV_AVX2 inline __m256 exp256_ps(__m256 x) {
  x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
  x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
  const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f)),
                                   _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(0.6931471805599453f)));
  __m256 p = _mm256_set1_ps(1.0f / 720);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 120));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
  const __m256i ni = _mm256_cvtps_epi32(n);
  const __m256 sc = _mm256_castsi256_ps(
      _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23));
  return _mm256_mul_ps(p, sc);
}

// SwiGLU: out = x/(1+exp(-x)) * up.  The scalar-per-lane version this replaces
// cost 1.25 ms per layer (35 ms per LM forward): 8 libm exp() calls plus two
// round trips through the stack for every 8 elements.
// out = rotate_half(x) with a precomputed cos/sin table, 8 lanes at a time.
OV_AVX2 void rope_rot_avx2(float* x, const float* cos_tab, const float* sin_tab, int64_t rows,
                           int n_heads, int head_dim) {
  const int half = head_dim / 2;
  for (int64_t r = 0; r < rows; ++r) {
    const float* c = cos_tab + r * half;
    const float* sn = sin_tab + r * half;
    float* row = x + r * n_heads * head_dim;
    for (int h = 0; h < n_heads; ++h) {
      float* v = row + h * head_dim;
      int i = 0;
      for (; i + 8 <= half; i += 8) {
        const __m256 va = _mm256_loadu_ps(v + i);
        const __m256 vb = _mm256_loadu_ps(v + i + half);
        const __m256 vc = _mm256_loadu_ps(c + i);
        const __m256 vs = _mm256_loadu_ps(sn + i);
        _mm256_storeu_ps(v + i, _mm256_fmsub_ps(va, vc, _mm256_mul_ps(vb, vs)));
        _mm256_storeu_ps(v + i + half, _mm256_fmadd_ps(va, vs, _mm256_mul_ps(vb, vc)));
      }
      for (; i < half; ++i) {
        const float a = v[i];
        const float b = v[i + half];
        v[i] = a * c[i] - b * sn[i];
        v[i + half] = b * c[i] + a * sn[i];
      }
    }
  }
}

OV_AVX2 void silu_mul_avx2(float* out, const float* gate, const float* up, int64_t n) {
  const __m256 one = _mm256_set1_ps(1.f);
  const __m256 zero = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 g = _mm256_loadu_ps(gate + i);
    const __m256 d = _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(zero, g)));
    _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_div_ps(g, d), _mm256_loadu_ps(up + i)));
  }
  for (; i < n; ++i) out[i] = silu(gate[i]) * up[i];
}

OV_AVX2 inline __m256 silu_up8(__m256 g, __m256 u) {
  const __m256 one = _mm256_set1_ps(1.f);
  const __m256 d = _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), g)));
  return _mm256_mul_ps(_mm256_div_ps(g, d), u);
}

OV_AVX2 void silu_mul_quant_avx2(int64_t M, int64_t K, const float* gate, const float* up,
                                 uint8_t* aq, float* asc) {
  const __m256 sign = _mm256_set1_ps(-0.0f);
  const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
  const __m256i zp = _mm256_set1_epi32(128);
  const int64_t Kb = (K + 31) / 32;
  const int64_t Kp = Kb * 32;
  const bool serial = M < 32;
#pragma omp parallel for schedule(static) if (!serial)
  for (int64_t r = 0; r < M; ++r) {
    const float* gr = gate + r * K;
    const float* ur = up + r * K;
    uint8_t* q = aq + r * Kp;
    float* sc = asc + r * Kb;
    for (int64_t b = 0; b < Kb; ++b) {
      const int64_t k0 = b * 32;
      const int64_t kn = (K - k0 < 32) ? (K - k0) : 32;
      alignas(32) float v[32];
      int64_t i = 0;
      for (; i + 8 <= kn; i += 8)
        _mm256_store_ps(v + i, silu_up8(_mm256_loadu_ps(gr + k0 + i),
                                        _mm256_loadu_ps(ur + k0 + i)));
      for (; i < kn; ++i) v[i] = silu(gr[k0 + i]) * ur[k0 + i];

      __m256 vmax = _mm256_setzero_ps();
      i = 0;
      for (; i + 8 <= kn; i += 8)
        vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign, _mm256_load_ps(v + i)));
      alignas(32) float lanes[8];
      _mm256_store_ps(lanes, vmax);
      float amax = 0.f;
      for (float z : lanes) amax = std::max(amax, z);
      for (; i < kn; ++i) amax = std::max(amax, std::fabs(v[i]));
      const float bs = amax / 127.f;
      const float inv = amax > 0.f ? 127.f / amax : 0.f;
      sc[b] = bs;
      const __m256 vinv = _mm256_set1_ps(inv);
      i = 0;
      for (; i + 8 <= kn; i += 8) {
        __m256i qi = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_load_ps(v + i), vinv));
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

OV_AVX2 void softmax_avx2(float* scores, const float* bias, int64_t rows, int64_t cols) {
  alignas(32) float tmp[8];
  for (int64_t r = 0; r < rows; ++r) {
    float* row = scores + r * cols;
    const float* bp = bias ? bias + r * cols : nullptr;

    __m256 mx = _mm256_set1_ps(-INFINITY);
    int64_t j = 0;
    for (; j + 8 <= cols; j += 8) {
      __m256 v = _mm256_loadu_ps(row + j);
      if (bp) v = _mm256_add_ps(v, _mm256_loadu_ps(bp + j));
      mx = _mm256_max_ps(mx, v);
    }
    _mm256_store_ps(tmp, mx);
    float m = tmp[0];
    for (int t = 1; t < 8; ++t) m = (tmp[t] > m) ? tmp[t] : m;
    for (; j < cols; ++j) {
      const float v = row[j] + (bp ? bp[j] : 0.f);
      m = (v > m) ? v : m;
    }
    if (m < -1e29f) {  // fully masked row
      std::memset(row, 0, sizeof(float) * static_cast<size_t>(cols));
      continue;
    }

    const __m256 vm = _mm256_set1_ps(m);
    __m256 sumv = _mm256_setzero_ps();
    j = 0;
    for (; j + 8 <= cols; j += 8) {
      __m256 v = _mm256_loadu_ps(row + j);
      if (bp) v = _mm256_add_ps(v, _mm256_loadu_ps(bp + j));
      v = exp256_ps(_mm256_sub_ps(v, vm));
      _mm256_storeu_ps(row + j, v);
      sumv = _mm256_add_ps(sumv, v);
    }
    _mm256_store_ps(tmp, sumv);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; j < cols; ++j) {
      const float e = std::exp(row[j] + (bp ? bp[j] : 0.f) - m);
      row[j] = e;
      sum += e;
    }

    const __m256 inv = _mm256_set1_ps(1.f / sum);
    j = 0;
    for (; j + 8 <= cols; j += 8) {
      _mm256_storeu_ps(row + j, _mm256_mul_ps(_mm256_loadu_ps(row + j), inv));
    }
    const float invs = 1.f / sum;
    for (; j < cols; ++j) row[j] *= invs;
  }
}


#ifdef OV_X86
OV_AVX2 inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

// One head: RMSNorm (same reduction/order as rmsnorm_avx2) followed by RoPE
// (same fmsub/fmadd shapes as rope_rot_avx2) in a single pass over the row.
OV_AVX2 inline void norm_rope_head_avx2(float* v, const float* w, const float* c,
                                        const float* sn, int D, float eps) {
  const int half = D / 2;
  __m256 acc = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= D; i += 8) {
    const __m256 x = _mm256_loadu_ps(v + i);
    acc = _mm256_fmadd_ps(x, x, acc);
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, acc);
  float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] +
              lanes[7];
  for (; i < D; ++i) sum += v[i] * v[i];
  const float scale = 1.f / std::sqrt(sum / static_cast<float>(D) + eps);
  const __m256 vs = _mm256_set1_ps(scale);
  i = 0;
  for (; i + 8 <= half; i += 8) {
    const __m256 va = _mm256_loadu_ps(v + i);
    const __m256 vb = _mm256_loadu_ps(v + i + half);
    const __m256 na = _mm256_mul_ps(_mm256_mul_ps(va, vs), _mm256_loadu_ps(w + i));
    const __m256 nb = _mm256_mul_ps(_mm256_mul_ps(vb, vs), _mm256_loadu_ps(w + i + half));
    const __m256 vc = _mm256_loadu_ps(c + i);
    const __m256 vsn = _mm256_loadu_ps(sn + i);
    _mm256_storeu_ps(v + i, _mm256_fmsub_ps(na, vc, _mm256_mul_ps(nb, vsn)));
    _mm256_storeu_ps(v + i + half, _mm256_fmadd_ps(na, vsn, _mm256_mul_ps(nb, vc)));
  }
  for (; i < half; ++i) {
    const float a = (v[i] * scale) * w[i];
    const float b = (v[i + half] * scale) * w[i + half];
    v[i] = a * c[i] - b * sn[i];
    v[i + half] = b * c[i] + a * sn[i];
  }
}

OV_AVX2 void qk_norm_rope_avx2(float* q, float* k, const float* qw, const float* kw, int64_t rows,
                               int n_heads, int n_kv_heads, int D, const float* cos_tab,
                               const float* sin_tab, float eps) {
  const int half = D / 2;
  const int64_t qrow = static_cast<int64_t>(n_heads) * D;
  const int64_t krow = static_cast<int64_t>(n_kv_heads) * D;
#pragma omp parallel for schedule(static) if (rows >= 16)
  for (int64_t r = 0; r < rows; ++r) {
    const float* c = cos_tab + r * half;
    const float* sn = sin_tab + r * half;
    float* qr = q + r * qrow;
    for (int h = 0; h < n_heads; ++h) norm_rope_head_avx2(qr + h * D, qw, c, sn, D, eps);
    float* kr = k + r * krow;
    for (int h = 0; h < n_kv_heads; ++h) norm_rope_head_avx2(kr + h * D, kw, c, sn, D, eps);
  }
}

// Masked positions produce exp(-1e30) ~ 1e-38, i.e. subnormals.  Subnormal
// arithmetic on x86 traps into a microcode assist (~100x slower) and the whole
// attention block crawls.  Flush-to-zero / denormals-are-zero is per-thread
// MXCSR state, so set it on entry (idempotent, one instruction).
// Two query rows per pass over K/V.
//
// The one-row version re-read the whole K (and V) for every query row: at S=109,
// D=128 that is 2 x 55 KB per row, i.e. ~12 MB per head per layer.  With 32
// attention heads running in parallel that saturates L2 and the kernel measured
// 5% of peak (60 GFLOPS single-thread, 223 GFLOPS aggregate).  Blocking two
// query rows halves the K/V traffic and reuses each K/V vector for two FMA
// chains, which is where the speedup comes from.
constexpr int kAttnRB = 2;   // query rows per pass (register budget: 8 acc + 6 tmp)

OV_AVX2 void attention_head_avx2_rb2(const float* q, int64_t q_stride, const float* k,
                                 const float* v, int64_t kv_stride, const float* bias,
                                 int S, int D, float scale, float* out, int64_t out_stride) {
  enable_flush_to_zero();
  static thread_local std::vector<float> kbuf, vbuf, pbuf;
  kbuf.resize(static_cast<size_t>(S) * D);
  vbuf.resize(static_cast<size_t>(S) * D);
  pbuf.resize(static_cast<size_t>(kAttnRB) * S);

  for (int t = 0; t < S; ++t) {
    std::memcpy(kbuf.data() + static_cast<size_t>(t) * D, k + static_cast<int64_t>(t) * kv_stride,
                sizeof(float) * D);
    std::memcpy(vbuf.data() + static_cast<size_t>(t) * D, v + static_cast<int64_t>(t) * kv_stride,
                sizeof(float) * D);
  }

  const __m256 vscale = _mm256_set1_ps(scale);
  int s0 = 0;
  for (; s0 + kAttnRB <= S; s0 += kAttnRB) {
    float* p0 = pbuf.data();
    float* p1 = pbuf.data() + S;
    const float* q0 = q + static_cast<int64_t>(s0 + 0) * q_stride;
    const float* q1 = q + static_cast<int64_t>(s0 + 1) * q_stride;
    const float* b0 = bias ? bias + static_cast<int64_t>(s0 + 0) * S : nullptr;
    const float* b1 = bias ? bias + static_cast<int64_t>(s0 + 1) * S : nullptr;

    // ---- scores: four keys per sweep, two rows sharing every K vector ----
    float mx0 = -INFINITY, mx1 = -INFINITY;
    int t = 0;
    for (; t + 4 <= S; t += 4) {
      __m256 a00 = _mm256_setzero_ps(), a01 = a00, a02 = a00, a03 = a00;
      __m256 a10 = _mm256_setzero_ps(), a11 = a10, a12 = a10, a13 = a10;
      const float* km = kbuf.data() + static_cast<size_t>(t) * D;
      const float* k0 = km;
      const float* k1 = km + D;
      const float* k2 = km + 2 * static_cast<int64_t>(D);
      const float* k3 = km + 3 * static_cast<int64_t>(D);
      for (int d = 0; d + 8 <= D; d += 8) {
        const __m256 b0v = _mm256_loadu_ps(k0 + d);
        const __m256 b1v = _mm256_loadu_ps(k1 + d);
        const __m256 b2v = _mm256_loadu_ps(k2 + d);
        const __m256 b3v = _mm256_loadu_ps(k3 + d);
        const __m256 qa = _mm256_loadu_ps(q0 + d);
        a00 = _mm256_fmadd_ps(qa, b0v, a00);
        a01 = _mm256_fmadd_ps(qa, b1v, a01);
        a02 = _mm256_fmadd_ps(qa, b2v, a02);
        a03 = _mm256_fmadd_ps(qa, b3v, a03);
        const __m256 qb = _mm256_loadu_ps(q1 + d);
        a10 = _mm256_fmadd_ps(qb, b0v, a10);
        a11 = _mm256_fmadd_ps(qb, b1v, a11);
        a12 = _mm256_fmadd_ps(qb, b2v, a12);
        a13 = _mm256_fmadd_ps(qb, b3v, a13);
      }
      float v00 = hsum256(_mm256_mul_ps(a00, vscale));
      float v01 = hsum256(_mm256_mul_ps(a01, vscale));
      float v02 = hsum256(_mm256_mul_ps(a02, vscale));
      float v03 = hsum256(_mm256_mul_ps(a03, vscale));
      float v10 = hsum256(_mm256_mul_ps(a10, vscale));
      float v11 = hsum256(_mm256_mul_ps(a11, vscale));
      float v12 = hsum256(_mm256_mul_ps(a12, vscale));
      float v13 = hsum256(_mm256_mul_ps(a13, vscale));
      if (b0) { v00 += b0[t]; v01 += b0[t + 1]; v02 += b0[t + 2]; v03 += b0[t + 3]; }
      if (b1) { v10 += b1[t]; v11 += b1[t + 1]; v12 += b1[t + 2]; v13 += b1[t + 3]; }
      p0[t] = v00; p0[t + 1] = v01; p0[t + 2] = v02; p0[t + 3] = v03;
      p1[t] = v10; p1[t + 1] = v11; p1[t + 2] = v12; p1[t + 3] = v13;
      mx0 = std::max(mx0, std::max(std::max(v00, v01), std::max(v02, v03)));
      mx1 = std::max(mx1, std::max(std::max(v10, v11), std::max(v12, v13)));
    }
    for (; t < S; ++t) {  // key tail
      const float* kt = kbuf.data() + static_cast<size_t>(t) * D;
      __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
      for (int d = 0; d + 8 <= D; d += 8) {
        const __m256 bv = _mm256_loadu_ps(kt + d);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + d), bv, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + d), bv, a1);
      }
      const float w0 = hsum256(_mm256_mul_ps(a0, vscale)) + (b0 ? b0[t] : 0.f);
      const float w1 = hsum256(_mm256_mul_ps(a1, vscale)) + (b1 ? b1[t] : 0.f);
      p0[t] = w0;
      p1[t] = w1;
      mx0 = std::max(mx0, w0);
      mx1 = std::max(mx1, w1);
    }

    if (mx0 < -1e29f && mx1 < -1e29f) {
      // Rows are not contiguous when out_stride > D (LM uses out_stride = QD),
      // so zero them one by one instead of one kAttnRB*D memcpy.
      for (int r = 0; r < kAttnRB; ++r) {
        std::memset(out + static_cast<int64_t>(s0 + r) * out_stride, 0,
                    sizeof(float) * static_cast<size_t>(D));
      }
      continue;
    }

    // ---- softmax, one row at a time (independent per row) ----
    const float mxs[2] = {mx0, mx1};
    for (int r = 0; r < kAttnRB; ++r) {
      float* pr = pbuf.data() + r * S;
      if (mxs[r] < -1e29f) {
        std::memset(out + static_cast<int64_t>(s0 + r) * out_stride, 0, sizeof(float) * D);
        std::memset(pr, 0, sizeof(float) * static_cast<size_t>(S));
        continue;
      }
      __m256 sumv = _mm256_setzero_ps();
      const __m256 vm = _mm256_set1_ps(mxs[r]);
      int i = 0;
      for (; i + 8 <= S; i += 8) {
        const __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(pr + i), vm));
        _mm256_storeu_ps(pr + i, e);
        sumv = _mm256_add_ps(sumv, e);
      }
      float sum = hsum256(sumv);
      for (; i < S; ++i) {
        const float e = std::exp(pr[i] - mxs[r]);
        pr[i] = e;
        sum += e;
      }
      const float inv = 1.f / sum;
      const __m256 vinv = _mm256_set1_ps(inv);
      for (i = 0; i + 8 <= S; i += 8)
        _mm256_storeu_ps(pr + i, _mm256_mul_ps(_mm256_loadu_ps(pr + i), vinv));
      for (; i < S; ++i) pr[i] *= inv;
    }

    // ---- out = P . V, 32 dims per pass for both rows ----
    float* o0 = out + static_cast<int64_t>(s0 + 0) * out_stride;
    float* o1 = out + static_cast<int64_t>(s0 + 1) * out_stride;
    int d0 = 0;
    for (; d0 + 32 <= D; d0 += 32) {
      __m256 r00 = _mm256_setzero_ps(), r01 = r00, r02 = r00, r03 = r00;
      __m256 r10 = _mm256_setzero_ps(), r11 = r10, r12 = r10, r13 = r10;
      for (int tt = 0; tt < S; ++tt) {
        const float* vr = vbuf.data() + static_cast<size_t>(tt) * D + d0;
        const __m256 x0 = _mm256_loadu_ps(vr + 0);
        const __m256 x1 = _mm256_loadu_ps(vr + 8);
        const __m256 x2 = _mm256_loadu_ps(vr + 16);
        const __m256 x3 = _mm256_loadu_ps(vr + 24);
        const __m256 w0 = _mm256_set1_ps(p0[tt]);
        r00 = _mm256_fmadd_ps(w0, x0, r00);
        r01 = _mm256_fmadd_ps(w0, x1, r01);
        r02 = _mm256_fmadd_ps(w0, x2, r02);
        r03 = _mm256_fmadd_ps(w0, x3, r03);
        const __m256 w1 = _mm256_set1_ps(p1[tt]);
        r10 = _mm256_fmadd_ps(w1, x0, r10);
        r11 = _mm256_fmadd_ps(w1, x1, r11);
        r12 = _mm256_fmadd_ps(w1, x2, r12);
        r13 = _mm256_fmadd_ps(w1, x3, r13);
      }
      _mm256_storeu_ps(o0 + d0 + 0, r00);
      _mm256_storeu_ps(o0 + d0 + 8, r01);
      _mm256_storeu_ps(o0 + d0 + 16, r02);
      _mm256_storeu_ps(o0 + d0 + 24, r03);
      _mm256_storeu_ps(o1 + d0 + 0, r10);
      _mm256_storeu_ps(o1 + d0 + 8, r11);
      _mm256_storeu_ps(o1 + d0 + 16, r12);
      _mm256_storeu_ps(o1 + d0 + 24, r13);
    }
    for (; d0 < D; d0 += 8) {
      __m256 r0 = _mm256_setzero_ps(), r1 = _mm256_setzero_ps();
      for (int tt = 0; tt < S; ++tt) {
        const __m256 xv = _mm256_loadu_ps(vbuf.data() + static_cast<size_t>(tt) * D + d0);
        r0 = _mm256_fmadd_ps(_mm256_set1_ps(p0[tt]), xv, r0);
        r1 = _mm256_fmadd_ps(_mm256_set1_ps(p1[tt]), xv, r1);
      }
      _mm256_storeu_ps(o0 + d0, r0);
      _mm256_storeu_ps(o1 + d0, r1);
    }
  }

  // Odd tail: one row left when S is odd (or S < kAttnRB).
  if (s0 < S) {
    float* pr = pbuf.data();
    const float* q0 = q + static_cast<int64_t>(s0) * q_stride;
    const float* b0 = bias ? bias + static_cast<int64_t>(s0) * S : nullptr;
    float mx = -INFINITY;
    int t = 0;
    for (; t + 4 <= S; t += 4) {
      __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
      const float* km = kbuf.data() + static_cast<size_t>(t) * D;
      for (int d = 0; d + 8 <= D; d += 8) {
        const __m256 qv = _mm256_loadu_ps(q0 + d);
        a0 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(km + d), a0);
        a1 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(km + D + d), a1);
        a2 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(km + 2 * static_cast<int64_t>(D) + d), a2);
        a3 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(km + 3 * static_cast<int64_t>(D) + d), a3);
      }
      float v0 = hsum256(_mm256_mul_ps(a0, vscale));
      float v1 = hsum256(_mm256_mul_ps(a1, vscale));
      float v2 = hsum256(_mm256_mul_ps(a2, vscale));
      float v3 = hsum256(_mm256_mul_ps(a3, vscale));
      if (b0) { v0 += b0[t]; v1 += b0[t + 1]; v2 += b0[t + 2]; v3 += b0[t + 3]; }
      pr[t] = v0; pr[t + 1] = v1; pr[t + 2] = v2; pr[t + 3] = v3;
      mx = std::max(mx, std::max(std::max(v0, v1), std::max(v2, v3)));
    }
    for (; t < S; ++t) {
      const float* kt = kbuf.data() + static_cast<size_t>(t) * D;
      __m256 a0 = _mm256_setzero_ps();
      for (int d = 0; d + 8 <= D; d += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + d), _mm256_loadu_ps(kt + d), a0);
      const float w0 = hsum256(_mm256_mul_ps(a0, vscale)) + (b0 ? b0[t] : 0.f);
      pr[t] = w0;
      mx = std::max(mx, w0);
    }
    float* o0 = out + static_cast<int64_t>(s0) * out_stride;
    if (mx < -1e29f) {
      std::memset(o0, 0, sizeof(float) * D);
    } else {
      __m256 sumv = _mm256_setzero_ps();
      const __m256 vm = _mm256_set1_ps(mx);
      int i = 0;
      for (; i + 8 <= S; i += 8) {
        const __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(pr + i), vm));
        _mm256_storeu_ps(pr + i, e);
        sumv = _mm256_add_ps(sumv, e);
      }
      float sum = hsum256(sumv);
      for (; i < S; ++i) { const float e = std::exp(pr[i] - mx); pr[i] = e; sum += e; }
      const __m256 vinv = _mm256_set1_ps(1.f / sum);
      for (i = 0; i + 8 <= S; i += 8)
        _mm256_storeu_ps(pr + i, _mm256_mul_ps(_mm256_loadu_ps(pr + i), vinv));
      for (; i < S; ++i) pr[i] *= 1.f / sum;
      int d0 = 0;
      for (; d0 + 32 <= D; d0 += 32) {
        __m256 r0 = _mm256_setzero_ps(), r1 = r0, r2 = r0, r3 = r0;
        for (int tt = 0; tt < S; ++tt) {
          const __m256 w = _mm256_set1_ps(pr[tt]);
          const float* vr = vbuf.data() + static_cast<size_t>(tt) * D + d0;
          r0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vr + 0), r0);
          r1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vr + 8), r1);
          r2 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vr + 16), r2);
          r3 = _mm256_fmadd_ps(w, _mm256_loadu_ps(vr + 24), r3);
        }
        _mm256_storeu_ps(o0 + d0 + 0, r0);
        _mm256_storeu_ps(o0 + d0 + 8, r1);
        _mm256_storeu_ps(o0 + d0 + 16, r2);
        _mm256_storeu_ps(o0 + d0 + 24, r3);
      }
      for (; d0 < D; d0 += 8) {
        __m256 r0 = _mm256_setzero_ps();
        for (int tt = 0; tt < S; ++tt)
          r0 = _mm256_fmadd_ps(_mm256_set1_ps(pr[tt]),
                               _mm256_loadu_ps(vbuf.data() + static_cast<size_t>(tt) * D + d0), r0);
        _mm256_storeu_ps(o0 + d0, r0);
      }
    }
  }
}

// Four query rows per pass.  The two-row kernel reuses every K/V vector for two
// dot products; four rows doubles that reuse while still keeping 32 independent
// attention tasks (one per segment/head) on this machine.  Score is processed
// two keys at a time (8 accumulators + 4 q + 2 k), PV 16 dims at a time
// (8 accumulators).  The tail (S%4) calls the same block with aliased rows and
// stores only the valid lanes.
OV_AVX2 inline void attention_block4(const float* kb, const float* vb, int S, int D, float scale,
                                     const float* q0, const float* q1, const float* q2,
                                     const float* q3, const float* b0, const float* b1,
                                     const float* b2, const float* b3, float* pbuf, float* o0,
                                     float* o1, float* o2, float* o3, int64_t out_stride,
                                     int rows) {
  const __m256 vscale = _mm256_set1_ps(scale);
  float* p0 = pbuf;
  float* p1 = pbuf + S;
  float* p2 = pbuf + 2 * S;
  float* p3 = pbuf + 3 * S;

  float mx0 = -INFINITY, mx1 = -INFINITY, mx2 = -INFINITY, mx3 = -INFINITY;
  int t = 0;
  for (; t + 2 <= S; t += 2) {
    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
    const float* k0 = kb + static_cast<int64_t>(t) * D;
    const float* k1 = k0 + D;
    for (int d = 0; d + 8 <= D; d += 8) {
      const __m256 k0v = _mm256_loadu_ps(k0 + d);
      const __m256 k1v = _mm256_loadu_ps(k1 + d);
      const __m256 qa = _mm256_loadu_ps(q0 + d);
      a00 = _mm256_fmadd_ps(qa, k0v, a00);
      a01 = _mm256_fmadd_ps(qa, k1v, a01);
      const __m256 qb = _mm256_loadu_ps(q1 + d);
      a10 = _mm256_fmadd_ps(qb, k0v, a10);
      a11 = _mm256_fmadd_ps(qb, k1v, a11);
      const __m256 qc = _mm256_loadu_ps(q2 + d);
      a20 = _mm256_fmadd_ps(qc, k0v, a20);
      a21 = _mm256_fmadd_ps(qc, k1v, a21);
      const __m256 qd = _mm256_loadu_ps(q3 + d);
      a30 = _mm256_fmadd_ps(qd, k0v, a30);
      a31 = _mm256_fmadd_ps(qd, k1v, a31);
    }
    float v00 = hsum256(_mm256_mul_ps(a00, vscale));
    float v01 = hsum256(_mm256_mul_ps(a01, vscale));
    float v10 = hsum256(_mm256_mul_ps(a10, vscale));
    float v11 = hsum256(_mm256_mul_ps(a11, vscale));
    float v20 = hsum256(_mm256_mul_ps(a20, vscale));
    float v21 = hsum256(_mm256_mul_ps(a21, vscale));
    float v30 = hsum256(_mm256_mul_ps(a30, vscale));
    float v31 = hsum256(_mm256_mul_ps(a31, vscale));
    if (b0) { v00 += b0[t]; v01 += b0[t + 1]; }
    if (b1) { v10 += b1[t]; v11 += b1[t + 1]; }
    if (b2) { v20 += b2[t]; v21 += b2[t + 1]; }
    if (b3) { v30 += b3[t]; v31 += b3[t + 1]; }
    p0[t] = v00; p0[t + 1] = v01;
    p1[t] = v10; p1[t + 1] = v11;
    p2[t] = v20; p2[t + 1] = v21;
    p3[t] = v30; p3[t + 1] = v31;
    mx0 = std::max(mx0, std::max(v00, v01));
    mx1 = std::max(mx1, std::max(v10, v11));
    mx2 = std::max(mx2, std::max(v20, v21));
    mx3 = std::max(mx3, std::max(v30, v31));
  }
  for (; t < S; ++t) {
    const float* kt = kb + static_cast<int64_t>(t) * D;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (int d = 0; d + 8 <= D; d += 8) {
      const __m256 kv = _mm256_loadu_ps(kt + d);
      a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + d), kv, a0);
      a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + d), kv, a1);
      a2 = _mm256_fmadd_ps(_mm256_loadu_ps(q2 + d), kv, a2);
      a3 = _mm256_fmadd_ps(_mm256_loadu_ps(q3 + d), kv, a3);
    }
    const float w0 = hsum256(_mm256_mul_ps(a0, vscale)) + (b0 ? b0[t] : 0.f);
    const float w1 = hsum256(_mm256_mul_ps(a1, vscale)) + (b1 ? b1[t] : 0.f);
    const float w2 = hsum256(_mm256_mul_ps(a2, vscale)) + (b2 ? b2[t] : 0.f);
    const float w3 = hsum256(_mm256_mul_ps(a3, vscale)) + (b3 ? b3[t] : 0.f);
    p0[t] = w0; p1[t] = w1; p2[t] = w2; p3[t] = w3;
    mx0 = std::max(mx0, w0); mx1 = std::max(mx1, w1);
    mx2 = std::max(mx2, w2); mx3 = std::max(mx3, w3);
  }

  const float mxs[4] = {mx0, mx1, mx2, mx3};
  float* ps[4] = {p0, p1, p2, p3};
  float* os[4] = {o0, o1, o2, o3};
  for (int r = 0; r < 4; ++r) {
    float* pr = ps[r];
    if (r >= rows || mxs[r] < -1e29f) {
      std::memset(pr, 0, sizeof(float) * static_cast<size_t>(S));
      if (r < rows) std::memset(os[r], 0, sizeof(float) * static_cast<size_t>(D));
      continue;
    }
    const __m256 vm = _mm256_set1_ps(mxs[r]);
    __m256 sumv = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= S; i += 8) {
      const __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(pr + i), vm));
      _mm256_storeu_ps(pr + i, e);
      sumv = _mm256_add_ps(sumv, e);
    }
    float sum = hsum256(sumv);
    for (; i < S; ++i) {
      const float e = std::exp(pr[i] - mxs[r]);
      pr[i] = e;
      sum += e;
    }
    const float inv = 1.f / sum;
    const __m256 vinv = _mm256_set1_ps(inv);
    for (i = 0; i + 8 <= S; i += 8)
      _mm256_storeu_ps(pr + i, _mm256_mul_ps(_mm256_loadu_ps(pr + i), vinv));
    for (; i < S; ++i) pr[i] *= inv;
  }

  for (int d0 = 0; d0 + 16 <= D; d0 += 16) {
    __m256 r00 = _mm256_setzero_ps(), r01 = r00, r10 = r00, r11 = r00;
    __m256 r20 = r00, r21 = r00, r30 = r00, r31 = r00;
    for (int tt = 0; tt < S; ++tt) {
      const float* vr = vb + static_cast<int64_t>(tt) * D + d0;
      const __m256 x0 = _mm256_loadu_ps(vr + 0);
      const __m256 x1 = _mm256_loadu_ps(vr + 8);
      const __m256 w0 = _mm256_set1_ps(p0[tt]);
      r00 = _mm256_fmadd_ps(w0, x0, r00);
      r01 = _mm256_fmadd_ps(w0, x1, r01);
      const __m256 w1 = _mm256_set1_ps(p1[tt]);
      r10 = _mm256_fmadd_ps(w1, x0, r10);
      r11 = _mm256_fmadd_ps(w1, x1, r11);
      const __m256 w2 = _mm256_set1_ps(p2[tt]);
      r20 = _mm256_fmadd_ps(w2, x0, r20);
      r21 = _mm256_fmadd_ps(w2, x1, r21);
      const __m256 w3 = _mm256_set1_ps(p3[tt]);
      r30 = _mm256_fmadd_ps(w3, x0, r30);
      r31 = _mm256_fmadd_ps(w3, x1, r31);
    }
    if (rows > 0) { _mm256_storeu_ps(o0 + d0, r00); _mm256_storeu_ps(o0 + d0 + 8, r01); }
    if (rows > 1) { _mm256_storeu_ps(o1 + d0, r10); _mm256_storeu_ps(o1 + d0 + 8, r11); }
    if (rows > 2) { _mm256_storeu_ps(o2 + d0, r20); _mm256_storeu_ps(o2 + d0 + 8, r21); }
    if (rows > 3) { _mm256_storeu_ps(o3 + d0, r30); _mm256_storeu_ps(o3 + d0 + 8, r31); }
  }
  for (int d0 = (D / 16) * 16; d0 + 8 <= D; d0 += 8) {
    __m256 r0 = _mm256_setzero_ps(), r1 = r0, r2 = r0, r3 = r0;
    for (int tt = 0; tt < S; ++tt) {
      const __m256 xv = _mm256_loadu_ps(vb + static_cast<int64_t>(tt) * D + d0);
      r0 = _mm256_fmadd_ps(_mm256_set1_ps(p0[tt]), xv, r0);
      r1 = _mm256_fmadd_ps(_mm256_set1_ps(p1[tt]), xv, r1);
      r2 = _mm256_fmadd_ps(_mm256_set1_ps(p2[tt]), xv, r2);
      r3 = _mm256_fmadd_ps(_mm256_set1_ps(p3[tt]), xv, r3);
    }
    if (rows > 0) _mm256_storeu_ps(o0 + d0, r0);
    if (rows > 1) _mm256_storeu_ps(o1 + d0, r1);
    if (rows > 2) _mm256_storeu_ps(o2 + d0, r2);
    if (rows > 3) _mm256_storeu_ps(o3 + d0, r3);
  }
}

OV_AVX2 void attention_head_avx2_rb4(const float* q, int64_t q_stride, const float* k,
                                     const float* v, int64_t kv_stride, const float* bias,
                                     int S, int D, float scale, float* out, int64_t out_stride) {
  enable_flush_to_zero();
  static thread_local std::vector<float> kbuf, vbuf, pbuf;
  kbuf.resize(static_cast<size_t>(S) * D);
  vbuf.resize(static_cast<size_t>(S) * D);
  pbuf.resize(static_cast<size_t>(4) * S);
  for (int t = 0; t < S; ++t) {
    std::memcpy(kbuf.data() + static_cast<size_t>(t) * D, k + static_cast<int64_t>(t) * kv_stride,
                sizeof(float) * D);
    std::memcpy(vbuf.data() + static_cast<size_t>(t) * D, v + static_cast<int64_t>(t) * kv_stride,
                sizeof(float) * D);
  }

  int s0 = 0;
  for (; s0 + 4 <= S; s0 += 4) {
    const float* q0 = q + static_cast<int64_t>(s0 + 0) * q_stride;
    const float* q1 = q + static_cast<int64_t>(s0 + 1) * q_stride;
    const float* q2 = q + static_cast<int64_t>(s0 + 2) * q_stride;
    const float* q3 = q + static_cast<int64_t>(s0 + 3) * q_stride;
    const float* b0 = bias ? bias + static_cast<int64_t>(s0 + 0) * S : nullptr;
    const float* b1 = bias ? bias + static_cast<int64_t>(s0 + 1) * S : nullptr;
    const float* b2 = bias ? bias + static_cast<int64_t>(s0 + 2) * S : nullptr;
    const float* b3 = bias ? bias + static_cast<int64_t>(s0 + 3) * S : nullptr;
    attention_block4(kbuf.data(), vbuf.data(), S, D, scale, q0, q1, q2, q3, b0, b1, b2, b3,
                     pbuf.data(), out + static_cast<int64_t>(s0 + 0) * out_stride,
                     out + static_cast<int64_t>(s0 + 1) * out_stride,
                     out + static_cast<int64_t>(s0 + 2) * out_stride,
                     out + static_cast<int64_t>(s0 + 3) * out_stride, out_stride, 4);
  }
  if (s0 < S) {
    const int rem = S - s0;
    const int r1 = rem > 1 ? 1 : 0;
    const int r2 = rem > 2 ? 2 : 0;
    const int r3 = rem > 3 ? 3 : 0;
    const float* q0 = q + static_cast<int64_t>(s0 + 0) * q_stride;
    const float* q1 = q + static_cast<int64_t>(s0 + r1) * q_stride;
    const float* q2 = q + static_cast<int64_t>(s0 + r2) * q_stride;
    const float* q3 = q + static_cast<int64_t>(s0 + r3) * q_stride;
    const float* b0 = bias ? bias + static_cast<int64_t>(s0 + 0) * S : nullptr;
    const float* b1 = bias ? bias + static_cast<int64_t>(s0 + r1) * S : nullptr;
    const float* b2 = bias ? bias + static_cast<int64_t>(s0 + r2) * S : nullptr;
    const float* b3 = bias ? bias + static_cast<int64_t>(s0 + r3) * S : nullptr;
    float* o0 = out + static_cast<int64_t>(s0 + 0) * out_stride;
    float* o1 = out + static_cast<int64_t>(s0 + r1) * out_stride;
    float* o2 = out + static_cast<int64_t>(s0 + r2) * out_stride;
    float* o3 = out + static_cast<int64_t>(s0 + r3) * out_stride;
    attention_block4(kbuf.data(), vbuf.data(), S, D, scale, q0, q1, q2, q3, b0, b1, b2, b3,
                     pbuf.data(), o0, o1, o2, o3, out_stride, rem);
  }
}

OV_AVX2 void attention_head_avx2(const float* q, int64_t q_stride, const float* k,
                                 const float* v, int64_t kv_stride, const float* bias, int S,
                                 int D, float scale, float* out, int64_t out_stride) {
  static const bool rb2 = [] {
    const char* e = std::getenv("OV_ATTN_RB2");
    return e && e[0] && e[0] != '0';
  }();
  if (rb2) {
    attention_head_avx2_rb2(q, q_stride, k, v, kv_stride, bias, S, D, scale, out, out_stride);
    return;
  }
  attention_head_avx2_rb4(q, q_stride, k, v, kv_stride, bias, S, D, scale, out, out_stride);
}

#endif  // OV_X86

#endif  // OV_X86

}  // namespace

void rmsnorm_f32(float* x, const float* w, int64_t rows, int64_t cols, float eps) {
  rmsnorm_into_f32(x, x, w, rows, cols, eps);
}

void rmsnorm_into_f32(float* dst, const float* src, const float* w, int64_t rows, int64_t cols,
                      float eps) {
#ifdef OV_X86
  if (__builtin_cpu_supports("avx2")) { rmsnorm_avx2(dst, src, w, rows, cols, eps); return; }
#endif
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = src + r * cols;
    float* out = dst + r * cols;
    double sum = 0.0;
    for (int64_t i = 0; i < cols; ++i) sum += static_cast<double>(row[i]) * row[i];
    const float scale = 1.f / std::sqrt(static_cast<float>(sum / cols) + eps);
    for (int64_t i = 0; i < cols; ++i) out[i] = row[i] * scale * w[i];
  }
}

void add_inplace_f32(float* dst, const float* src, int64_t n) {
#ifdef OV_X86
  if (__builtin_cpu_supports("avx2")) { add_inplace_avx2(dst, src, n); return; }
#endif
  for (int64_t i = 0; i < n; ++i) dst[i] += src[i];
}

void silu_mul_f32(float* out, const float* gate, const float* up, int64_t n) {
#ifdef OV_X86
  if (__builtin_cpu_supports("avx2")) { silu_mul_avx2(out, gate, up, n); return; }
#endif
  for (int64_t i = 0; i < n; ++i) out[i] = silu(gate[i]) * up[i];
}

void rope_f32(float* x, int64_t rows, int n_heads, int head_dim, const int64_t* pos, float theta) {
  const int half = head_dim / 2;
  // cos/sin depend only on (position, dim), not on the head: precompute them
  // once per row instead of once per (row, head).  That is a 16x reduction in
  // transcendental calls for this model (28 layers x 64.6k -> 20k per layer).
  static thread_local std::vector<float> inv_freq;
  static thread_local int cached_dim = -1;
  static thread_local float cached_theta = -1.f;
  if (cached_dim != head_dim || cached_theta != theta) {
    inv_freq.resize(static_cast<size_t>(half));
    for (int i = 0; i < half; ++i) {
      inv_freq[i] = static_cast<float>(std::pow(static_cast<double>(theta),
                                                -2.0 * i / static_cast<double>(head_dim)));
    }
    cached_dim = head_dim;
    cached_theta = theta;
  }

  // The table depends only on (position list, head_dim, theta), and the LM calls
  // RoPE 56 times per forward with the identical position list, so cache it and
  // rebuild only when the positions actually change.
  static thread_local std::vector<float> cos_tab, sin_tab, cached_pos;
  bool stale = static_cast<int64_t>(cached_pos.size()) != rows;
  if (!stale) {
    for (int64_t r = 0; r < rows; ++r) {
      if (cached_pos[r] != pos[r]) { stale = true; break; }
    }
  }
  if (stale) {
    cos_tab.resize(static_cast<size_t>(rows) * half);
    sin_tab.resize(static_cast<size_t>(rows) * half);
    cached_pos.assign(pos, pos + rows);
    for (int64_t r = 0; r < rows; ++r) {
      const float p = static_cast<float>(pos[r]);
      float* c = cos_tab.data() + r * half;
      float* s = sin_tab.data() + r * half;
      for (int i = 0; i < half; ++i) {
        const float angle = p * inv_freq[i];
        c[i] = std::cos(angle);
        s[i] = std::sin(angle);
      }
    }
  }

#ifdef OV_X86
  if (__builtin_cpu_supports("avx2") && half % 8 == 0) {
    rope_rot_avx2(x, cos_tab.data(), sin_tab.data(), rows, n_heads, head_dim);
    return;
  }
#endif
  for (int64_t r = 0; r < rows; ++r) {
    const float* c = cos_tab.data() + r * half;
    const float* sn = sin_tab.data() + r * half;
    float* row = x + r * n_heads * head_dim;
    for (int h = 0; h < n_heads; ++h) {
      float* v = row + h * head_dim;
      for (int i = 0; i < half; ++i) {
        const float a = v[i];
        const float b = v[i + half];
        v[i] = a * c[i] - b * sn[i];
        v[i + half] = b * c[i] + a * sn[i];
      }
    }
  }
}

void attention_head_f32(const float* q, int64_t q_stride, const float* k, const float* v,
                        int64_t kv_stride, const float* bias, int S, int D, float scale,
                        float* out, int64_t out_stride) {
#ifdef OV_X86
  if (__builtin_cpu_supports("avx2")) {
    attention_head_avx2(q, q_stride, k, v, kv_stride, bias, S, D, scale, out, out_stride);
    return;
  }
#endif
  // Scalar fallback: same math, no blocking.
  std::vector<float> p(static_cast<size_t>(S));
  for (int s = 0; s < S; ++s) {
    const float* qrow = q + static_cast<int64_t>(s) * q_stride;
    float rowmax = -INFINITY;
    for (int t = 0; t < S; ++t) {
      const float* krow = k + static_cast<int64_t>(t) * kv_stride;
      double acc = 0.0;
      for (int d = 0; d < D; ++d) acc += static_cast<double>(qrow[d]) * krow[d];
      const float val = static_cast<float>(acc) * scale + (bias ? bias[s * S + t] : 0.f);
      p[static_cast<size_t>(t)] = val;
      rowmax = std::max(rowmax, val);
    }
    float* orow = out + static_cast<int64_t>(s) * out_stride;
    if (rowmax < -1e29f) {  // fully masked row: same contract as the AVX2 path
      std::memset(orow, 0, sizeof(float) * static_cast<size_t>(D));
      continue;
    }
    float sum = 0.f;
    for (int t = 0; t < S; ++t) { p[t] = std::exp(p[t] - rowmax); sum += p[t]; }
    for (int d = 0; d < D; ++d) orow[d] = 0.f;
    for (int t = 0; t < S; ++t) {
      const float w = p[static_cast<size_t>(t)] / sum;
      const float* vr = v + static_cast<int64_t>(t) * kv_stride;
      for (int d = 0; d < D; ++d) orow[d] += w * vr[d];
    }
  }
}

void qk_norm_rope_f32(float* q, float* k, const float* qw, const float* kw, int64_t rows,
                      int n_heads, int n_kv_heads, int head_dim, const int64_t* pos,
                      float theta, float eps) {
  if (rows <= 0 || head_dim <= 0) return;
  const int half = head_dim / 2;
  static thread_local std::vector<float> inv_freq, cos_tab, sin_tab, cached_pos;
  static thread_local int cached_dim = -1;
  static thread_local float cached_theta = -1.f;
  if (cached_dim != head_dim || cached_theta != theta) {
    inv_freq.resize(static_cast<size_t>(half));
    for (int i = 0; i < half; ++i) {
      inv_freq[static_cast<size_t>(i)] =
          static_cast<float>(std::pow(static_cast<double>(theta),
                                      -2.0 * i / static_cast<double>(head_dim)));
    }
    cached_dim = head_dim;
    cached_theta = theta;
  }
  bool stale = static_cast<int64_t>(cached_pos.size()) != rows;
  if (!stale) {
    for (int64_t r = 0; r < rows; ++r) {
      if (cached_pos[static_cast<size_t>(r)] != pos[r]) {
        stale = true;
        break;
      }
    }
  }
  if (stale) {
    cos_tab.resize(static_cast<size_t>(rows) * half);
    sin_tab.resize(static_cast<size_t>(rows) * half);
    cached_pos.assign(pos, pos + rows);
    for (int64_t r = 0; r < rows; ++r) {
      const float p = static_cast<float>(pos[r]);
      float* c = cos_tab.data() + r * half;
      float* s = sin_tab.data() + r * half;
      for (int i = 0; i < half; ++i) {
        const float angle = p * inv_freq[static_cast<size_t>(i)];
        c[i] = std::cos(angle);
        s[i] = std::sin(angle);
      }
    }
  }
#ifdef OV_X86
  if (__builtin_cpu_supports("avx2") && half % 8 == 0) {
    qk_norm_rope_avx2(q, k, qw, kw, rows, n_heads, n_kv_heads, head_dim, cos_tab.data(),
                      sin_tab.data(), eps);
    return;
  }
#endif
  // Scalar fallback (not used on AVX2 machines).
  for (int64_t r = 0; r < rows; ++r) {
    const float* c = cos_tab.data() + r * half;
    const float* sn = sin_tab.data() + r * half;
    for (int pass = 0; pass < 2; ++pass) {
      float* base = pass == 0 ? q + r * n_heads * head_dim : k + r * n_kv_heads * head_dim;
      const float* w = pass == 0 ? qw : kw;
      const int nh = pass == 0 ? n_heads : n_kv_heads;
      for (int h = 0; h < nh; ++h) {
        float* v = base + h * head_dim;
        float sum = 0.f;
        for (int i = 0; i < head_dim; ++i) sum += v[i] * v[i];
        const float scale = 1.f / std::sqrt(sum / static_cast<float>(head_dim) + eps);
        for (int i = 0; i < half; ++i) {
          const float a = (v[i] * scale) * w[i];
          const float b = (v[i + half] * scale) * w[i + half];
          v[i] = a * c[i] - b * sn[i];
          v[i + half] = b * c[i] + a * sn[i];
        }
      }
    }
  }
}

void layer_norm_f32(float* x, const float* w, const float* b, int64_t rows, int64_t cols,
                    float eps) {
  for (int64_t r = 0; r < rows; ++r) {
    float* row = x + r * cols;
    double mean = 0.0;
    for (int64_t i = 0; i < cols; ++i) mean += row[i];
    mean /= static_cast<double>(cols);
    double var = 0.0;
    for (int64_t i = 0; i < cols; ++i) {
      const double d = row[i] - mean;
      var += d * d;
    }
    var /= static_cast<double>(cols);
    const float inv = 1.f / std::sqrt(static_cast<float>(var) + eps);
    const float m = static_cast<float>(mean);
    for (int64_t i = 0; i < cols; ++i) {
      row[i] = (row[i] - m) * inv * w[i] + (b ? b[i] : 0.f);
    }
  }
}

void elu_f32(float* x, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    x[i] = (x[i] > 0.f) ? x[i] : (std::exp(x[i]) - 1.f);
  }
}

void gelu_f32(float* x, int64_t n) {
  constexpr float kInvSqrt2 = 0.7071067811865476f;
  for (int64_t i = 0; i < n; ++i) x[i] = 0.5f * x[i] * (1.f + std::erf(x[i] * kInvSqrt2));
}

void group_norm_channels_f32(float* x, const float* w, const float* b, int channels, int t,
                             float eps) {
  for (int c = 0; c < channels; ++c) {
    double mean = 0.0;
    for (int i = 0; i < t; ++i) mean += x[static_cast<size_t>(i) * channels + c];
    mean /= t;
    double var = 0.0;
    for (int i = 0; i < t; ++i) {
      const double d = x[static_cast<size_t>(i) * channels + c] - mean;
      var += d * d;
    }
    var /= t;
    const float inv = 1.f / std::sqrt(static_cast<float>(var) + eps);
    const float m = static_cast<float>(mean);
    const float sw = w ? w[c] : 1.f;
    const float sb = b ? b[c] : 0.f;
    for (int i = 0; i < t; ++i) {
      float& v = x[static_cast<size_t>(i) * channels + c];
      v = (v - m) * inv * sw + sb;
    }
  }
}

void conv1d_generic_f32(const Conv1dSpec& c, const float* w_in, const float* x, int t_in, float* y,
                        int t_out) {
  const float* w = w_in ? w_in : c.w;
  if (!w) throw std::runtime_error("conv1d_generic_f32: no weights");
  const int cog = c.cout / c.groups;
  const int cig = c.cin / c.groups;
  const int CK = cig * c.k;
  const int64_t tb = cpu_detail::panel_rows(CK);
  // Grow-only scratch: the old version allocated three vectors per call, and the
  // panel alone reaches ~100 MB for the deepest encoder block.
  std::vector<float>& wtb = cpu_detail::scratch(1);
  std::vector<float>& col = cpu_detail::scratch(0);
  wtb.resize(static_cast<size_t>(CK) * cog);
  col.resize(static_cast<size_t>(tb) * CK);
  const size_t tap_bytes = sizeof(float) * static_cast<size_t>(cig);

  for (int g = 0; g < c.groups; ++g) {
    // PyTorch layout is [cout, cin/groups, k]; a group is a contiguous slice.
    // Transpose to tap-major [k, cin/groups, cout] so that tap kk is a
    // contiguous [cig, cog] GEMM operand and the panel fill is a memcpy.
    const float* wg = w + static_cast<size_t>(g) * cog * CK;
    for (int kk = 0; kk < c.k; ++kk) {
      for (int i = 0; i < cig; ++i) {
        for (int o = 0; o < cog; ++o) {
          wtb[(static_cast<size_t>(kk) * cig + i) * cog + o] =
              wg[(static_cast<size_t>(o) * cig + i) * c.k + kk];
        }
      }
    }
    for (int64_t t0 = 0; t0 < t_out; t0 += tb) {
      const int64_t nb = std::min<int64_t>(tb, t_out - t0);
      for (int64_t r = 0; r < nb; ++r) {
        const int64_t t = t0 + r;
        float* row = col.data() + r * CK;
        const int64_t base = t * c.stride - c.pad;
        for (int kk = 0; kk < c.k; ++kk) {
          const int64_t src = base + kk * c.dil;
          float* dst = row + static_cast<size_t>(kk) * cig;
          if (src >= 0 && src < t_in) {
            std::memcpy(dst, x + src * c.cin + static_cast<size_t>(g) * cig, tap_bytes);
          } else {
            std::memset(dst, 0, tap_bytes);
          }
        }
      }
      // Write straight into the output rows (stride cout, offset g*cog) instead
      // of a temporary [t_out, cog] block plus a scatter loop.
      linear_f32_ld(nb, cog, CK, col.data(), wtb.data(),
                    y + t0 * c.cout + static_cast<size_t>(g) * cog, c.cout, /*accumulate=*/false);
    }
  }
  if (c.b) {
    for (int t = 0; t < t_out; ++t) {
      float* row = y + static_cast<size_t>(t) * c.cout;
      for (int o = 0; o < c.cout; ++o) row[o] += c.b[o];
    }
  }
}

void enable_flush_to_zero() {
#ifdef OV_X86
  _mm_setcsr(_mm_getcsr() | 0x8040);  // FTZ (bit 15) | DAZ (bit 6)
#endif
}

void silu_mul_quantize_f32(int64_t M, int64_t K, const float* gate, const float* up,
                           uint8_t* aq, float* asc) {
  if (M <= 0 || K <= 0) return;
#ifdef OV_X86
  if (__builtin_cpu_supports("avx2")) {
    silu_mul_quant_avx2(M, K, gate, up, aq, asc);
    return;
  }
#endif
  const int64_t Kb = (K + 31) / 32;
  const int64_t Kp = Kb * 32;
  for (int64_t r = 0; r < M; ++r) {
    for (int64_t b = 0; b < Kb; ++b) {
      const int64_t k0 = b * 32, kn = std::min<int64_t>(32, K - k0);
      float amax = 0.f;
      for (int64_t i = 0; i < kn; ++i) {
        const float v = silu(gate[r * K + k0 + i]) * up[r * K + k0 + i];
        amax = std::max(amax, std::fabs(v));
      }
      const float bs = amax / 127.f, inv = amax > 0.f ? 127.f / amax : 0.f;
      asc[r * Kb + b] = bs;
      for (int64_t i = 0; i < kn; ++i) {
        const float v = silu(gate[r * K + k0 + i]) * up[r * K + k0 + i];
        int z = static_cast<int>(std::nearbyint(v * inv));
        z = z < -127 ? -127 : (z > 127 ? 127 : z);
        aq[r * Kp + k0 + i] = static_cast<uint8_t>(z + 128);
      }
      for (int64_t i = kn; i < 32; ++i) aq[r * Kp + k0 + i] = 0;
    }
  }
}

void softmax_bias_f32(float* scores, const float* bias, int64_t rows, int64_t cols) {
#ifdef OV_X86
  if (__builtin_cpu_supports("avx2")) { softmax_avx2(scores, bias, rows, cols); return; }
#endif
  for (int64_t r = 0; r < rows; ++r) {
    float* row = scores + r * cols;
    const float* b = bias ? bias + r * cols : nullptr;
    float m = -INFINITY;
    for (int64_t j = 0; j < cols; ++j) {
      const float v = b ? row[j] + b[j] : row[j];
      if (v > m) m = v;
    }
    if (m < -1e29f) {  // whole row masked out
      std::memset(row, 0, sizeof(float) * static_cast<size_t>(cols));
      continue;
    }
    float sum = 0.f;
    for (int64_t j = 0; j < cols; ++j) {
      const float v = b ? row[j] + b[j] : row[j];
      const float e = std::exp(v - m);
      row[j] = e;
      sum += e;
    }
    const float inv = 1.f / sum;
    for (int64_t j = 0; j < cols; ++j) row[j] *= inv;
  }
}

}  // namespace ov

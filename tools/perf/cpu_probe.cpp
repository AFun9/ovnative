// CPU kernel probe: current implementation vs candidate replacements, at the
// shapes the LM and the DAC decoder actually use.
#include <immintrin.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "model/audio_tokenizer.h"
#include "model/conv.h"
#include "model/mat.h"
#include "model/ops.h"
#include "omnivoice/ov_gemm.h"
#include "omnivoice/ov_weights.h"

using namespace ov;

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
template <class F>
static double bench(int reps, F fn) {
  fn();
  const double t = now();
  for (int i = 0; i < reps; ++i) fn();
  return (now() - t) / reps;
}

#define AVX __attribute__((target("avx2,fma")))

static AVX __m256 exp256(__m256 x) {
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
  const __m256 sc =
      _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23));
  return _mm256_mul_ps(p, sc);
}

static AVX void silu_mul_vec(float* out, const float* g, const float* u, int64_t n) {
  const __m256 one = _mm256_set1_ps(1.f);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 x = _mm256_loadu_ps(g + i);
    const __m256 d = _mm256_add_ps(one, exp256(_mm256_sub_ps(_mm256_setzero_ps(), x)));
    _mm256_storeu_ps(out + i,
                     _mm256_mul_ps(_mm256_div_ps(x, d), _mm256_loadu_ps(u + i)));
  }
  for (; i < n; ++i) out[i] = (g[i] / (1.f + std::exp(-g[i]))) * u[i];
}

// sin(x) with round-to-nearest reduction to [-pi/2, pi/2] and a degree-11 odd
// polynomial; absolute error < 2e-10.
static AVX __m256 sin256(__m256 x) {
  const __m256 pi = _mm256_set1_ps(3.14159265358979323846f);
  const __m256 k = _mm256_round_ps(_mm256_div_ps(x, pi),
                                   _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  // Cody-Waite: pi = 3.140625 + 0.0009670257568359375 + 6.277194e-07
  __m256 r = _mm256_fnmadd_ps(k, _mm256_set1_ps(3.140625f), x);
  r = _mm256_fnmadd_ps(k, _mm256_set1_ps(0.0009670257568359375f), r);
  r = _mm256_fnmadd_ps(k, _mm256_set1_ps(6.277194106520059e-07f), r);
  const __m256 r2 = _mm256_mul_ps(r, r);
  __m256 p = _mm256_set1_ps(-1.f / 39916800.f);
  p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(1.f / 362880.f));
  p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(-1.f / 5040.f));
  p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(1.f / 120.f));
  p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(-1.f / 6.f));
  p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(1.f));
  __m256 s = _mm256_mul_ps(r, p);
  // flip sign where k is odd
  const __m256i ki = _mm256_cvtps_epi32(k);
  const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(ki, _mm256_set1_epi32(1)), 31);
  return _mm256_xor_ps(s, _mm256_castsi256_ps(sign));
}

// Channel-vectorised Snake: v += sin^2(a v)/a, one ymm per 8 channels.
static AVX void snake_vec(float* x, const float* alpha, int ch, int t) {
  const __m256 one = _mm256_set1_ps(1.f);
  for (int i = 0; i < t; ++i) {
    float* row = x + static_cast<size_t>(i) * ch;
    int c = 0;
    for (; c + 8 <= ch; c += 8) {
      const __m256 a = _mm256_loadu_ps(alpha + c);
      const __m256 v = _mm256_loadu_ps(row + c);
      const __m256 s = sin256(_mm256_mul_ps(a, v));
      _mm256_storeu_ps(row + c,
                       _mm256_fmadd_ps(_mm256_div_ps(_mm256_mul_ps(s, s), a), one, v));
    }
    for (; c < ch; ++c) {
      const float a = alpha[c];
      const float s = std::sin(a * row[c]);
      row[c] += s * s / a;
    }
  }
}

// Accuracy-preserving: row-major traversal + OpenMP over frames, still sinf().
static void snake_omp(float* x, const float* alpha, int ch, int t) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < t; ++i) {
    float* row = x + static_cast<size_t>(i) * ch;
    for (int c = 0; c < ch; ++c) {
      const float a = alpha[c];
      const float s = std::sin(a * row[c]);
      row[c] += s * s / a;
    }
  }
}

int main(int argc, char** argv) {
  const int H = 1024, INTER = 3072;
  printf("kernel: %s\n", cpu_kernel_name());

  // ------------------------------------------------ GEMM at LM shapes
  for (int64_t M : {int64_t{218}, int64_t{512}}) {
    const int64_t N = INTER, K = H;
    std::vector<float> A(M * K, 0.01f), Wf(K * N, 0.01f), C(M * N);
    std::vector<int8_t> Wq(K * N, 3);      // old [K,N] layout
    std::vector<int8_t> Wnk(N * K, 3);     // new [N,K] layout
    std::vector<float> sc(N, 0.01f), rsc(N, 0.01f);
    std::vector<int8_t> packed(s8_nk_packed_bytes(N, K));
    std::vector<int32_t> wsum(s8_nk_wsum_elems(N, K));
    s8_nk_pack(N, K, Wnk.data(), packed.data(), wsum.data());
    const double fl = 2.0 * M * N * K;
    const double tf = bench(20, [&] { linear_f32(M, N, K, A.data(), Wf.data(), C.data()); });
    const double tq =
        bench(20, [&] { linear_s8(M, N, K, A.data(), Wq.data(), sc.data(), C.data()); });
    const double tn =
        bench(20, [&] { linear_s8_nk(M, N, K, A.data(), Wnk.data(), rsc.data(), C.data()); });
    const double tv = bench(20, [&] {
      linear_s8_nk_packed(M, N, K, A.data(), packed.data(), wsum.data(), rsc.data(), C.data());
    });
    printf("[gemm]   M=%3ld N=%4ld K=%4d  fp32 %5.0f | s8[K,N] %5.0f | s8[N,K] %5.0f | "
           "s8[N,K]+VNNI %5.0f GFLOPS   (%.2f/%.2f/%.2f ms)\n",
           (long)M, (long)N, (int)K, fl / tf * 1e-9, fl / tq * 1e-9, fl / tn * 1e-9,
           fl / tv * 1e-9, tf * 1e3, tq * 1e3, tv * 1e3);
  }

  // ------------------------------------------------ silu_mul
  {
    const int64_t n = 218 * INTER;
    std::vector<float> g(n, 0.5f), u(n, 2.f), o(n);
    const double tc = bench(20, [&] { silu_mul_f32(o.data(), g.data(), u.data(), n); });
    const double tv = bench(20, [&] { silu_mul_vec(o.data(), g.data(), u.data(), n); });
    // accuracy of the vector version
    silu_mul_f32(o.data(), g.data(), u.data(), n);
    std::vector<float> r(o);
    silu_mul_vec(o.data(), g.data(), u.data(), n);
    float md = 0;
    for (int64_t i = 0; i < n; ++i) md = std::max(md, std::fabs(o[i] - r[i]));
    printf("[silu]   n=%7ld  current %6.2f ms  vectorised %6.2f ms  (x%.1f)  max|d|=%.1e   per forward(28L): %.1f -> %.1f ms\n",
           (long)n, tc * 1e3, tv * 1e3, tc / tv, md, tc * 28e3, tv * 28e3);
  }

  // ------------------------------------------------ snake at decoder sizes
  {
    float maxd = 0;
    printf("[snake]  (ch,t)          current      omp+rowmajor   vectorised   speedup   max|d|\n");
    const int shapes[5][2] = {{512, 1024}, {256, 5120}, {128, 20480}, {64, 40960}, {32, 122880}};
    for (auto& s : shapes) {
      const int ch = s[0], t = s[1];
      std::vector<float> x((size_t)ch * t), alpha(ch);
      std::mt19937 rng(1);
      std::uniform_real_distribution<float> u(-2.f, 2.f);
      for (auto& v : x) v = u(rng);
      for (int c = 0; c < ch; ++c) alpha[c] = 0.2f + 0.001f * c;
      std::vector<float> ref(x);
      const double tc = bench(5, [&] { snake_f32(x.data(), alpha.data(), ch, t); });
      const double tv = bench(5, [&] { snake_vec(x.data(), alpha.data(), ch, t); });
      const double to = bench(5, [&] { snake_omp(x.data(), alpha.data(), ch, t); });
      // accuracy check on a fresh copy
      ref = x;
      snake_f32(ref.data(), alpha.data(), ch, t);
      snake_vec(x.data(), alpha.data(), ch, t);
      for (size_t i = 0; i < x.size(); ++i) maxd = std::max(maxd, std::fabs(x[i] - ref[i]));
      printf("         (%4d,%6d)  %8.2f ms  %8.2f ms  %8.2f ms   x%5.1f / x%5.1f  %.1e\n", ch, t,
             tc * 1e3, to * 1e3, tv * 1e3, tc / to, tc / tv, maxd);
    }
  }

  // ------------------------------------------------ rope
  {
    const int T = 218, NH = 16, D = 128;
    std::vector<float> x((size_t)T * NH * D, 0.01f);
    std::vector<int64_t> pos(T);
    for (int i = 0; i < T; ++i) pos[i] = i;
    const double t = bench(10, [&] { rope_f32(x.data(), T, NH, D, pos.data(), 1e6f); });
    printf("[rope]   %d rows x %d heads: %.2f ms per call -> %.1f ms per forward (2 calls x 28 layers)\n",
           T, NH, t * 1e3, t * 2 * 28 * 1e3);
  }

  // ------------------------------------------------ group norm (HuBERT stem)
  {
    const int ch = 512, t = 11000;
    std::vector<float> x((size_t)ch * t, 0.5f), w(ch, 1.f), b(ch, 0.f);
    const double tc = bench(3, [&] { group_norm_channels_f32(x.data(), w.data(), b.data(), ch, t, 1e-5f); });
    printf("[gnorm]  ch=512 t=11000 (HuBERT stem)  %.1f ms\n", tc * 1e3);
  }

  // ------------------------------------------------ gelu (erf, scalar)
  {
    const int64_t n = 11000 * 512;
    std::vector<float> x(n, 0.3f);
    const double tc = bench(3, [&] { gelu_f32(x.data(), n); });
    printf("[gelu]   n=%ld (stem)  %.1f ms\n", (long)n, tc * 1e3);
  }

  // ------------------------------------------------ im2col allocation cost
  {
    const int t_out = 122880, CK = 32 * 7;
    const double t = bench(3, [&] {
      std::vector<float> col(static_cast<size_t>(t_out) * CK);
      col[0] = 1.f;
      col[col.size() - 1] = 2.f;
    });
    printf("[im2col] fresh vector<float>(%d x %d) = %.1f MB per conv call: %.1f ms (page faults alone)\n",
           t_out, CK, 4.0 * t_out * CK / 1e6, t * 1e3);
  }

  // ------------------------------------------------ full decoder
  if (argc > 1) {
    const WeightFile wf = WeightFile::Open(argv[1]);
    AcousticDecoder dec(wf);
    const int frames = 86;
    std::mt19937 rng(7);
    std::vector<int64_t> codes((size_t)8 * frames);
    for (auto& c : codes) c = rng() % 1024;
    std::vector<float> audio((size_t)frames * 960);
    dec.decode(codes.data(), frames, audio.data());
    const double td = bench(3, [&] { dec.decode(codes.data(), frames, audio.data()); });
    printf("[decoder] 86 frames (3.44 s audio): %.3f s  (RTF %.3f)\n", td, td / 3.44);
  }
  return 0;
}

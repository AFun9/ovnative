#include "model/conv.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "backend/cpu/scratch.h"
#include "ovnative/ov_gemm.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define OV_X86 1
#endif

namespace ov {

namespace cpu_detail {
std::vector<float>& scratch(int slot) {
  static thread_local std::vector<float> buf[2];
  return buf[slot & 1];
}
}  // namespace cpu_detail

namespace {

}  // namespace

void Conv1d::prepare(const float* w) {
  // Tap-major so that tap `kk` is a contiguous [cin, cout] matrix: the im2col
  // panel can then be filled with one memcpy per (frame, tap) instead of a
  // scalar gather, and the same layout feeds the ConvTranspose phase GEMM.
  wt.resize(static_cast<size_t>(cin) * k * cout);
  for (int kk = 0; kk < k; ++kk) {
    for (int i = 0; i < cin; ++i) {
      for (int o = 0; o < cout; ++o) {
        wt[(static_cast<size_t>(kk) * cin + i) * cout + o] =
            w[(static_cast<size_t>(o) * cin + i) * k + kk];
      }
    }
  }
}

void ConvT1d::prepare(const float* w) {
  wt.resize(static_cast<size_t>(cin) * k * cout);
  for (int kk = 0; kk < k; ++kk) {
    for (int i = 0; i < cin; ++i) {
      for (int o = 0; o < cout; ++o) {
        wt[(static_cast<size_t>(kk) * cin + i) * cout + o] =
            w[(static_cast<size_t>(i) * cout + o) * k + kk];
      }
    }
  }
}

// im2col + GEMM, blocked over output frames so the panel is a reused L3-sized
// slab instead of a fresh multi-MB allocation per call.
void conv1d_f32(const Conv1d& c, const float* x, int t_in, float* y) {
  const int t_out = c.out_len(t_in);
  const int CK = c.cin * c.k;
  if (t_out <= 0) return;

  const int64_t tb = cpu_detail::panel_rows(CK);
  std::vector<float>& col = cpu_detail::scratch(0);
  col.resize(static_cast<size_t>(tb) * CK);
  const size_t row_bytes = sizeof(float) * static_cast<size_t>(c.cin);

  for (int64_t t0 = 0; t0 < t_out; t0 += tb) {
    const int64_t nb = std::min<int64_t>(tb, t_out - t0);
    for (int64_t r = 0; r < nb; ++r) {
      const int64_t t = t0 + r;
      float* row = col.data() + r * CK;
      const int64_t base = t * c.stride - c.pad;
      for (int kk = 0; kk < c.k; ++kk) {
        const int64_t src = base + kk * c.dil;
        float* dst = row + static_cast<size_t>(kk) * c.cin;
        if (src >= 0 && src < t_in) {
          std::memcpy(dst, x + src * c.cin, row_bytes);
        } else {
          std::memset(dst, 0, row_bytes);
        }
      }
    }
    linear_f32(nb, c.cout, CK, col.data(), c.wt.data(), y + t0 * c.cout);
  }
  if (c.b) {
    for (int t = 0; t < t_out; ++t) {
      float* row = y + static_cast<size_t>(t) * c.cout;
      for (int o = 0; o < c.cout; ++o) row[o] += c.b[o];
    }
  }
}

// Transposed convolution as `k` accumulating GEMMs.
//
//   t_out = t*stride - pad + kk   =>   Y[(t0+i)*stride + (kk-pad), :] += X[t0+i, :] · W[kk]
//
// so every tap writes a strided slice of Y and no im2col buffer, no scalar axpy
// and no scatter is needed.  This replaces a cin x t x k loop of scalar axpy
// calls (140万 for the first block, 242 ms of the decoder).
void conv_transpose1d_f32(const ConvT1d& c, const float* x, int t_in, float* y) {
  const int t_out = c.out_len(t_in);
  if (t_out <= 0) return;
  std::fill(y, y + static_cast<size_t>(t_out) * c.cout, 0.f);
  if (t_in <= 0) return;

  for (int kk = 0; kk < c.k; ++kk) {
    const int off = kk - c.pad;
    // Rows t whose output index t*stride + off falls inside [0, t_out).
    int64_t t0 = 0, t1 = t_in - 1;
    if (off < 0) t0 = (-off + c.stride - 1) / c.stride;
    const int64_t last = t_out - 1 - off;
    if (last < 0) continue;
    t1 = std::min<int64_t>(t1, last / c.stride);
    if (t1 < t0) continue;
    const int64_t m = t1 - t0 + 1;
    const float* wkk = c.wt.data() + static_cast<size_t>(kk) * c.cin * c.cout;
    linear_f32_ld(m, c.cout, c.cin, x + t0 * c.cin, wkk,
                  y + static_cast<size_t>(t0 * c.stride + off) * c.cout,
                  static_cast<int64_t>(c.stride) * c.cout, /*accumulate=*/true);
  }
  if (c.b) {
    for (int t = 0; t < t_out; ++t) {
      float* row = y + static_cast<size_t>(t) * c.cout;
      for (int o = 0; o < c.cout; ++o) row[o] += c.b[o];
    }
  }
}

void snake_f32(float* x, const float* alpha, int channels, int t) {
  // x is time-major [t, channels], so the channel loop must be the *inner* one:
  // the old channel-outer loop walked a column of the matrix, i.e. one fresh
  // cache line per element.  Row-major also lets OpenMP split over frames, which
  // is worth 6-11x on the big decoder blocks (see docs/perf-review.md).
  const bool parallel = static_cast<int64_t>(t) * channels >= 65536;
#pragma omp parallel for schedule(static) if (parallel)
  for (int i = 0; i < t; ++i) {
    float* row = x + static_cast<size_t>(i) * channels;
    for (int c = 0; c < channels; ++c) {
      const float a = alpha[c] + 1e-9f;
      const float s = std::sin(a * row[c]);
      row[c] += (1.f / a) * s * s;
    }
  }
}

}  // namespace ov

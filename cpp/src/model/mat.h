// Weight views shared by every model (LM, audio tokenizer).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ovnative/ov_weights.h"

namespace ov {

// Linear layer, fp32 or int8.  Two INT8 encodings exist:
//   * pre-transposed [K, N] + per-column scale (OVW1 layout=1, the old default);
//   * output-major     [N, K] + per-row scale    (OVW1 layout=0, --int8-layout nk).
// The second one is what the AVX-VNNI kernel consumes; `packed`/`wsum` hold its
// 4x8 tiled form, built once at load so `mul` is a plain pointer call.
struct Mat {
  const float* w = nullptr;
  const int8_t* q = nullptr;
  const float* scale = nullptr;
  int64_t K = 0, N = 0;
  bool nk = false;              // q is [N, K] with per-row scales
  std::vector<int8_t> packed;   // VNNI 4x8 tiles (empty until prepared)
  std::vector<int32_t> wsum;    // per-32k-block, per-column Σq (u8 zero-point correction)
  std::vector<float> scale_storage;  // owned scales for a combined/sliced Mat
  const float* row_scale() const { return scale_storage.empty() ? scale : scale_storage.data(); }
  void mul(int64_t M, const float* A, float* C) const;  // C[M,N] = A[M,K]·W[K,N]
  // C[M,N] += A[M,K]·W[K,N] where the kernel supports it (fp32 or packed VNNI).
  bool add_supported() const;
  void mul_add(int64_t M, const float* A, float* C) const;

  // VNNI only: true when the 4x8 tiles are built, i.e. the caller can hoist the
  // activation quantisation with s8_nk_quantize() and reuse it across the
  // projections that share this input (q/k/v, gate/up).
  bool vnni_packed() const { return nk && !packed.empty(); }
  // C[M,N] = A·W with A already quantised.
  void mul_q(int64_t M, const uint8_t* aq, const float* asc, float* C) const;
  void mul_q_add(int64_t M, const uint8_t* aq, const float* asc, float* C) const;
};

// Lookup table [rows, cols], row-major; fp32 or int8 + per-row scale.
struct Emb {
  const float* w = nullptr;
  const int8_t* q = nullptr;
  const float* scale = nullptr;
  int64_t rows = 0, cols = 0;
  void gather(int64_t id, float* out) const;
};

// Loaders: throw with the tensor name on any mismatch.
Mat make_linear(const WeightFile& wf, const std::string& name);
Emb make_emb(const WeightFile& wf, const std::string& name);
const float* make_norm(const WeightFile& wf, const std::string& name, int64_t expect);

}  // namespace ov

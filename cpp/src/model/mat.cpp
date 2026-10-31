#include "model/mat.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "ovnative/ov_gemm.h"

namespace ov {

void Mat::mul(int64_t M, const float* A, float* C) const {
  if (w) {
    linear_f32(M, N, K, A, w, C);
  } else if (q) {
    if (nk) {
      if (!packed.empty()) {
        linear_s8_nk_packed(M, N, K, A, packed.data(), wsum.data(), row_scale(), C);
      } else {
        linear_s8_nk(M, N, K, A, q, row_scale(), C);
      }
    } else {
      linear_s8(M, N, K, A, q, row_scale(), C);
    }
  } else {
    throw std::runtime_error("Mat::mul: empty weight");
  }
}

bool Mat::add_supported() const { return w != nullptr || (nk && !packed.empty()); }

void Mat::mul_q_add(int64_t M, const uint8_t* aq, const float* asc, float* C) const {
  if (nk && !packed.empty()) {
    linear_s8_nk_packed_q_add(M, N, K, aq, asc, packed.data(), wsum.data(), row_scale(), C);
    return;
  }
  throw std::runtime_error("Mat::mul_q_add: weight does not support accumulation");
}

void Mat::mul_add(int64_t M, const float* A, float* C) const {
  if (w) {
    linear_f32_ld(M, N, K, A, w, C, N, true);
    return;
  }
  if (nk && !packed.empty()) {
    linear_s8_nk_packed_add(M, N, K, A, packed.data(), wsum.data(), row_scale(), C);
    return;
  }
  throw std::runtime_error("Mat::mul_add: weight does not support accumulation");
}

void Mat::mul_q(int64_t M, const uint8_t* aq, const float* asc, float* C) const {
  if (!vnni_packed()) throw std::runtime_error("Mat::mul_q: weight is not AVX-VNNI packed");
  linear_s8_nk_packed_q(M, N, K, aq, asc, packed.data(), wsum.data(), row_scale(), C);
}

void Emb::gather(int64_t id, float* out) const {
  if (id < 0 || id >= rows) {
    throw std::runtime_error("embedding id out of range: " + std::to_string(id));
  }
  if (w) {
    std::memcpy(out, w + id * cols, sizeof(float) * static_cast<size_t>(cols));
  } else {
    const int8_t* row = q + id * cols;
    const float s = scale[id];
    for (int64_t c = 0; c < cols; ++c) out[c] = static_cast<float>(row[c]) * s;
  }
}

Mat make_linear(const WeightFile& wf, const std::string& name) {
  const Tensor& t = wf.Get(name);
  if (t.shape.ndim != 2) throw std::runtime_error(name + ": expected a 2-D weight");
  Mat m;
  if (t.transposed) {
    // Old / fp32 encoding: [K, N], scales per output column.
    m.K = t.shape[0];
    m.N = t.shape[1];
    if (t.dtype == DType::F32) {
      m.w = t.f32();
    } else if (t.dtype == DType::I8 && t.scales) {
      m.q = t.i8();
      m.scale = t.scales;
      if (t.n_scales != m.N) throw std::runtime_error(name + ": scale count != N");
    } else {
      throw std::runtime_error(name + ": unsupported dtype for a linear weight");
    }
    return m;
  }
  if (t.dtype == DType::I8 && t.scales) {
    // Output-major int8 ([N, K], layout=0, one scale per output row).  The
    // stored tensor has the same encoding as an embedding table, but it is
    // consumed through Mat, not Gather.
    m.nk = true;
    m.N = t.shape[0];
    m.K = t.shape[1];
    m.q = t.i8();
    m.scale = t.scales;
    if (t.n_scales != m.N) throw std::runtime_error(name + ": scale count != N (nk layout)");
    if (cpu_vnni_available()) {
      m.packed.resize(static_cast<size_t>(s8_nk_packed_bytes(m.N, m.K)));
      m.wsum.assign(static_cast<size_t>(s8_nk_wsum_elems(m.N, m.K)), 0);
      s8_nk_pack(m.N, m.K, m.q, m.packed.data(), m.wsum.data());
    }
    return m;
  }
  throw std::runtime_error(name + ": expected a pre-transposed weight or an int8 [N,K] weight");
}

Emb make_emb(const WeightFile& wf, const std::string& name) {
  const Tensor& t = wf.Get(name);
  if (t.shape.ndim != 2) throw std::runtime_error(name + ": expected a 2-D table");
  if (t.transposed) throw std::runtime_error(name + ": embedding must stay row-major");
  Emb e;
  e.rows = t.shape[0];
  e.cols = t.shape[1];
  if (t.dtype == DType::F32) {
    e.w = t.f32();
  } else if (t.dtype == DType::I8 && t.scales) {
    e.q = t.i8();
    e.scale = t.scales;
    if (t.n_scales != e.rows) throw std::runtime_error(name + ": scale count != rows");
  } else {
    throw std::runtime_error(name + ": unsupported dtype for an embedding");
  }
  return e;
}

const float* make_norm(const WeightFile& wf, const std::string& name, int64_t expect) {
  const Tensor& t = wf.Get(name);
  if (t.dtype != DType::F32) throw std::runtime_error(name + ": norm must be fp32");
  if (expect >= 0 && t.shape.numel() != expect) {
    throw std::runtime_error(name + ": expected " + std::to_string(expect) + " elements, got " +
                             std::to_string(t.shape.numel()));
  }
  return t.f32();
}

}  // namespace ov

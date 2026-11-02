// Self-contained correctness tests for the CPU GEMM kernels.
// The reference is a naive double-precision triple loop, so these tests catch
// both micro-kernel bugs and tail/edge-case bugs without needing Python.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>

#include "model/ops.h"
#include "ovnative/ov_gemm.h"
#include "ovnative/ov_weights.h"

namespace {

void fill(std::vector<float>& v, std::mt19937& rng, float lo = -1.f, float hi = 1.f) {
  std::uniform_real_distribution<float> d(lo, hi);
  for (auto& x : v) x = d(rng);
}

// Reference: C = A[M,K] @ W[K,N], accumulated in double.
void naive_f32(int64_t M, int64_t N, int64_t K, const float* A, const float* W,
               const float* col_scale, float* C) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int64_t k = 0; k < K; ++k) acc += static_cast<double>(A[i * K + k]) * W[k * N + j];
      C[i * N + j] = static_cast<float>(acc * (col_scale ? col_scale[j] : 1.0));
    }
  }
}

bool check(const char* name, const std::vector<float>& got, const std::vector<float>& ref,
           double tol = 1e-5) {
  double maxd = 0.0, maxr = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    maxd = std::max(maxd, std::fabs(static_cast<double>(got[i]) - ref[i]));
    maxr = std::max(maxr, std::fabs(static_cast<double>(ref[i])));
  }
  const double rel = maxd / (maxr + 1e-12);
  const bool ok = rel <= tol;
  std::printf("  %-34s max|delta|=%.3e  rel=%.2e  %s\n", name, maxd, rel, ok ? "OK" : "FAIL");
  return ok;
}

bool test_f32(int64_t M, int64_t N, int64_t K, std::mt19937& rng) {
  std::vector<float> A(M * K), W(K * N), C(M * N), R(M * N);
  fill(A, rng);
  fill(W, rng);
  ov::linear_f32(M, N, K, A.data(), W.data(), C.data());
  naive_f32(M, N, K, A.data(), W.data(), nullptr, R.data());
  char name[128];
  std::snprintf(name, sizeof name, "f32 %lldx%lldx%lld", (long long)M, (long long)N, (long long)K);
  return check(name, C, R);
}

bool test_s8(int64_t M, int64_t N, int64_t K, std::mt19937& rng) {
  std::vector<float> A(M * K), W(K * N), C(M * N), R(M * N), Wref(K * N), scale(N);
  fill(A, rng);
  fill(W, rng);

  // per-column symmetric INT8, mirroring the OVW1 convention
  std::vector<int8_t> q(K * N);
  for (int64_t j = 0; j < N; ++j) {
    float amax = 0.f;
    for (int64_t k = 0; k < K; ++k) amax = std::max(amax, std::fabs(W[k * N + j]));
    scale[j] = std::max(amax / 127.f, 1e-12f);
    for (int64_t k = 0; k < K; ++k) {
      const float v = std::nearbyint(W[k * N + j] / scale[j]);
      q[k * N + j] = static_cast<int8_t>(std::clamp(v, -127.f, 127.f));
      Wref[k * N + j] = static_cast<float>(q[k * N + j]) * scale[j];
    }
  }

  ov::linear_s8(M, N, K, A.data(), q.data(), scale.data(), C.data());
  naive_f32(M, N, K, A.data(), Wref.data(), nullptr, R.data());
  char name[128];
  std::snprintf(name, sizeof name, "s8  %lldx%lldx%lld", (long long)M, (long long)N, (long long)K);
  return check(name, C, R);
}


// [N, K] int8 + per-row scale, exact float activations (the non-VNNI path).
bool test_s8_nk(int64_t M, int64_t N, int64_t K, std::mt19937& rng) {
  std::vector<float> A(M * K), C(M * N), R(M * N);
  std::vector<int8_t> q(N * K);
  std::vector<float> scale(N);
  fill(A, rng);
  std::uniform_int_distribution<int> wd(-127, 127);
  for (auto& v : q) v = static_cast<int8_t>(wd(rng));
  std::uniform_real_distribution<float> sd(1e-4f, 1e-2f);
  for (auto& v : scale) v = sd(rng);
  ov::linear_s8_nk(M, N, K, A.data(), q.data(), scale.data(), C.data());
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int64_t k = 0; k < K; ++k) acc += static_cast<double>(A[i * K + k]) * q[j * K + k];
      R[i * N + j] = static_cast<float>(acc * scale[j]);
    }
  }
  char name[128];
  std::snprintf(name, sizeof name, "s8nk %lldx%lldx%lld", (long long)M, (long long)N,
                (long long)K);
  return check(name, C, R);
}

// Packed 4x8 VNNI path vs a double reference.  Activations are multiples of
// 0.5 with a 63.5 maximum in every 32-k block, so amax/127 == 0.5 exactly and
// the engine's per-block quantisation is lossless; any mismatch is a tile /
// zero-point / scale bug, not quantisation noise.
bool test_s8_nk_vnni(int64_t M, int64_t N, int64_t K, std::mt19937& rng) {
  if (!ov::cpu_vnni_available()) {
    std::printf("  %-34s SKIP (no AVX-VNNI)\n", "s8nk+vnni");
    return true;
  }
  std::vector<float> A(M * K), C(M * N), R(M * N);
  std::vector<int8_t> q(N * K);
  std::vector<float> scale(N);
  std::uniform_int_distribution<int> ad(-127, 127);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t k = 0; k < K; ++k) A[i * K + k] = 0.5f * static_cast<float>(ad(rng));
    for (int64_t k0 = 0; k0 < K; k0 += 32) A[i * K + k0] = 63.5f;
  }
  std::uniform_int_distribution<int> wd(-127, 127);
  for (auto& v : q) v = static_cast<int8_t>(wd(rng));
  std::uniform_real_distribution<float> sd(1e-4f, 1e-2f);
  for (auto& v : scale) v = sd(rng);
  std::vector<int8_t> packed(ov::s8_nk_packed_bytes(N, K));
  std::vector<int32_t> wsum(ov::s8_nk_wsum_elems(N, K));
  ov::s8_nk_pack(N, K, q.data(), packed.data(), wsum.data());
  ov::linear_s8_nk_packed(M, N, K, A.data(), packed.data(), wsum.data(), scale.data(), C.data());
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int64_t k = 0; k < K; ++k) acc += static_cast<double>(A[i * K + k]) * q[j * K + k];
      R[i * N + j] = static_cast<float>(acc * scale[j]);
    }
  }
  char name[128];
  std::snprintf(name, sizeof name, "s8nk+vnni %lldx%lldx%lld", (long long)M, (long long)N,
                (long long)K);
  return check(name, C, R);
}


// The hoisted quantisation (s8_nk_quantize + linear_s8_nk_packed_q) must be
// bit-identical to the one-shot path that quantises inside linear_s8_nk_packed.
bool test_s8_nk_pool(int64_t M, int64_t N, int64_t K, std::mt19937& rng) {
  if (!ov::cpu_vnni_available()) {
    std::printf("  %-34s SKIP (no AVX-VNNI)\n", "s8nk pooled");
    return true;
  }
  std::vector<float> A(M * K), C1(M * N), C2(M * N);
  fill(A, rng);
  std::vector<int8_t> q(N * K);
  std::uniform_int_distribution<int> wd(-127, 127);
  for (auto& v : q) v = static_cast<int8_t>(wd(rng));
  std::uniform_real_distribution<float> sd(1e-4f, 1e-2f);
  std::vector<float> scale(N);
  for (auto& v : scale) v = sd(rng);
  std::vector<int8_t> packed(ov::s8_nk_packed_bytes(N, K));
  std::vector<int32_t> wsum(ov::s8_nk_wsum_elems(N, K));
  ov::s8_nk_pack(N, K, q.data(), packed.data(), wsum.data());
  std::vector<uint8_t> aq(ov::s8_nk_quant_bytes(M, K));
  std::vector<float> asc(ov::s8_nk_quant_scales(M, K));
  ov::linear_s8_nk_packed(M, N, K, A.data(), packed.data(), wsum.data(), scale.data(), C1.data());
  ov::s8_nk_quantize(M, K, A.data(), aq.data(), asc.data());
  ov::linear_s8_nk_packed_q(M, N, K, aq.data(), asc.data(), packed.data(), wsum.data(),
                            scale.data(), C2.data());
  const bool same = std::memcmp(C1.data(), C2.data(), C1.size() * sizeof(float)) == 0;
  char name[128];
  std::snprintf(name, sizeof name, "s8nk pooled %lldx%lldx%lld", (long long)M, (long long)N,
                (long long)K);
  std::printf("  %-34s %s\n", name, same ? "bit-identical OK" : "FAIL");
  return same;
}

// Fused silu(gate)*up + quantise must byte-match silu_mul_f32 then s8_nk_quantize.
bool test_silu_quant_fused(int64_t M, int64_t K, std::mt19937& rng) {
  if (!ov::cpu_vnni_available()) {
    std::printf("  %-34s SKIP (no AVX-VNNI)\n", "silu+quant");
    return true;
  }
  std::vector<float> gate(M * K), up(M * K), act(M * K);
  fill(gate, rng);
  fill(up, rng);
  std::vector<uint8_t> a1(ov::s8_nk_quant_bytes(M, K)), a2(a1.size());
  std::vector<float> s1(ov::s8_nk_quant_scales(M, K)), s2(s1.size());
  ov::silu_mul_f32(act.data(), gate.data(), up.data(), M * K);
  ov::s8_nk_quantize(M, K, act.data(), a1.data(), s1.data());
  ov::silu_mul_quantize_f32(M, K, gate.data(), up.data(), a2.data(), s2.data());
  const bool same = std::memcmp(a1.data(), a2.data(), a1.size()) == 0 &&
                    std::memcmp(s1.data(), s2.data(), s1.size() * sizeof(float)) == 0;
  char name[128];
  std::snprintf(name, sizeof name, "silu+quant %lldx%lld", (long long)M, (long long)K);
  std::printf("  %-34s %s\n", name, same ? "bit-identical OK" : "FAIL");
  return same;
}

// Pre-quantised residual-add kernel must match q GEMM into P then C += P.
bool test_s8_nk_qadd(int64_t M, int64_t N, int64_t K, std::mt19937& rng) {
  if (!ov::cpu_vnni_available()) {
    std::printf("  %-34s SKIP (no AVX-VNNI)\n", "s8nk qadd");
    return true;
  }
  std::vector<float> A(M * K), C1(M * N), C2(M * N), P(M * N);
  fill(A, rng);
  fill(C1, rng);
  C2 = C1;
  std::vector<uint8_t> aq(ov::s8_nk_quant_bytes(M, K));
  std::vector<float> asc(ov::s8_nk_quant_scales(M, K));
  ov::s8_nk_quantize(M, K, A.data(), aq.data(), asc.data());
  std::vector<int8_t> q(N * K);
  std::uniform_int_distribution<int> wd(-127, 127);
  for (auto& v : q) v = static_cast<int8_t>(wd(rng));
  std::uniform_real_distribution<float> sd(1e-4f, 1e-2f);
  std::vector<float> scale(N);
  for (auto& v : scale) v = sd(rng);
  std::vector<int8_t> packed(ov::s8_nk_packed_bytes(N, K));
  std::vector<int32_t> wsum(ov::s8_nk_wsum_elems(N, K));
  ov::s8_nk_pack(N, K, q.data(), packed.data(), wsum.data());
  ov::linear_s8_nk_packed_q(M, N, K, aq.data(), asc.data(), packed.data(), wsum.data(),
                            scale.data(), P.data());
  for (size_t i = 0; i < P.size(); ++i) C1[i] += P[i];
  ov::linear_s8_nk_packed_q_add(M, N, K, aq.data(), asc.data(), packed.data(), wsum.data(),
                                scale.data(), C2.data());
  const bool same = std::memcmp(C1.data(), C2.data(), C1.size() * sizeof(float)) == 0;
  char name[128];
  std::snprintf(name, sizeof name, "s8nk qadd %lldx%lldx%lld", (long long)M, (long long)N,
                (long long)K);
  std::printf("  %-34s %s\n", name, same ? "bit-identical OK" : "FAIL");
  return same;
}

// Sliced combined GEMM must produce exactly the same bytes as running each
// slice through linear_s8_nk_packed_q separately.
bool test_s8_nk_sliced(int64_t M, int64_t K, std::mt19937& rng) {
  if (!ov::cpu_vnni_available()) {
    std::printf("  %-34s SKIP (no AVX-VNNI)\n", "s8nk sliced");
    return true;
  }
  const int64_t Ns[2] = {128, 64};
  std::vector<int8_t> p0, p1;
  std::vector<int32_t> w0, w1;
  std::vector<float> s0(static_cast<size_t>(Ns[0])), s1(static_cast<size_t>(Ns[1]));
  std::uniform_int_distribution<int> wd(-127, 127);
  std::uniform_real_distribution<float> sd(1e-4f, 1e-2f);
  auto fill_slice = [&](int64_t N, std::vector<int8_t>& p, std::vector<int32_t>& w,
                        std::vector<float>& sc) {
    std::vector<int8_t> q(static_cast<size_t>(N * K));
    for (auto& v : q) v = static_cast<int8_t>(wd(rng));
    for (auto& v : sc) v = sd(rng);
    p.resize(static_cast<size_t>(ov::s8_nk_packed_bytes(N, K)));
    w.resize(static_cast<size_t>(ov::s8_nk_wsum_elems(N, K)));
    ov::s8_nk_pack(N, K, q.data(), p.data(), w.data());
  };
  fill_slice(Ns[0], p0, w0, s0);
  fill_slice(Ns[1], p1, w1, s1);

  const int64_t Kb = (K + 31) / 32, Npad0 = ((Ns[0] + 7) / 8) * 8,
                Npad1 = ((Ns[1] + 7) / 8) * 8, NpadT = Npad0 + Npad1;
  std::vector<int8_t> packed(p0.size() + p1.size());
  std::memcpy(packed.data(), p0.data(), p0.size());
  std::memcpy(packed.data() + p0.size(), p1.data(), p1.size());
  std::vector<int32_t> wsum(static_cast<size_t>(Kb * NpadT), 0);
  for (int64_t b = 0; b < Kb; ++b) {
    std::memcpy(wsum.data() + b * NpadT, w0.data() + b * Npad0, sizeof(int32_t) * Npad0);
    std::memcpy(wsum.data() + b * NpadT + Npad0, w1.data() + b * Npad1,
                sizeof(int32_t) * Npad1);
  }
  std::vector<float> scale(static_cast<size_t>(Ns[0] + Ns[1]));
  std::memcpy(scale.data(), s0.data(), sizeof(float) * Ns[0]);
  std::memcpy(scale.data() + Ns[0], s1.data(), sizeof(float) * Ns[1]);

  std::vector<float> A(static_cast<size_t>(M * K)), C0(static_cast<size_t>(M * Ns[0])),
      C1b(static_cast<size_t>(M * Ns[1])), D0(C0.size()), D1(C1b.size());
  fill(A, rng);
  std::vector<uint8_t> aq(static_cast<size_t>(ov::s8_nk_quant_bytes(M, K)));
  std::vector<float> asc(static_cast<size_t>(ov::s8_nk_quant_scales(M, K)));
  ov::s8_nk_quantize(M, K, A.data(), aq.data(), asc.data());
  ov::linear_s8_nk_packed_q(M, Ns[0], K, aq.data(), asc.data(), p0.data(), w0.data(),
                            s0.data(), C0.data());
  ov::linear_s8_nk_packed_q(M, Ns[1], K, aq.data(), asc.data(), p1.data(), w1.data(),
                            s1.data(), C1b.data());
  const ov::S8NkSlice slices[2] = {{Ns[0], D0.data()}, {Ns[1], D1.data()}};
  ov::linear_s8_nk_packed_q_sliced(M, K, aq.data(), asc.data(), packed.data(), wsum.data(),
                                   scale.data(), slices, 2);
  const bool same = std::memcmp(C0.data(), D0.data(), C0.size() * sizeof(float)) == 0 &&
                    std::memcmp(C1b.data(), D1.data(), C1b.size() * sizeof(float)) == 0;
  char name[128];
  std::snprintf(name, sizeof name, "s8nk sliced %lldx%lld", (long long)M, (long long)K);
  std::printf("  %-34s %s\n", name, same ? "bit-identical OK" : "FAIL");
  return same;
}

// The residual-add epilogue must be byte-identical to computing P then C+=P.
bool test_s8_nk_add(int64_t M, int64_t N, int64_t K, std::mt19937& rng) {
  if (!ov::cpu_vnni_available()) {
    std::printf("  %-34s SKIP (no AVX-VNNI)\n", "s8nk add");
    return true;
  }
  std::vector<float> A(M * K), C1(M * N), C2(M * N), P(M * N);
  fill(A, rng);
  fill(C1, rng);
  C2 = C1;
  std::vector<int8_t> q(N * K);
  std::uniform_int_distribution<int> wd(-127, 127);
  for (auto& v : q) v = static_cast<int8_t>(wd(rng));
  std::uniform_real_distribution<float> sd(1e-4f, 1e-2f);
  std::vector<float> scale(N);
  for (auto& v : scale) v = sd(rng);
  std::vector<int8_t> packed(ov::s8_nk_packed_bytes(N, K));
  std::vector<int32_t> wsum(ov::s8_nk_wsum_elems(N, K));
  ov::s8_nk_pack(N, K, q.data(), packed.data(), wsum.data());
  ov::linear_s8_nk_packed(M, N, K, A.data(), packed.data(), wsum.data(), scale.data(), P.data());
  for (size_t i = 0; i < P.size(); ++i) C1[i] += P[i];
  ov::linear_s8_nk_packed_add(M, N, K, A.data(), packed.data(), wsum.data(), scale.data(),
                              C2.data());
  const bool same = std::memcmp(C1.data(), C2.data(), C1.size() * sizeof(float)) == 0;
  char name[128];
  std::snprintf(name, sizeof name, "s8nk add %lldx%lldx%lld", (long long)M, (long long)N,
                (long long)K);
  std::printf("  %-34s %s\n", name, same ? "bit-identical OK" : "FAIL");
  return same;
}

// Fused RMSNorm+quantise must be byte-identical to writing xn and quantising it.
bool test_norm_quant_fused(int64_t M, int64_t K, std::mt19937& rng) {
  if (!ov::cpu_vnni_available()) {
    std::printf("  %-34s SKIP (no AVX-VNNI)\n", "rmsnorm+quant");
    return true;
  }
  std::vector<float> x(M * K), w(K), xn(M * K);
  fill(x, rng);
  std::uniform_real_distribution<float> wd(0.1f, 1.1f);
  for (auto& v : w) v = wd(rng);
  std::vector<uint8_t> a1(ov::s8_nk_quant_bytes(M, K)), a2(a1.size());
  std::vector<float> s1(ov::s8_nk_quant_scales(M, K)), s2(s1.size());
  ov::rmsnorm_into_f32(xn.data(), x.data(), w.data(), M, K, 1e-6f);
  ov::s8_nk_quantize(M, K, xn.data(), a1.data(), s1.data());
  ov::rmsnorm_quantize_f32(M, K, x.data(), w.data(), 1e-6f, a2.data(), s2.data());
  const bool same = std::memcmp(a1.data(), a2.data(), a1.size()) == 0 &&
                    std::memcmp(s1.data(), s2.data(), s1.size() * sizeof(float)) == 0;
  char name[128];
  std::snprintf(name, sizeof name, "rmsnorm+quant %lldx%lld", (long long)M, (long long)K);
  std::printf("  %-34s %s\n", name, same ? "bit-identical OK" : "FAIL");
  return same;
}


// Minimal OVW1 file: header (128 B) + one descriptor + name + 4 floats. This
// exercises the mmap/CreateFileMapping path on every platform, including
// Windows, where it is otherwise only used by full model weights.
bool test_weight_file() {
  const auto path = (std::filesystem::temp_directory_path() / "ovnative_ovw1_test.bin").string();
  std::vector<uint8_t> b(272, 0);
  auto put32 = [&](size_t off, uint32_t v) { std::memcpy(b.data() + off, &v, 4); };
  auto put64 = [&](size_t off, uint64_t v) { std::memcpy(b.data() + off, &v, 8); };
  std::memcpy(b.data(), "OMNIVOIC", 8);
  put32(8, 1);      // version
  put32(12, 0);     // dtype_mode
  put32(16, 1);     // n_tensors
  put32(20, 128);   // header_bytes
  put64(24, 128);   // index_off
  put64(32, 192);   // names_off
  put64(40, 256);   // data_off
  put64(48, 272);   // file_size
  put64(56, 0);     // flags
  put64(64, 4);     // n_params
  // descriptor at 128
  put32(128, 0);    // name_off
  put32(132, 1);    // name_len
  b[136] = 0;       // FP32
  b[137] = 2;       // ndim
  b[138] = 0;       // layout
  b[139] = 0;       // quant
  put32(140, 2);    // dims[0]
  put32(144, 2);    // dims[1]
  put64(160, 0);    // hash
  put64(168, 256);  // data_off
  put64(176, 16);   // nbytes
  put64(184, 0);    // aux_off
  b[192] = 'x';
  const float vals[4] = {1.f, 2.f, 3.f, 4.f};
  std::memcpy(b.data() + 256, vals, sizeof(vals));
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const size_t wrote = std::fwrite(b.data(), 1, b.size(), f);
  std::fclose(f);
  bool ok = wrote == b.size();
  try {
    ov::WeightFile wf = ov::WeightFile::Open(path);
    const ov::Tensor& t = wf.Get("x");
    ok = ok && t.shape.ndim == 2 && t.shape.dims[0] == 2 && t.shape.dims[1] == 2;
    ok = ok && t.nbytes == 16 && t.f32()[0] == 1.f && t.f32()[3] == 4.f;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "weight-file test: %s\n", e.what());
    ok = false;
  }
  std::remove(path.c_str());
  std::printf("  %-34s %s\n", "ovw1 mmap", ok ? "OK" : "FAIL");
  return ok;
}

// Fused per-head RMSNorm+RoPE must byte-match the four separate passes.
bool test_qk_norm_rope(std::mt19937& rng) {
  const int T = 7, NH = 4, NKV = 2, D = 128;
  std::vector<float> q(static_cast<size_t>(T) * NH * D), k(static_cast<size_t>(T) * NKV * D),
      qw(D), kw(D), q2, k2;
  fill(q, rng);
  fill(k, rng);
  fill(qw, rng, 0.5f, 1.5f);
  fill(kw, rng, 0.5f, 1.5f);
  std::vector<int64_t> pos(T);
  for (int i = 0; i < T; ++i) pos[i] = i * 3;
  q2 = q;
  k2 = k;
  ov::rmsnorm_f32(q.data(), qw.data(), static_cast<int64_t>(T) * NH, D, 1e-6f);
  ov::rmsnorm_f32(k.data(), kw.data(), static_cast<int64_t>(T) * NKV, D, 1e-6f);
  ov::rope_f32(q.data(), T, NH, D, pos.data(), 1e6f);
  ov::rope_f32(k.data(), T, NKV, D, pos.data(), 1e6f);
  ov::qk_norm_rope_f32(q2.data(), k2.data(), qw.data(), kw.data(), T, NH, NKV, D, pos.data(), 1e6f,
                       1e-6f);
  const bool same = std::memcmp(q.data(), q2.data(), q.size() * sizeof(float)) == 0 &&
                    std::memcmp(k.data(), k2.data(), k.size() * sizeof(float)) == 0;
  std::printf("  %-34s %s\n", "qk norm+rope", same ? "bit-identical OK" : "FAIL");
  return same;
}

// kAttnRB=2 zeroes a fully-masked pair of query rows.  out_stride is deliberately
// larger than D (the LM passes QD), which used to make the pair memset cross
// into the padding instead of the second row.
bool test_attention_masked() {
  if (std::strcmp(ov::cpu_kernel_name(), "scalar") == 0) {
    std::printf("  %-34s SKIP (scalar)\n", "attention fully-masked");
    return true;
  }
  const int S = 6, D = 128;
  const int64_t stride = D * 2;
  std::mt19937 rng(5);
  std::vector<float> q(static_cast<size_t>(S) * stride), k(static_cast<size_t>(S) * D),
      v(static_cast<size_t>(S) * D), bias(static_cast<size_t>(S) * S, -1e30f),
      out(static_cast<size_t>(S) * stride, 1.f);
  fill(q, rng);
  fill(k, rng);
  fill(v, rng);
  ov::attention_head_f32(q.data(), stride, k.data(), v.data(), D, bias.data(), S, D, 0.1f,
                         out.data(), stride);
  bool ok = true;
  for (int s = 0; s < S; ++s)
    for (int d = 0; d < D; ++d)
      if (out[static_cast<size_t>(s) * stride + d] != 0.f) ok = false;
  std::printf("  %-34s %s\n", "attention fully-masked", ok ? "OK" : "FAIL");
  return ok;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::mt19937 rng(1234);
  std::printf("kernel: %s\n", ov::cpu_kernel_name());
  bool ok = true;

  // row block (4) and column block (8) aligned
  ok &= test_f32(4, 8, 16, rng);
  // every tail path: M%4, N%8, small K
  ok &= test_f32(7, 13, 5, rng);
  // single row (decode step)
  ok &= test_f32(1, 33, 9, rng);
  // realistic projection
  ok &= test_f32(2, 1024, 1024, rng);
  ok &= test_s8(4, 32, 64, rng);
  ok &= test_s8(3, 9, 7, rng);
  ok &= test_s8(2, 1024, 1024, rng);
  // [N,K] layout: aligned, every tail (M%4, N%8, K%32), and a realistic shape.
  ok &= test_s8_nk(4, 8, 16, rng);
  ok &= test_s8_nk(7, 13, 5, rng);
  ok &= test_s8_nk(1, 33, 9, rng);
  ok &= test_s8_nk(3, 9, 7, rng);
  ok &= test_s8_nk(2, 1024, 1024, rng);
  ok &= test_s8_nk_vnni(4, 8, 16, rng);
  ok &= test_s8_nk_vnni(5, 17, 33, rng);
  ok &= test_s8_nk_vnni(9, 7, 69, rng);
  ok &= test_s8_nk_vnni(1, 33, 9, rng);
  ok &= test_s8_nk_vnni(2, 1024, 1024, rng);
  ok &= test_s8_nk_pool(7, 13, 69, rng);
  ok &= test_s8_nk_pool(8, 1024, 1024, rng);
  ok &= test_s8_nk_add(7, 69, 33, rng);
  ok &= test_s8_nk_add(8, 1024, 1024, rng);
  ok &= test_s8_nk_sliced(7, 69, rng);
  ok &= test_s8_nk_sliced(8, 1024, rng);
  ok &= test_s8_nk_qadd(7, 69, 33, rng);
  ok &= test_s8_nk_qadd(8, 1024, 1024, rng);
  ok &= test_silu_quant_fused(7, 69, rng);
  ok &= test_silu_quant_fused(8, 1024, rng);
  ok &= test_norm_quant_fused(7, 33, rng);
  ok &= test_norm_quant_fused(8, 1024, rng);
  ok &= test_qk_norm_rope(rng);
  ok &= test_attention_masked();
  ok &= test_weight_file();

  std::printf("%s\n", ok ? "ALL GEMM TESTS PASSED" : "GEMM TESTS FAILED");
  return ok ? 0 : 1;
}

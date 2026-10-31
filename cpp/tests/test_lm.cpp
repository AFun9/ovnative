// Acceptance test for the LM forward pass.
//
// Compares against logits produced by the ONNX FP32 reference
// (tools/gen_lm_reference.py).  Fails on any meaningful drift — this is the
// gate that keeps the C++ engine honest while kernels are being optimised.
//
//   ov_test_lm <weights.ovw> <reference dir>
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "model/lm.h"
#ifdef OV_ENABLE_CUDA
#include <memory>
#include "backend/cuda/cuda_lm.h"
#endif
#include "ovnative/ov_weights.h"

namespace {

long file_size(const std::string& p) {
  struct stat st {};
  if (::stat(p.c_str(), &st) != 0) {
    std::fprintf(stderr, "cannot stat %s\n", p.c_str());
    std::exit(2);
  }
  return static_cast<long>(st.st_size);
}

template <typename T>
std::vector<T> read_vec(const std::string& p, size_t n) {
  std::vector<T> v(n);
  std::FILE* f = std::fopen(p.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", p.c_str());
    std::exit(2);
  }
  const size_t got = std::fread(v.data(), sizeof(T), n, f);
  std::fclose(f);
  if (got != n) {
    std::fprintf(stderr, "short read %s: %zu/%zu\n", p.c_str(), got, n);
    std::exit(2);
  }
  return v;
}


// The CPU ragged-CFG forward packs cond(S) + the T active uncond rows and skips
// the unused tail.  Verify it against the exact B=2 forward on a small synthetic
// case: cond rows must be bit-identical, uncond rows may differ only by the
// sub-ULP attention tail (well inside the int8 quality gate).
bool test_cfg_forward(ov::LM& lm, bool use_cuda) {
  if (use_cuda) return true;  // forward_cfg is CPU-only
  const int C = lm.config().n_codebooks, V = lm.config().audio_vocab;
  const int S = 109, T = 86, B = 2;
  std::mt19937 rng(7);
  std::vector<int64_t> ids2(static_cast<size_t>(B) * C * S), pos2(static_cast<size_t>(B) * S);
  std::vector<uint8_t> am2(static_cast<size_t>(B) * S, 1),
      attn2(static_cast<size_t>(B) * S * S, 0);
  for (auto& v : ids2) v = static_cast<int64_t>(rng() % 1024);
  for (int b = 0; b < B; ++b)
    for (int s = 0; s < S; ++s) pos2[static_cast<size_t>(b) * S + s] = s;
  for (int s = 0; s < S; ++s)
    for (int k = 0; k < S; ++k) attn2[static_cast<size_t>(s) * S + k] = 1;
  for (int s = 0; s < S; ++s)
    for (int k = 0; k < S; ++k)
      attn2[static_cast<size_t>(S) * S + static_cast<size_t>(s) * S + k] =
          (s < T ? (k < T) : (k == s));

  std::vector<int64_t> ids(static_cast<size_t>(C) * (S + T)), pos(S + T);
  std::vector<uint8_t> am(S + T, 1), attn_c(static_cast<size_t>(S) * S, 1),
      attn_u(static_cast<size_t>(T) * T, 1);
  for (int c = 0; c < C; ++c) {
    for (int s = 0; s < S; ++s) ids[static_cast<size_t>(c) * (S + T) + s] = ids2[c * S + s];
    for (int t = 0; t < T; ++t)
      ids[static_cast<size_t>(c) * (S + T) + S + t] = ids2[C * S + c * S + t];
  }
  for (int s = 0; s < S; ++s) pos[s] = s;
  for (int t = 0; t < T; ++t) pos[S + t] = t;

  std::vector<float> l2(static_cast<size_t>(B) * C * S * V),
      lc(static_cast<size_t>(S + T) * C * V);
  lm.forward(B, S, ids2.data(), am2.data(), attn2.data(), pos2.data(), l2.data());
  lm.forward_cfg(S, T, ids.data(), am.data(), attn_c.data(), attn_u.data(), pos.data(),
                 lc.data());

  double cond_max = 0, maxd = 0, sumd = 0, dot = 0, na = 0, nb = 0;
  size_t bad = 0, n = 0;
  for (int c = 0; c < C; ++c)
    for (int s = 0; s < S; ++s)
      for (int v = 0; v < V; ++v) {
        const double d = std::fabs((double)l2[((size_t)0 * C + c) * S * V + (size_t)s * V + v] -
                                   lc[(size_t)s * C * V + (size_t)c * V + v]);
        cond_max = std::max(cond_max, d);
      }
  size_t agree = 0, total = 0;
  for (int c = 0; c < C; ++c)
    for (int t = 0; t < T; ++t) {
      const float* a = &l2[((size_t)1 * C + c) * S * V + (size_t)t * V];
      const float* b = &lc[(size_t)(S + t) * C * V + (size_t)c * V];
      int ia = 0, ib = 0;
      for (int v = 0; v < V; ++v) {
        const double d = std::fabs((double)a[v] - b[v]);
        maxd = std::max(maxd, d);
        sumd += d;
        dot += (double)a[v] * b[v];
        na += (double)a[v] * a[v];
        nb += (double)b[v] * b[v];
        if (d > 0.05) ++bad;
        ++n;
        if (a[v] > a[ia]) ia = v;
        if (b[v] > b[ib]) ib = v;
      }
      agree += (ia == ib);
      ++total;
    }
  const double mean = sumd / n;
  const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
  const double agree_rate = static_cast<double>(agree) / total;
  const bool ok = cond_max <= 1e-6 && cos > 0.99999 && mean < 0.15 && agree_rate >= 0.90;
  std::printf("cfg-fwd  : cond max|Δ|=%.2e  uncond mean|Δ|=%.3e  cos=%.8f  argmax %.2f%%  %s\n",
              cond_max, mean, cos, agree_rate * 100, ok ? "OK" : "FAIL");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: ov_test_lm <weights.ovw> <reference-dir> [--cuda]\n");
    return 2;
  }
  const std::string weights = argv[1];
  const std::string ref = argv[2];
  bool use_cuda = false;
  for (int i = 3; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--cuda")) use_cuda = true;
  }

  // Shapes are derivable from the file sizes, so no JSON parsing is needed.
  const long logits_bytes = file_size(ref + "/logits.f32");
  const long attn_bytes = file_size(ref + "/attention_mask.u8");
  const int64_t bs = logits_bytes / (8 * 1025 * 4);
  const int64_t seq = attn_bytes / bs;
  const int64_t batch = bs / seq;
  if (bs * 8 * 1025 * 4 != logits_bytes || seq * bs != attn_bytes) {
    std::fprintf(stderr, "reference files are inconsistent (%ld, %ld)\n", logits_bytes, attn_bytes);
    return 2;
  }
  std::printf("reference: B=%lld S=%lld  (%s)\n", (long long)batch, (long long)seq, ref.c_str());

  ov::WeightFile wf = ov::WeightFile::Open(weights);
  ov::LM lm(wf);
#ifdef OV_ENABLE_CUDA
  ov::cuda::LmRunnerPtr runner{nullptr, &ov::cuda::LmRunnerDelete};
  if (use_cuda) {
    runner = ov::cuda::LmRunnerCreate(wf, lm.config());
    if (!runner) {
      std::fprintf(stderr, "CUDA backend unavailable\n");
      return 2;
    }
    lm.set_cuda_backend(runner.get(), &ov::cuda::LmRunnerForwardThunk);
    std::printf("backend  : cuda\n");
  }
#else
  if (use_cuda) {
    std::fprintf(stderr, "built without CUDA (rebuild with -DOV_ENABLE_CUDA=ON)\n");
    return 2;
  }
#endif
  std::printf("engine   : %s  layers=%d  hidden=%d  heads=%d/%d  dtype_mode=%d\n",
              weights.c_str(), lm.config().n_layers, lm.config().hidden, lm.config().n_heads,
              lm.config().n_kv_heads, wf.dtype_mode());
  if (use_cuda && lm.has_nk_weights()) {
    std::fprintf(stderr,
                 "this checkpoint uses [N,K] int8 weights (CPU/VNNI layout); the CUDA runner "
                 "needs the pre-transposed int8 file\n");
    return 2;
  }
  std::printf("workspace: %.1f MB\n", lm.workspace_bytes(batch, seq) / 1e6);

  const auto ids = read_vec<int64_t>(ref + "/input_ids.i64", bs * 8);
  const auto amask = read_vec<uint8_t>(ref + "/audio_mask.u8", bs);
  const auto attn = read_vec<uint8_t>(ref + "/attention_mask.u8", bs * seq);
  const auto pos = read_vec<int64_t>(ref + "/position_ids.i64", bs);
  const auto want = read_vec<float>(ref + "/logits.f32", bs * 8 * 1025);

  std::vector<float> got(want.size());
  lm.forward(static_cast<int>(batch), static_cast<int>(seq), ids.data(), amask.data(),
             attn.data(), pos.data(), got.data());

  double maxd = 0, maxr = 0, dot = 0, na = 0, nb = 0, sumd = 0;
  size_t bad = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double a = got[i], b = want[i];
    const double d = std::fabs(a - b);
    maxd = std::max(maxd, d);
    maxr = std::max(maxr, std::fabs(b));
    sumd += d;
    dot += a * b;
    na += a * a;
    nb += b * b;
    if (d > 0.05) ++bad;
  }
  const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
  const double mean = sumd / want.size();
  const double rel = maxd / (maxr + 1e-30);

  // argmax agreement over every (b, codebook, s) position
  size_t agree = 0, pos_n = want.size() / 1025;
  for (size_t p = 0; p < pos_n; ++p) {
    const float* a = got.data() + p * 1025;
    const float* b = want.data() + p * 1025;
    int ia = 0, ib = 0;
    for (int v = 1; v < 1025; ++v) {
      if (a[v] > a[ia]) ia = v;
      if (b[v] > b[ib]) ib = v;
    }
    if (ia == ib) ++agree;
  }
  const double agree_rate = pos_n ? static_cast<double>(agree) / static_cast<double>(pos_n) : 0.0;

  // FP32 must be a near-bit-exact reproduction; INT8 is judged the same way the
  // ONNX toolkit judges int8hq (cosine + mean error + argmax agreement).
  bool ok;
  if (wf.dtype_mode() == 1) {
    ok = cos > 0.99999 && mean < 0.15 && agree_rate >= 0.80;
  } else {
    ok = rel < 1e-4 && cos > 0.99999 && bad == 0;
  }

  ok &= test_cfg_forward(lm, use_cuda);

  std::printf("logits   : n=%zu  max|Δ|=%.3e  mean|Δ|=%.3e  rel=%.2e  cos=%.8f  >0.05: %zu\n",
              want.size(), maxd, mean, rel, cos, bad);
  std::printf("argmax   : %zu/%zu agree (%.2f%%)%s\n", agree, pos_n, agree_rate * 100,
              wf.dtype_mode() == 1 ? "   [int8 gate: cos>0.99999, mean<0.15, agree>=80%]"
                                   : "");
  std::printf("%s\n", ok ? (wf.dtype_mode() == 1 ? "INT8 LM WITHIN QUALITY GATE"
                                                  : "LM FORWARD MATCHES ONNX REFERENCE")
                          : "LM FORWARD MISMATCH");
  return ok ? 0 : 1;
}

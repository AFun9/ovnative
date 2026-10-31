// Acceptance test for the C++ acoustic decoder (P3a).
//
//   ov_test_at <at.ovw> <reference dir>
//
// The reference is produced by tools/gen_at_reference.py (ONNX decoder on random
// codes), so this checks RVQ lookup, fc2, the Snake/ConvTranspose stack and the
// exact output length (frames × 960).
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "model/audio_tokenizer.h"
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
    std::fprintf(stderr, "short read %s\n", p.c_str());
    std::exit(2);
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: ov_test_at <at.ovw> <reference-dir>\n");
    return 2;
  }
  const std::string weights = argv[1];
  const std::string ref = argv[2];

  const long code_bytes = file_size(ref + "/codes.i64");
  const long audio_bytes = file_size(ref + "/audio.f32");
  const int frames = static_cast<int>(code_bytes / (8 * 8));
  if (static_cast<long>(frames) * 8 * 8 != code_bytes ||
      static_cast<long>(frames) * 960 * 4 != audio_bytes) {
    std::fprintf(stderr, "reference files inconsistent\n");
    return 2;
  }
  std::printf("reference: frames=%d samples=%d\n", frames, frames * 960);

  ov::WeightFile wf = ov::WeightFile::Open(weights);
  ov::AcousticDecoder dec(wf);
  std::printf("engine   : %s  dtype_mode=%d  quantizers=%d  workspace=%.1f MB\n", weights.c_str(),
              wf.dtype_mode(), dec.config().n_quantizers,
              dec.workspace_bytes(frames) / 1e6);

  const auto codes = read_vec<int64_t>(ref + "/codes.i64", static_cast<size_t>(frames) * 8);
  const auto want = read_vec<float>(ref + "/audio.f32", static_cast<size_t>(frames) * 960);
  std::vector<float> got(want.size());

  dec.decode(codes.data(), frames, got.data());

  double maxd = 0, maxr = 0, dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double a = got[i], b = want[i];
    maxd = std::max(maxd, std::fabs(a - b));
    maxr = std::max(maxr, std::fabs(b));
    dot += a * b;
    na += a * a;
    nb += b * b;
  }
  const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
  const bool ok = wf.dtype_mode() == 1 ? (cos > 0.999 && maxd < 0.05)
                                       : (cos > 0.99999 && maxd <= 2e-5);

  std::printf("waveform : n=%zu  max|Δ|=%.3e  peak=%.4f  cos=%.8f\n", want.size(), maxd, maxr, cos);
  std::printf("%s\n", ok ? "DECODER MATCHES ONNX REFERENCE" : "DECODER MISMATCH");
  return ok ? 0 : 1;
}

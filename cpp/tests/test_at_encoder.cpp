// Stage-level acceptance test for the audio-tokenizer encoder (P3b).
//
//   ov_test_at_enc <at.ovw> <ref-dir>
//
// Compares, in order:
//   1. 24 kHz -> 16 kHz resample         vs input_16k.f32
//   2. HuBERT semantic features          vs semantic_downsampled.f32
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "model/at_encoder.h"
#include "model/audio_io.h"
#include "ovnative/ov_weights.h"

namespace {

long file_size(const std::string& p) {
  struct stat st {};
  if (::stat(p.c_str(), &st) != 0) {
    std::fprintf(stderr, "cannot stat %s\n", p.c_str());
    return -1;
  }
  return static_cast<long>(st.st_size);
}

std::vector<float> read_f32(const std::string& p, long* nbytes = nullptr) {
  const long sz = file_size(p);
  if (sz < 0) std::exit(2);
  if (nbytes) *nbytes = sz;
  std::vector<float> v(static_cast<size_t>(sz / 4));
  std::FILE* f = std::fopen(p.c_str(), "rb");
  if (!f) std::exit(2);
  if (std::fread(v.data(), 4, v.size(), f) != v.size()) std::exit(2);
  std::fclose(f);
  return v;
}

// [C, T] (channel-major, as dumped by PyTorch) -> [T, C]
std::vector<float> transpose(const std::vector<float>& in, int channels, int frames) {
  std::vector<float> out(in.size());
  for (int t = 0; t < frames; ++t)
    for (int c = 0; c < channels; ++c) out[static_cast<size_t>(t) * channels + c] = in[static_cast<size_t>(c) * frames + t];
  return out;
}

bool g_ok = true;

void compare(const char* name, const std::vector<float>& got, const std::vector<float>& want) {
  if (got.size() != want.size()) {
    std::printf("  %-22s SIZE MISMATCH got=%zu want=%zu\n", name, got.size(), want.size());
    g_ok = false;
    return;
  }
  double maxd = 0, maxr = 0, dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double a = got[i], b = want[i];
    maxd = std::max(maxd, std::fabs(a - b));
    maxr = std::max(maxr, std::fabs(b));
    dot += a * b;
    na += a * a;
    nb += b * b;
  }
  const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
  std::printf("  %-22s n=%-7zu max|Δ|=%.3e  rel=%.2e  cos=%.8f\n", name, got.size(), maxd,
              maxd / (maxr + 1e-30), cos);
  if (cos < 0.99999) g_ok = false;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: ov_test_at_enc <at.ovw> <ref-dir>\n");
    return 2;
  }
  ov::WeightFile wf = ov::WeightFile::Open(argv[1]);
  const std::string ref_dir = argv[2];
  const std::string ref = ref_dir;

  const auto wav24 = read_f32(ref + "/input_24k.f32");
  const auto want16 = read_f32(ref + "/input_16k.f32");
  std::printf("input: %zu samples @24k, reference 16k has %zu\n", wav24.size(), want16.size());

  int n16 = 0;
  const std::vector<float> wav16 = ov::AudioEncoder::resample_16k(wav24.data(),
                                                                  static_cast<int>(wav24.size()), n16);
  compare("resample 16k", wav16, want16);

  // ---- voice-clone reference preprocessing ------------------------------
  {
    const std::string wav_path = (argc > 3) ? argv[3] : "assets/zero_shot_prompt.wav";
    std::vector<float> ref = ov::read_wav_mono(wav_path, 24000);
    ov::normalize_ref_rms(ref);
    ov::remove_silence(ref, 24000, 200, 100, 200);
    compare("ref preprocessed", ref, read_f32(ref_dir + "/ref_preprocessed.f32"));
  }

  ov::AudioEncoder enc(wf);
  std::vector<float> sem;
  int frames = 0;
  enc.semantic_features(wav24.data(), static_cast<int>(wav24.size()), sem, frames);
  const auto want_sem = read_f32(ref + "/semantic_downsampled.f32");
  std::printf("semantic: frames=%d dim=%zu (reference %zu floats)\n", frames,
              frames ? sem.size() / frames : 0, want_sem.size());
  compare("semantic features", sem, want_sem);

  // ---- remaining stages ------------------------------------------------
  ov::AudioEncoder::Stages st;
  std::vector<int64_t> codes;
  int codec_frames = 0;
  enc.encode_with_stages(wav24.data(), static_cast<int>(wav24.size()), &st, codes, codec_frames);
  std::printf("encode: frames=%d codes=%zu\n", codec_frames, codes.size());
  compare("e_semantic", st.es, transpose(read_f32(ref + "/e_semantic.f32"), 768, codec_frames));
  compare("e_acoustic", st.ea, transpose(read_f32(ref + "/e_acoustic.f32"), 256, codec_frames));
  compare("embeddings", st.emb, transpose(read_f32(ref + "/embeddings.f32"), 1024, codec_frames));

  // codes: reference is [Q, 1, T] float; ours is [Q, T] int64
  {
    const auto want_c = read_f32(ref + "/codes.f32");
    if (want_c.size() == codes.size()) {
      size_t same = 0;
      for (size_t i = 0; i < codes.size(); ++i) {
        if (static_cast<int>(want_c[i]) == static_cast<int>(codes[i])) ++same;
      }
      std::printf("  %-22s exact match = %.2f %% (%zu/%zu)\n", "codes", 100.0 * same / codes.size(),
                  same, codes.size());
      if (same != codes.size()) g_ok = false;
    } else {
      std::printf("  codes SIZE MISMATCH %zu vs %zu\n", codes.size(), want_c.size());
      g_ok = false;
    }
  }
  std::printf("%s\n", g_ok ? "ENCODER MATCHES PYTORCH" : "ENCODER MISMATCH");
  return g_ok ? 0 : 1;
}

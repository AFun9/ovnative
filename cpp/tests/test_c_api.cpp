// Smoke test for the C ABI: create -> query metadata -> synthesise -> free.
//
//   ov_test_c_api <lm.ovw> <at.ovw> <tokenizer.ovt> [out.wav]
//
// This is the contract an embedder (Unity / C# / Rust / Go) sees, so it is kept
// deliberately free of C++ types on the caller side.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ovnative/ovnative.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: ov_test_c_api <lm.ovw> <at.ovw> <tokenizer.ovt> [out.wav]\n");
    return 2;
  }
  char err[512] = {0};
  OvEngine* engine = ov_engine_create(argv[1], argv[2], argv[3], err, sizeof(err));
  if (!engine) {
    std::fprintf(stderr, "ov_engine_create failed: %s\n", err);
    return 1;
  }
  std::printf("sample_rate=%d codebooks=%d backend=%s\n", ov_engine_sample_rate(engine),
              ov_engine_num_codebooks(engine), ov_engine_backend(engine));

  OvTtsOptions opt = ov_tts_options_default();
  opt.num_steps = 8;  // short run: this test gates the ABI, not the audio quality
  opt.language = "English";
  opt.seed = 1;

  float* samples = nullptr;
  size_t count = 0;
  const int rc = ov_tts(engine, "C ABI smoke test.", &opt, &samples, &count, err, sizeof(err));
  if (rc != OV_OK) {
    std::fprintf(stderr, "ov_tts rc=%d: %s\n", rc, err);
    ov_engine_destroy(engine);
    return 1;
  }
  if (!samples || count == 0) {
    std::fprintf(stderr, "ov_tts returned no samples\n");
    ov_engine_destroy(engine);
    return 1;
  }
  double peak = 0.0, acc = 0.0;
  for (size_t i = 0; i < count; ++i) {
    peak = std::max(peak, std::fabs(static_cast<double>(samples[i])));
    acc += static_cast<double>(samples[i]) * samples[i];
  }
  std::printf("rc=0 samples=%zu (%.2fs) peak=%.3f rms=%.4f\n", count, count / 24000.0, peak,
              std::sqrt(acc / count));

  if (argc > 4) {
    std::FILE* f = std::fopen(argv[4], "wb");
    if (f) {
      const uint32_t rate = 24000, data = static_cast<uint32_t>(count * 4), riff = data + 36;
      const uint16_t pcm = 1, ch = 1, bits = 32;
      std::fwrite("RIFF", 1, 4, f);
      std::fwrite(&riff, 4, 1, f);
      std::fwrite("WAVEfmt ", 1, 8, f);
      const uint32_t fmt = 16;
      const uint16_t align = 4, bps = 32;
      const uint32_t zero = 0;
      std::fwrite(&fmt, 4, 1, f);
      std::fwrite(&pcm, 2, 1, f);
      std::fwrite(&ch, 2, 1, f);
      std::fwrite(&rate, 4, 1, f);
      std::fwrite(&rate, 4, 1, f);  // byte rate = rate*ch*bps/8
      std::fwrite(&align, 2, 1, f);
      std::fwrite(&bits, 2, 1, f);
      std::fwrite("data", 1, 4, f);
      std::fwrite(&data, 4, 1, f);
      (void)bps;
      (void)zero;
      std::fwrite(samples, 4, count, f);
      std::fclose(f);
      std::printf("wrote %s\n", argv[4]);
    }
  }

  ov_samples_free(samples);
  ov_engine_destroy(engine);
  std::printf("C ABI OK\n");
  return (peak > 0.0 && peak < 1.0) ? 0 : 1;
}

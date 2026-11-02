// ovdecode — turn codec tokens into a wav file, no Python involved.
//
//   ovdecode <at.ovw> <codes.i64> <frames> <out.wav> [--fp16] [--int8]
//
// codes layout: [n_quantizers, frames] int64, little-endian (the layout the
// sampler produces and the LM emits: 8 codebooks × T frames).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/utf8.h"
#include "model/audio_tokenizer.h"
#include "ovnative/ov_weights.h"

namespace {

void write_u32(std::FILE* f, uint32_t v) { std::fwrite(&v, 4, 1, f); }
void write_u16(std::FILE* f, uint16_t v) { std::fwrite(&v, 2, 1, f); }

bool write_wav(const std::string& path, const float* pcm, size_t n, int sample_rate) {
  std::FILE* f = ov::fopen_utf8(path, "wb");
  if (!f) return false;
  const uint32_t data_bytes = static_cast<uint32_t>(n * 2);
  std::fwrite("RIFF", 1, 4, f);
  write_u32(f, 36 + data_bytes);
  std::fwrite("WAVEfmt ", 1, 8, f);
  write_u32(f, 16);
  write_u16(f, 1);  // PCM
  write_u16(f, 1);  // mono
  write_u32(f, static_cast<uint32_t>(sample_rate));
  write_u32(f, static_cast<uint32_t>(sample_rate * 2));
  write_u16(f, 2);
  write_u16(f, 16);
  std::fwrite("data", 1, 4, f);
  write_u32(f, data_bytes);

  std::vector<int16_t> pcm16(n);
  double peak = 0.0;
  for (float v : std::vector<float>(pcm, pcm + n)) peak = std::max(peak, std::fabs((double)v));
  // Only normalise when the signal would clip; keeps the numeric identity of
  // the decoder output intact for regression listening.
  const double gain = peak > 0.999 ? 0.999 / peak : 1.0;
  for (size_t i = 0; i < n; ++i) {
    double v = pcm[i] * gain;
    v = std::max(-1.0, std::min(1.0, v));
    pcm16[i] = static_cast<int16_t>(std::lrint(v * 32767.0));
  }
  std::fwrite(pcm16.data(), sizeof(int16_t), n, f);
  std::fclose(f);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: ovdecode <at.ovw> <codes.i64> <frames> <out.wav>\n");
    return 2;
  }
  const std::string weights = argv[1];
  const std::string codes_path = argv[2];
  const int frames = std::atoi(argv[3]);
  const std::string out_path = argv[4];
  if (frames <= 0) {
    std::fprintf(stderr, "frames must be > 0\n");
    return 2;
  }

  std::vector<int64_t> codes(static_cast<size_t>(frames) * 8);
  std::FILE* f = ov::fopen_utf8(codes_path, "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", codes_path.c_str());
    return 2;
  }
  const size_t got = std::fread(codes.data(), sizeof(int64_t), codes.size(), f);
  std::fclose(f);
  if (got != codes.size()) {
    std::fprintf(stderr, "codes file too short: %zu/%zu\n", got, codes.size());
    return 2;
  }

  try {
    ov::WeightFile wf = ov::WeightFile::Open(weights);
    ov::AcousticDecoder dec(wf);
    std::vector<float> audio(static_cast<size_t>(frames) * 960);
    dec.decode(codes.data(), frames, audio.data());

    double peak = 0, rms = 0;
    for (float v : audio) { peak = std::max(peak, std::fabs((double)v)); rms += (double)v * v; }
    rms = std::sqrt(rms / audio.size());

    if (!write_wav(out_path, audio.data(), audio.size(), 24000)) {
      std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
      return 2;
    }
    std::printf("wrote %s  samples=%zu  dur=%.2fs  peak=%.3f  rms=%.4f\n", out_path.c_str(),
                audio.size(), audio.size() / 24000.0, peak, rms);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}

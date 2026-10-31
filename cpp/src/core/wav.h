// Minimal RIFF/WAVE writer (16-bit PCM mono), header-only.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace ov {

inline bool write_wav_mono16(const std::string& path, const float* pcm, size_t n,
                             int sample_rate = 24000) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const uint32_t data_bytes = static_cast<uint32_t>(n * 2);
  auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
  auto w16 = [&](uint16_t v) { std::fwrite(&v, 2, 1, f); };

  std::fwrite("RIFF", 1, 4, f);
  w32(36 + data_bytes);
  std::fwrite("WAVEfmt ", 1, 8, f);
  w32(16);
  w16(1);
  w16(1);
  w32(static_cast<uint32_t>(sample_rate));
  w32(static_cast<uint32_t>(sample_rate * 2));
  w16(2);
  w16(16);
  std::fwrite("data", 1, 4, f);
  w32(data_bytes);

  double peak = 0.0;
  for (size_t i = 0; i < n; ++i) peak = std::max(peak, std::fabs(static_cast<double>(pcm[i])));
  const double gain = peak > 0.999 ? 0.999 / peak : 1.0;
  std::vector<int16_t> pcm16(n);
  for (size_t i = 0; i < n; ++i) {
    double v = pcm[i] * gain;
    v = std::max(-1.0, std::min(1.0, v));
    pcm16[i] = static_cast<int16_t>(std::lrint(v * 32767.0));
  }
  std::fwrite(pcm16.data(), sizeof(int16_t), n, f);
  std::fclose(f);
  return true;
}

}  // namespace ov

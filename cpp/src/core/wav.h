// Minimal RIFF/WAVE writer (16-bit PCM mono), header-only.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "core/utf8.h"

namespace ov {

inline std::vector<uint8_t> wav_bytes_mono16(const float* pcm, size_t n, int sample_rate = 24000) {
  const uint32_t data_bytes = static_cast<uint32_t>(n * 2);
  std::vector<uint8_t> out;
  out.reserve(44 + data_bytes);
  auto w32 = [&](uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
  };
  auto w16 = [&](uint16_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
  };
  auto wstr = [&](const char* s, size_t nbytes) {
    out.insert(out.end(), s, s + nbytes);
  };

  wstr("RIFF", 4);
  w32(36 + data_bytes);
  wstr("WAVEfmt ", 8);
  w32(16);
  w16(1);
  w16(1);
  w32(static_cast<uint32_t>(sample_rate));
  w32(static_cast<uint32_t>(sample_rate * 2));
  w16(2);
  w16(16);
  wstr("data", 4);
  w32(data_bytes);

  double peak = 0.0;
  for (size_t i = 0; i < n; ++i) peak = std::max(peak, std::fabs(static_cast<double>(pcm[i])));
  const double gain = peak > 0.999 ? 0.999 / peak : 1.0;
  for (size_t i = 0; i < n; ++i) {
    double v = pcm[i] * gain;
    v = std::max(-1.0, std::min(1.0, v));
    const int16_t s = static_cast<int16_t>(std::lrint(v * 32767.0));
    w16(static_cast<uint16_t>(s));
  }
  return out;
}

inline bool write_wav_mono16(const std::string& path, const float* pcm, size_t n,
                             int sample_rate = 24000) {
  const std::vector<uint8_t> bytes = wav_bytes_mono16(pcm, n, sample_rate);
  std::FILE* f = fopen_utf8(path, "wb");
  if (!f) return false;
  const size_t wrote = std::fwrite(bytes.data(), 1, bytes.size(), f);
  std::fclose(f);
  return wrote == bytes.size();
}

}  // namespace ov

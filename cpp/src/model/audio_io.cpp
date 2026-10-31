#include "model/audio_io.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ov {
namespace {

constexpr float kSilenceDb = -50.f;  // pydub default used by remove_silence

inline float db_to_float(float db) { return std::pow(10.f, db / 20.f); }

inline int16_t to_i16(float x) {
  float v = x * 32768.f;
  v = std::max(-32768.f, std::min(32767.f, v));
  return static_cast<int16_t>(v);
}

float rms_i16(const int16_t* p, size_t n) {
  if (n == 0) return 0.f;
  double acc = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double v = p[i];
    acc += v * v;
  }
  return static_cast<float>(std::sqrt(acc / static_cast<double>(n)));
}

// pydub's AudioSegment.dBFS, with max_possible_amplitude = 2^15.
float dbfs(const int16_t* p, size_t n) {
  const float r = rms_i16(p, n);
  if (r <= 0.f) return -INFINITY;
  return 20.f * std::log10(r / 32768.f);
}

// detect_silence(): window scan every 10 ms, threshold on the int16 RMS.
std::vector<std::pair<int, int>> detect_silent_ranges(const std::vector<int16_t>& x, int sr,
                                                      int min_silence_len_ms, float thresh_db,
                                                      int seek_step_ms) {
  const int spms = sr / 1000;  // samples per millisecond
  const int seg_len_ms = static_cast<int>(x.size() / static_cast<size_t>(spms));
  std::vector<std::pair<int, int>> out;
  if (seg_len_ms < min_silence_len_ms) return out;

  const float thresh = db_to_float(thresh_db) * 32768.f;
  std::vector<int> starts;
  const int last = seg_len_ms - min_silence_len_ms;
  for (int i = 0; i <= last; i += seek_step_ms) starts.push_back(i);
  if (last % seek_step_ms) starts.push_back(last);

  std::vector<int> silent_starts;
  for (int i : starts) {
    const size_t begin = static_cast<size_t>(i) * spms;
    const size_t len = static_cast<size_t>(min_silence_len_ms) * spms;
    if (begin + len > x.size()) continue;
    if (rms_i16(x.data() + begin, len) <= thresh) silent_starts.push_back(i);
  }
  if (silent_starts.empty()) return out;

  int prev = silent_starts[0];
  int range_start = prev;
  for (size_t k = 1; k < silent_starts.size(); ++k) {
    const int cur = silent_starts[k];
    const bool continuous = (cur == prev + seek_step_ms);
    const bool has_gap = (cur > prev + min_silence_len_ms);
    if (!continuous && has_gap) {
      out.emplace_back(range_start, prev + min_silence_len_ms);
      range_start = cur;
    }
    prev = cur;
  }
  out.emplace_back(range_start, prev + min_silence_len_ms);
  return out;
}

std::vector<std::pair<int, int>> detect_nonsilent(const std::vector<int16_t>& x, int sr,
                                                  int min_silence_len_ms, float thresh_db,
                                                  int seek_step_ms) {
  const int spms = sr / 1000;
  const int seg_len_ms = static_cast<int>(x.size() / static_cast<size_t>(spms));
  const auto silent = detect_silent_ranges(x, sr, min_silence_len_ms, thresh_db, seek_step_ms);
  std::vector<std::pair<int, int>> out;
  if (silent.empty()) {
    out.emplace_back(0, seg_len_ms);
    return out;
  }
  if (silent.front().first == 0 && silent.front().second == seg_len_ms) return out;

  int prev_end = 0;
  for (const auto& s : silent) {
    out.emplace_back(prev_end, s.first);
    prev_end = s.second;
  }
  if (silent.back().second != seg_len_ms) out.emplace_back(prev_end, seg_len_ms);
  if (!out.empty() && out.front() == std::make_pair(0, 0)) out.erase(out.begin());
  return out;
}

int detect_leading_silence_ms(const std::vector<int16_t>& x, int sr, float thresh_db,
                              int chunk_ms) {
  const int spms = sr / 1000;
  const int total = static_cast<int>(x.size());
  int trim = 0;
  while (trim < total) {
    const size_t begin = static_cast<size_t>(trim) * spms;
    const size_t len = std::min<size_t>(static_cast<size_t>(chunk_ms) * spms, x.size() - begin);
    if (len == 0) break;
    if (!(dbfs(x.data() + begin, len) < thresh_db)) break;
    trim += chunk_ms;
  }
  return trim;
}

}  // namespace

std::vector<float> read_wav_mono(const std::string& path, int target_rate) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open wav: " + path);
  std::fseek(f, 0, SEEK_END);
  const long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(static_cast<size_t>(size));
  if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
    std::fclose(f);
    throw std::runtime_error("short read: " + path);
  }
  std::fclose(f);
  if (buf.size() < 44 || std::memcmp(buf.data(), "RIFF", 4) != 0 ||
      std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
    throw std::runtime_error("not a RIFF/WAVE file: " + path);
  }

  int channels = 1, rate = 0, bits = 16, format = 1;
  const uint8_t* data = nullptr;
  size_t data_bytes = 0;
  size_t pos = 12;
  while (pos + 8 <= buf.size()) {
    char id[5] = {0};
    std::memcpy(id, buf.data() + pos, 4);
    uint32_t chunk = 0;
    std::memcpy(&chunk, buf.data() + pos + 4, 4);
    pos += 8;
    if (std::memcmp(id, "fmt ", 4) == 0 && pos + 16 <= buf.size()) {
      uint16_t fmt = 0, ch = 0;
      uint32_t sr = 0;
      uint16_t bps = 0;
      std::memcpy(&fmt, buf.data() + pos, 2);
      std::memcpy(&ch, buf.data() + pos + 2, 2);
      std::memcpy(&sr, buf.data() + pos + 4, 4);
      std::memcpy(&bps, buf.data() + pos + 14, 2);
      format = fmt;
      channels = ch;
      rate = static_cast<int>(sr);
      bits = bps;
    } else if (std::memcmp(id, "data", 4) == 0) {
      data = buf.data() + pos;
      data_bytes = std::min<size_t>(chunk, buf.size() - pos);
    }
    pos += chunk + (chunk & 1);
  }
  if (!data || rate <= 0 || channels <= 0) throw std::runtime_error("malformed wav: " + path);
  if (format != 1 && format != 3) throw std::runtime_error("unsupported wav format: " + path);

  const int bps = bits / 8;
  const size_t frames = data_bytes / (static_cast<size_t>(bps) * channels);
  std::vector<float> mono(frames, 0.f);
  for (size_t i = 0; i < frames; ++i) {
    double acc = 0.0;
    for (int c = 0; c < channels; ++c) {
      const uint8_t* p = data + (i * channels + c) * bps;
      if (format == 3 && bits == 32) {
        float v;
        std::memcpy(&v, p, 4);
        acc += v;
      } else if (bits == 16) {
        int16_t v;
        std::memcpy(&v, p, 2);
        acc += static_cast<double>(v) / 32768.0;
      } else if (bits == 24) {
        int32_t v = (static_cast<int32_t>(p[2]) << 16) | (p[1] << 8) | p[0];
        if (v & 0x800000) v |= ~0xFFFFFF;
        acc += static_cast<double>(v) / 8388608.0;
      } else if (bits == 32) {
        int32_t v;
        std::memcpy(&v, p, 4);
        acc += static_cast<double>(v) / 2147483648.0;
      } else {
        throw std::runtime_error("unsupported bit depth: " + std::to_string(bits));
      }
    }
    mono[i] = static_cast<float>(acc / channels);
  }

  if (rate == target_rate) return mono;

  // Simple windowed-sinc resample (only used when the file is not 24 kHz).
  const double ratio = static_cast<double>(target_rate) / rate;
  const size_t out_n = static_cast<size_t>(std::ceil(mono.size() * ratio));
  std::vector<float> out(out_n, 0.f);
  const int width = 8;
  for (size_t i = 0; i < out_n; ++i) {
    const double src = i / ratio;
    const int centre = static_cast<int>(std::floor(src));
    double acc = 0.0, wsum = 0.0;
    for (int k = -width + 1; k <= width; ++k) {
      const int idx = centre + k;
      if (idx < 0 || idx >= static_cast<int>(mono.size())) continue;
      const double t = src - idx;
      const double sinc = (std::fabs(t) < 1e-9) ? 1.0 : std::sin(M_PI * t) / (M_PI * t);
      const double win = 0.5 + 0.5 * std::cos(M_PI * t / width);
      acc += mono[static_cast<size_t>(idx)] * sinc * win;
      wsum += sinc * win;
    }
    out[i] = static_cast<float>(wsum != 0.0 ? acc / wsum : 0.0);
  }
  return out;
}

void normalize_ref_rms(std::vector<float>& wav) {
  if (wav.empty()) return;
  double acc = 0.0;
  for (float v : wav) acc += static_cast<double>(v) * v;
  const double rms = std::sqrt(acc / wav.size());
  if (rms > 0.0 && rms < 0.1) {
    const float gain = static_cast<float>(0.1 / rms);
    for (float& v : wav) v *= gain;
  }
}

void remove_silence(std::vector<float>& wav, int sr, int mid_sil, int lead_sil, int trail_sil) {
  if (wav.empty()) return;
  const int spms = sr / 1000;
  std::vector<int16_t> x(wav.size());
  for (size_t i = 0; i < wav.size(); ++i) x[i] = to_i16(wav[i]);

  // 1) cut interior silences, keeping mid_sil ms of context around each chunk
  const auto ranges = detect_nonsilent(x, sr, mid_sil, kSilenceDb, 10);
  std::vector<int16_t> joined;
  for (const auto& r : ranges) {
    int start = std::max(0, r.first - mid_sil);
    int end = r.second + mid_sil;
    end = std::min(end, static_cast<int>(x.size() / static_cast<size_t>(spms)));
    const size_t b = static_cast<size_t>(start) * spms;
    const size_t e = std::min(x.size(), static_cast<size_t>(end) * spms);
    if (e > b) joined.insert(joined.end(), x.begin() + static_cast<long>(b), x.begin() + static_cast<long>(e));
  }
  if (!joined.empty()) x.swap(joined);

  // 2) trim the leading / trailing silence
  int start_ms = detect_leading_silence_ms(x, sr, kSilenceDb, 10);
  start_ms = std::max(0, start_ms - lead_sil);
  x.erase(x.begin(), x.begin() + static_cast<long>(std::min<size_t>(x.size(), static_cast<size_t>(start_ms) * spms)));

  std::vector<int16_t> rev(x.rbegin(), x.rend());
  int end_ms = detect_leading_silence_ms(rev, sr, kSilenceDb, 10);
  end_ms = std::max(0, end_ms - trail_sil);
  rev.erase(rev.begin(), rev.begin() + static_cast<long>(std::min<size_t>(rev.size(), static_cast<size_t>(end_ms) * spms)));
  x.assign(rev.rbegin(), rev.rend());

  wav.resize(x.size());
  for (size_t i = 0; i < x.size(); ++i) wav[i] = static_cast<float>(x[i]) / 32768.f;
}

}  // namespace ov

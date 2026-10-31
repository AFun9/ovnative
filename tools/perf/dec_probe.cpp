#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "model/audio_tokenizer.h"
#include "omnivoice/ov_weights.h"
namespace ov { void perf_report(double total); }
using namespace ov;
int main(int argc, char** argv) {
  const WeightFile wf = WeightFile::Open(argv[1]);
  AcousticDecoder dec(wf);
  const int frames = argc > 2 ? atoi(argv[2]) : 86;
  std::mt19937 rng(7);
  std::vector<int64_t> codes((size_t)8 * frames);
  for (auto& c : codes) c = rng() % 1024;
  std::vector<float> audio((size_t)frames * 960);
  dec.decode(codes.data(), frames, audio.data());
  dec.decode(codes.data(), frames, audio.data());
  auto t0 = std::chrono::steady_clock::now();
  dec.decode(codes.data(), frames, audio.data());
  double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  perf_report(dt);
  printf("frames=%d  audio=%.2fs  decode=%.3fs  RTF=%.3f\n", frames, frames * 960 / 24000.0, dt, dt / (frames * 960 / 24000.0));
  return 0;
}

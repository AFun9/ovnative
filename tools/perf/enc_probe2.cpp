#include <chrono>
#include <cstdio>
#include <vector>
#include "model/at_encoder.h"
#include "model/audio_io.h"
#include "ovnative/ov_weights.h"
namespace ov { void enc_report(double); }
using namespace ov;
int main(int argc, char** argv) {
  const WeightFile wf = WeightFile::Open(argv[1]);
  AudioEncoder enc(wf);
  std::vector<float> wav = read_wav_mono(argv[2], 24000);
  normalize_ref_rms(wav); remove_silence(wav, 24000, 200, 100, 200);
  std::vector<int64_t> codes; int frames = 0;
  enc.encode(wav.data(), (int)wav.size(), codes, frames);
  auto t0 = std::chrono::steady_clock::now();
  enc.encode(wav.data(), (int)wav.size(), codes, frames);
  double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  enc_report(dt);
  printf("encoder %.3f s (%.2f s audio, %d frames)\n", dt, wav.size()/24000.0, frames);
  return 0;
}

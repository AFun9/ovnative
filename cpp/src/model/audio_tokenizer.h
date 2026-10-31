// Audio tokenizer — phase P3.
//
// Implemented so far: the acoustic decoder path used by `decode(codes)`:
//
//   codes [8, T] --RVQ lookup+project_out--> Σ q_i --fc2--> [T,256]
//        --conv1(k7)--> Snake --> 5 × (Snake, ConvTranspose, 3 ResUnits)
//        --> Snake --> conv2(k7) --> waveform [T*960]
//
// The encoder path (HuBERT + resampler + semantic encoder + RVQ encode) lands in
// P3b; it is only needed for voice cloning, not for auto/voice-design modes.
#pragma once

#include <vector>

#include "model/conv.h"
#include "model/mat.h"
#include "ovnative/ov_weights.h"

namespace ov {

namespace cuda { class DecoderRunner; }

struct ATConfig {
  int hidden = 1024;        // RVQ hidden size (== LM hidden)
  int codebook_dim = 64;
  int codebook_size = 1024;
  int n_quantizers = 8;     // 4.0 kbps / (10 bits × 25 Hz) = 8
  int dec_in = 256;         // acoustic decoder input channels
  int dec_ch = 1024;        // decoder hidden width
  int hop = 960;            // samples per codec frame @ 24 kHz
  std::vector<int> up_ratios{8, 5, 4, 2, 3};
};

class AcousticDecoder {
 public:
  friend class cuda::DecoderRunner;
  explicit AcousticDecoder(const WeightFile& wf, ATConfig cfg = {});

  const ATConfig& config() const { return cfg_; }
  int64_t workspace_bytes(int frames) const;

  // codes: [n_quantizers, frames] int64 (row q = codebook q)
  // audio: [frames * hop] float32
  void decode(const int64_t* codes, int frames, float* audio);

 private:
  struct ResUnit {
    std::vector<float> snake1, snake2;
    Conv1d conv1, conv2;
  };
  struct DecBlock {
    std::vector<float> snake;
    ConvT1d convt;
    ResUnit ru[3];
  };
  struct Rvq {
    Mat project_out;
    const float* project_out_bias = nullptr;  // [hidden]
    const float* embed = nullptr;             // [codebook_size, codebook_dim]
  };

  ATConfig cfg_;
  std::vector<Rvq> rvq_;
  Mat fc2_;
  const float* fc2_bias_ = nullptr;
  Conv1d conv1_, conv2_;
  std::vector<float> snake_out_;
  std::vector<DecBlock> blocks_;
};

}  // namespace ov

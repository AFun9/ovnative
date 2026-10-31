// Audio-tokenizer encoder (phase P3b).
//
// Stage 1 (this file, this round): 24 kHz -> 16 kHz resample and the HuBERT
// semantic tower (7-conv stem -> projection -> pos-conv -> 12 post-norm
// transformer layers -> mean over all hidden states -> 2x temporal downsample).
//
// The remaining stages (semantic conv blocks, DAC acoustic encoder, RVQ encode)
// land next; `encode()` is wired for all of them but those two are not
// implemented yet and throw.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "model/mat.h"
#include "model/ops.h"
#include "ovnative/ov_weights.h"

namespace ov {

namespace cuda { class EncoderRunner; }

struct ATEncoderConfig {
  int sample_rate = 24000;
  int semantic_rate = 16000;

  // HuBERT
  int feat_dim = 512;          // conv stem width
  int sem_hidden = 768;
  int sem_heads = 12;
  int sem_inter = 3072;
  int sem_layers = 12;
  float layer_norm_eps = 1e-5f;
  int pos_conv_kernel = 128;
  int pos_conv_groups = 16;
  int semantic_downsample = 2;
  int acoustic_pad = 480;   // hop_length / 2

  // conv stem geometry
  int stem_kernel[7] = {10, 3, 3, 3, 3, 2, 2};
  int stem_stride[7] = {5, 2, 2, 2, 2, 2, 2};
};

class AudioEncoder {
 public:
  friend class cuda::EncoderRunner;
  using CudaEncodeFn = void (*)(void* ctx, const float* wav, int n_samples,
                                std::vector<int64_t>& codes, int& frames);
  void set_cuda_encode(void* ctx, CudaEncodeFn fn) { cuda_ctx_ = ctx; cuda_encode_ = fn; }
  bool cuda_encode_enabled() const { return cuda_encode_ != nullptr; }
  explicit AudioEncoder(const WeightFile& wf, ATEncoderConfig cfg = {});

  // 24 kHz waveform -> HuBERT semantic features, [frames, 768] (time-major).
  void semantic_features(const float* wav24k, int n_samples, std::vector<float>& out,
                         int& frames);

  // Full encode -> codes [8, frames] (row q = codebook q).
  void encode(const float* wav24k, int n_samples, std::vector<int64_t>& codes, int& frames);

  // Intermediate tensors, for stage-level tests (nullable in the call below).
  struct Stages {
    std::vector<float> sem;  // [T, 768]
    std::vector<float> es;   // [T, 768]
    std::vector<float> ea;   // [T, 256]
    std::vector<float> emb;  // [T, 1024]
    int frames = 0;
  };
  void encode_with_stages(const float* wav24k, int n_samples, Stages* stages,
                          std::vector<int64_t>& codes, int& frames);

  // Exposed for stage-level tests.
  static std::vector<float> resample_16k(const float* wav, int n_samples, int& out_n);

 private:
  struct StemConv {
    const float* w = nullptr;   // [cout, cin, k]
    const float* b = nullptr;
    int cin = 0, cout = 0, k = 0, stride = 1;
  };

  struct HubertLayer {
    Mat q, k, v, o;
    const float* qb = nullptr;
    const float* kb = nullptr;
    const float* vb = nullptr;
    const float* ob = nullptr;
    Mat f1, f2;
    const float* f1b = nullptr;
    const float* f2b = nullptr;
    const float* ln1w = nullptr;
    const float* ln1b = nullptr;
    const float* ln2w = nullptr;
    const float* ln2b = nullptr;
  };

  void* cuda_ctx_ = nullptr;
  CudaEncodeFn cuda_encode_ = nullptr;
  void run_hubert(const float* wav16, int n16, std::vector<float>& hidden, int& frames);
  void semantic_encode(const float* sem, int frames, std::vector<float>& out);   // [T,768]
  void acoustic_encode(const float* wav24k, int n, std::vector<float>& out, int& frames);  // [T,256]

  struct ResUnit {
    Conv1dSpec c1, c2;
    const float* sa1 = nullptr;   // snake alphas (DAC units only)
    const float* sa2 = nullptr;
  };
  struct SemBlock {
    ResUnit units[2];
    Conv1dSpec conv;
  };
  struct AcBlock {
    ResUnit units[3];
    const float* snake = nullptr;
    Conv1dSpec conv;
  };
  struct Vq {
    Mat project_in;
    const float* pin_b = nullptr;
    const float* embed = nullptr;   // [1024, 64] row-major
    Mat project_out;
    const float* pout_b = nullptr;
  };

  ATEncoderConfig cfg_;
  // stem
  std::vector<StemConv> stem_;
  const float* stem_gn_w_ = nullptr;
  const float* stem_gn_b_ = nullptr;
  // projection
  const float* proj_ln_w_ = nullptr;
  const float* proj_ln_b_ = nullptr;
  Mat proj_;
  const float* proj_b_ = nullptr;
  // encoder
  std::vector<float> pos_conv_weight_;  // [768, 48, 128], weight-norm resolved
  const float* pos_conv_b_ = nullptr;
  const float* enc_ln_w_ = nullptr;
  const float* enc_ln_b_ = nullptr;
  std::vector<HubertLayer> layers_;

  // semantic conv stack
  Conv1dSpec sem_pre_;
  std::vector<SemBlock> sem_blocks_;
  // DAC acoustic encoder
  Conv1dSpec ac_conv1_, ac_conv2_;
  const float* ac_snake_ = nullptr;
  std::vector<AcBlock> ac_blocks_;
  // concat + fc
  Mat fc_;
  const float* fc_b_ = nullptr;
  // RVQ
  std::vector<Vq> vq_;
  bool pad_for_acoustic_ = false;
};

}  // namespace ov

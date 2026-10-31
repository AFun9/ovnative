// OmniVoice language model: Qwen3-0.6B backbone + 8-way audio head.
//
// Mirrors `OmniVoice.forward` exactly:
//   embeds = where(audio_mask, Σ_c audio_emb[id + c·vocab], token_emb[id[0]])
//   hidden = llm(embeds, position_ids, additive mask)
//   logits = audio_heads(hidden) -> [B, 8, S, 1025]
#pragma once

#include <cstdint>
#include <vector>

#include "model/mat.h"
#include "ovnative/ov_weights.h"

namespace ov {

struct LMConfig {
  int n_layers = 28;
  int hidden = 1024;
  int n_heads = 16;
  int n_kv_heads = 8;
  int head_dim = 128;
  int inter = 3072;
  int vocab = 151676;
  int audio_vocab = 1025;
  int n_codebooks = 8;
  float rms_eps = 1e-6f;
  float rope_theta = 1e6f;
};

struct LMLayer {
  Mat q_proj, k_proj, v_proj, o_proj, gate, up, down;
  // Optional concatenated [Nq+Nk+Nv][K] / [Ng+Nu][K] packed views used by the
  // sliced VNNI GEMM (separate output buffers, one OpenMP region).
  Mat qkv_proj, gate_up_proj;
  const float* input_ln = nullptr;
  const float* post_ln = nullptr;
  const float* q_norm = nullptr;
  const float* k_norm = nullptr;
};

// One attention segment: `rows` query rows starting at q_off in the packed
// activation buffers, attending to the matching `rows` keys starting at kv_off.
// A normal batch has one segment per batch row (rows=seq); the CPU CFG path uses
// two segments (cond rows=S, uncond rows=T) packed back-to-back.
struct AttnSegment {
  const float* bias = nullptr;
  int rows = 0;
  int64_t q_off = 0;
  int64_t kv_off = 0;
};

class LM {
 public:
  static constexpr int kMaxLayers = 64;

  explicit LM(const WeightFile& wf, LMConfig cfg = {});

  const LMConfig& config() const { return cfg_; }
  // True when the checkpoint uses the output-major [N,K] int8 encoding.  The
  // CUDA runner only understands the pre-transposed layout, so front-ends must
  // reject this combination instead of silently computing garbage.
  bool has_nk_weights() const;
  // Bytes of scratch buffers for a given batch/sequence (for budgeting).
  int64_t workspace_bytes(int batch, int seq) const;

  // input_ids : [B, 8, S]   int64
  // audio_mask: [B, S]      uint8 (1 = audio position)
  // attn_mask : [B, S, S]   uint8 (1 = attend)
  // position  : [B, S]      int64
  // logits    : [B, 8, S, 1025] float32
  void forward(int batch, int seq, const int64_t* input_ids, const uint8_t* audio_mask,
               const uint8_t* attn_mask, const int64_t* position_ids, float* logits);

  // CPU-only CFG forward with a ragged batch: `cond_seq` rows of the conditional
  // prompt followed by `uncond_seq` rows of the target-only unconditional input,
  // packed contiguously.  This is the same math as forward(2, cond_seq, ...) for
  // the first cond_seq+uncond_seq rows, but it skips the unused tail rows of the
  // uncond plane (and streams each weight only once).
  //
  //   input_ids  : [C, cond_seq+uncond_seq]  (codebook-major, packed)
  //   audio_mask : [cond_seq+uncond_seq]
  //   cond_attn  : [cond_seq, cond_seq]
  //   uncond_attn: [uncond_seq, uncond_seq]
  //   position   : [cond_seq+uncond_seq]
  //   logits     : [cond_seq+uncond_seq, C*V]  (token-major, no permute)
  void forward_cfg(int cond_seq, int uncond_seq, const int64_t* input_ids,
                   const uint8_t* audio_mask, const uint8_t* cond_attn,
                   const uint8_t* uncond_attn, const int64_t* position_ids, float* logits);

  // Optional accelerator hook (the CUDA backend installs itself from outside so
  // that ovcore never links CUDA).
  using CudaForwardFn = void (*)(void* ctx, int batch, int seq, const int64_t* ids,
                                 const uint8_t* amask, const uint8_t* attn, const int64_t* pos,
                                 float* logits);
  using CudaLogitsFn = const void* (*)(void* ctx);
  void set_cuda_backend(void* ctx, CudaForwardFn fn, CudaLogitsFn logits_fn = nullptr);
  bool cuda_enabled() const { return cuda_fn_ != nullptr; }

  // Optional accelerator hook for the per-step guidance + argmax pass.  When
  // installed, forward() leaves the logits on the device and this fills
  // pred/scores [n_codebooks, target_len] without the S*C*V round trip.
  // Two logits planes are passed because the CFG batch runs as two forwards:
  // the conditional one over the full prompt (seq = cond_seq) and the
  // unconditional one over just the target slice (seq = uncond_seq).
  using CudaSampleFn = void (*)(void* ctx, const void* cond, int cond_seq, const void* uncond,
                                int uncond_seq, int target_len, float guidance,
                                float layer_penalty, int32_t* pred, float* scores);
  void set_cuda_sampler(void* ctx, CudaSampleFn fn) { cuda_sample_ = fn; cuda_sample_ctx_ = ctx; }
  bool cuda_sampler_enabled() const { return cuda_sample_ != nullptr; }
  // Returns false when no accelerator is installed.
  bool sample_guided(const void* cond, int cond_seq, const void* uncond, int uncond_seq,
                     int target_len, float guidance, float layer_penalty, int32_t* pred,
                     float* scores);

  // Device address of the logits left by the last forward() (nullptr on the CPU
  // path).  Callers must grab it right after each forward that they need.
  const void* last_logits_dev() const { return cuda_logits_ ? cuda_logits_(cuda_ctx_) : nullptr; }

 private:
  void embed(int batch, int seq, const int64_t* ids, const uint8_t* mask, float* out);
  void embed_cfg(int64_t rows, const int64_t* ids, const uint8_t* mask, float* out);
  void run_layer(int64_t rows, const AttnSegment* segs, int n_segs, int layer, const int64_t* pos,
                 float* x);

  LMConfig cfg_;
  std::vector<LMLayer> layers_;
  Mat audio_heads_;
  Emb token_emb_, audio_emb_;
  std::vector<int64_t> codebook_offsets_;
  const float* final_norm_ = nullptr;

  std::vector<float> x_, xn_, q_, k_, v_, attn_, proj_, gate_, up_;
  std::vector<float> bias_, bias_c_, bias_u_, logits_flat_, token_scratch_;
  void* cuda_ctx_ = nullptr;
  CudaForwardFn cuda_fn_ = nullptr;
  CudaLogitsFn cuda_logits_ = nullptr;
  CudaSampleFn cuda_sample_ = nullptr;
  void* cuda_sample_ctx_ = nullptr;
  int last_seq_ = 0;
};

}  // namespace ov

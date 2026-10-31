// End-to-end text-to-speech pipeline (phase P4b).
//
// Mirrors `OmniVoice.generate` for a single item, batch = 1:
//
//   style tokens (<|denoise|>? <|lang_*> <|instruct_*>) + text tokens
//   + T × <audio_mask>  ->  iterative unmasking with CFG (cond/uncond batch of 2)
//   -> codes [8, T]  ->  AcousticDecoder  ->  waveform
//
// Defaults match OmniVoiceGenerationConfig: 32 steps, CFG 2.0, t_shift 0.1,
// layer penalty 5.0, position temperature 5.0, greedy class sampling.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <memory>

#include "model/at_encoder.h"
#include "model/audio_tokenizer.h"
#include "model/lm.h"
#include "tokenizer/bpe.h"

namespace ov {

struct TtsOptions {
  int num_steps = 32;
  float guidance_scale = 2.0f;
  float t_shift = 0.1f;
  float layer_penalty = 5.0f;
  float position_temperature = 5.0f;
  std::string language;   // "English" / "Chinese" / "en" / ... ; empty = None
  std::string instruct;   // closed vocabulary, comma separated; empty = None
  std::string ref_audio;  // voice-clone reference wav (24 kHz mono); empty = auto voice
  std::string ref_text;   // transcript of ref_audio
  uint32_t seed = 1234;
};

class Pipeline {
 public:
  Pipeline(const WeightFile& lm_weights, const WeightFile& at_weights,
           const std::string& tokenizer_path);

  // One-shot synthesis of `text`; returns mono float32 at 24 kHz.
  std::vector<float> synthesize(const std::string& text, const TtsOptions& opt = {});

  // Exposed for tests / debugging: build the prompt the same way generate() does.
  struct Prompt {
    std::vector<int64_t> ids;      // [n_codebooks, seq]
    std::vector<uint8_t> audio_mask;  // [seq]
    int seq = 0;
    int target_len = 0;
    int ref_frames = 0;            // 0 when not cloning
  };
  Prompt build_prompt(const std::string& text, const TtsOptions& opt);
  std::vector<int64_t> run_sampling(const Prompt& prompt, const TtsOptions& opt);
  // Decode codec tokens [n_codebooks, frames] into mono 24 kHz float samples.
  std::vector<float> decode_codes(const std::vector<int64_t>& codes, int frames);

  // Optional accelerator decode hook (installed by ovtts); when set, decode_codes
  // runs entirely on the device and no intermediate PCM copy exists on the host.
  using CudaDecodeFn = void (*)(void* ctx, const int64_t* codes, int frames, float* audio);
  void set_cuda_decoder(void* ctx, CudaDecodeFn fn) {
    cuda_decode_ctx_ = ctx;
    cuda_decode_ = fn;
  }
  bool cuda_decoder_enabled() const { return cuda_decode_ != nullptr; }
  const AcousticDecoder& decoder() const { return decoder_; }

  int sample_rate() const { return 24000; }
  const Tokenizer& tokenizer() const { return tok_; }
  const LM& lm() const { return lm_; }
  LM& lm() { return lm_; }
  AudioEncoder& encoder() { return *encoder_; }
  bool has_encoder() const { return encoder_ != nullptr; }

 private:
  Tokenizer tok_;
  LM lm_;
  std::unique_ptr<AudioEncoder> encoder_;  // voice clone reference encoder
  AcousticDecoder decoder_;
  void* cuda_decode_ctx_ = nullptr;
  CudaDecodeFn cuda_decode_ = nullptr;
};

}  // namespace ov

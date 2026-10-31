// GPU audio-tokenizer encoder (voice clone): 24 kHz PCM -> RVQ codes.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace ov {
class AudioEncoder;
namespace cuda {

class EncoderRunner;
void EncoderRunnerDelete(EncoderRunner* r);
using EncoderRunnerPtr = std::unique_ptr<EncoderRunner, void (*)(EncoderRunner*)>;

EncoderRunnerPtr EncoderRunnerCreate(const AudioEncoder& encoder);

void EncoderRunnerEncodeThunk(void* ctx, const float* wav, int n_samples,
                              std::vector<int64_t>& codes, int& frames);

}  // namespace cuda
}  // namespace ov

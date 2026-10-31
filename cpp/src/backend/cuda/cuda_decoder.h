// GPU acoustic decoder: same weights/contract as the CPU AcousticDecoder, but
// everything after the codec codes stays on the device (only codes in, PCM out).
#pragma once

#include <cstdint>
#include <memory>

namespace ov {
class AcousticDecoder;
namespace cuda {

class DecoderRunner;
void DecoderRunnerDelete(DecoderRunner* r);
using DecoderRunnerPtr = std::unique_ptr<DecoderRunner, void (*)(DecoderRunner*)>;

// Returns null when no CUDA device is usable.
DecoderRunnerPtr DecoderRunnerCreate(const AcousticDecoder& decoder);

void DecoderRunnerDecodeThunk(void* ctx, const int64_t* codes, int frames, float* audio);

}  // namespace cuda
}  // namespace ov

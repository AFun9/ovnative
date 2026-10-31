// Device-resident LM forward.
//
// Design notes
// ------------
// * Weights live in the OVW1 file as int8 (+ per-column scale) or fp32.  On
//   first use each weight is dequantised and cached by host pointer.  The fp32
//   checkpoint uses exact SGEMM; the int8 checkpoint uses TF32 and, for
//   qkv/gate-up/down/o, fp16 weights+activations with fp32 accumulation
//   (cublasGemmEx).  qkv and gate-up weights are merged once at load time.
// * Activations never leave the device during the forward: only inputs and the
//   final logits cross PCIe.
// * GEMM shape: our weights are stored pre-transposed as [K, N] and cuBLAS is
//   column-major, so a row-major C[M,N] = A[M,K]·W[K,N] maps to
//   cublasSgemm(N, M, K, W, N, A, K, C, N) — the classic row-major swap.
#pragma once

#include <cstdint>
#include <memory>

#include "model/lm.h"
#include "ovnative/ov_weights.h"

namespace ov {
namespace cuda {

class LmRunner;
void LmRunnerDelete(LmRunner* runner);
using LmRunnerPtr = std::unique_ptr<LmRunner, void (*)(LmRunner*)>;

// Returns a null pointer payload when no CUDA device is usable.
LmRunnerPtr LmRunnerCreate(const WeightFile& wf, const LMConfig& cfg);

// void*-signature thunk for LM::set_cuda_backend.
void LmRunnerForwardThunk(void* ctx, int batch, int seq, const int64_t* input_ids,
                          const uint8_t* audio_mask, const uint8_t* attn_mask,
                          const int64_t* position_ids, float* logits);

// void*-signature thunk for LM::set_cuda_sampler: runs the guidance/argmax pass on
// the logits left on the device by the last forward().
void LmRunnerSampleThunk(void* ctx, const void* cond, int cond_seq, const void* uncond,
                         int uncond_seq, int target_len, float guidance, float layer_penalty,
                         int32_t* pred, float* scores);

// Device address of the logits produced by the last LmRunnerForward (for LM::last_logits_dev).
const void* LmRunnerLastLogitsThunk(void* ctx);

// When the caller consumes predictions through LmRunnerSampleThunk, forward() can
// skip the [B, C, S, V] device-to-host copy entirely.
void LmRunnerSetDeviceSampling(LmRunner& runner, bool on);

// Same contract as LM::forward; logits is host memory [B, C, S, V].
void LmRunnerForward(LmRunner& runner, int batch, int seq, const int64_t* input_ids,
                     const uint8_t* audio_mask, const uint8_t* attn_mask,
                     const int64_t* position_ids, float* logits);

// True when a CUDA device was found and initialised.
bool Available();

}  // namespace cuda
}  // namespace ov

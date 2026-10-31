/* OVNative C API — stable ABI for embedders (Unity / C# / Rust / Go ...).
 *
 * Status: full TTS path available on CPU and optional CUDA backend.
 * Thread-safety: an OvEngine must not be used concurrently from two threads;
 * create one engine per decoding thread.
 */
#ifndef OVNATIVE_H
#define OVNATIVE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OV_ABI_VERSION 1

/* status codes */
#define OV_OK 0
#define OV_ERR_INVALID_ARG (-1)
#define OV_ERR_IO (-2)
#define OV_ERR_NOT_IMPLEMENTED (-3)
#define OV_ERR_INTERNAL (-4)

typedef struct OvEngine OvEngine;

typedef struct OvTtsOptions {
  int num_steps;         /* diffusion steps, default 32 */
  float guidance_scale;  /* CFG scale, default 2.0 */
  float t_shift;         /* schedule shift, default 0.1 */
  const char* language;  /* "English", "Chinese", ... or NULL for auto */
  const char* instruct;  /* voice-design vocabulary, comma separated, or NULL */
  const char* ref_audio; /* 24 kHz mono wav path for voice cloning, or NULL */
  const char* ref_text;  /* transcription of ref_audio, or NULL */
  int seed;              /* 0 = nondeterministic */
} OvTtsOptions;

OvTtsOptions ov_tts_options_default(void);

/* Load LM + audio-tokenizer weights (OVW1) and the OVTOK1 tokenizer.
 * Returns NULL on failure and fills `err` with a NUL-terminated message. */
OvEngine* ov_engine_create(const char* lm_weights, const char* at_weights,
                           const char* tokenizer, char* err, size_t err_len);
void ov_engine_destroy(OvEngine* engine);

int ov_engine_sample_rate(const OvEngine* engine);   /* 24000 */
int ov_engine_num_codebooks(const OvEngine* engine); /* 8 */
const char* ov_engine_backend(const OvEngine* engine); /* "cpu:avx2+fma" */

/* Synthesise `text`. On success returns OV_OK, sets *out_samples to a malloc'd
 * buffer of mono float32 at ov_engine_sample_rate() and *out_count to its
 * length. The buffer is peak-normalised to <= 0.999 (same policy as the CLI's
 * wav writer). Free it with ov_samples_free(). */
int ov_tts(OvEngine* engine, const char* text, const OvTtsOptions* opts,
           float** out_samples, size_t* out_count, char* err, size_t err_len);

void ov_samples_free(float* samples);

#ifdef __cplusplus
}
#endif

#endif /* OVNATIVE_H */

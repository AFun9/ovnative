#include "ovnative/ovnative.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <string>

#include <memory>

#include "model/tts.h"
#include "ovnative/ov_gemm.h"
#include "ovnative/ov_weights.h"

struct OvEngine {
  ov::WeightFile lm;
  ov::WeightFile at;
  std::unique_ptr<ov::Pipeline> pipe;
  std::string backend;
};

namespace {

void set_err(char* err, size_t err_len, const std::string& msg) {
  if (!err || err_len == 0) return;
  std::snprintf(err, err_len, "%s", msg.c_str());
}

int fail(char* err, size_t err_len, int code, const std::string& msg) {
  set_err(err, err_len, msg);
  return code;
}

}  // namespace

extern "C" {

OvTtsOptions ov_tts_options_default(void) {
  OvTtsOptions o{};
  o.num_steps = 32;
  o.guidance_scale = 2.0f;
  o.t_shift = 0.1f;
  o.language = nullptr;
  o.instruct = nullptr;
  o.ref_audio = nullptr;
  o.ref_text = nullptr;
  o.seed = 0;
  return o;
}

OvEngine* ov_engine_create(const char* lm_weights, const char* at_weights,
                           const char* tokenizer, char* err, size_t err_len) {
  if (!lm_weights || !at_weights || !tokenizer) {
    set_err(err, err_len, "lm_weights, at_weights and tokenizer are required");
    return nullptr;
  }
  OvEngine* e = new (std::nothrow) OvEngine();
  if (!e) {
    set_err(err, err_len, "out of memory");
    return nullptr;
  }
  try {
    e->lm = ov::WeightFile::Open(lm_weights);
    e->at = ov::WeightFile::Open(at_weights);
    e->pipe = std::make_unique<ov::Pipeline>(e->lm, e->at, tokenizer);
    e->backend = std::string("cpu:") + ov::cpu_kernel_name();
  } catch (const std::exception& ex) {
    set_err(err, err_len, ex.what());
    delete e;
    return nullptr;
  }
  return e;
}

void ov_engine_destroy(OvEngine* engine) { delete engine; }

int ov_engine_sample_rate(const OvEngine*) { return 24000; }
int ov_engine_num_codebooks(const OvEngine*) { return 8; }

const char* ov_engine_backend(const OvEngine* engine) {
  return engine ? engine->backend.c_str() : "none";
}

int ov_tts(OvEngine* engine, const char* text, const OvTtsOptions* opts,
           float** out_samples, size_t* out_count, char* err, size_t err_len) {
  if (!engine || !engine->pipe) return fail(err, err_len, OV_ERR_INVALID_ARG, "engine is null");
  if (!text || !*text) return fail(err, err_len, OV_ERR_INVALID_ARG, "text is empty");
  if (!out_samples || !out_count) {
    return fail(err, err_len, OV_ERR_INVALID_ARG, "out_samples/out_count are required");
  }
  *out_samples = nullptr;
  *out_count = 0;

  ov::TtsOptions o;
  if (opts) {
    if (opts->num_steps > 0) o.num_steps = opts->num_steps;
    if (opts->guidance_scale > 0.f) o.guidance_scale = opts->guidance_scale;
    if (opts->t_shift > 0.f) o.t_shift = opts->t_shift;
    if (opts->language) o.language = opts->language;
    if (opts->instruct) o.instruct = opts->instruct;
    if (opts->ref_audio) o.ref_audio = opts->ref_audio;
    if (opts->ref_text) o.ref_text = opts->ref_text;
    if (opts->seed > 0) o.seed = static_cast<uint32_t>(opts->seed);
  }
  try {
    std::vector<float> audio = engine->pipe->synthesize(text, o);
    float* buf = static_cast<float*>(std::malloc(sizeof(float) * audio.size()));
    if (!buf) return fail(err, err_len, OV_ERR_INTERNAL, "out of memory");
    // Match the CLI's wav writer: keep the peak at or below 0.999 so embedders
    // never receive samples that clip a 16-bit output stage.
    double peak = 0.0;
    for (float v : audio) peak = std::max(peak, static_cast<double>(std::fabs(v)));
    const double gain = peak > 0.999 ? 0.999 / peak : 1.0;
    if (gain != 1.0) {
      for (float& v : audio) v = static_cast<float>(v * gain);
    }
    std::memcpy(buf, audio.data(), sizeof(float) * audio.size());
    *out_samples = buf;
    *out_count = audio.size();
  } catch (const std::exception& ex) {
    return fail(err, err_len, OV_ERR_INTERNAL, ex.what());
  }
  return OV_OK;
}

void ov_samples_free(float* samples) { std::free(samples); }

}  // extern "C"

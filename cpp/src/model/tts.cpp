#include "model/tts.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>

#include "model/audio_io.h"
#include "model/ops.h"

namespace ov {
namespace {

constexpr int kMaskId = 1024;

std::string to_lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

// _combine_text: join, strip newlines, normalise parentheses/spaces, and drop
// spaces adjacent to CJK characters.
std::string combine_text(const std::string& text, const std::string& ref_text) {
  std::string out;
  if (!ref_text.empty()) out = trim(ref_text) + " " + trim(text);
  else out = trim(text);

  std::string cleaned;
  cleaned.reserve(out.size());
  for (size_t i = 0; i < out.size();) {
    const unsigned char c = static_cast<unsigned char>(out[i]);
    if (c == '\r' || c == '\n') { ++i; continue; }
    if (out.compare(i, 3, "\xef\xbc\x88") == 0) { cleaned += '('; i += 3; continue; }  // （
    if (out.compare(i, 3, "\xef\xbc\x89") == 0) { cleaned += ')'; i += 3; continue; }  // ）
    if (c == '\t') { cleaned += ' '; ++i; continue; }
    cleaned += out[i];
    ++i;
  }
  return cleaned;
}

// Language/instruction are part of the prompt; keep them byte-identical to what
// the Python API produces.
std::string resolve_language(const std::string& lang) {
  if (lang.empty()) return "None";
  const std::string low = to_lower(lang);
  if (low == "english") return "en";
  if (low == "chinese" || low == "mandarin") return "zh";
  if (low == "japanese") return "ja";
  if (low == "korean") return "ko";
  if (low == "german") return "de";
  if (low == "french") return "fr";
  if (low == "spanish") return "es";
  if (low == "russian") return "ru";
  if (low == "italian") return "it";
  if (low == "portuguese") return "pt";
  return lang;  // already an id like "en"
}

std::string normalise_instruct(const std::string& instruct) {
  if (trim(instruct).empty()) return "None";
  std::string out;
  size_t i = 0;
  while (i < instruct.size()) {
    size_t j = i;
    while (j < instruct.size() && instruct[j] != ',' && instruct[j] != '\xef') ++j;
    // handle full-width comma U+FF0C
    if (j < instruct.size() && instruct[j] == '\xef' && instruct.compare(j, 3, "\xef\xbc\x8c") == 0) {
      std::string item = trim(instruct.substr(i, j - i));
      if (!item.empty()) { if (!out.empty()) out += ", "; out += to_lower(item); }
      i = j + 3;
      continue;
    }
    std::string item = trim(instruct.substr(i, j - i));
    if (!item.empty()) { if (!out.empty()) out += ", "; out += to_lower(item); }
    i = (j < instruct.size()) ? j + 1 : j;
  }
  return out.empty() ? "None" : out;
}

void log_softmax(const float* x, int n, float* out) {
  float m = -INFINITY;
  for (int i = 0; i < n; ++i) m = std::max(m, x[i]);
  float sum = 0.f;
  for (int i = 0; i < n; ++i) {
    out[i] = std::exp(x[i] - m);
    sum += out[i];
  }
  const float lse = std::log(sum) + m;
  for (int i = 0; i < n; ++i) out[i] = x[i] - lse;
}

}  // namespace

Pipeline::Pipeline(const WeightFile& lm_weights, const WeightFile& at_weights,
                   const std::string& tokenizer_path)
    : tok_(tokenizer_path), lm_(lm_weights), decoder_(at_weights) {
  // The reference encoder is only needed for voice cloning, but building it is
  // cheap (all weight views) and keeps `synthesize` allocation-free.
  encoder_ = std::make_unique<AudioEncoder>(at_weights);
}

Pipeline::Prompt Pipeline::build_prompt(const std::string& text, const TtsOptions& opt) {
  const std::string lang = resolve_language(opt.language);
  const std::string instruct = normalise_instruct(opt.instruct);
  const bool has_ref = !opt.ref_audio.empty();

  // ---- voice clone: reference wav -> codes ------------------------------
  std::vector<int64_t> ref_codes;
  int ref_frames = 0;
  if (has_ref) {
    std::vector<float> wav = read_wav_mono(opt.ref_audio, 24000);
    normalize_ref_rms(wav);
    remove_silence(wav, 24000, 200, 100, 200);
    encoder_->encode(wav.data(), static_cast<int>(wav.size()), ref_codes, ref_frames);
  }

  const std::string style_text = (has_ref ? std::string("<|denoise|>") : std::string()) +
                                 "<|lang_start|>" + lang + "<|lang_end|>" +
                                 "<|instruct_start|>" + instruct + "<|instruct_end|>";
  const std::string full_text = combine_text(text, has_ref ? opt.ref_text : "");
  const std::string wrapped = "<|text_start|>" + full_text + "<|text_end|>";

  const std::vector<int32_t> style_ids = tok_.encode(style_text);
  const std::vector<int32_t> text_ids = tok_.encode(wrapped);

  // _estimate_target_tokens: fall back to a 25-frame reference when cloning is
  // not used; otherwise use the real reference length and transcript.
  const double frames =
      has_ref ? tok_.estimate_frames(text, opt.ref_text, static_cast<double>(ref_frames))
              : tok_.estimate_frames(text, "Nice to meet you.", 25.0);
  const int target_len = std::max(1, static_cast<int>(frames));

  const int C = lm_.config().n_codebooks;
  const int seq = static_cast<int>(style_ids.size() + text_ids.size()) + ref_frames + target_len;

  Prompt p;
  p.seq = seq;
  p.target_len = target_len;
  p.ref_frames = ref_frames;
  p.ids.assign(static_cast<size_t>(C) * seq, kMaskId);
  p.audio_mask.assign(static_cast<size_t>(seq), 0);

  for (int c = 0; c < C; ++c) {
    int64_t* row = p.ids.data() + static_cast<size_t>(c) * seq;
    size_t k = 0;
    for (int32_t id : style_ids) row[k++] = id;
    for (int32_t id : text_ids) row[k++] = id;
    for (int t = 0; t < ref_frames; ++t) row[k++] = ref_codes[static_cast<size_t>(c) * ref_frames + t];
  }
  for (int t = seq - target_len - ref_frames; t < seq; ++t) p.audio_mask[static_cast<size_t>(t)] = 1;
  return p;
}

std::vector<int64_t> Pipeline::run_sampling(const Prompt& prompt, const TtsOptions& opt) {
  const int C = lm_.config().n_codebooks;
  const int V = lm_.config().audio_vocab;
  const int T = prompt.target_len;
  const int S = prompt.seq;
  const int B2 = 2;

  // ---- batch: row 0 = cond (full prompt), row 1 = uncond (target slice) ----
  std::vector<int64_t> ids(static_cast<size_t>(B2) * C * S, kMaskId);
  std::vector<uint8_t> amask(static_cast<size_t>(B2) * S, 0);
  std::vector<uint8_t> attn(static_cast<size_t>(B2) * S * S, 0);

  for (int c = 0; c < C; ++c) {
    std::memcpy(ids.data() + static_cast<size_t>(c) * S,
                prompt.ids.data() + static_cast<size_t>(c) * S, sizeof(int64_t) * S);
  }
  std::memcpy(amask.data(), prompt.audio_mask.data(), S);
  for (int i = 0; i < S; ++i) {
    for (int j = 0; j < S; ++j) attn[static_cast<size_t>(i) * S + j] = 1;
  }

  for (int c = 0; c < C; ++c) {
    for (int t = 0; t < T; ++t) {
      ids[static_cast<size_t>(1) * C * S + static_cast<size_t>(c) * S + t] =
          prompt.ids[static_cast<size_t>(c) * S + (S - T + t)];
    }
  }
  // Uncond row: full attention inside the first u_len positions (this is
  // `batch_attention_mask[B+i, :, :u_len, :u_len] = True` upstream), plus the
  // diagonal for the unused tail so those queries still see one key.
  for (int t = 0; t < T; ++t) {
    amask[static_cast<size_t>(S) + t] = 1;
    for (int j = 0; j < T; ++j) attn[(static_cast<size_t>(S) + t) * S + j] = 1;
  }
  for (int t = T; t < S; ++t) attn[(static_cast<size_t>(S) + t) * S + t] = 1;

  std::vector<int64_t> pos(static_cast<size_t>(B2) * S);
  for (int b = 0; b < B2; ++b)
    for (int i = 0; i < S; ++i) pos[static_cast<size_t>(b) * S + i] = i;

  // CPU CFG path: pack cond rows [0,S) and only the T active uncond rows
  // [S,S+T) into one batch.  forward_cfg then skips the unused uncond tail
  // (S-T rows) while streaming each weight once; CUDA keeps the B=2 path.
  const bool use_cuda = lm_.cuda_enabled();
  // OV_CPU_EXACT_CFG=1 restores the full B=2 forward on the CPU (bit-identical
  // to the pre-ragged path); the default skips the unused uncond tail rows.
  const char* exact_env = std::getenv("OV_CPU_EXACT_CFG");
  const bool exact_cfg = exact_env && exact_env[0] && exact_env[0] != '0';
  const bool use_b2 = use_cuda || exact_cfg;
  const int rows = S + T;
  std::vector<int64_t> ids_cfg(static_cast<size_t>(C) * rows, kMaskId);
  std::vector<uint8_t> amask_cfg(static_cast<size_t>(rows), 0);
  std::vector<uint8_t> attn_c(static_cast<size_t>(S) * S, 1);
  std::vector<uint8_t> attn_u(static_cast<size_t>(T) * T, 1);
  std::vector<int64_t> pos_cfg(static_cast<size_t>(rows));
  for (int c = 0; c < C; ++c) {
    std::memcpy(ids_cfg.data() + static_cast<size_t>(c) * rows,
                prompt.ids.data() + static_cast<size_t>(c) * S, sizeof(int64_t) * S);
    for (int t = 0; t < T; ++t) {
      ids_cfg[static_cast<size_t>(c) * rows + S + t] =
          prompt.ids[static_cast<size_t>(c) * S + (S - T + t)];
    }
  }
  std::memcpy(amask_cfg.data(), prompt.audio_mask.data(), S);
  for (int t = 0; t < T; ++t) amask_cfg[static_cast<size_t>(S) + t] = 1;
  for (int i = 0; i < S; ++i) pos_cfg[static_cast<size_t>(i)] = i;
  for (int t = 0; t < T; ++t) pos_cfg[static_cast<size_t>(S) + t] = t;

  // ---- diffusion schedule (mirrors _generate_iterative) -------------------
  std::vector<double> ts(static_cast<size_t>(opt.num_steps) + 1);
  for (int i = 0; i <= opt.num_steps; ++i) {
    const double t = static_cast<double>(i) / opt.num_steps;
    ts[static_cast<size_t>(i)] = opt.t_shift * t / (1.0 + (opt.t_shift - 1.0) * t);
  }
  const int64_t total_mask = static_cast<int64_t>(T) * C;
  std::vector<int> sched(static_cast<size_t>(opt.num_steps), 0);
  int64_t rem = total_mask;
  for (int s = 0; s < opt.num_steps; ++s) {
    int64_t num = (s == opt.num_steps - 1)
                      ? rem
                      : std::min<int64_t>(
                            static_cast<int64_t>(std::ceil(total_mask * (ts[s + 1] - ts[s]))), rem);
    sched[static_cast<size_t>(s)] = static_cast<int>(num);
    rem -= num;
  }

  std::vector<int64_t> tokens(static_cast<size_t>(C) * T, kMaskId);
  std::vector<float> logits(use_b2 ? static_cast<size_t>(B2) * C * S * V : 0);
  std::vector<float> logits_cfg(use_b2 ? 0 : static_cast<size_t>(rows) * C * V);
  std::vector<float> scores(static_cast<size_t>(C) * T);
  std::vector<int32_t> pred(static_cast<size_t>(C) * T);

  std::mt19937 rng(opt.seed);
  std::uniform_real_distribution<float> uni(0.f, 1.f);

  for (int step = 0; step < opt.num_steps; ++step) {
    const int k = sched[static_cast<size_t>(step)];
    if (k <= 0) continue;

    if (use_b2) {
      lm_.forward(B2, S, ids.data(), amask.data(), attn.data(), pos.data(), logits.data());
    } else {
      lm_.forward_cfg(S, T, ids_cfg.data(), amask_cfg.data(), attn_c.data(), attn_u.data(),
                      pos_cfg.data(), logits_cfg.data());
    }

    // With an accelerator sampler installed the whole guidance/argmax pass runs on
    // the device and only C*T values come back (no S*C*V host round trip).  Both
    // CFG planes are views of the same [2*S, C*V] buffer.
    const void* dev = use_cuda ? lm_.last_logits_dev() : nullptr;
    const void* dev_u = dev ? static_cast<const void*>(
                                  static_cast<const float*>(dev) +
                                  static_cast<size_t>(S) * C * V)
                            : nullptr;
    if (use_cuda &&
        lm_.sample_guided(dev, S, dev_u, S, T, opt.guidance_scale, opt.layer_penalty,
                          pred.data(), scores.data())) {
      // pred/scores already filled
    } else {
    // C*T independent rows, each a few passes over V: 12 ms of the 108 ms step
    // before this was parallelised (~0.9 ms on 32 threads).  Every row keeps the
    // exact scalar math, so the result is unchanged.
    const int n_row = C * T;
#pragma omp parallel
    {
      std::vector<float> lp_c(static_cast<size_t>(V)), lp_u(static_cast<size_t>(V)),
          lp_mix(static_cast<size_t>(V));
#pragma omp for schedule(static)
      for (int idx = 0; idx < n_row; ++idx) {
        const int c = idx / T;
        const int t = idx - c * T;
        const float* lc;
        const float* lu;
        if (use_b2) {
          lc = logits.data() + (static_cast<size_t>(0) * C + c) * S * V + (S - T + t) * V;
          lu = logits.data() + (static_cast<size_t>(1) * C + c) * S * V + static_cast<size_t>(t) * V;
        } else {
          lc = logits_cfg.data() + static_cast<size_t>(S - T + t) * C * V +
               static_cast<size_t>(c) * V;
          lu = logits_cfg.data() + static_cast<size_t>(S + t) * C * V +
               static_cast<size_t>(c) * V;
        }
        log_softmax(lc, V, lp_c.data());
        log_softmax(lu, V, lp_u.data());
        if (opt.guidance_scale != 0.f) {
          for (int v = 0; v < V; ++v) {
            lp_mix[v] = lp_c[v] + opt.guidance_scale * (lp_c[v] - lp_u[v]);
          }
          log_softmax(lp_mix.data(), V, lp_mix.data());
        } else {
          std::memcpy(lp_mix.data(), lp_c.data(), sizeof(float) * V);
        }
        lp_mix[kMaskId] = -INFINITY;

        int best = 0;
        float best_val = -INFINITY;
        for (int v = 0; v < V; ++v) {
          if (lp_mix[v] > best_val) { best_val = lp_mix[v]; best = v; }
        }
        pred[static_cast<size_t>(idx)] = best;
        scores[static_cast<size_t>(idx)] = best_val - c * opt.layer_penalty;
      }
    }
    }

    // position selection: gumbel noise, then top-k over still-masked positions
    for (size_t i = 0; i < scores.size(); ++i) {
      float u = uni(rng);
      u = std::max(u, 1e-10f);
      const float g = -std::log(-std::log(u) + 1e-10f);
      scores[i] = scores[i] / opt.position_temperature + g;
    }
    for (int c = 0; c < C; ++c) {
      for (int t = 0; t < T; ++t) {
        if (tokens[static_cast<size_t>(c) * T + t] != kMaskId) {
          scores[static_cast<size_t>(c) * T + t] = -INFINITY;
        }
      }
    }

    std::vector<size_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    const int kk = std::min<int>(k, static_cast<int>(order.size()));
    std::partial_sort(order.begin(), order.begin() + kk, order.end(),
                      [&](size_t a, size_t b) { return scores[a] > scores[b]; });
    for (int i = 0; i < kk; ++i) {
      const size_t idx = order[static_cast<size_t>(i)];
      if (!std::isfinite(scores[idx])) break;
      tokens[idx] = pred[idx];
      const int c = static_cast<int>(idx / T);
      const int t = static_cast<int>(idx % T);
      ids[(static_cast<size_t>(0) * C + c) * S + (S - T + t)] = pred[idx];
      ids[(static_cast<size_t>(1) * C + c) * S + t] = pred[idx];
      ids_cfg[static_cast<size_t>(c) * rows + (S - T + t)] = pred[idx];
      ids_cfg[static_cast<size_t>(c) * rows + S + t] = pred[idx];
    }
  }

  return tokens;
}

std::vector<float> Pipeline::decode_codes(const std::vector<int64_t>& codes, int frames) {
  std::vector<float> audio(static_cast<size_t>(frames) * 960);
  if (cuda_decode_) {
    cuda_decode_(cuda_decode_ctx_, codes.data(), frames, audio.data());
  } else {
    decoder_.decode(codes.data(), frames, audio.data());
  }
  return audio;
}

std::vector<float> Pipeline::synthesize(const std::string& text, const TtsOptions& opt) {
  const Prompt p = build_prompt(text, opt);
  const std::vector<int64_t> codes = run_sampling(p, opt);
  std::vector<float> audio = decode_codes(codes, p.target_len);
  if (opt.trim_silence) {
    trim_output_silence(audio, sample_rate(), opt.trim_lead_ms, opt.trim_trail_ms,
                        opt.max_internal_silence_ms, opt.keep_internal_silence_ms);
  }
  return audio;
}

}  // namespace ov

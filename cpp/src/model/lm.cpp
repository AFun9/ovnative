#include "model/lm.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "model/mat.h"
#include "model/ops.h"
#include "ovnative/ov_gemm.h"

namespace ov {

namespace {
// Activation scratch for the AVX-VNNI path, reused across the projections of one
// layer that share their input (q/k/v and gate/up).
thread_local std::vector<uint8_t> t_aq;
thread_local std::vector<float> t_asc;

// Coarse phase timers, enabled with OV_PROFILE=1. Overhead is two clock reads
// per phase per layer, i.e. negligible next to the GEMMs.
struct Profile {
  bool on = std::getenv("OV_PROFILE") != nullptr;
  double embed = 0, qkv = 0, attn = 0, oproj = 0, mlp = 0, head = 0;
  double a_prep = 0, a_qk = 0, a_soft = 0, a_pv = 0, a_scatter = 0;
  double t0 = 0;
  void start() { t0 = now(); }
  double stop() { return now() - t0; }
  static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void report(int B, int S) {
    if (!on) return;
    const double tot = embed + qkv + attn + oproj + mlp + head;
    std::fprintf(stderr,
                 "[profile] B=%d S=%d total=%.3fs | embed %.1f%% qkv %.1f%% attn %.1f%% "
                 "oproj %.1f%% mlp %.1f%% head %.1f%% | attn: prep %.0fms qk %.0fms soft %.0fms pv %.0fms scatter %.0fms\n",
                 B, S, tot, 100 * embed / tot, 100 * qkv / tot, 100 * attn / tot,
                 100 * oproj / tot, 100 * mlp / tot, 100 * head / tot,
                 a_prep * 1e3, a_qk * 1e3, a_soft * 1e3, a_pv * 1e3, a_scatter * 1e3);
    embed = qkv = attn = oproj = mlp = head = 0;
    a_prep = a_qk = a_soft = a_pv = a_scatter = 0;
  }
};
Profile g_prof;

}  // namespace

// Concatenate several packed [N][K] Mats into one logical matrix.  The packed
// tiles concatenate directly (same K, same gb); wsum is block-major so it is
// interleaved per 32-k block; scales concatenate.  The source Mats keep their
// raw q pointer/scale for the non-VNNI fallback, but their packed/wsum copies
// are released after the move to keep steady-state memory flat.
Mat combine_packed(std::initializer_list<Mat*> mats) {
  Mat c;
  c.nk = true;
  c.K = mats.begin()[0]->K;
  int64_t total_n = 0, npad_total = 0;
  for (Mat* m : mats) {
    if (!m->nk || m->packed.empty() || m->K != c.K || m->N % 64 != 0) {
      throw std::runtime_error("combine_packed: incompatible packed matrix");
    }
    total_n += m->N;
    npad_total += ((m->N + 7) / 8) * 8;
  }
  const int64_t Kb = (c.K + 31) / 32;
  c.N = total_n;
  c.scale_storage.resize(static_cast<size_t>(total_n));
  c.wsum.assign(static_cast<size_t>(Kb * npad_total), 0);
  size_t packed_bytes = 0;
  for (Mat* m : mats) packed_bytes += m->packed.size();
  c.packed.resize(packed_bytes);
  size_t poff = 0;
  int64_t col = 0;
  for (Mat* m : mats) {
    std::memcpy(c.packed.data() + poff, m->packed.data(), m->packed.size());
    poff += m->packed.size();
    std::memcpy(c.scale_storage.data() + col, m->row_scale(), sizeof(float) * m->N);
    const int64_t npad = ((m->N + 7) / 8) * 8;
    for (int64_t b = 0; b < Kb; ++b) {
      std::memcpy(c.wsum.data() + b * npad_total + col, m->wsum.data() + b * npad,
                  sizeof(int32_t) * npad);
    }
    col += m->N;
  }
  for (Mat* m : mats) {
    std::vector<int8_t>().swap(m->packed);
    std::vector<int32_t>().swap(m->wsum);
  }
  return c;
}

LM::LM(const WeightFile& wf, LMConfig cfg) : cfg_(cfg) {
  token_emb_ = make_emb(wf, "llm.embed_tokens.weight");
  audio_emb_ = make_emb(wf, "audio_embeddings.weight");
  audio_heads_ = make_linear(wf, "audio_heads.weight");
  final_norm_ = make_norm(wf, "llm.norm.weight", cfg_.hidden);

  const Tensor& offs = wf.Get("codebook_layer_offsets");
  codebook_offsets_.assign(offs.i64(), offs.i64() + offs.shape.numel());
  if (static_cast<int>(codebook_offsets_.size()) != cfg_.n_codebooks) {
    throw std::runtime_error("codebook_layer_offsets: unexpected size");
  }

  if (token_emb_.cols != cfg_.hidden) throw std::runtime_error("token embedding hidden mismatch");
  if (audio_emb_.cols != cfg_.hidden) throw std::runtime_error("audio embedding hidden mismatch");
  if (audio_emb_.rows != static_cast<int64_t>(cfg_.n_codebooks) * cfg_.audio_vocab) {
    throw std::runtime_error("audio embedding rows mismatch");
  }
  if (audio_heads_.K != cfg_.hidden ||
      audio_heads_.N != static_cast<int64_t>(cfg_.n_codebooks) * cfg_.audio_vocab) {
    throw std::runtime_error("audio head shape mismatch");
  }
  if (cfg_.n_layers > kMaxLayers) throw std::runtime_error("too many layers");

  layers_.resize(static_cast<size_t>(cfg_.n_layers));
  for (int i = 0; i < cfg_.n_layers; ++i) {
    const std::string p = "llm.layers." + std::to_string(i) + ".";
    LMLayer& L = layers_[static_cast<size_t>(i)];
    L.input_ln = make_norm(wf, p + "input_layernorm.weight", cfg_.hidden);
    L.post_ln = make_norm(wf, p + "post_attention_layernorm.weight", cfg_.hidden);
    L.q_norm = make_norm(wf, p + "self_attn.q_norm.weight", cfg_.head_dim);
    L.k_norm = make_norm(wf, p + "self_attn.k_norm.weight", cfg_.head_dim);
    L.q_proj = make_linear(wf, p + "self_attn.q_proj.weight");
    L.k_proj = make_linear(wf, p + "self_attn.k_proj.weight");
    L.v_proj = make_linear(wf, p + "self_attn.v_proj.weight");
    L.o_proj = make_linear(wf, p + "self_attn.o_proj.weight");
    L.gate = make_linear(wf, p + "mlp.gate_proj.weight");
    L.up = make_linear(wf, p + "mlp.up_proj.weight");
    L.down = make_linear(wf, p + "mlp.down_proj.weight");

    if (L.q_proj.N != static_cast<int64_t>(cfg_.n_heads) * cfg_.head_dim ||
        L.k_proj.N != static_cast<int64_t>(cfg_.n_kv_heads) * cfg_.head_dim ||
        L.v_proj.N != static_cast<int64_t>(cfg_.n_kv_heads) * cfg_.head_dim ||
        L.o_proj.K != static_cast<int64_t>(cfg_.n_heads) * cfg_.head_dim ||
        L.o_proj.N != cfg_.hidden || L.gate.N != cfg_.inter || L.down.K != cfg_.inter) {
      throw std::runtime_error("layer " + std::to_string(i) + ": projection shape mismatch");
    }

    if (L.q_proj.vnni_packed() && L.k_proj.vnni_packed() && L.v_proj.vnni_packed() &&
        L.q_proj.N % 64 == 0 && L.k_proj.N % 64 == 0 && L.v_proj.N % 64 == 0) {
      L.qkv_proj = combine_packed({&L.q_proj, &L.k_proj, &L.v_proj});
    }
    if (L.gate.vnni_packed() && L.up.vnni_packed() && L.gate.N % 64 == 0 &&
        L.up.N % 64 == 0) {
      L.gate_up_proj = combine_packed({&L.gate, &L.up});
    }
  }
}

bool LM::has_nk_weights() const {
  if (audio_heads_.nk) return true;
  for (const auto& L : layers_) {
    if (L.q_proj.nk || L.k_proj.nk || L.v_proj.nk || L.o_proj.nk || L.gate.nk || L.up.nk ||
        L.down.nk) {
      return true;
    }
  }
  return false;
}

int64_t LM::workspace_bytes(int batch, int seq) const {
  const int64_t T = int64_t(batch) * seq;
  const int64_t H = cfg_.hidden, Q = int64_t(cfg_.n_heads) * cfg_.head_dim;
  const int64_t KV = int64_t(cfg_.n_kv_heads) * cfg_.head_dim;
  return T * (2 * H + 2 * Q + 2 * KV + int64_t(cfg_.inter)) * 4 +   // x..up
         int64_t(batch) * seq * seq * 4 * 2 +                       // bias + scores
         int64_t(seq) * (Q + KV) * 4;                               // head scratch
}

void LM::embed(int batch, int seq, const int64_t* ids, const uint8_t* mask, float* out) {
  const int C = cfg_.n_codebooks, H = cfg_.hidden;
  std::vector<float> acc(static_cast<size_t>(H));
  for (int b = 0; b < batch; ++b) {
    const int64_t* base = ids + static_cast<int64_t>(b) * C * seq;
    for (int s = 0; s < seq; ++s) {
      float* o = out + (static_cast<int64_t>(b) * seq + s) * H;
      if (mask[static_cast<int64_t>(b) * seq + s]) {
        std::fill(acc.begin(), acc.end(), 0.f);
        for (int c = 0; c < C; ++c) {
          const int64_t id = base[static_cast<int64_t>(c) * seq + s] + codebook_offsets_[c];
          token_scratch_.resize(static_cast<size_t>(H));
          audio_emb_.gather(id, token_scratch_.data());
          for (int h = 0; h < H; ++h) acc[h] += token_scratch_[h];
        }
        std::memcpy(o, acc.data(), sizeof(float) * static_cast<size_t>(H));
      } else {
        token_emb_.gather(base[s], o);
      }
    }
  }
}

void LM::embed_cfg(int64_t rows, const int64_t* ids, const uint8_t* mask, float* out) {
  const int C = cfg_.n_codebooks, H = cfg_.hidden;
  std::vector<float> acc(static_cast<size_t>(H));
  token_scratch_.resize(static_cast<size_t>(H));
  for (int64_t r = 0; r < rows; ++r) {
    float* o = out + r * H;
    if (mask[r]) {
      std::fill(acc.begin(), acc.end(), 0.f);
      for (int c = 0; c < C; ++c) {
        const int64_t id = ids[static_cast<int64_t>(c) * rows + r] + codebook_offsets_[c];
        audio_emb_.gather(id, token_scratch_.data());
        for (int h = 0; h < H; ++h) acc[h] += token_scratch_[h];
      }
      std::memcpy(o, acc.data(), sizeof(float) * static_cast<size_t>(H));
    } else {
      token_emb_.gather(ids[r], o);
    }
  }
}

void LM::run_layer(int64_t T, const AttnSegment* segs, int n_segs, int li, const int64_t* pos,
                   float* x) {
  const int H = cfg_.hidden, NH = cfg_.n_heads, NKV = cfg_.n_kv_heads, D = cfg_.head_dim;
  const int64_t QD = int64_t(NH) * D, KVD = int64_t(NKV) * D;
  LMLayer& L = layers_[static_cast<size_t>(li)];

  g_prof.start();

  // The three projections read the same xn_: on the VNNI path fuse the norm and
  // the one-per-layer quantisation (7 quantisations per layer -> 4) and never
  // materialise xn_ in global memory.
  if (L.qkv_proj.vnni_packed() || (L.q_proj.vnni_packed() && L.k_proj.vnni_packed() &&
                                   L.v_proj.vnni_packed())) {
    t_aq.resize(static_cast<size_t>(s8_nk_quant_bytes(T, H)));
    t_asc.resize(static_cast<size_t>(s8_nk_quant_scales(T, H)));
    rmsnorm_quantize_f32(T, H, x, L.input_ln, cfg_.rms_eps, t_aq.data(), t_asc.data());
    if (L.qkv_proj.vnni_packed()) {
      const S8NkSlice slices[3] = {
          {L.q_proj.N, q_.data()}, {L.k_proj.N, k_.data()}, {L.v_proj.N, v_.data()}};
      linear_s8_nk_packed_q_sliced(T, H, t_aq.data(), t_asc.data(), L.qkv_proj.packed.data(),
                                   L.qkv_proj.wsum.data(), L.qkv_proj.row_scale(), slices, 3);
    } else {
      L.q_proj.mul_q(T, t_aq.data(), t_asc.data(), q_.data());
      L.k_proj.mul_q(T, t_aq.data(), t_asc.data(), k_.data());
      L.v_proj.mul_q(T, t_aq.data(), t_asc.data(), v_.data());
    }
  } else {
    rmsnorm_into_f32(xn_.data(), x, L.input_ln, T, H, cfg_.rms_eps);
    L.q_proj.mul(T, xn_.data(), q_.data());
    L.k_proj.mul(T, xn_.data(), k_.data());
    L.v_proj.mul(T, xn_.data(), v_.data());
  }
  g_prof.qkv += g_prof.stop();

  g_prof.start();
  // Qwen3 applies per-head RMSNorm *before* RoPE; fuse both to one pass.
  qk_norm_rope_f32(q_.data(), k_.data(), L.q_norm, L.k_norm, T, NH, NKV, D, pos,
                   cfg_.rope_theta, cfg_.rms_eps);
  g_prof.a_prep += g_prof.stop();
  const float scale = 1.f / std::sqrt(static_cast<float>(D));
  const int group = NH / NKV;

  // Parallelise over (segment, KV head, query head) and keep the per-head GEMMs
  // serial: a 158x158x128 gemm is ~6 MFLOP, so opening a 32-thread team for it
  // costs more than the math.  A normal B=2 batch has 32 tasks; the CFG path has
  // 16 tasks per segment (32 total) and skips the unused uncond tail rows.
#pragma omp parallel for collapse(3) schedule(static)
  for (int seg = 0; seg < n_segs; ++seg) {
    for (int kv = 0; kv < NKV; ++kv) {
      for (int g = 0; g < group; ++g) {
        const int h = kv * group + g;
        const AttnSegment& a = segs[seg];
        const int64_t qoff = a.q_off + static_cast<int64_t>(h) * D;
        const int64_t koff = a.kv_off + static_cast<int64_t>(kv) * D;
        ov::attention_head_f32(q_.data() + qoff, QD, k_.data() + koff, v_.data() + koff, KVD,
                               a.bias, a.rows, D, scale, attn_.data() + qoff, QD);
      }
    }
  }

  g_prof.attn += g_prof.stop();

  g_prof.start();
  if (L.o_proj.add_supported()) {
    L.o_proj.mul_add(T, attn_.data(), x);
  } else {
    L.o_proj.mul(T, attn_.data(), proj_.data());
    add_inplace_f32(x, proj_.data(), T * H);
  }
  g_prof.oproj += g_prof.stop();

  g_prof.start();
  if (L.gate_up_proj.vnni_packed() || (L.gate.vnni_packed() && L.up.vnni_packed())) {
    t_aq.resize(static_cast<size_t>(s8_nk_quant_bytes(T, H)));
    t_asc.resize(static_cast<size_t>(s8_nk_quant_scales(T, H)));
    rmsnorm_quantize_f32(T, H, x, L.post_ln, cfg_.rms_eps, t_aq.data(), t_asc.data());
    if (L.gate_up_proj.vnni_packed()) {
      const S8NkSlice slices[2] = {{L.gate.N, gate_.data()}, {L.up.N, up_.data()}};
      linear_s8_nk_packed_q_sliced(T, H, t_aq.data(), t_asc.data(),
                                   L.gate_up_proj.packed.data(), L.gate_up_proj.wsum.data(),
                                   L.gate_up_proj.row_scale(), slices, 2);
    } else {
      L.gate.mul_q(T, t_aq.data(), t_asc.data(), gate_.data());
      L.up.mul_q(T, t_aq.data(), t_asc.data(), up_.data());
    }
  } else {
    rmsnorm_into_f32(xn_.data(), x, L.post_ln, T, H, cfg_.rms_eps);
    L.gate.mul(T, xn_.data(), gate_.data());
    L.up.mul(T, xn_.data(), up_.data());
  }
  if (L.down.vnni_packed()) {
    t_aq.resize(static_cast<size_t>(s8_nk_quant_bytes(T, cfg_.inter)));
    t_asc.resize(static_cast<size_t>(s8_nk_quant_scales(T, cfg_.inter)));
    silu_mul_quantize_f32(T, cfg_.inter, gate_.data(), up_.data(), t_aq.data(), t_asc.data());
    L.down.mul_q_add(T, t_aq.data(), t_asc.data(), x);
  } else {
    silu_mul_f32(gate_.data(), gate_.data(), up_.data(), T * cfg_.inter);
    if (L.down.add_supported()) {
      L.down.mul_add(T, gate_.data(), x);
    } else {
      L.down.mul(T, gate_.data(), proj_.data());
      add_inplace_f32(x, proj_.data(), T * H);
    }
  }
  g_prof.mlp += g_prof.stop();
}

bool LM::sample_guided(const void* cond, int cond_seq, const void* uncond, int uncond_seq,
                       int target_len, float guidance, float layer_penalty, int32_t* pred,
                       float* scores) {
  if (!cuda_sample_) return false;
  cuda_sample_(cuda_sample_ctx_, cond, cond_seq, uncond, uncond_seq, target_len, guidance,
               layer_penalty, pred, scores);
  return true;
}

void LM::set_cuda_backend(void* ctx, CudaForwardFn fn, CudaLogitsFn logits_fn) {
  cuda_ctx_ = ctx;
  cuda_fn_ = fn;
  cuda_logits_ = logits_fn;
}

void LM::forward(int batch, int seq, const int64_t* input_ids, const uint8_t* audio_mask,
                 const uint8_t* attn_mask, const int64_t* position_ids, float* logits) {
  if (cuda_fn_) {
    last_seq_ = seq;
    cuda_fn_(cuda_ctx_, batch, seq, input_ids, audio_mask, attn_mask, position_ids, logits);
    return;
  }
  // Every worker thread that will touch masked-softmax output needs FTZ/DAZ.
#pragma omp parallel
  { enable_flush_to_zero(); }

  const int H = cfg_.hidden, C = cfg_.n_codebooks, V = cfg_.audio_vocab;
  const int64_t T = int64_t(batch) * seq;
  const int64_t QD = int64_t(cfg_.n_heads) * cfg_.head_dim;
  const int64_t KVD = int64_t(cfg_.n_kv_heads) * cfg_.head_dim;

  x_.resize(static_cast<size_t>(T * H));
  xn_.resize(static_cast<size_t>(T * H));
  q_.resize(static_cast<size_t>(T * QD));
  k_.resize(static_cast<size_t>(T * KVD));
  v_.resize(static_cast<size_t>(T * KVD));
  attn_.resize(static_cast<size_t>(T * QD));
  proj_.resize(static_cast<size_t>(T * H));
  gate_.resize(static_cast<size_t>(T * cfg_.inter));
  up_.resize(static_cast<size_t>(T * cfg_.inter));
  bias_.resize(static_cast<size_t>(T * seq));

  for (int64_t i = 0, n = T * seq; i < n; ++i) bias_[i] = attn_mask[i] ? 0.f : -1e30f;

  g_prof.start();
  embed(batch, seq, input_ids, audio_mask, x_.data());
  g_prof.embed += g_prof.stop();
  std::vector<AttnSegment> segs(static_cast<size_t>(batch));
  for (int b = 0; b < batch; ++b) {
    segs[static_cast<size_t>(b)] = {bias_.data() + static_cast<int64_t>(b) * seq * seq, seq,
                                    static_cast<int64_t>(b) * seq * QD,
                                    static_cast<int64_t>(b) * seq * KVD};
  }
  for (int li = 0; li < cfg_.n_layers; ++li) {
    run_layer(T, segs.data(), batch, li, position_ids, x_.data());
  }

  g_prof.start();
  last_seq_ = seq;
  rmsnorm_f32(x_.data(), final_norm_, T, H, cfg_.rms_eps);
  logits_flat_.resize(static_cast<size_t>(T * C * V));
  audio_heads_.mul(T, x_.data(), logits_flat_.data());

  // [B*S, C*V] -> [B, C, S, V]
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < C; ++c) {
      for (int s = 0; s < seq; ++s) {
        const float* src = logits_flat_.data() +
                           (static_cast<int64_t>(b) * seq + s) * (int64_t(C) * V) +
                           static_cast<int64_t>(c) * V;
        float* dst = logits + ((static_cast<int64_t>(b) * C + c) * seq + s) * V;
        std::memcpy(dst, src, sizeof(float) * static_cast<size_t>(V));
      }
    }
  }
  g_prof.head += g_prof.stop();
  g_prof.report(batch, seq);
}

void LM::forward_cfg(int cond_seq, int uncond_seq, const int64_t* input_ids,
                     const uint8_t* audio_mask, const uint8_t* cond_attn,
                     const uint8_t* uncond_attn, const int64_t* position_ids, float* logits) {
  if (cond_seq <= 0 || uncond_seq <= 0) {
    throw std::runtime_error("forward_cfg: both segments must be non-empty");
  }
  if (cuda_fn_) throw std::runtime_error("forward_cfg: CPU path only");

  // Every worker thread that will touch masked-softmax output needs FTZ/DAZ.
#pragma omp parallel
  { enable_flush_to_zero(); }

  const int H = cfg_.hidden;
  const int64_t rows = int64_t(cond_seq) + uncond_seq;
  const int64_t QD = int64_t(cfg_.n_heads) * cfg_.head_dim;
  const int64_t KVD = int64_t(cfg_.n_kv_heads) * cfg_.head_dim;

  x_.resize(static_cast<size_t>(rows * H));
  xn_.resize(static_cast<size_t>(rows * H));
  q_.resize(static_cast<size_t>(rows * QD));
  k_.resize(static_cast<size_t>(rows * KVD));
  v_.resize(static_cast<size_t>(rows * KVD));
  attn_.resize(static_cast<size_t>(rows * QD));
  proj_.resize(static_cast<size_t>(rows * H));
  gate_.resize(static_cast<size_t>(rows * cfg_.inter));
  up_.resize(static_cast<size_t>(rows * cfg_.inter));

  bias_c_.resize(static_cast<size_t>(int64_t(cond_seq) * cond_seq));
  bias_u_.resize(static_cast<size_t>(int64_t(uncond_seq) * uncond_seq));
  for (int64_t i = 0, n = int64_t(cond_seq) * cond_seq; i < n; ++i)
    bias_c_[i] = cond_attn[i] ? 0.f : -1e30f;
  for (int64_t i = 0, n = int64_t(uncond_seq) * uncond_seq; i < n; ++i)
    bias_u_[i] = uncond_attn[i] ? 0.f : -1e30f;

  g_prof.start();
  embed_cfg(rows, input_ids, audio_mask, x_.data());
  g_prof.embed += g_prof.stop();

  const AttnSegment segs[2] = {
      {bias_c_.data(), cond_seq, 0, 0},
      {bias_u_.data(), uncond_seq, int64_t(cond_seq) * QD, int64_t(cond_seq) * KVD}};
  for (int li = 0; li < cfg_.n_layers; ++li) {
    run_layer(rows, segs, 2, li, position_ids, x_.data());
  }

  g_prof.start();
  last_seq_ = cond_seq;
  rmsnorm_f32(x_.data(), final_norm_, rows, H, cfg_.rms_eps);
  audio_heads_.mul(rows, x_.data(), logits);
  g_prof.head += g_prof.stop();
  g_prof.report(1, static_cast<int>(rows));
}

}  // namespace ov

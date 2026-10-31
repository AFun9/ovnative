#include "model/at_encoder.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "model/conv.h"
#include "model/ops.h"

namespace ov {
namespace {

// torchaudio.functional.resample(24000 -> 16000), sinc_interp_hann,
// lowpass_filter_width=6, rolloff=0.99.  Ratios reduce to orig=3, new=2, so the
// filter has 2 polyphase branches of 2*width+orig taps.
std::vector<float> make_resample_kernel(int orig, int neu, int& width) {
  constexpr int kWidth = 6;
  constexpr double kRolloff = 0.99;
  const double base_freq = static_cast<double>(std::min(orig, neu)) * kRolloff;
  width = static_cast<int>(std::ceil(kWidth * orig / base_freq));

  const double scale = base_freq / static_cast<double>(orig);
  const int taps = 2 * width + orig;
  std::vector<float> k(static_cast<size_t>(neu) * taps, 0.f);
  for (int phase = 0; phase < neu; ++phase) {
    for (int tap = 0; tap < taps; ++tap) {
      double t = (static_cast<double>(tap - width) / orig - static_cast<double>(phase) / neu) * base_freq;
      t = std::max(-static_cast<double>(kWidth), std::min(static_cast<double>(kWidth), t));
      const double window = std::cos(t * M_PI / kWidth / 2.0);
      const double sinc = (t == 0.0) ? 1.0 : std::sin(t * M_PI) / (t * M_PI);
      k[static_cast<size_t>(phase) * taps + tap] =
          static_cast<float>(sinc * window * window * scale);
    }
  }
  return k;
}

}  // namespace

std::vector<float> AudioEncoder::resample_16k(const float* wav, int n, int& out_n) {
  constexpr int orig = 3, neu = 2;  // 24000/8000, 16000/8000
  // `width` must outlive the kernel initialisation: as a plain local it would be
  // 0 on every call after the first one.
  static int width = 0;
  static const std::vector<float> kern = make_resample_kernel(orig, neu, width);
  const int taps = 2 * width + orig;

  const int padded = n + 2 * width + orig;
  std::vector<float> x(static_cast<size_t>(padded), 0.f);
  std::memcpy(x.data() + width, wav, sizeof(float) * static_cast<size_t>(n));

  const int t_out = (padded - taps) / orig + 1;
  std::vector<float> y;
  y.reserve(static_cast<size_t>(t_out) * neu);
  for (int t = 0; t < t_out; ++t) {
    for (int p = 0; p < neu; ++p) {
      const float* kp = kern.data() + static_cast<size_t>(p) * taps;
      const float* xp = x.data() + static_cast<size_t>(t) * orig;
      float acc = 0.f;
      for (int i = 0; i < taps; ++i) acc += xp[i] * kp[i];
      y.push_back(acc);
    }
  }
  const int target = static_cast<int>(std::ceil(static_cast<double>(neu) * n / orig));
  if (static_cast<int>(y.size()) > target) y.resize(static_cast<size_t>(target));
  out_n = static_cast<int>(y.size());
  return y;
}

AudioEncoder::AudioEncoder(const WeightFile& wf, ATEncoderConfig cfg) : cfg_(cfg) {
  // ---- conv stem --------------------------------------------------------
  int cin = 1;
  for (int i = 0; i < 7; ++i) {
    const std::string p = "semantic_model.feature_extractor.conv_layers." + std::to_string(i) + ".";
    const Tensor& w = wf.Get(p + "conv.weight");  // [cout, cin, k]
    StemConv sc;
    sc.w = w.f32();
    sc.cin = cin;
    sc.cout = static_cast<int>(w.shape[0]);
    sc.k = static_cast<int>(w.shape[2]);
    sc.stride = cfg_.stem_stride[i];
    const Tensor* b = wf.Find(p + "conv.bias");
    sc.b = b ? b->f32() : nullptr;
    stem_.push_back(sc);
    cin = sc.cout;
    if (sc.cout != cfg_.feat_dim) throw std::runtime_error("stem width mismatch");
  }
  stem_gn_w_ = make_norm(wf, "semantic_model.feature_extractor.conv_layers.0.layer_norm.weight", cfg_.feat_dim);
  stem_gn_b_ = make_norm(wf, "semantic_model.feature_extractor.conv_layers.0.layer_norm.bias", cfg_.feat_dim);

  // ---- feature projection ----------------------------------------------
  proj_ln_w_ = make_norm(wf, "semantic_model.feature_projection.layer_norm.weight", cfg_.feat_dim);
  proj_ln_b_ = make_norm(wf, "semantic_model.feature_projection.layer_norm.bias", cfg_.feat_dim);
  proj_ = make_linear(wf, "semantic_model.feature_projection.projection.weight");
  proj_b_ = make_norm(wf, "semantic_model.feature_projection.projection.bias", cfg_.sem_hidden);

  // ---- positional conv (weight-norm over dims 0,1; dim=2 keeps the kernel) --
  {
    const Tensor& g = wf.Get("semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0");
    const Tensor& v = wf.Get("semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1");
    const int64_t cout = v.shape[0], cig = v.shape[1], k = v.shape[2];
    pos_conv_weight_.assign(static_cast<size_t>(cout * cig * k), 0.f);
    const float* gp = g.f32();
    const float* vp = v.f32();
    for (int64_t kk = 0; kk < k; ++kk) {
      double norm = 0.0;
      for (int64_t o = 0; o < cout; ++o)
        for (int64_t i = 0; i < cig; ++i) {
          const double val = vp[(o * cig + i) * k + kk];
          norm += val * val;
        }
      const double scale = static_cast<double>(gp[kk]) / std::sqrt(norm + 1e-24);
      for (int64_t o = 0; o < cout; ++o)
        for (int64_t i = 0; i < cig; ++i) {
          pos_conv_weight_[(o * cig + i) * k + kk] =
              static_cast<float>(vp[(o * cig + i) * k + kk] * scale);
        }
    }
    pos_conv_b_ = make_norm(wf, "semantic_model.encoder.pos_conv_embed.conv.bias", cfg_.sem_hidden);
  }
  enc_ln_w_ = make_norm(wf, "semantic_model.encoder.layer_norm.weight", cfg_.sem_hidden);
  enc_ln_b_ = make_norm(wf, "semantic_model.encoder.layer_norm.bias", cfg_.sem_hidden);

  // ---- transformer layers ----------------------------------------------
  layers_.resize(static_cast<size_t>(cfg_.sem_layers));
  for (int i = 0; i < cfg_.sem_layers; ++i) {
    const std::string p = "semantic_model.encoder.layers." + std::to_string(i) + ".";
    HubertLayer& L = layers_[static_cast<size_t>(i)];
    L.q = make_linear(wf, p + "attention.q_proj.weight");
    L.k = make_linear(wf, p + "attention.k_proj.weight");
    L.v = make_linear(wf, p + "attention.v_proj.weight");
    L.o = make_linear(wf, p + "attention.out_proj.weight");
    L.qb = make_norm(wf, p + "attention.q_proj.bias", cfg_.sem_hidden);
    L.kb = make_norm(wf, p + "attention.k_proj.bias", cfg_.sem_hidden);
    L.vb = make_norm(wf, p + "attention.v_proj.bias", cfg_.sem_hidden);
    L.ob = make_norm(wf, p + "attention.out_proj.bias", cfg_.sem_hidden);
    L.f1 = make_linear(wf, p + "feed_forward.intermediate_dense.weight");
    L.f2 = make_linear(wf, p + "feed_forward.output_dense.weight");
    L.f1b = make_norm(wf, p + "feed_forward.intermediate_dense.bias", cfg_.sem_inter);
    L.f2b = make_norm(wf, p + "feed_forward.output_dense.bias", cfg_.sem_hidden);
    L.ln1w = make_norm(wf, p + "layer_norm.weight", cfg_.sem_hidden);
    L.ln1b = make_norm(wf, p + "layer_norm.bias", cfg_.sem_hidden);
    L.ln2w = make_norm(wf, p + "final_layer_norm.weight", cfg_.sem_hidden);
    L.ln2b = make_norm(wf, p + "final_layer_norm.bias", cfg_.sem_hidden);
  }

  // ---- semantic conv stack (encoder_semantic) ---------------------------
  {
    const Tensor& w = wf.Get("encoder_semantic.conv.weight");  // [768,768,3], no bias
    sem_pre_.cin = static_cast<int>(w.shape[1]);
    sem_pre_.cout = static_cast<int>(w.shape[0]);
    sem_pre_.k = static_cast<int>(w.shape[2]);
    sem_pre_.pad = sem_pre_.k / 2;
    sem_pre_.w = w.f32();
    sem_blocks_.resize(2);
    for (int bi = 0; bi < 2; ++bi) {
      const std::string p = "encoder_semantic.conv_blocks." + std::to_string(bi) + ".";
      SemBlock& b = sem_blocks_[static_cast<size_t>(bi)];
      for (int u = 0; u < 2; ++u) {
        const std::string rp = p + "res_units." + std::to_string(u) + ".";
        const Tensor& c1 = wf.Get(rp + "conv1.weight");
        const Tensor& c2 = wf.Get(rp + "conv2.weight");
        ResUnit& ru = b.units[u];
        ru.c1.cin = ru.c1.cout = static_cast<int>(c1.shape[0]);
        ru.c1.k = static_cast<int>(c1.shape[2]);
        ru.c1.pad = ru.c1.k / 2;
        ru.c1.w = c1.f32();
        ru.c2.cin = ru.c2.cout = static_cast<int>(c2.shape[0]);
        ru.c2.k = static_cast<int>(c2.shape[2]);
        ru.c2.pad = 0;
        ru.c2.w = c2.f32();
      }
      const Tensor& cb = wf.Get(p + "conv.weight");
      b.conv.cin = static_cast<int>(cb.shape[1]);
      b.conv.cout = static_cast<int>(cb.shape[0]);
      b.conv.k = static_cast<int>(cb.shape[2]);
      b.conv.pad = b.conv.k / 2;
      b.conv.w = cb.f32();
      b.conv.b = make_norm(wf, p + "conv.bias", b.conv.cout);
    }
  }

  // ---- DAC acoustic encoder --------------------------------------------
  {
    const Tensor& w1 = wf.Get("acoustic_encoder.conv1.weight");  // [64,1,7]
    ac_conv1_.cin = static_cast<int>(w1.shape[1]);
    ac_conv1_.cout = static_cast<int>(w1.shape[0]);
    ac_conv1_.k = static_cast<int>(w1.shape[2]);
    ac_conv1_.pad = ac_conv1_.k / 2;
    ac_conv1_.w = w1.f32();
    ac_conv1_.b = make_norm(wf, "acoustic_encoder.conv1.bias", ac_conv1_.cout);

    const int strides[5] = {8, 5, 4, 2, 3};
    ac_blocks_.resize(5);
    for (int bi = 0; bi < 5; ++bi) {
      const int dimension = 64 * (1 << (bi + 1));
      const std::string p = "acoustic_encoder.block." + std::to_string(bi) + ".";
      AcBlock& b = ac_blocks_[static_cast<size_t>(bi)];
      const int rc = dimension / 2;
      for (int u = 0; u < 3; ++u) {
        const std::string rp = p + "res_unit" + std::to_string(u + 1) + ".";
        const Tensor& c1 = wf.Get(rp + "conv1.weight");
        const Tensor& c2 = wf.Get(rp + "conv2.weight");
        ResUnit& ru = b.units[u];
        const int dil = (u == 0) ? 1 : (u == 1 ? 3 : 9);
        ru.c1.cin = ru.c1.cout = rc;
        ru.c1.k = static_cast<int>(c1.shape[2]);
        ru.c1.dil = dil;
        ru.c1.pad = (ru.c1.k - 1) / 2 * dil;
        ru.c1.w = c1.f32();
        ru.c1.b = make_norm(wf, rp + "conv1.bias", rc);
        ru.c2.cin = ru.c2.cout = rc;
        ru.c2.k = static_cast<int>(c2.shape[2]);
        ru.c2.w = c2.f32();
        ru.c2.b = make_norm(wf, rp + "conv2.bias", rc);
        ru.sa1 = make_norm(wf, rp + "snake1.alpha", rc);
        ru.sa2 = make_norm(wf, rp + "snake2.alpha", rc);
      }
      b.snake = make_norm(wf, p + "snake1.alpha", rc);
      const Tensor& cw = wf.Get(p + "conv1.weight");
      const int stride = strides[bi];
      b.conv.cin = static_cast<int>(cw.shape[1]);
      b.conv.cout = static_cast<int>(cw.shape[0]);
      b.conv.k = static_cast<int>(cw.shape[2]);
      b.conv.stride = stride;
      b.conv.pad = (stride + 1) / 2;
      b.conv.w = cw.f32();
      b.conv.b = make_norm(wf, p + "conv1.bias", b.conv.cout);
    }
    const Tensor& w2 = wf.Get("acoustic_encoder.conv2.weight");
    ac_conv2_.cin = static_cast<int>(w2.shape[1]);
    ac_conv2_.cout = static_cast<int>(w2.shape[0]);
    ac_conv2_.k = static_cast<int>(w2.shape[2]);
    ac_conv2_.pad = ac_conv2_.k / 2;
    ac_conv2_.w = w2.f32();
    ac_conv2_.b = make_norm(wf, "acoustic_encoder.conv2.bias", ac_conv2_.cout);
    ac_snake_ = make_norm(wf, "acoustic_encoder.snake1.alpha", 2048);
  }

  // ---- fc + RVQ --------------------------------------------------------
  fc_ = make_linear(wf, "fc.weight");
  fc_b_ = make_norm(wf, "fc.bias", 1024);
  vq_.resize(8);
  for (int q = 0; q < 8; ++q) {
    const std::string p = "quantizer.quantizers." + std::to_string(q) + ".";
    Vq& v = vq_[static_cast<size_t>(q)];
    v.project_in = make_linear(wf, p + "project_in.weight");
    v.pin_b = make_norm(wf, p + "project_in.bias", 64);
    v.embed = make_norm(wf, p + "codebook.embed", 1024 * 64);
    v.project_out = make_linear(wf, p + "project_out.weight");
    v.pout_b = make_norm(wf, p + "project_out.bias", 1024);
  }
}

void AudioEncoder::run_hubert(const float* wav16, int n16, std::vector<float>& out, int& frames) {
  const int H = cfg_.sem_hidden;
  std::vector<float> x(wav16, wav16 + n16);
  int t = n16;
  std::vector<float> y;

    // ---- conv stem (GroupNorm only on the first layer, GELU after every one) --
  for (int i = 0; i < 7; ++i) {
    const StemConv& sc = stem_[static_cast<size_t>(i)];
    Conv1dSpec spec;
    spec.cin = sc.cin;
    spec.cout = sc.cout;
    spec.k = sc.k;
    spec.stride = sc.stride;
    spec.b = sc.b;
    const int t_out = (t - spec.k) / spec.stride + 1;
    y.assign(static_cast<size_t>(t_out) * spec.cout, 0.f);
    conv1d_generic_f32(spec, sc.w, x.data(), t, y.data(), t_out);
    if (i == 0) {
            group_norm_channels_f32(y.data(), stem_gn_w_, stem_gn_b_, cfg_.feat_dim, t_out,
                              cfg_.layer_norm_eps);
    }
    gelu_f32(y.data(), static_cast<int64_t>(t_out) * cfg_.feat_dim);
        x.swap(y);
    t = t_out;
  }

  if (const char* dd = std::getenv("OV_DUMP_DIR")) {
    std::FILE* f = std::fopen((std::string(dd) + "/cpp_stem.f32").c_str(), "wb");
    if (f) { std::fwrite(x.data(), 4, x.size(), f); std::fclose(f); }
  }

  // ---- feature projection: LayerNorm(512) -> Linear(512,768) -------------
  std::vector<float> stem_out(x);  // [T, 512]
  layer_norm_f32(stem_out.data(), proj_ln_w_, proj_ln_b_, t, cfg_.feat_dim, cfg_.layer_norm_eps);
  std::vector<float> proj_out(static_cast<size_t>(t) * H);
  proj_.mul(t, stem_out.data(), proj_out.data());
  for (int i = 0; i < t; ++i) {
    float* row = proj_out.data() + static_cast<size_t>(i) * H;
    for (int c = 0; c < H; ++c) row[c] += proj_b_[c];
  }

  if (const char* dd = std::getenv("OV_DUMP_DIR")) {
    std::FILE* f = std::fopen((std::string(dd) + "/cpp_proj.f32").c_str(), "wb");
    if (f) { std::fwrite(proj_out.data(), 4, proj_out.size(), f); std::fclose(f); }
  }

  // ---- positional conv (groups=16, k=128, pad=64) + trim + LayerNorm -----
  std::vector<float> h(proj_out);
  {
    Conv1dSpec spec;
    spec.cin = H;
    spec.cout = H;
    spec.k = cfg_.pos_conv_kernel;
    spec.pad = cfg_.pos_conv_kernel / 2;
    spec.groups = cfg_.pos_conv_groups;
    spec.b = pos_conv_b_;
    const int t_out = t + 1;  // even kernel -> one extra frame, dropped below
    std::vector<float> pc(static_cast<size_t>(t_out) * H, 0.f);
    conv1d_generic_f32(spec, pos_conv_weight_.data(), proj_out.data(), t, pc.data(), t_out);
    gelu_f32(pc.data(), static_cast<int64_t>(t_out) * H);  // module applies GELU after the conv
        for (int i = 0; i < t; ++i) {
      float* dst = h.data() + static_cast<size_t>(i) * H;
      const float* src = pc.data() + static_cast<size_t>(i) * H;
      for (int c = 0; c < H; ++c) dst[c] += src[c];
    }
  }
  layer_norm_f32(h.data(), enc_ln_w_, enc_ln_b_, t, H, cfg_.layer_norm_eps);

  if (const char* dd = std::getenv("OV_DUMP_DIR")) {
    std::FILE* f = std::fopen((std::string(dd) + "/cpp_enc_in.f32").c_str(), "wb");
    if (f) { std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
  }

  // ---- 12 post-norm layers; hidden states = [projection, layer inputs...] --
  const int heads = cfg_.sem_heads, hd = H / heads;
  const float scale = 1.f / std::sqrt(static_cast<float>(hd));
  std::vector<float> hs_sum(static_cast<size_t>(t) * H, 0.f);
  int n_states = 0;
  auto accumulate = [&](const float* v) {
    for (size_t i = 0, n = static_cast<size_t>(t) * H; i < n; ++i) hs_sum[i] += v[i];
    ++n_states;
  };
  std::vector<float> q(static_cast<size_t>(t) * H), k(static_cast<size_t>(t) * H),
      v(static_cast<size_t>(t) * H), attn(static_cast<size_t>(t) * H),
      proj(static_cast<size_t>(t) * H), ffn(static_cast<size_t>(t) * cfg_.sem_inter);

  for (int li = 0; li < cfg_.sem_layers; ++li) {
    HubertLayer& L = layers_[static_cast<size_t>(li)];
    accumulate(h.data());

    L.q.mul(t, h.data(), q.data());
    L.k.mul(t, h.data(), k.data());
    L.v.mul(t, h.data(), v.data());
    for (int i = 0; i < t; ++i) {
      float* rq = q.data() + static_cast<size_t>(i) * H;
      float* rk = k.data() + static_cast<size_t>(i) * H;
      float* rv = v.data() + static_cast<size_t>(i) * H;
      for (int c = 0; c < H; ++c) {
        rq[c] += L.qb[c];
        rk[c] += L.kb[c];
        rv[c] += L.vb[c];
      }
    }
    for (int hh = 0; hh < heads; ++hh) {
      attention_head_f32(q.data() + hh * hd, H, k.data() + hh * hd, v.data() + hh * hd, H, nullptr,
                         t, hd, scale, attn.data() + hh * hd, H);
    }
    L.o.mul(t, attn.data(), proj.data());
    for (int i = 0; i < t; ++i) {
      float* r = h.data() + static_cast<size_t>(i) * H;
      const float* a = proj.data() + static_cast<size_t>(i) * H;
      for (int c = 0; c < H; ++c) r[c] += a[c] + L.ob[c];
    }
    layer_norm_f32(h.data(), L.ln1w, L.ln1b, t, H, cfg_.layer_norm_eps);

    L.f1.mul(t, h.data(), ffn.data());
    for (int i = 0; i < t; ++i) {
      float* r = ffn.data() + static_cast<size_t>(i) * cfg_.sem_inter;
      for (int c = 0; c < cfg_.sem_inter; ++c) r[c] += L.f1b[c];
    }
    gelu_f32(ffn.data(), static_cast<int64_t>(t) * cfg_.sem_inter);
    L.f2.mul(t, ffn.data(), proj.data());
    for (int i = 0; i < t; ++i) {
      float* r = h.data() + static_cast<size_t>(i) * H;
      const float* a = proj.data() + static_cast<size_t>(i) * H;
      for (int c = 0; c < H; ++c) r[c] += a[c] + L.f2b[c];
    }
    layer_norm_f32(h.data(), L.ln2w, L.ln2b, t, H, cfg_.layer_norm_eps);
    if (li == 0) {
      if (const char* dd = std::getenv("OV_DUMP_DIR")) {
        std::FILE* f = std::fopen((std::string(dd) + "/cpp_layer0.f32").c_str(), "wb");
        if (f) { std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
      }
    }
  }

    // The encoder's hidden-state tuple is [input of layer 0 .. input of layer 11,
  // final output] — the projection output is NOT part of it (verified against
  // PyTorch: mean over the tuple reproduces hubert_mean_hidden bit-exactly).
  accumulate(h.data());

  // ---- mean over hidden states, then 2x temporal downsample -------------
  frames = t / cfg_.semantic_downsample;
  out.assign(static_cast<size_t>(frames) * H, 0.f);
  const float inv = 1.f / static_cast<float>(n_states);
  for (int f = 0; f < frames; ++f) {
    const int src = f * cfg_.semantic_downsample;
    for (int c = 0; c < H; ++c) {
      out[static_cast<size_t>(f) * H + c] = hs_sum[static_cast<size_t>(src) * H + c] * inv;
    }
  }
}

void AudioEncoder::semantic_features(const float* wav24k, int n_samples, std::vector<float>& out,
                                     int& frames) {
  int n16 = 0;
  const std::vector<float> wav16 = resample_16k(wav24k, n_samples, n16);

  constexpr int kPad = 160;
  std::vector<float> padded(static_cast<size_t>(n16 + 2 * kPad), 0.f);
  std::memcpy(padded.data() + kPad, wav16.data(), sizeof(float) * static_cast<size_t>(n16));
  if (const char* dd = std::getenv("OV_DUMP_DIR")) {
    std::FILE* f = std::fopen((std::string(dd) + "/cpp_wav16.f32").c_str(), "wb");
    if (f) { std::fwrite(wav16.data(), 4, wav16.size(), f); std::fclose(f); }
  }
  run_hubert(padded.data(), static_cast<int>(padded.size()), out, frames);
}

void AudioEncoder::semantic_encode(const float* sem, int frames, std::vector<float>& out) {
  const int C = cfg_.sem_hidden;
  const size_t n = static_cast<size_t>(frames) * C;
  out.assign(sem, sem + n);

  std::vector<float> y(n);
  conv1d_generic_f32(sem_pre_, nullptr, out.data(), frames, y.data(), frames);
  out.swap(y);

  for (int bi = 0; bi < 2; ++bi) {
    SemBlock& b = sem_blocks_[static_cast<size_t>(bi)];
    for (int u = 0; u < 2; ++u) {
      ResUnit& ru = b.units[u];
      std::vector<float> x0(out);
      elu_f32(out.data(), static_cast<int64_t>(n));
      conv1d_generic_f32(ru.c1, nullptr, out.data(), frames, y.data(), frames);
      elu_f32(y.data(), static_cast<int64_t>(n));
      conv1d_generic_f32(ru.c2, nullptr, y.data(), frames, out.data(), frames);
      for (size_t i = 0; i < n; ++i) out[i] += x0[i];
    }
    conv1d_generic_f32(b.conv, nullptr, out.data(), frames, y.data(), frames);
    out.swap(y);
  }
}

void AudioEncoder::acoustic_encode(const float* wav24k, int n_samples, std::vector<float>& out,
                                   int& frames) {
  auto out_len = [&](int len) {
    int t = len;  // conv1: k7 pad3 stride1
    const int strides[5] = {8, 5, 4, 2, 3};
    for (int bi = 0; bi < 5; ++bi) {
      const int s = strides[bi], k = 2 * s, pad = (s + 1) / 2;
      t = (t + 2 * pad - k) / s + 1;
    }
    return t;  // conv2: k3 pad1 stride1
  };

  // Mirror `encode`: pad only when the unpadded length disagrees with the
  // semantic frame count (decided by the caller).
  std::vector<float> padded;
  const float* src = wav24k;
  int t_in = n_samples;
  if (pad_for_acoustic_) {
    const int pad = cfg_.acoustic_pad;
    padded.assign(static_cast<size_t>(n_samples) + 2 * pad, 0.f);
    std::memcpy(padded.data() + pad, wav24k, sizeof(float) * static_cast<size_t>(n_samples));
    src = padded.data();
    t_in = n_samples + 2 * pad;
  }

  // conv1: 1 -> 64, k=7 pad=3 stride=1 (length preserved)
  int t = t_in;
  std::vector<float> a(static_cast<size_t>(t) * ac_conv1_.cout);
  conv1d_generic_f32(ac_conv1_, nullptr, src, t_in, a.data(), t_in);

  std::vector<float> y;
  for (int bi = 0; bi < 5; ++bi) {
    AcBlock& b = ac_blocks_[static_cast<size_t>(bi)];
    const int rc = b.units[0].c1.cout;
    for (int u = 0; u < 3; ++u) {
      ResUnit& ru = b.units[u];
      std::vector<float> x0(a);
      snake_f32(a.data(), ru.sa1, rc, t);
      std::vector<float> tmp(static_cast<size_t>(t) * rc);
      conv1d_generic_f32(ru.c1, nullptr, a.data(), t, tmp.data(), t);
      snake_f32(tmp.data(), ru.sa2, rc, t);
      conv1d_generic_f32(ru.c2, nullptr, tmp.data(), t, a.data(), t);
      for (size_t i = 0, n2 = a.size(); i < n2; ++i) a[i] += x0[i];
    }
    snake_f32(a.data(), b.snake, rc, t);
    const int t_out = (t + 2 * b.conv.pad - b.conv.k) / b.conv.stride + 1;
    y.assign(static_cast<size_t>(t_out) * b.conv.cout, 0.f);
    conv1d_generic_f32(b.conv, nullptr, a.data(), t, y.data(), t_out);
    a.swap(y);
    t = t_out;
  }
  snake_f32(a.data(), ac_snake_, ac_conv2_.cin, t);
  const int t_out = (t + 2 * ac_conv2_.pad - ac_conv2_.k) + 1;
  out.assign(static_cast<size_t>(t_out) * ac_conv2_.cout, 0.f);
  conv1d_generic_f32(ac_conv2_, nullptr, a.data(), t, out.data(), t_out);
  frames = t_out;
}

void AudioEncoder::encode(const float* wav24k, int n_samples, std::vector<int64_t>& codes,
                          int& frames) {
  if (cuda_encode_) {
    cuda_encode_(cuda_ctx_, wav24k, n_samples, codes, frames);
    return;
  }
  encode_with_stages(wav24k, n_samples, nullptr, codes, frames);
}

void AudioEncoder::encode_with_stages(const float* wav24k, int n_samples, Stages* stages,
                                      std::vector<int64_t>& codes, int& frames) {
  // 1) semantic tower
  std::vector<float> sem;
  semantic_features(wav24k, n_samples, sem, frames);
  const int T = frames;
  const int H = cfg_.sem_hidden;

  // 2) semantic conv stack
  std::vector<float> es;
  semantic_encode(sem.data(), T, es);

  // 3) acoustic encoder (pad only if the unpadded frame count differs)
  auto enc_frames = [&](int len) {
    int t = len;
    const int strides[5] = {8, 5, 4, 2, 3};
    for (int i = 0; i < 5; ++i) {
      const int st = strides[i], k = 2 * st, pad = (st + 1) / 2;
      t = (t + 2 * pad - k) / st + 1;
    }
    return t;
  };
  pad_for_acoustic_ = (enc_frames(n_samples) != T);

  std::vector<float> ea;
  int Ta = 0;
  acoustic_encode(wav24k, n_samples, ea, Ta);
  if (Ta != T) {
    throw std::runtime_error("acoustic/semantic frame count mismatch: " + std::to_string(Ta) +
                             " vs " + std::to_string(T));
  }

  // 4) concat [acoustic 256 | semantic 768] and fc
  std::vector<float> emb(static_cast<size_t>(T) * 1024);
  for (int i = 0; i < T; ++i) {
    float* row = emb.data() + static_cast<size_t>(i) * 1024;
    std::memcpy(row, ea.data() + static_cast<size_t>(i) * 256, sizeof(float) * 256);
    std::memcpy(row + 256, es.data() + static_cast<size_t>(i) * H, sizeof(float) * H);
  }
  std::vector<float> z(static_cast<size_t>(T) * 1024);
  fc_.mul(T, emb.data(), z.data());
  for (int i = 0; i < T; ++i) {
    float* row = z.data() + static_cast<size_t>(i) * 1024;
    for (int c = 0; c < 1024; ++c) row[c] += fc_b_[c];
  }

  if (stages) {
    stages->sem = sem;
    stages->es = es;
    stages->ea = ea;
    stages->emb = z;
    stages->frames = T;
  }

  // 5) RVQ encode (8 quantizers for the 2.0 kbps setting)
  const int Q = 8, D = 64, V = 1024;
  codes.assign(static_cast<size_t>(Q) * T, 0);
  std::vector<float> residual(z), x64(static_cast<size_t>(T) * D), qq(static_cast<size_t>(T) * 1024),
      emb64(static_cast<size_t>(T) * D);
  std::vector<float> norm2(V);

  for (int qi = 0; qi < Q; ++qi) {
    Vq& v = vq_[static_cast<size_t>(qi)];
    v.project_in.mul(T, residual.data(), x64.data());
    for (int i = 0; i < T; ++i) {
      float* row = x64.data() + static_cast<size_t>(i) * D;
      for (int c = 0; c < D; ++c) row[c] += v.pin_b[c];
    }
    for (int c = 0; c < V; ++c) {
      const float* e = v.embed + static_cast<size_t>(c) * D;
      float n2 = 0.f;
      for (int d = 0; d < D; ++d) n2 += e[d] * e[d];
      norm2[static_cast<size_t>(c)] = n2;
    }
    for (int i = 0; i < T; ++i) {
      const float* xr = x64.data() + static_cast<size_t>(i) * D;
      int best = 0;
      float best_score = -INFINITY;
      for (int c = 0; c < V; ++c) {
        const float* e = v.embed + static_cast<size_t>(c) * D;
        float dot = 0.f;
        for (int d = 0; d < D; ++d) dot += xr[d] * e[d];
        const float score = 2.f * dot - norm2[static_cast<size_t>(c)];
        if (score > best_score) {
          best_score = score;
          best = c;
        }
      }
      codes[static_cast<size_t>(qi) * T + i] = best;
      std::memcpy(emb64.data() + static_cast<size_t>(i) * D, v.embed + static_cast<size_t>(best) * D,
                  sizeof(float) * D);
    }
    v.project_out.mul(T, emb64.data(), qq.data());
    for (int i = 0; i < T; ++i) {
      float* r = residual.data() + static_cast<size_t>(i) * 1024;
      const float* q = qq.data() + static_cast<size_t>(i) * 1024;
      for (int c = 0; c < 1024; ++c) r[c] -= (q[c] + v.pout_b[c]);
    }
  }
}

}  // namespace ov

#include "model/audio_tokenizer.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "model/conv.h"

namespace ov {
namespace {

void copy_f32(std::vector<float>& dst, const WeightFile& wf, const std::string& name,
              int64_t expect) {
  const float* p = make_norm(wf, name, expect);
  dst.assign(p, p + expect);
}

}  // namespace

AcousticDecoder::AcousticDecoder(const WeightFile& wf, ATConfig cfg) : cfg_(cfg) {
  if (cfg_.up_ratios.size() != 5) throw std::runtime_error("expected 5 upsampling ratios");

  // ---- RVQ: 8 × (codebook lookup -> project_out 64 -> 1024) --------------
  rvq_.resize(static_cast<size_t>(cfg_.n_quantizers));
  for (int q = 0; q < cfg_.n_quantizers; ++q) {
    const std::string p = "quantizer.quantizers." + std::to_string(q) + ".";
    Rvq& r = rvq_[static_cast<size_t>(q)];
    r.project_out = make_linear(wf, p + "project_out.weight");
    r.project_out_bias = make_norm(wf, p + "project_out.bias", cfg_.hidden);
    const Tensor& e = wf.Get(p + "codebook.embed");
    if (e.dtype != DType::F32) throw std::runtime_error("codebook.embed must be fp32");
    if (e.transposed) throw std::runtime_error("codebook.embed must stay row-major");
    r.embed = e.f32();
    const int64_t expect = static_cast<int64_t>(cfg_.codebook_size) * cfg_.codebook_dim;
    if (e.shape.numel() != expect) throw std::runtime_error("codebook.embed: unexpected size");
    if (r.project_out.K != cfg_.codebook_dim) throw std::runtime_error("project_out: bad K");
  }

  fc2_ = make_linear(wf, "fc2.weight");
  fc2_bias_ = make_norm(wf, "fc2.bias", cfg_.dec_in);
  if (fc2_.K != cfg_.hidden || fc2_.N != cfg_.dec_in) {
    throw std::runtime_error("fc2: expected [1024 -> 256]");
  }

  // ---- conv1: 256 -> 1024, k=7, pad=3 ------------------------------------
  conv1_.cin = cfg_.dec_in;
  conv1_.cout = cfg_.dec_ch;
  conv1_.k = 7;
  conv1_.pad = 3;
  conv1_.b = make_norm(wf, "acoustic_decoder.conv1.bias", cfg_.dec_ch);
  conv1_.prepare(make_norm(wf, "acoustic_decoder.conv1.weight", -1));
  // NOTE: DacDecoder has no Snake after conv1 — `acoustic_decoder.snake1` is the final one.

  // ---- 5 upsampling blocks ----------------------------------------------
  blocks_.resize(cfg_.up_ratios.size());
  int ch = cfg_.dec_ch;
  for (size_t i = 0; i < cfg_.up_ratios.size(); ++i) {
    const int stride = cfg_.up_ratios[i];
    const int out_ch = ch / 2;
    const std::string p = "acoustic_decoder.block." + std::to_string(i) + ".";
    DecBlock& b = blocks_[i];

    copy_f32(b.snake, wf, p + "snake1.alpha", ch);

    b.convt.cin = ch;
    b.convt.cout = out_ch;
    b.convt.k = 2 * stride;
    b.convt.stride = stride;
    b.convt.pad = (stride + 1) / 2;   // ceil(stride/2)
    b.convt.out_pad = stride % 2;     // _adjust_dac_decoder
    b.convt.b = make_norm(wf, p + "conv_t1.bias", out_ch);
    b.convt.prepare(make_norm(wf, p + "conv_t1.weight", -1));

    const int dilations[3] = {1, 3, 9};
    for (int j = 0; j < 3; ++j) {
      const std::string rp = p + "res_unit" + std::to_string(j + 1) + ".";
      ResUnit& u = b.ru[j];
      copy_f32(u.snake1, wf, rp + "snake1.alpha", out_ch);
      copy_f32(u.snake2, wf, rp + "snake2.alpha", out_ch);

      u.conv1.cin = out_ch;
      u.conv1.cout = out_ch;
      u.conv1.k = 7;
      u.conv1.dil = dilations[j];
      u.conv1.pad = 3 * dilations[j];
      u.conv1.b = make_norm(wf, rp + "conv1.bias", out_ch);
      u.conv1.prepare(make_norm(wf, rp + "conv1.weight", -1));

      u.conv2.cin = out_ch;
      u.conv2.cout = out_ch;
      u.conv2.k = 1;
      u.conv2.pad = 0;
      u.conv2.b = make_norm(wf, rp + "conv2.bias", out_ch);
      u.conv2.prepare(make_norm(wf, rp + "conv2.weight", -1));
    }
    ch = out_ch;
  }

  // ---- final snake + 32 -> 1 projection ---------------------------------
  copy_f32(snake_out_, wf, "acoustic_decoder.snake1.alpha", ch);
  conv2_.cin = ch;
  conv2_.cout = 1;
  conv2_.k = 7;
  conv2_.pad = 3;
  conv2_.b = make_norm(wf, "acoustic_decoder.conv2.bias", 1);
  conv2_.prepare(make_norm(wf, "acoustic_decoder.conv2.weight", -1));
}

int64_t AcousticDecoder::workspace_bytes(int frames) const {
  int64_t samples = int64_t(frames) * cfg_.hop;
  return samples * 8 * static_cast<int64_t>(sizeof(float)) + int64_t(frames) * cfg_.hidden * 8;
}

void AcousticDecoder::decode(const int64_t* codes, int frames, float* audio) {
  const int T = frames, H = cfg_.hidden, D = cfg_.codebook_dim;

  // 1) RVQ: Σ_q project_out(embed[codes[q]])
  std::vector<float> q(static_cast<size_t>(T) * H, 0.f);
  std::vector<float> e(static_cast<size_t>(T) * D);
  std::vector<float> tmp(static_cast<size_t>(T) * H);
  for (int n = 0; n < cfg_.n_quantizers; ++n) {
    const Rvq& r = rvq_[static_cast<size_t>(n)];
    for (int t = 0; t < T; ++t) {
      const int64_t id = codes[static_cast<size_t>(n) * T + t];
      if (id < 0 || id >= cfg_.codebook_size) {
        throw std::runtime_error("code index out of range: " + std::to_string(id));
      }
      std::memcpy(e.data() + static_cast<size_t>(t) * D, r.embed + id * D,
                  sizeof(float) * static_cast<size_t>(D));
    }
    r.project_out.mul(T, e.data(), tmp.data());
    // project_out carries a bias; add it once per frame together with the sum
    for (int t = 0; t < T; ++t) {
      const float* src = tmp.data() + static_cast<size_t>(t) * H;
      float* dst = q.data() + static_cast<size_t>(t) * H;
      for (int j = 0; j < H; ++j) dst[j] += src[j] + r.project_out_bias[j];
    }
  }

  // 2) fc2: [T,1024] -> [T,256]
  std::vector<float> h(static_cast<size_t>(T) * cfg_.dec_in);
  fc2_.mul(T, q.data(), h.data());
  for (int t = 0; t < T; ++t) {
    float* row = h.data() + static_cast<size_t>(t) * cfg_.dec_in;
    for (int j = 0; j < cfg_.dec_in; ++j) row[j] += fc2_bias_[j];
  }

  // 3) conv1 + snake
  int t_cur = conv1_.out_len(T);
  int ch = cfg_.dec_ch;
  std::vector<float> a(static_cast<size_t>(t_cur) * ch);
  conv1d_f32(conv1_, h.data(), T, a.data());

  // 4) blocks
  for (size_t bi = 0; bi < blocks_.size(); ++bi) {
    DecBlock& blk = blocks_[bi];
    const int out_ch = blk.convt.cout;
    snake_f32(a.data(), blk.snake.data(), ch, t_cur);

    const int t_out = blk.convt.out_len(t_cur);
    std::vector<float> b(static_cast<size_t>(t_out) * out_ch);
    conv_transpose1d_f32(blk.convt, a.data(), t_cur, b.data());

    for (int j = 0; j < 3; ++j) {
      ResUnit& u = blk.ru[j];
      std::vector<float> x0(b.begin(), b.end());
      snake_f32(b.data(), u.snake1.data(), out_ch, t_out);
      std::vector<float> y(static_cast<size_t>(t_out) * out_ch);
      conv1d_f32(u.conv1, b.data(), t_out, y.data());
      snake_f32(y.data(), u.snake2.data(), out_ch, t_out);
      conv1d_f32(u.conv2, y.data(), t_out, b.data());
      for (size_t i = 0, n = b.size(); i < n; ++i) b[i] += x0[i];
    }

    a.swap(b);
    ch = out_ch;
    t_cur = t_out;
  }

  // 5) final snake + conv2 -> mono waveform
  snake_f32(a.data(), snake_out_.data(), ch, t_cur);
  std::vector<float> y(static_cast<size_t>(t_cur));  // cout == 1
  conv1d_f32(conv2_, a.data(), t_cur, y.data());

  const int64_t samples = int64_t(frames) * cfg_.hop;
  if (static_cast<int64_t>(t_cur) != samples) {
    throw std::runtime_error("decoder produced " + std::to_string(t_cur) + " samples, expected " +
                             std::to_string(samples));
  }
  std::memcpy(audio, y.data(), sizeof(float) * static_cast<size_t>(samples));
}

}  // namespace ov

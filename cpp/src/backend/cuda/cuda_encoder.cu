// GPU audio-tokenizer encoder (voice clone): PCM -> HuBERT semantic tower ->
// semantic/acoustic conv stacks -> concat/fc -> 8x RVQ encode.
#include "backend/cuda/cuda_encoder.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "model/at_encoder.h"
#include "model/conv.h"
#include "model/mat.h"

namespace ov {
namespace cuda {
namespace {

#define OV_ECHECK(x)                                                                  \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess)                                                            \
      throw std::runtime_error(std::string("cuda: ") + cudaGetErrorString(e_));       \
  } while (0)

#define OV_ECUBLAS(x)                                                                 \
  do {                                                                                \
    cublasStatus_t s_ = (x);                                                          \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                                  \
      throw std::runtime_error("cublas error " + std::to_string(static_cast<int>(s_))); \
  } while (0)

class Arena {
 public:
  ~Arena() { cudaFree(base_); }
  void reserve(size_t bytes) {
    if (bytes <= cap_) return;
    cudaFree(base_);
    base_ = nullptr;
    OV_ECHECK(cudaMalloc(&base_, bytes));
    cap_ = bytes;
  }
  void reset() { used_ = 0; }
  template <typename T>
  T* get(size_t n) {
    const size_t off = (used_ + 255) & ~static_cast<size_t>(255);
    used_ = off + n * sizeof(T);
    if (used_ > cap_) throw std::runtime_error("encoder workspace overflow");
    return reinterpret_cast<T*>(static_cast<char*>(base_) + off);
  }
 private:
  void* base_ = nullptr;
  size_t cap_ = 0, used_ = 0;
};

struct EConv {
  int cin = 0, cout = 0, k = 0, stride = 1, pad = 0, dil = 1, groups = 1;
  const float* w = nullptr;
  const float* b = nullptr;
  int out_len(int t_in) const {
    return (t_in + 2 * pad - dil * (k - 1) - 1) / stride + 1;
  }
};

struct EMat {
  int K = 0, N = 0;
  const float* w = nullptr;
  const float* b = nullptr;
};

__global__ void conv1d_kernel(const float* __restrict__ x, const float* __restrict__ w,
                              const float* __restrict__ b, int t_in, int cin, int cout, int k,
                              int stride, int pad, int dil, int groups, int t_out,
                              float* __restrict__ y) {
  const int64_t n = static_cast<int64_t>(t_out) * cout;
  const int cig = cin / groups, cog = cout / groups;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int t = static_cast<int>(idx / cout);
    const int o = static_cast<int>(idx - static_cast<int64_t>(t) * cout);
    const int g = o / cog;
    const int i0 = g * cig;
    float acc = b ? b[o] : 0.f;
    const int base = t * stride - pad;
    for (int kk = 0; kk < k; ++kk) {
      const int src = base + kk * dil;
      if (src < 0 || src >= t_in) continue;
      const float* xr = x + static_cast<int64_t>(src) * cin + i0;
      const float* wr = w + static_cast<int64_t>(o) * cig * k + kk;
      for (int ii = 0; ii < cig; ++ii) acc += xr[ii] * wr[static_cast<int64_t>(ii) * k];
    }
    y[idx] = acc;
  }
}

__global__ void im2col_kernel(const float* __restrict__ x, int t_in, int cin, int cig, int k,
                              int stride, int pad, int dil, int t_out, int i0,
                              float* __restrict__ col) {
  const int CK = cig * k;
  const int64_t n = static_cast<int64_t>(t_out) * CK;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int t = static_cast<int>(idx / CK);
    const int rem = static_cast<int>(idx - static_cast<int64_t>(t) * CK);
    const int kk = rem / cig;
    const int ii = rem - kk * cig;
    const int src = t * stride - pad + kk * dil;
    col[idx] = (src >= 0 && src < t_in)
                   ? x[static_cast<int64_t>(src) * cin + i0 + ii]
                   : 0.f;
  }
}

__global__ void prepare_conv_kernel(const float* __restrict__ raw, float* __restrict__ prep,
                                    int groups, int k, int cig, int cog) {
  const int64_t n = static_cast<int64_t>(groups) * k * cig * cog;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t r = idx;
    const int o = static_cast<int>(r % cog);
    r /= cog;
    const int i = static_cast<int>(r % cig);
    r /= cig;
    const int kk = static_cast<int>(r % k);
    const int g = static_cast<int>(r / k);
    prep[idx] = raw[(static_cast<int64_t>(g) * cog + o) * cig * k + static_cast<int64_t>(i) * k + kk];
  }
}

__global__ void layer_norm_kernel(float* __restrict__ x, const float* __restrict__ w,
                                  const float* __restrict__ b, int rows, int cols, float eps) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  float* row = x + static_cast<int64_t>(r) * cols;
  double sum = 0.0;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) sum += row[i];
  __shared__ double red[256];
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const double mean = red[0] / cols;
  __syncthreads();
  double var = 0.0;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const double d = row[i] - mean;
    var += d * d;
  }
  red[threadIdx.x] = var;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = rsqrtf(static_cast<float>(red[0] / cols) + eps);
  const float m = static_cast<float>(mean);
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    row[i] = (row[i] - m) * inv * w[i] + (b ? b[i] : 0.f);
}

__global__ void group_norm_kernel(float* __restrict__ x, const float* __restrict__ w,
                                  const float* __restrict__ b, int channels, int t, float eps) {
  const int c = blockIdx.x;
  if (c >= channels) return;
  double sum = 0.0;
  for (int i = threadIdx.x; i < t; i += blockDim.x) sum += x[static_cast<int64_t>(i) * channels + c];
  __shared__ double red[256];
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const double mean = red[0] / t;
  __syncthreads();
  double var = 0.0;
  for (int i = threadIdx.x; i < t; i += blockDim.x) {
    const double d = x[static_cast<int64_t>(i) * channels + c] - mean;
    var += d * d;
  }
  red[threadIdx.x] = var;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = rsqrtf(static_cast<float>(red[0] / t) + eps);
  const float m = static_cast<float>(mean);
  const float sw = w ? w[c] : 1.f, sb = b ? b[c] : 0.f;
  for (int i = threadIdx.x; i < t; i += blockDim.x) {
    float& v = x[static_cast<int64_t>(i) * channels + c];
    v = (v - m) * inv * sw + sb;
  }
}

__global__ void gelu_kernel(float* __restrict__ x, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) x[i] = 0.5f * x[i] * (1.f + erff(x[i] * 0.7071067811865476f));
}

__global__ void elu_kernel(float* __restrict__ x, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) x[i] = (x[i] > 0.f) ? x[i] : (expf(x[i]) - 1.f);
}

__global__ void snake_kernel(float* __restrict__ x, const float* __restrict__ alpha, int t,
                             int c) {
  const int64_t n = static_cast<int64_t>(t) * c;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int ch = static_cast<int>(i % c);
    const float a = alpha[ch] + 1e-9f;
    const float s = sinf(a * x[i]);
    x[i] += (1.f / a) * s * s;
  }
}

__global__ void add_bias_kernel(float* __restrict__ x, const float* __restrict__ b, int rows,
                                int cols) {
  const int64_t n = static_cast<int64_t>(rows) * cols;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) x[i] += b[i % cols];
}

__global__ void add_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

__global__ void copy_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

__global__ void concat_kernel(const float* __restrict__ ea, const float* __restrict__ es, int T,
                              float* __restrict__ emb) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = static_cast<int64_t>(T) * 1024;
  if (i < n) {
    const int t = static_cast<int>(i / 1024), c = static_cast<int>(i % 1024);
    emb[i] = c < 256 ? ea[static_cast<int64_t>(t) * 256 + c]
                     : es[static_cast<int64_t>(t) * 768 + (c - 256)];
  }
}

__global__ void softmax_rows_kernel(float* __restrict__ s, int rows, int cols) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  float* row = s + static_cast<int64_t>(r) * cols;
  float mx = -INFINITY;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) mx = fmaxf(mx, row[i]);
  __shared__ float red[256];
  red[threadIdx.x] = mx;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + st]);
    __syncthreads();
  }
  mx = red[0];
  __syncthreads();
  float sum = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float e = expf(row[i] - mx);
    row[i] = e;
    sum += e;
  }
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
    __syncthreads();
  }
  const float inv = 1.f / red[0];
  for (int i = threadIdx.x; i < cols; i += blockDim.x) row[i] *= inv;
}

__global__ void resample_kernel(const float* __restrict__ wav, int n, const float* __restrict__ k,
                                int width, int taps, int orig, int neu, int t_out,
                                float* __restrict__ out) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(t_out) * neu) return;
  const int t = static_cast<int>(idx / neu);
  const int p = static_cast<int>(idx - static_cast<int64_t>(t) * neu);
  const float* kp = k + static_cast<int64_t>(p) * taps;
  float acc = 0.f;
  for (int i = 0; i < taps; ++i) {
    const int src = t * orig + i - width;
    if (src >= 0 && src < n) acc += wav[src] * kp[i];
  }
  out[idx] = acc;
}

__global__ void rvq_argmin_kernel(const float* __restrict__ x, int T, const float* __restrict__ e,
                                  int V, int D, int64_t* __restrict__ codes) {
  const int t = blockIdx.x;
  if (t >= T) return;
  const float* xr = x + static_cast<int64_t>(t) * D;
  float best = -INFINITY;
  int bc = 0;
  for (int c = threadIdx.x; c < V; c += blockDim.x) {
    const float* er = e + static_cast<int64_t>(c) * D;
    float dot = 0.f, n2 = 0.f;
    for (int d = 0; d < D; ++d) {
      dot += xr[d] * er[d];
      n2 += er[d] * er[d];
    }
    const float score = 2.f * dot - n2;
    if (score > best) {
      best = score;
      bc = c;
    }
  }
  __shared__ float sb[256];
  __shared__ int sc[256];
  sb[threadIdx.x] = best;
  sc[threadIdx.x] = bc;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st && sb[threadIdx.x + st] > sb[threadIdx.x]) {
      sb[threadIdx.x] = sb[threadIdx.x + st];
      sc[threadIdx.x] = sc[threadIdx.x + st];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) codes[t] = sc[0];
}

__global__ void gather_code_kernel(const float* __restrict__ e, const int64_t* __restrict__ codes,
                                   int T, int V, int D, float* __restrict__ out) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(T) * D) return;
  const int t = static_cast<int>(idx / D);
  const int d = static_cast<int>(idx % D);
  const int64_t c = codes[t];
  out[idx] = e[c * D + d];
}

__global__ void mean_down_kernel(const float* __restrict__ hs, int frames, int H, int stride,
                                 int states, float* __restrict__ out) {
  const int64_t n = static_cast<int64_t>(frames) * H;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int f = static_cast<int>(i / H), c = static_cast<int>(i % H);
  out[i] = hs[static_cast<int64_t>(f * stride) * H + c] / static_cast<float>(states);
}

__global__ void sub_kernel(float* __restrict__ r, const float* __restrict__ q, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) r[i] -= q[i];
}

inline int blocks_for(int64_t n) { return static_cast<int>((n + 255) / 256); }

}  // namespace

namespace {
struct EStageProf {
  bool on = false;
  cudaStream_t st = nullptr;
  std::vector<cudaEvent_t> ev;
  std::vector<const char*> lab;
  void mark(const char* l) {
    if (!on) return;
    cudaEvent_t e; cudaEventCreate(&e); cudaEventRecord(e, st);
    ev.push_back(e); lab.push_back(l);
  }
  void finish() {
    if (!on || ev.empty()) return;
    cudaEventSynchronize(ev.back());
    std::map<std::string, double> sum;
    for (size_t i = 1; i < ev.size(); ++i) {
      float ms = 0; cudaEventElapsedTime(&ms, ev[i - 1], ev[i]);
      sum[lab[i]] += ms;
    }
    std::fprintf(stderr, "[enc-stage]");
    for (auto& kv : sum) std::fprintf(stderr, " %s=%.2f", kv.first.c_str(), kv.second);
    std::fprintf(stderr, "\n");
    for (auto e : ev) cudaEventDestroy(e);
    ev.clear(); lab.clear();
  }
};
}  // namespace

class EncoderRunner {
 public:
  explicit EncoderRunner(const AudioEncoder& e) : cfg_(e.cfg_) {
    OV_ECHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    OV_ECUBLAS(cublasCreate(&blas_));
    OV_ECUBLAS(cublasSetStream(blas_, stream_));

    stem_.resize(e.stem_.size());
    for (size_t i = 0; i < e.stem_.size(); ++i) {
      const auto& s = e.stem_[i];
      stem_[i] = conv(s.w, s.b, s.cout, s.cin, s.k, 1, s.stride, 0, 1);
    }
    stem_gn_w_ = vec(e.stem_gn_w_, cfg_.feat_dim);
    stem_gn_b_ = vec(e.stem_gn_b_, cfg_.feat_dim);
    proj_ln_w_ = vec(e.proj_ln_w_, cfg_.feat_dim);
    proj_ln_b_ = vec(e.proj_ln_b_, cfg_.feat_dim);
    proj_ = mat(e.proj_);
    proj_.b = vec(e.proj_b_, cfg_.sem_hidden);
    pos_conv_ = conv(e.pos_conv_weight_.data(), e.pos_conv_b_, cfg_.sem_hidden, cfg_.sem_hidden,
                     cfg_.pos_conv_kernel, cfg_.pos_conv_groups, 1, cfg_.pos_conv_kernel / 2, 1);
    enc_ln_w_ = vec(e.enc_ln_w_, cfg_.sem_hidden);
    enc_ln_b_ = vec(e.enc_ln_b_, cfg_.sem_hidden);

    layers_.resize(e.layers_.size());
    for (size_t i = 0; i < e.layers_.size(); ++i) {
      const auto& s = e.layers_[i];
      auto& d = layers_[i];
      d.q = mat(s.q); d.k = mat(s.k); d.v = mat(s.v); d.o = mat(s.o);
      d.qb = vec(s.qb, cfg_.sem_hidden); d.kb = vec(s.kb, cfg_.sem_hidden);
      d.vb = vec(s.vb, cfg_.sem_hidden); d.ob = vec(s.ob, cfg_.sem_hidden);
      d.f1 = mat(s.f1); d.f2 = mat(s.f2);
      d.f1b = vec(s.f1b, cfg_.sem_inter); d.f2b = vec(s.f2b, cfg_.sem_hidden);
      d.ln1w = vec(s.ln1w, cfg_.sem_hidden); d.ln1b = vec(s.ln1b, cfg_.sem_hidden);
      d.ln2w = vec(s.ln2w, cfg_.sem_hidden); d.ln2b = vec(s.ln2b, cfg_.sem_hidden);
    }

    sem_pre_ = conv_spec(e.sem_pre_);
    sem_blocks_.resize(e.sem_blocks_.size());
    for (size_t i = 0; i < e.sem_blocks_.size(); ++i) {
      const auto& s = e.sem_blocks_[i];
      auto& d = sem_blocks_[i];
      for (int u = 0; u < 2; ++u) {
        d.ru[u].c1 = conv_spec(s.units[u].c1);
        d.ru[u].c2 = conv_spec(s.units[u].c2);
      }
      d.conv = conv_spec(s.conv);
    }

    ac_conv1_ = conv_spec(e.ac_conv1_);
    ac_blocks_.resize(e.ac_blocks_.size());
    for (size_t i = 0; i < e.ac_blocks_.size(); ++i) {
      const auto& s = e.ac_blocks_[i];
      auto& d = ac_blocks_[i];
      d.snake = vec(s.snake, s.units[0].c1.cout);
      for (int u = 0; u < 3; ++u) {
        d.ru[u].snake1 = vec(s.units[u].sa1, s.units[u].c1.cout);
        d.ru[u].snake2 = vec(s.units[u].sa2, s.units[u].c1.cout);
        d.ru[u].c1 = conv_spec(s.units[u].c1);
        d.ru[u].c2 = conv_spec(s.units[u].c2);
      }
      d.conv = conv_spec(s.conv);
    }
    ac_snake_ = vec(e.ac_snake_, e.ac_conv2_.cin);
    ac_conv2_ = conv_spec(e.ac_conv2_);
    fc_ = mat(e.fc_);
    fc_.b = vec(e.fc_b_, 1024);

    vq_.resize(e.vq_.size());
    for (size_t i = 0; i < e.vq_.size(); ++i) {
      const auto& s = e.vq_[i];
      auto& d = vq_[i];
      d.pin = mat(s.project_in);
      d.pin.b = vec(s.pin_b, 64);
      d.embed = vec(s.embed, int64_t(1024) * 64);
      d.pout = mat(s.project_out);
      d.pout.b = vec(s.pout_b, 1024);
    }
  }

  ~EncoderRunner() {
    for (void* p : owned_) cudaFree(p);
    if (blas_) cublasDestroy(blas_);
    if (stream_) cudaStreamDestroy(stream_);
  }

  void encode(const float* wav24, int n_samples, std::vector<int64_t>& codes, int& frames);

 private:
  const float* vec(const float* src, int64_t n) {
    if (!src || n <= 0) return nullptr;
    float* p = nullptr;
    OV_ECHECK(cudaMalloc(&p, sizeof(float) * n));
    OV_ECHECK(cudaMemcpyAsync(p, src, sizeof(float) * n, cudaMemcpyHostToDevice, stream_));
    OV_ECHECK(cudaGetLastError());
    owned_.push_back(p);
    return p;
  }

  EMat mat(const Mat& m) {
    if (!m.w) throw std::runtime_error("cuda encoder: expected fp32 Mat weights");
    EMat r;
    r.K = static_cast<int>(m.K);
    r.N = static_cast<int>(m.N);
    r.w = vec(m.w, m.K * m.N);
    return r;
  }

  EConv conv(const float* w, const float* b, int cout, int cin, int k, int groups, int stride,
             int pad, int dil) {
    EConv r;
    r.cin = cin; r.cout = cout; r.k = k; r.groups = groups; r.stride = stride; r.pad = pad;
    r.dil = dil;
    const int cig = cin / groups, cog = cout / groups;
    const int64_t n = static_cast<int64_t>(groups) * k * cig * cog;
    float* raw = nullptr;
    float* prep = nullptr;
    OV_ECHECK(cudaMalloc(&raw, sizeof(float) * n));
    OV_ECHECK(cudaMemcpyAsync(raw, w, sizeof(float) * n, cudaMemcpyHostToDevice, stream_));
    OV_ECHECK(cudaMalloc(&prep, sizeof(float) * n));
    prepare_conv_kernel<<<blocks_for(n), 256, 0, stream_>>>(raw, prep, groups, k, cig, cog);
    OV_ECHECK(cudaGetLastError());
    OV_ECHECK(cudaFree(raw));
    owned_.push_back(prep);
    r.w = prep;
    r.b = b ? vec(b, cout) : nullptr;
    return r;
  }

  EConv conv_spec(const Conv1dSpec& c) {
    return conv(c.w, c.b, c.cout, c.cin, c.k, c.groups, c.stride, c.pad, c.dil);
  }

  void gemm(int M, int N, int K, const float* A, const float* W, float* C) {
    const float alpha = 1.f, beta = 0.f;
    OV_ECUBLAS(cublasSgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, W, N, A, K, &beta,
                           C, N));
  }

  void gemm_ld(int M, int N, int K, const float* A, const float* W, float* C, int ldc) {
    const float alpha = 1.f, beta = 0.f;
    OV_ECUBLAS(cublasSgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, W, N, A, K, &beta,
                           C, ldc));
  }

  void linear(const EMat& m, int M, const float* A, float* C) {
    gemm(M, m.N, m.K, A, m.w, C);
    if (m.b) {
      add_bias_kernel<<<blocks_for(static_cast<int64_t>(M) * m.N), 256, 0, stream_>>>(C, m.b, M,
                                                                                      m.N);
      OV_ECHECK(cudaGetLastError());
    }
  }

  void conv_run(const EConv& c, const float* x, int t_in, float* y) {
    const int t_out = c.out_len(t_in);
    const int cig = c.cin / c.groups, cog = c.cout / c.groups;
    const int CK = cig * c.k;
    for (int g = 0; g < c.groups; ++g) {
      const int64_t n = static_cast<int64_t>(t_out) * CK;
      im2col_kernel<<<blocks_for(n), 256, 0, stream_>>>(x, t_in, c.cin, cig, c.k, c.stride, c.pad,
                                                        c.dil, t_out, g * cig, col_);
      OV_ECHECK(cudaGetLastError());
      gemm_ld(t_out, cog, CK, col_, c.w + static_cast<int64_t>(g) * CK * cog,
              y + static_cast<int64_t>(g) * cog, c.cout);
    }
    if (c.b) {
      add_bias_kernel<<<blocks_for(static_cast<int64_t>(t_out) * c.cout), 256, 0, stream_>>>(
          y, c.b, t_out, c.cout);
      OV_ECHECK(cudaGetLastError());
    }
  }

  void gelu(float* x, int64_t n) {
    gelu_kernel<<<blocks_for(n), 256, 0, stream_>>>(x, n);
    OV_ECHECK(cudaGetLastError());
  }
  void elu(float* x, int64_t n) {
    elu_kernel<<<blocks_for(n), 256, 0, stream_>>>(x, n);
    OV_ECHECK(cudaGetLastError());
  }
  void snake(float* x, const float* a, int t, int c) {
    snake_kernel<<<blocks_for(static_cast<int64_t>(t) * c), 256, 0, stream_>>>(x, a, t, c);
    OV_ECHECK(cudaGetLastError());
  }

  ATEncoderConfig cfg_;
  cudaStream_t stream_ = nullptr;
  cublasHandle_t blas_ = nullptr;
  std::vector<void*> owned_;
  Arena arena_;
  float* col_ = nullptr;

  struct ELayer {
    EMat q, k, v, o, f1, f2;
    const float *qb = nullptr, *kb = nullptr, *vb = nullptr, *ob = nullptr;
    const float *f1b = nullptr, *f2b = nullptr;
    const float *ln1w = nullptr, *ln1b = nullptr, *ln2w = nullptr, *ln2b = nullptr;
  };
  struct ERes { EConv c1, c2; const float *snake1 = nullptr, *snake2 = nullptr; };
  struct ESem { ERes ru[2]; EConv conv; };
  struct EAc { ERes ru[3]; const float* snake = nullptr; EConv conv; };
  struct EVq { EMat pin, pout; const float* embed = nullptr; };

  std::vector<EConv> stem_;
  const float *stem_gn_w_ = nullptr, *stem_gn_b_ = nullptr;
  const float *proj_ln_w_ = nullptr, *proj_ln_b_ = nullptr;
  EMat proj_;
  EConv pos_conv_;
  const float *enc_ln_w_ = nullptr, *enc_ln_b_ = nullptr;
  std::vector<ELayer> layers_;
  EConv sem_pre_;
  std::vector<ESem> sem_blocks_;
  EConv ac_conv1_;
  std::vector<EAc> ac_blocks_;
  const float* ac_snake_ = nullptr;
  EConv ac_conv2_;
  EMat fc_;
  std::vector<EVq> vq_;
};

void EncoderRunner::encode(const float* wav24, int n_samples, std::vector<int64_t>& codes,
                           int& frames) {
  // same resampler coefficients as the CPU implementation
  constexpr int orig = 3, neu = 2;
  constexpr int kWidth = 6;
  constexpr double kRolloff = 0.99;
  const double base_freq = static_cast<double>(std::min(orig, neu)) * kRolloff;
  const int width = static_cast<int>(std::ceil(kWidth * orig / base_freq));
  const int taps = 2 * width + orig;
  const double scale = base_freq / static_cast<double>(orig);
  std::vector<float> kern(static_cast<size_t>(neu) * taps, 0.f);
  for (int phase = 0; phase < neu; ++phase)
    for (int tap = 0; tap < taps; ++tap) {
      double t = (static_cast<double>(tap - width) / orig - static_cast<double>(phase) / neu) *
                 base_freq;
      t = std::max(-static_cast<double>(kWidth), std::min(static_cast<double>(kWidth), t));
      const double win = std::cos(t * M_PI / kWidth / 2.0);
      const double sinc = (t == 0.0) ? 1.0 : std::sin(t * M_PI) / (t * M_PI);
      kern[static_cast<size_t>(phase) * taps + tap] =
          static_cast<float>(sinc * win * win * scale);
    }
  const float* d_kern = vec(kern.data(), static_cast<int64_t>(kern.size()));
  const int n16 = static_cast<int>(std::ceil(2.0 * n_samples / 3.0));
  const int64_t max_tc = static_cast<int64_t>(n16) * 128;
  const int64_t Tmax = std::max<int64_t>(n16 / 320 + 4, 8);
  arena_.reserve(sizeof(float) * (6 * max_tc + static_cast<int64_t>(n_samples) * 448 +
                                 4 * Tmax * 1024 + Tmax * Tmax + 1024 * 1024) +
                 sizeof(int64_t) * 8 * Tmax);
  arena_.reset();

  float* d_wav = arena_.get<float>(n_samples);
  float* d_pad = arena_.get<float>(static_cast<int64_t>(n16) + 2 * width + orig + 320);
  float* d_w16 = arena_.get<float>(static_cast<int64_t>(n16) + 320);
  float* d_acpad = arena_.get<float>(n_samples + 2 * cfg_.acoustic_pad);
  col_ = arena_.get<float>(static_cast<int64_t>(n_samples) * 64 * 7);
  float* a = arena_.get<float>(max_tc);
  float* b = arena_.get<float>(max_tc);
  float* cbuf = arena_.get<float>(max_tc);
  float* x0 = arena_.get<float>(max_tc);
  float* h = arena_.get<float>(Tmax * cfg_.sem_hidden);
  float* proj = arena_.get<float>(Tmax * cfg_.sem_hidden);
  float* q = arena_.get<float>(Tmax * cfg_.sem_hidden);
  float* k = arena_.get<float>(Tmax * cfg_.sem_hidden);
  float* v = arena_.get<float>(Tmax * cfg_.sem_hidden);
  float* attn = arena_.get<float>(Tmax * cfg_.sem_hidden);
  float* ffn = arena_.get<float>(Tmax * cfg_.sem_inter);
  float* hs = arena_.get<float>(Tmax * cfg_.sem_hidden);
  float* scores = arena_.get<float>(Tmax * Tmax);
  float* sem = arena_.get<float>(Tmax * cfg_.sem_hidden);
  float* es = arena_.get<float>(Tmax * cfg_.sem_hidden);
  float* emb = arena_.get<float>(Tmax * 1024);
  float* z = arena_.get<float>(Tmax * 1024);
  float* r64 = arena_.get<float>(Tmax * 64);
  float* e64 = arena_.get<float>(Tmax * 64);
  float* qq = arena_.get<float>(Tmax * 1024);
  int64_t* d_codes = arena_.get<int64_t>(8 * Tmax);

  EStageProf sp; sp.on = std::getenv("OV_ENC_PROFILE") != nullptr; sp.st = stream_; sp.mark("start");
  OV_ECHECK(cudaMemcpyAsync(d_wav, wav24, sizeof(float) * n_samples, cudaMemcpyHostToDevice,
                            stream_));
  OV_ECHECK(cudaMemsetAsync(d_pad, 0, sizeof(float) * (n16 + 2 * width + orig + 320), stream_));
  const int rt_out = n16 / neu;
  resample_kernel<<<blocks_for(n16), 256, 0, stream_>>>(
      d_wav, n_samples, d_kern, width, taps, orig, neu, rt_out, d_w16);
  OV_ECHECK(cudaGetLastError());
  // HuBERT input is zero-padded by 160 on both sides.
  OV_ECHECK(cudaMemcpyAsync(d_pad + 160, d_w16, sizeof(float) * n16, cudaMemcpyDeviceToDevice,
                            stream_));
  
  sp.mark("resample");
  int t = n16 + 320;
  float* cur = d_pad;
  float* stem0 = a, *stem1 = cbuf;
  float* out = stem0;
  for (size_t i = 0; i < stem_.size(); ++i) {
    const EConv& sc = stem_[i];
    const int t_out = sc.out_len(t);
    conv_run(sc, cur, t, out);
    if (i == 0) {
      group_norm_kernel<<<cfg_.feat_dim, 256, 0, stream_>>>(out, stem_gn_w_, stem_gn_b_,
                                                            cfg_.feat_dim, t_out,
                                                            cfg_.layer_norm_eps);
      OV_ECHECK(cudaGetLastError());
    }
    gelu(out, static_cast<int64_t>(t_out) * sc.cout);
    cur = out;
    out = (out == stem0) ? stem1 : stem0;
    t = t_out;
  }

  
  sp.mark("stem");
  layer_norm_kernel<<<t, 256, 0, stream_>>>(cur, proj_ln_w_, proj_ln_b_, t, cfg_.feat_dim,
                                            cfg_.layer_norm_eps);
  OV_ECHECK(cudaGetLastError());
  linear(proj_, t, cur, proj);
  conv_run(pos_conv_, proj, t, b);  // t_out = t+1, dropped below
  gelu(b, static_cast<int64_t>(t + 1) * cfg_.sem_hidden);
  for (int i = 0; i < t; ++i) {
    add_kernel<<<blocks_for(cfg_.sem_hidden), 256, 0, stream_>>>(
        proj + static_cast<int64_t>(i) * cfg_.sem_hidden,
        b + static_cast<int64_t>(i) * cfg_.sem_hidden, cfg_.sem_hidden);
  }
  OV_ECHECK(cudaGetLastError());
  layer_norm_kernel<<<t, 256, 0, stream_>>>(proj, enc_ln_w_, enc_ln_b_, t, cfg_.sem_hidden,
                                            cfg_.layer_norm_eps);
  OV_ECHECK(cudaGetLastError());
  
  OV_ECHECK(cudaMemcpyAsync(h, proj, sizeof(float) * static_cast<size_t>(t) * cfg_.sem_hidden,
                            cudaMemcpyDeviceToDevice, stream_));
  OV_ECHECK(cudaMemsetAsync(hs, 0, sizeof(float) * static_cast<size_t>(t) * cfg_.sem_hidden,
                            stream_));
  const int heads = cfg_.sem_heads, hd = cfg_.sem_hidden / heads;
  const float attn_scale = 1.f / std::sqrt(static_cast<float>(hd));
  const float one = 1.f, zero = 0.f;
  int n_states = 0;
  auto accumulate = [&]() {
    add_kernel<<<blocks_for(static_cast<int64_t>(t) * cfg_.sem_hidden), 256, 0, stream_>>>(
        hs, h, static_cast<int64_t>(t) * cfg_.sem_hidden);
    OV_ECHECK(cudaGetLastError());
    ++n_states;
  };

  sp.mark("proj+pos+ln");
  for (size_t li = 0; li < layers_.size(); ++li) {
    const ELayer& L = layers_[li];
    accumulate();
    linear(L.q, t, h, q);
    linear(L.k, t, h, k);
    linear(L.v, t, h, v);
    add_bias_kernel<<<blocks_for(static_cast<int64_t>(t) * cfg_.sem_hidden), 256, 0, stream_>>>(
        q, L.qb, t, cfg_.sem_hidden);
    add_bias_kernel<<<blocks_for(static_cast<int64_t>(t) * cfg_.sem_hidden), 256, 0, stream_>>>(
        k, L.kb, t, cfg_.sem_hidden);
    add_bias_kernel<<<blocks_for(static_cast<int64_t>(t) * cfg_.sem_hidden), 256, 0, stream_>>>(
        v, L.vb, t, cfg_.sem_hidden);
    OV_ECHECK(cudaGetLastError());
    for (int hh = 0; hh < heads; ++hh) {
      OV_ECUBLAS(cublasSgemm(blas_, CUBLAS_OP_T, CUBLAS_OP_N, t, t, hd, &attn_scale,
                             k + hh * hd, cfg_.sem_hidden, q + hh * hd, cfg_.sem_hidden, &zero,
                             scores, t));
      softmax_rows_kernel<<<t, 256, 0, stream_>>>(scores, t, t);
      OV_ECHECK(cudaGetLastError());
      OV_ECUBLAS(cublasSgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, hd, t, t, &one, v + hh * hd,
                             cfg_.sem_hidden, scores, t, &zero, attn + hh * hd,
                             cfg_.sem_hidden));
    }
    linear(L.o, t, attn, proj);
    add_bias_kernel<<<blocks_for(static_cast<int64_t>(t) * cfg_.sem_hidden), 256, 0, stream_>>>(
        proj, L.ob, t, cfg_.sem_hidden);
    add_kernel<<<blocks_for(static_cast<int64_t>(t) * cfg_.sem_hidden), 256, 0, stream_>>>(
        h, proj, static_cast<int64_t>(t) * cfg_.sem_hidden);
    layer_norm_kernel<<<t, 256, 0, stream_>>>(h, L.ln1w, L.ln1b, t, cfg_.sem_hidden,
                                              cfg_.layer_norm_eps);
    OV_ECHECK(cudaGetLastError());

    linear(L.f1, t, h, ffn);
    add_bias_kernel<<<blocks_for(static_cast<int64_t>(t) * cfg_.sem_inter), 256, 0, stream_>>>(
        ffn, L.f1b, t, cfg_.sem_inter);
    gelu(ffn, static_cast<int64_t>(t) * cfg_.sem_inter);
    linear(L.f2, t, ffn, proj);
    add_bias_kernel<<<blocks_for(static_cast<int64_t>(t) * cfg_.sem_hidden), 256, 0, stream_>>>(
        proj, L.f2b, t, cfg_.sem_hidden);
    add_kernel<<<blocks_for(static_cast<int64_t>(t) * cfg_.sem_hidden), 256, 0, stream_>>>(
        h, proj, static_cast<int64_t>(t) * cfg_.sem_hidden);
    layer_norm_kernel<<<t, 256, 0, stream_>>>(h, L.ln2w, L.ln2b, t, cfg_.sem_hidden,
                                              cfg_.layer_norm_eps);
    OV_ECHECK(cudaGetLastError());
  }
  accumulate();
  sp.mark("hubert");
  frames = t / cfg_.semantic_downsample;
  mean_down_kernel<<<blocks_for(static_cast<int64_t>(frames) * cfg_.sem_hidden), 256, 0,
                     stream_>>>(hs, frames, cfg_.sem_hidden, cfg_.semantic_downsample, n_states,
                                sem);
  OV_ECHECK(cudaGetLastError());

  sp.mark("mean");
  // semantic conv stack (768 -> 768, ELU residual units)
  OV_ECHECK(cudaMemcpyAsync(es, sem, sizeof(float) * static_cast<size_t>(frames) * cfg_.sem_hidden,
                            cudaMemcpyDeviceToDevice, stream_));
  conv_run(sem_pre_, es, frames, b);
  std::swap(es, b);
  for (size_t bi = 0; bi < sem_blocks_.size(); ++bi) {
    const ESem& blk = sem_blocks_[bi];
    for (int u = 0; u < 2; ++u) {
      const ERes& ru = blk.ru[u];
      const int64_t n = static_cast<int64_t>(frames) * cfg_.sem_hidden;
      copy_kernel<<<blocks_for(n), 256, 0, stream_>>>(x0, es, n);
      elu(es, n);
      conv_run(ru.c1, es, frames, b);
      elu(b, n);
      conv_run(ru.c2, b, frames, es);
      add_kernel<<<blocks_for(n), 256, 0, stream_>>>(es, x0, n);
      OV_ECHECK(cudaGetLastError());
    }
    conv_run(blk.conv, es, frames, b);
    std::swap(es, b);
  }

  

  sp.mark("semantic");
  // acoustic encoder: pad only when the unpadded length disagrees with T
  auto enc_frames = [&](int len) {
    int tt = len;
    const int strides[5] = {8, 5, 4, 2, 3};
    for (int i = 0; i < 5; ++i) {
      const int st = strides[i], kk = 2 * st, pad = (st + 1) / 2;
      tt = (tt + 2 * pad - kk) / st + 1;
    }
    return tt;
  };
  const bool pad_ac = (enc_frames(n_samples) != frames);
  const float* ac_src = d_wav;
  int ac_t = n_samples;
  if (pad_ac) {
    const int pad = cfg_.acoustic_pad;
    OV_ECHECK(cudaMemsetAsync(d_acpad, 0, sizeof(float) * (n_samples + 2 * pad), stream_));
    OV_ECHECK(cudaMemcpyAsync(d_acpad + pad, d_wav, sizeof(float) * n_samples,
                              cudaMemcpyDeviceToDevice, stream_));
    ac_src = d_acpad;
    ac_t = n_samples + 2 * pad;
  }
  conv_run(ac_conv1_, ac_src, ac_t, a);
  int at = ac_t;
  for (size_t bi = 0; bi < ac_blocks_.size(); ++bi) {
    const EAc& blk = ac_blocks_[bi];
    const int rc = blk.ru[0].c1.cout;
    for (int u = 0; u < 3; ++u) {
      const ERes& ru = blk.ru[u];
      const int64_t n = static_cast<int64_t>(at) * rc;
      copy_kernel<<<blocks_for(n), 256, 0, stream_>>>(x0, a, n);
      snake(a, ru.snake1, at, rc);
      conv_run(ru.c1, a, at, b);
      snake(b, ru.snake2, at, rc);
      conv_run(ru.c2, b, at, a);
      add_kernel<<<blocks_for(n), 256, 0, stream_>>>(a, x0, n);
      OV_ECHECK(cudaGetLastError());
    }
    snake(a, blk.snake, at, rc);
    conv_run(blk.conv, a, at, b);
    std::swap(a, b);
    at = blk.conv.out_len(at);
  }
  snake(a, ac_snake_, at, ac_conv2_.cin);
  conv_run(ac_conv2_, a, at, b);
  if (at != frames) throw std::runtime_error("cuda encoder: frame mismatch");

  
  sp.mark("acoustic");
  concat_kernel<<<blocks_for(static_cast<int64_t>(frames) * 1024), 256, 0, stream_>>>(
      b, es, frames, emb);
  OV_ECHECK(cudaGetLastError());
  linear(fc_, frames, emb, z);
  

  sp.mark("fc");
  for (int qi = 0; qi < 8; ++qi) {
    linear(vq_[qi].pin, frames, z, r64);
    rvq_argmin_kernel<<<frames, 256, 0, stream_>>>(r64, frames, vq_[qi].embed, 1024,
                                                 64, d_codes + qi * frames);
    gather_code_kernel<<<blocks_for(static_cast<int64_t>(frames) * 64), 256, 0,
                         stream_>>>(vq_[qi].embed, d_codes + qi * frames, frames,
                                    1024, 64, e64);
    linear(vq_[qi].pout, frames, e64, qq);
    sub_kernel<<<blocks_for(static_cast<int64_t>(frames) * 1024), 256, 0, stream_>>>(
        z, qq, static_cast<int64_t>(frames) * 1024);
    OV_ECHECK(cudaGetLastError());
  }
  codes.resize(static_cast<size_t>(8) * frames);
  sp.mark("rvq");
  OV_ECHECK(cudaMemcpyAsync(codes.data(), d_codes, sizeof(int64_t) * 8 * frames,
                            cudaMemcpyDeviceToHost, stream_));
  OV_ECHECK(cudaStreamSynchronize(stream_));
  sp.mark("d2h"); sp.finish();
}

EncoderRunnerPtr EncoderRunnerCreate(const AudioEncoder& encoder) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0)
    return {nullptr, &EncoderRunnerDelete};
  try {
    return {new EncoderRunner(encoder), &EncoderRunnerDelete};
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[cuda] encoder unavailable: %s\n", e.what());
    return {nullptr, &EncoderRunnerDelete};
  }
}

void EncoderRunnerDelete(EncoderRunner* r) { delete r; }

void EncoderRunnerEncodeThunk(void* ctx, const float* wav, int n_samples,
                              std::vector<int64_t>& codes, int& frames) {
  static_cast<EncoderRunner*>(ctx)->encode(wav, n_samples, codes, frames);
}

}  // namespace cuda
}  // namespace ov

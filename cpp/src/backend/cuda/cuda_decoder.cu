// GPU acoustic decoder.
//
// Mirrors src/model/audio_tokenizer.cpp: RVQ lookup + project_out, fc2, then
// conv1 / 5 x (Snake, ConvTranspose, 3 ResUnits) / Snake / conv2.  Everything is
// time-major [T, C], the same layout as the CPU tensors and the prepared
// Conv1d::wt [k, cin, cout] weights, so no transpose is needed at the boundary.
#include "backend/cuda/cuda_decoder.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/audio_tokenizer.h"
#include "model/conv.h"
#include "model/mat.h"

namespace ov {
namespace cuda {
namespace {

#define OV_DCHECK(x)                                                                  \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess)                                                            \
      throw std::runtime_error(std::string("cuda: ") + cudaGetErrorString(e_));       \
  } while (0)

#define OV_DCUBLAS(x)                                                                 \
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
    OV_DCHECK(cudaMalloc(&base_, bytes));
    cap_ = bytes;
  }
  void reset() { used_ = 0; }
  template <typename T>
  T* get(size_t n) {
    const size_t off = (used_ + 255) & ~static_cast<size_t>(255);
    used_ = off + n * sizeof(T);
    if (used_ > cap_) throw std::runtime_error("decoder workspace overflow");
    return reinterpret_cast<T*>(static_cast<char*>(base_) + off);
  }
 private:
  void* base_ = nullptr;
  size_t cap_ = 0, used_ = 0;
};

struct DConv {
  int cin = 0, cout = 0, k = 0, stride = 1, pad = 0, dil = 1, out_pad = 0;
  const float* w = nullptr;
  const float* b = nullptr;
  int out_len(int t_in) const { return (t_in + 2 * pad - dil * (k - 1) - 1) / stride + 1; }
  int convt_out_len(int t_in) const {
    return (t_in - 1) * stride - 2 * pad + k + out_pad;
  }
};

struct DMat {
  int K = 0, N = 0;
  const float* w = nullptr;
  const float* b = nullptr;
};

__global__ void gather_embed_kernel(float* __restrict__ e, const float* __restrict__ embed,
                                    const int64_t* __restrict__ codes, int T, int D, int q) {
  const int64_t n = static_cast<int64_t>(T) * D;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int t = static_cast<int>(i / D);
    const int d = static_cast<int>(i - static_cast<int64_t>(t) * D);
    const int64_t id = codes[static_cast<int64_t>(q) * T + t];
    e[i] = embed[id * D + d];
  }
}

__global__ void accum_project_kernel(float* __restrict__ q, const float* __restrict__ tmp,
                                     const float* __restrict__ bias, int64_t n, int H) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) q[i] += tmp[i] + bias[i % H];
}

__global__ void add_bias_kernel(float* __restrict__ y, const float* __restrict__ b, int t,
                                int c) {
  const int64_t n = static_cast<int64_t>(t) * c;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    y[i] += b[i % c];
}

__global__ void conv1d_direct_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                     const float* __restrict__ b, int t_in, int cin, int cout,
                                     int k, int stride, int pad, int dil, int t_out,
                                     float* __restrict__ y) {
  const int64_t n = static_cast<int64_t>(t_out) * cout;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int t = static_cast<int>(idx / cout);
    const int o = static_cast<int>(idx - static_cast<int64_t>(t) * cout);
    float acc = b ? b[o] : 0.f;
    const int base = t * stride - pad;
    for (int kk = 0; kk < k; ++kk) {
      const int src = base + kk * dil;
      if (src < 0 || src >= t_in) continue;
      const float* xr = x + static_cast<int64_t>(src) * cin;
      const float* wr = w + static_cast<int64_t>(kk) * cin * cout;
      for (int i = 0; i < cin; ++i) acc += xr[i] * wr[static_cast<int64_t>(i) * cout + o];
    }
    y[idx] = acc;
  }
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

__global__ void add_inplace_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                   int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

__global__ void copy_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

inline int blocks_for(int64_t n) {
  return static_cast<int>((n + 255) / 256);
}

}  // namespace

class DecoderRunner {
 public:
  explicit DecoderRunner(const AcousticDecoder& d) : cfg_(d.cfg_) {
    OV_DCHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    OV_DCUBLAS(cublasCreate(&blas_));
    OV_DCUBLAS(cublasSetStream(blas_, stream_));

    rvq_.resize(d.rvq_.size());
    for (size_t q = 0; q < d.rvq_.size(); ++q) {
      const auto& src = d.rvq_[q];
      auto& dst = rvq_[q];
      dst.project = upload_mat(src.project_out);
      dst.bias = upload(src.project_out_bias, cfg_.hidden);
      dst.embed = upload(src.embed, static_cast<int64_t>(cfg_.codebook_size) * cfg_.codebook_dim);
    }
    fc2_ = upload_mat(d.fc2_);
    fc2_.b = upload(d.fc2_bias_, cfg_.dec_in);
    conv1_ = upload_conv(d.conv1_);
    conv2_ = upload_conv(d.conv2_);
    snake_out_ = upload(d.snake_out_.data(), static_cast<int64_t>(d.snake_out_.size()));

    blocks_.resize(d.blocks_.size());
    for (size_t bi = 0; bi < d.blocks_.size(); ++bi) {
      const auto& sb = d.blocks_[bi];
      auto& db = blocks_[bi];
      db.snake = upload(sb.snake.data(), static_cast<int64_t>(sb.snake.size()));
      db.convt = upload_convt(sb.convt);
      for (int j = 0; j < 3; ++j) {
        const auto& su = sb.ru[j];
        auto& du = db.ru[j];
        du.snake1 = upload(su.snake1.data(), static_cast<int64_t>(su.snake1.size()));
        du.snake2 = upload(su.snake2.data(), static_cast<int64_t>(su.snake2.size()));
        du.conv1 = upload_conv(su.conv1);
        du.conv2 = upload_conv(su.conv2);
      }
    }
  }

  ~DecoderRunner() {
    for (void* p : owned_) cudaFree(p);
    if (blas_) cublasDestroy(blas_);
    if (stream_) cudaStreamDestroy(stream_);
  }

  void decode(const int64_t* codes, int frames, float* audio);

 private:
  const float* upload(const float* src, int64_t n) {
    if (!src || n <= 0) return nullptr;
    float* p = nullptr;
    OV_DCHECK(cudaMalloc(&p, sizeof(float) * n));
    OV_DCHECK(cudaMemcpyAsync(p, src, sizeof(float) * n, cudaMemcpyHostToDevice, stream_));
    OV_DCHECK(cudaGetLastError());
    owned_.push_back(p);
    return p;
  }

  DMat upload_mat(const Mat& m) {
    if (!m.w) throw std::runtime_error("cuda decoder: expected fp32 Mat weights");
    DMat r;
    r.K = static_cast<int>(m.K);
    r.N = static_cast<int>(m.N);
    r.w = upload(m.w, m.K * m.N);
    return r;
  }

  DConv upload_conv(const Conv1d& c) {
    DConv r;
    r.cin = c.cin; r.cout = c.cout; r.k = c.k; r.stride = c.stride; r.pad = c.pad; r.dil = c.dil;
    r.w = upload(c.wt.data(), static_cast<int64_t>(c.wt.size()));
    r.b = c.b ? upload(c.b, c.cout) : nullptr;
    return r;
  }

  DConv upload_convt(const ConvT1d& c) {
    DConv r;
    r.cin = c.cin; r.cout = c.cout; r.k = c.k; r.stride = c.stride; r.pad = c.pad; r.dil = 1;
    r.out_pad = c.out_pad;
    r.w = upload(c.wt.data(), static_cast<int64_t>(c.wt.size()));
    r.b = c.b ? upload(c.b, c.cout) : nullptr;
    return r;
  }

  void gemm(int M, int N, int K, const float* A, const float* W, float* C, float beta = 0.f) {
    const float alpha = 1.f;
    OV_DCUBLAS(cublasSgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, W, N, A, K, &beta,
                           C, N));
  }

  void gemm_ld(int M, int N, int K, const float* A, const float* W, float* C, int ldc,
               float beta) {
    const float alpha = 1.f;
    OV_DCUBLAS(cublasSgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, W, N, A, K, &beta,
                           C, ldc));
  }

  void conv1d(const DConv& c, const float* x, int t_in, float* y) {
    const int t_out = c.out_len(t_in);
    const int64_t n = static_cast<int64_t>(t_out) * c.cout;
    conv1d_direct_kernel<<<blocks_for(n), 256, 0, stream_>>>(
        x, c.w, c.b, t_in, c.cin, c.cout, c.k, c.stride, c.pad, c.dil, t_out, y);
    OV_DCHECK(cudaGetLastError());
  }

  void convt(const DConv& c, const float* x, int t_in, float* y) {
    const int t_out = c.convt_out_len(t_in);
    OV_DCHECK(cudaMemsetAsync(y, 0, sizeof(float) * static_cast<size_t>(t_out) * c.cout, stream_));
    for (int kk = 0; kk < c.k; ++kk) {
      const int off = kk - c.pad;
      int64_t t0 = 0, t1 = t_in - 1;
      if (off < 0) t0 = (-off + c.stride - 1) / c.stride;
      const int64_t last = t_out - 1 - off;
      if (last < 0) continue;
      t1 = std::min<int64_t>(t1, last / c.stride);
      if (t1 < t0) continue;
      const int64_t m = t1 - t0 + 1;
      gemm_ld(static_cast<int>(m), c.cout, c.cin, x + t0 * c.cin,
              c.w + static_cast<int64_t>(kk) * c.cin * c.cout,
              y + static_cast<int64_t>(t0 * c.stride + off) * c.cout,
              c.stride * c.cout, 1.f);
    }
    if (c.b) {
      add_bias_kernel<<<blocks_for(static_cast<int64_t>(t_out) * c.cout), 256, 0, stream_>>>(
          y, c.b, t_out, c.cout);
      OV_DCHECK(cudaGetLastError());
    }
  }

  void snake(float* x, const float* alpha, int t, int ch) {
    const int64_t n = static_cast<int64_t>(t) * ch;
    snake_kernel<<<blocks_for(n), 256, 0, stream_>>>(x, alpha, t, ch);
    OV_DCHECK(cudaGetLastError());
  }

  ATConfig cfg_;
  cudaStream_t stream_ = nullptr;
  cublasHandle_t blas_ = nullptr;
  std::vector<void*> owned_;
  Arena arena_;

  struct DRvq {
    DMat project;
    const float* bias = nullptr;
    const float* embed = nullptr;
  };
  struct DRes {
    const float* snake1 = nullptr;
    const float* snake2 = nullptr;
    DConv conv1, conv2;
  };
  struct DBlock {
    const float* snake = nullptr;
    DConv convt;
    DRes ru[3];
  };
  std::vector<DRvq> rvq_;
  DMat fc2_;
  DConv conv1_, conv2_;
  const float* snake_out_ = nullptr;
  std::vector<DBlock> blocks_;
};

void DecoderRunner::decode(const int64_t* codes, int frames, float* audio) {
  if (!codes || frames <= 0) return;
  const int T = frames, H = cfg_.hidden, D = cfg_.codebook_dim, IN = cfg_.dec_in;
  const int64_t samples = static_cast<int64_t>(T) * cfg_.hop;
  for (int q = 0; q < cfg_.n_quantizers; ++q) {
    for (int t = 0; t < T; ++t) {
      const int64_t id = codes[static_cast<int64_t>(q) * T + t];
      if (id < 0 || id >= cfg_.codebook_size)
        throw std::runtime_error("cuda decoder: code index out of range");
    }
  }

  const int64_t max_tc = samples * 32;  // final channel count is the largest t*c
  const size_t bytes = sizeof(float) * (2 * T * H + 2 * T * IN + T * D + 4 * max_tc) +
                       sizeof(int64_t) * static_cast<int64_t>(cfg_.n_quantizers) * T;
  arena_.reserve(bytes);
  arena_.reset();

  int64_t* d_codes = arena_.get<int64_t>(static_cast<int64_t>(cfg_.n_quantizers) * T);
  float* q = arena_.get<float>(static_cast<int64_t>(T) * H);
  float* tmp = arena_.get<float>(static_cast<int64_t>(T) * H);
  float* e = arena_.get<float>(static_cast<int64_t>(T) * D);
  float* h = arena_.get<float>(static_cast<int64_t>(T) * IN);
  float* a = arena_.get<float>(max_tc);
  float* b = arena_.get<float>(max_tc);
  float* x0 = arena_.get<float>(max_tc);
  float* y = arena_.get<float>(max_tc);

  OV_DCHECK(cudaMemcpyAsync(d_codes, codes,
                            sizeof(int64_t) * static_cast<size_t>(cfg_.n_quantizers) * T,
                            cudaMemcpyHostToDevice, stream_));
  OV_DCHECK(cudaMemsetAsync(q, 0, sizeof(float) * static_cast<size_t>(T) * H, stream_));

  for (int n = 0; n < cfg_.n_quantizers; ++n) {
    gather_embed_kernel<<<blocks_for(static_cast<int64_t>(T) * D), 256, 0, stream_>>>(
        e, rvq_[n].embed, d_codes, T, D, n);
    OV_DCHECK(cudaGetLastError());
    gemm(T, H, D, e, rvq_[n].project.w, tmp);
    accum_project_kernel<<<blocks_for(static_cast<int64_t>(T) * H), 256, 0, stream_>>>(
        q, tmp, rvq_[n].bias, static_cast<int64_t>(T) * H, H);
    OV_DCHECK(cudaGetLastError());
  }
  gemm(T, IN, H, q, fc2_.w, h);
  add_bias_kernel<<<blocks_for(static_cast<int64_t>(T) * IN), 256, 0, stream_>>>(h, fc2_.b, T,
                                                                                   IN);
  OV_DCHECK(cudaGetLastError());

  int t_cur = conv1_.out_len(T);
  int ch = cfg_.dec_ch;
  conv1d(conv1_, h, T, a);

  for (size_t bi = 0; bi < blocks_.size(); ++bi) {
    const DBlock& blk = blocks_[bi];
    const int out_ch = blk.convt.cout;
    snake(a, blk.snake, t_cur, ch);
    const int t_out = blk.convt.convt_out_len(t_cur);
    convt(blk.convt, a, t_cur, b);

    for (int j = 0; j < 3; ++j) {
      const DRes& u = blk.ru[j];
      const int64_t n = static_cast<int64_t>(t_out) * out_ch;
      copy_kernel<<<blocks_for(n), 256, 0, stream_>>>(x0, b, n);
      OV_DCHECK(cudaGetLastError());
      snake(b, u.snake1, t_out, out_ch);
      conv1d(u.conv1, b, t_out, y);
      snake(y, u.snake2, t_out, out_ch);
      conv1d(u.conv2, y, t_out, b);
      add_inplace_kernel<<<blocks_for(n), 256, 0, stream_>>>(b, x0, n);
      OV_DCHECK(cudaGetLastError());
    }
    std::swap(a, b);
    ch = out_ch;
    t_cur = t_out;
  }

  snake(a, snake_out_, t_cur, ch);
  conv1d(conv2_, a, t_cur, y);
  if (static_cast<int64_t>(t_cur) != samples) {
    throw std::runtime_error("cuda decoder: wrong output length");
  }
  OV_DCHECK(cudaMemcpyAsync(audio, y, sizeof(float) * static_cast<size_t>(samples),
                            cudaMemcpyDeviceToHost, stream_));
  OV_DCHECK(cudaStreamSynchronize(stream_));
}

DecoderRunnerPtr DecoderRunnerCreate(const AcousticDecoder& decoder) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0)
    return {nullptr, &DecoderRunnerDelete};
  try {
    return {new DecoderRunner(decoder), &DecoderRunnerDelete};
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[cuda] decoder unavailable: %s\n", e.what());
    return {nullptr, &DecoderRunnerDelete};
  }
}

void DecoderRunnerDelete(DecoderRunner* r) { delete r; }

void DecoderRunnerDecodeThunk(void* ctx, const int64_t* codes, int frames, float* audio) {
  static_cast<DecoderRunner*>(ctx)->decode(codes, frames, audio);
}

}  // namespace cuda
}  // namespace ov

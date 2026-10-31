// Measurement probe for OmniVoice-Native CUDA LM forward.
// Times: (1) the repo's per-head attention kernel verbatim, (2) a batched-cuBLAS
// variant, (3) the forward's SGEMMs in fp32 / TF32 / fp16 tensor cores,
// (4) the per-forward cudaMalloc churn, (5) the logits D2H copy.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                     \
  do {                                                            \
    cudaError_t e_ = (x);                                         \
    if (e_ != cudaSuccess) {                                      \
      printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__);  \
      exit(1);                                                    \
    }                                                             \
  } while (0)

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- repo kernel
__global__ void attention_kernel_repo(const float* q, const float* k, const float* v,
                                      const float* bias, int seq, int D, int QD, int KVD, int nh,
                                      int nkv_ratio, float scale, float* out) {
  extern __shared__ float smem[];
  float* sc = smem;
  float* red = smem + 1024;
  const int bh = blockIdx.x;
  const int b = bh / nh;
  const int h = bh % nh;
  const int kv = h / nkv_ratio;
  const float* qh = q + static_cast<size_t>(b) * seq * QD + static_cast<size_t>(h) * D;
  const float* kh = k + static_cast<size_t>(b) * seq * KVD + static_cast<size_t>(kv) * D;
  const float* vh = v + static_cast<size_t>(b) * seq * KVD + static_cast<size_t>(kv) * D;
  float* oh = out + static_cast<size_t>(b) * seq * QD + static_cast<size_t>(h) * D;
  const float* bias_b = bias ? bias + static_cast<size_t>(b) * seq * seq : nullptr;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  for (int i = 0; i < seq; ++i) {
    const float* qrow = qh + static_cast<size_t>(i) * QD;
    for (int j = tid; j < seq; j += nthreads) {
      const float* krow = kh + static_cast<size_t>(j) * KVD;
      float acc = 0.f;
      for (int d = 0; d < D; ++d) acc += qrow[d] * krow[d];
      sc[j] = acc * scale + (bias_b ? bias_b[i * seq + j] : 0.f);
    }
    __syncthreads();
    float m = -INFINITY;
    for (int j = tid; j < seq; j += nthreads) m = fmaxf(m, sc[j]);
    red[tid] = m;
    __syncthreads();
    for (int st = nthreads / 2; st > 0; st >>= 1) {
      if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
      __syncthreads();
    }
    const float rowmax = red[0];
    __syncthreads();
    float sum = 0.f;
    if (rowmax < -1e29f) {
      for (int j = tid; j < seq; j += nthreads) sc[j] = 0.f;
    } else {
      for (int j = tid; j < seq; j += nthreads) {
        const float e = __expf(sc[j] - rowmax);
        sc[j] = e;
        sum += e;
      }
    }
    red[tid] = sum;
    __syncthreads();
    for (int st = nthreads / 2; st > 0; st >>= 1) {
      if (tid < st) red[tid] += red[tid + st];
      __syncthreads();
    }
    const float inv = (rowmax < -1e29f) ? 0.f : 1.f / red[0];
    __syncthreads();
    if (tid < D) {
      float acc = 0.f;
      for (int j = 0; j < seq; ++j) acc += sc[j] * inv * vh[static_cast<size_t>(j) * KVD + tid];
      oh[static_cast<size_t>(i) * QD + tid] = acc;
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------- improved
__global__ void softmax_rows_kernel(float* s, int rows, int cols) {
  const int r = blockIdx.x;
  float* row = s + static_cast<size_t>(r) * cols;
  float m = -INFINITY;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) m = fmaxf(m, row[j]);
  __shared__ float red[256];
  red[threadIdx.x] = m;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + st]);
    __syncthreads();
  }
  m = red[0];
  __syncthreads();
  float sum = 0.f;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    const float e = __expf(row[j] - m);
    row[j] = e;
    sum += e;
  }
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
    __syncthreads();
  }
  const float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < cols; j += blockDim.x) row[j] *= inv;
}

// Batched GQA attention: one strided-batched GEMM per (b, qhead-in-group),
// covering all 8 KV heads at once.  Scores are kept in [t][s] layout.
static void attention_batched(cublasHandle_t h, const float* dq, const float* dk, const float* dv,
                              int B, int S, int D, int QD, int KVD, int NH, int NKV, float scale,
                              float* dout, float* dscr) {
  const int G = NH / NKV;
  const float one = 1.f, zero = 0.f;
  for (int b = 0; b < B; ++b) {
    for (int g = 0; g < G; ++g) {
      cublasSgemmStridedBatched(
          h, CUBLAS_OP_T, CUBLAS_OP_N, S, S, D, &one,
          dq + static_cast<size_t>(b) * S * QD + static_cast<size_t>(g) * D, QD, static_cast<long long>(G) * D,
          dk + static_cast<size_t>(b) * S * KVD, KVD, static_cast<long long>(D),
          &zero, dscr + static_cast<size_t>(g) * S * S, S, static_cast<long long>(G) * S * S, NKV);
      for (int kv = 0; kv < NKV; ++kv)
        softmax_rows_kernel<<<S, 128>>>(dscr + static_cast<size_t>(kv * G + g) * S * S, S, S);
      cublasSgemmStridedBatched(
          h, CUBLAS_OP_T, CUBLAS_OP_N, D, S, S, &one,
          dv + static_cast<size_t>(b) * S * KVD, KVD, static_cast<long long>(D),
          dscr + static_cast<size_t>(g) * S * S, S, static_cast<long long>(G) * S * S,
          &zero, dout + static_cast<size_t>(b) * S * QD + static_cast<size_t>(g) * D, QD,
          static_cast<long long>(G) * D, NKV);
    }
  }
}

__global__ void f2h(const float* src, __half* dst, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}

int main() {
  cublasHandle_t blas;
  cublasCreate(&blas);
  cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH);
  const int H = 1024, NH = 16, NKV = 8, D = 128, INTER = 3072, CV = 8200;
  const int QD = NH * D, KVD = NKV * D;

  for (int S : {109, 256}) {
    const int B = 2, T = B * S;
    std::vector<float> hq(T * QD, 0.01f), hk(T * KVD, 0.01f), hv(T * KVD, 0.01f),
        ho(T * QD, 0.f), ha(T * H, 0.01f), hw((size_t)INTER * H, 0.01f);
    float *dq, *dk, *dv, *dout, *dscr, *da, *dw, *dc, *dlog, *dwh;
    CK(cudaMalloc(&dq, 4 * hq.size()));
    CK(cudaMalloc(&dk, 4 * hk.size()));
    CK(cudaMalloc(&dv, 4 * hv.size()));
    CK(cudaMalloc(&dout, 4 * ho.size()));
    CK(cudaMalloc(&dscr, 4 * (size_t)NH * B * S * S));
    CK(cudaMalloc(&da, 4 * ha.size()));
    CK(cudaMalloc(&dw, 4 * hw.size()));
    CK(cudaMalloc(&dc, 4 * (size_t)T * INTER));
    CK(cudaMalloc(&dlog, 4 * (size_t)T * CV));
    CK(cudaMalloc(&dwh, 4 * (size_t)CV * H));
    CK(cudaMemset(dwh, 0, 4 * (size_t)CV * H));
    CK(cudaMemcpy(dq, hq.data(), 4 * hq.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk, hk.data(), 4 * hk.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, hv.data(), 4 * hv.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(da, ha.data(), 4 * ha.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, hw.data(), 4 * hw.size(), cudaMemcpyHostToDevice));
    const float scale = 1.f / std::sqrt(static_cast<float>(D));

    // ---- 1. repo attention, per forward (28 layers) ----
    const size_t smem = (1024 + 256) * sizeof(float);
    attention_kernel_repo<<<B * NH, 128, smem>>>(dq, dk, dv, nullptr, S, D, QD, KVD, NH, NH / NKV,
                                                 scale, dout);
    CK(cudaDeviceSynchronize());
    double t0 = now();
    for (int i = 0; i < 28 * 3; ++i)
      attention_kernel_repo<<<B * NH, 128, smem>>>(dq, dk, dv, nullptr, S, D, QD, KVD, NH, NH / NKV,
                                                   scale, dout);
    CK(cudaDeviceSynchronize());
    const double attn_repo = (now() - t0) / (28 * 3) * 1e3;

    // ---- 2. batched-cuBLAS attention, per forward ----
    attention_batched(blas, dq, dk, dv, B, S, D, QD, KVD, NH, NKV, scale, dout, dscr);
    CK(cudaDeviceSynchronize());
    t0 = now();
    for (int i = 0; i < 28 * 3; ++i)
      attention_batched(blas, dq, dk, dv, B, S, D, QD, KVD, NH, NKV, scale, dout, dscr);
    CK(cudaDeviceSynchronize());
    const double attn_batched = (now() - t0) / (28 * 3) * 1e3;

    // ---- 3. the forward's GEMMs: fp32 / tf32 / fp16 ----
    auto time_gemms = [&](cublasMath_t mode) {
      cublasSetMathMode(blas, mode);
      CK(cudaDeviceSynchronize());
      const double t = now();
      for (int rep = 0; rep < 3; ++rep) {
        for (int li = 0; li < 28; ++li) {
          const float one = 1.f, zero = 0.f;
          cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, QD, T, H, &one, dw, QD, da, H, &zero, dc, QD);
          cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, KVD, T, H, &one, dw, KVD, da, H, &zero, dc, KVD);
          cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, KVD, T, H, &one, dw, KVD, da, H, &zero, dc, KVD);
          cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, H, T, QD, &one, dw, H, dc, QD, &zero, da, H);
          cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, INTER, T, H, &one, dw, INTER, da, H, &zero, dc, INTER);
          cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, INTER, T, H, &one, dw, INTER, da, H, &zero, dc, INTER);
          cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, H, T, INTER, &one, dw, H, dc, INTER, &zero, da, H);
        }
        const float one = 1.f, zero = 0.f;
        cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, CV, T, H, &one, dwh, CV, da, H, &zero, dlog, CV);
      }
      CK(cudaDeviceSynchronize());
      cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH);
      return (now() - t) / 3 * 1e3;
    };
    const double gemm_fp32 = time_gemms(CUBLAS_DEFAULT_MATH);
    const double gemm_tf32 = time_gemms(CUBLAS_TF32_TENSOR_OP_MATH);

    // fp16 weights + fp16 activations, fp32 accumulate (tensor cores)
    __half *dhw, *dha, *dhc;
    CK(cudaMalloc(&dhw, 2 * hw.size()));
    CK(cudaMalloc(&dha, 2 * ha.size()));
    CK(cudaMalloc(&dhc, 2 * (size_t)T * INTER));
    f2h<<<(hw.size() + 255) / 256, 256>>>(dw, dhw, hw.size());
    f2h<<<(ha.size() + 255) / 256, 256>>>(da, dha, ha.size());
    CK(cudaDeviceSynchronize());
    auto time_gemms_f16 = [&]() {
      const double t = now();
      for (int rep = 0; rep < 3; ++rep) {
        for (int li = 0; li < 28; ++li) {
          const float one = 1.f, zero = 0.f;
          auto g = [&](int N, int M, int K, __half* A, __half* B, __half* C) {
            cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, A, CUDA_R_16F, N, B,
                         CUDA_R_16F, K, &zero, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
          };
          g(QD, T, H, dhw, dha, dhc);
          g(KVD, T, H, dhw, dha, dhc);
          g(KVD, T, H, dhw, dha, dhc);
          g(H, T, QD, dhw, dhc, dha);
          g(INTER, T, H, dhw, dha, dhc);
          g(INTER, T, H, dhw, dha, dhc);
          g(H, T, INTER, dhw, dhc, dha);
        }
      }
      CK(cudaDeviceSynchronize());
      return (now() - t) / 3 * 1e3;
    };
    const double gemm_f16 = time_gemms_f16();

    // ---- 4. per-forward cudaMalloc churn (15 buffers) ----
    t0 = now();
    for (int i = 0; i < 32; ++i) {
      std::vector<void*> ps;
      for (int j = 0; j < 15; ++j) {
        void* p;
        CK(cudaMalloc(&p, sizeof(float) * (size_t)T * INTER));
        ps.push_back(p);
      }
      for (void* p : ps) CK(cudaFree(p));
    }
    const double t_malloc = (now() - t0) / 32 * 1e3;

    // ---- 5. pageable logits D2H ----
    std::vector<float> hlog((size_t)T * CV);
    t0 = now();
    for (int i = 0; i < 8; ++i)
      CK(cudaMemcpy(hlog.data(), dlog, 4 * hlog.size(), cudaMemcpyDeviceToHost));
    const double t_d2h = (now() - t0) / 8 * 1e3;

    printf(
        "S=%3d | attn: repo %6.2f ms  batched %5.2f ms (x%4.1f) | gemm/forward: fp32 %5.2f  tf32 %5.2f  "
        "fp16-TC %5.2f ms | malloc %5.2f ms | logits D2H %5.2f ms\n",
        S, attn_repo, attn_batched, attn_repo / attn_batched, gemm_fp32, gemm_tf32, gemm_f16, t_malloc,
        t_d2h);
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(dscr); cudaFree(da);
    cudaFree(dw); cudaFree(dc); cudaFree(dlog); cudaFree(dwh); cudaFree(dhw); cudaFree(dha); cudaFree(dhc);
  }
  double t0 = now();
  float* big;
  CK(cudaMalloc(&big, (size_t)2450340992ull));
  printf("cudaMalloc(2.45 GB) = %.2f ms   (upload of fp32 LM weights, once)\n", (now() - t0) * 1e3);
  cudaFree(big);
  cublasDestroy(blas);
  return 0;
}

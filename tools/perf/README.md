# 性能探针（对应 `docs/perf-review.md`）

这组小程序**不属于引擎**、不被 CMake 构建，只用来做周期性测量。
结论和数字都在 `docs/perf-review.md`，这里只放怎么跑。

## 1. CUDA：`probe.cu`

逐字复刻 `src/backend/cuda/cuda_lm.cu` 的 `attention_kernel`，并和 cuBLAS
strided-batched 版本对比；同时测一次 forward 的全部 SGEMM（fp32 / TF32 / fp16 tensor core）、
每 forward 的 `cudaMalloc` 开销、logits D2H。

```bash
nvcc -O3 -arch=sm_89 --use_fast_math -o /tmp/probe tools/perf/probe.cu -lcublas
/tmp/probe
```

本机（4090）输出示例：

```
S=109 | attn: repo   1.20 ms  batched  0.20 ms (x 6.0) | gemm/forward: fp32  9.28  tf32 13.42  fp16-TC 14.68 ms | malloc  1.67 ms | logits D2H  1.16 ms
S=256 | attn: repo   5.30 ms  batched  0.26 ms (x20.7) | gemm/forward: fp32 14.07  tf32 10.54  fp16-TC  8.08 ms | malloc  2.10 ms | logits D2H  2.65 ms
```

## 2. CPU kernel：`cpu_probe.cpp`

GEMM（fp32 / 旧 `[K,N]` int8 / 新 `[N,K]` int8 / `[N,K]`+AVX-VNNI，M=218 与 512）、
`silu_mul`（现状 vs `exp256_ps` 版）、`snake`
（现状 vs 换循环序+OpenMP vs AVX2 多项式版）、`rope`、`group_norm`、`gelu`、
im2col 的分配/缺页成本，最后可选跑一次完整解码器。

```bash
g++ -O3 -fopenmp -std=c++17 -Icpp/include -Icpp/src \
    -o /tmp/cpu_probe tools/perf/cpu_probe.cpp cpp/build/libovcore.a -lgomp
OMP_NUM_THREADS=32 /tmp/cpu_probe output/weights/omnivoice-at-fp32.ovw
```

（第二个参数可选，给了就会跑一次 `AcousticDecoder::decode` 计时。）

本机（i9-14900KF, 32 线程）示例：

```
kernel: avx2+fma+vnni
[gemm] M=218 N=3072 K=1024  fp32 719 | s8[K,N] 1339 | s8[N,K] 824 | s8[N,K]+VNNI 3341 GFLOPS
[gemm] M=512 N=3072 K=1024  fp32 909 | s8[K,N] 1368 | s8[N,K] 785 | s8[N,K]+VNNI 3459 GFLOPS
```

`s8[N,K]` 是未预打包的 AVX2 回退（`OV_CPU_NO_VNNI=1` 走它），`s8[N,K]+VNNI` 是
加载时预打包 + per-32-k 激活量化的路径。改内核后必须重新编译探针，不要用旧二进制。

## 3. 分阶段打点：`dec_probe.cpp` / `enc_probe2.cpp`

这两个探针依赖在**源码副本**里插入的 6 个计时点（仓库源码保持干净）。
做法：把 `cpp/src/model/audio_tokenizer.cpp` 或 `at_encoder.cpp` 拷进临时目录，
在下列位置各插一对 `t0 = now()` / `ph("name") += now() - t0`，然后和探针一起编译。

`audio_tokenizer.cpp::decode` 的插入点：

| 位置 | 标签 |
|---|---|
| RVQ 循环之前 / `// 2) fc2` 之前 | `rvq` |
| `// 3) conv1 + snake` 之前 | `fc2` |
| `// 4) blocks` 之前 | `conv1` |
| `snake_f32(blk.snake...)` / `conv_transpose1d_f32(...)` 前后 | `snake` / `convt` |
| res-unit 里的 `snake1` / `conv1d_f32(conv1)` / `snake2` / `conv1d_f32(conv2)` | `snake` / `conv_k7` / `snake` / `conv_k1` |

`at_encoder.cpp::encode_with_stages` 的插入点：

| 位置 | 标签 |
|---|---|
| `semantic_features` 之后 | `hubert` |
| `semantic_encode` 之后 | `sem convs` |
| `acoustic_encode` 之后 | `acoustic enc` |
| `fc_.mul` 之后 | `fc` |
| RVQ 循环之后 | `rvq` |

编译（以解码器为例）：

```bash
g++ -O3 -fopenmp -std=c++17 -Icpp/include -Icpp/src -I/tmp/patched \
    -o /tmp/dec_probe tools/perf/dec_probe.cpp /tmp/patched/audio_tokenizer.cpp \
    cpp/src/model/mat.cpp cpp/src/core/weight_file.cpp \
    cpp/src/backend/cpu/{gemm,ops_cpu,conv_cpu}.cpp -lgomp
OMP_NUM_THREADS=32 /tmp/dec_probe output/weights/omnivoice-at-fp32.ovw 86
```

本机输出（86 帧）：

```
[decode] total 723.1 ms
   conv_k7  277 ms (38%)   convt 242 ms (33%)   snake 144 ms (20%)   conv_k1 36 ms (5%)
```

## 4. 端到端

```bash
# 单步成本 = (sample(steps=N) - sample(steps=1)) / (N-1)
for n in 1 2 4 8; do ./cpp/build/ovtts ... --steps $n --backend cuda; done

# 线程数对 GEMM 的影响（8 P-core + 16 E-core 上 12 线程会出现断崖）
for t in 1 2 4 8 12 16 24 32; do OMP_NUM_THREADS=$t ./cpp/build/ovtts ... ; done
```

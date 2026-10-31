# 新 session 提示词：CPU int8 改 AVX-VNNI 点积

> 直接整段粘进新 session 即可。最后一段「先做的事」会带它进入正确的第一步。

---

你是在 `/home/model/project/Ominivoice-Native` 这个仓库里工作。这是一个从零手写的
OmniVoice TTS 推理引擎（C++17，无 ONNX Runtime / 无 PyTorch 运行时）：Qwen3-0.6B 语言模型
+ DAC/HuBERT 音频 tokenizer，权重是自定义的 `OVW1` 二进制格式（mmap 加载）。有两个后端：
`cpp/src/backend/cpu/`（AVX2+FMA + OpenMP）和 `cpp/src/backend/cuda/`（fused attention +
cuBLAS GEMM）。

**开始前必须先读这三个文件**，它们包含了此前的全部上下文与已实测数据：

- `docs/perf-review.md` —— 性能剖析、已落地的优化、优先级表（含状态列）
- `docs/weight-format.md` —— OVW1 格式（`layout` / `quant` / scales 的语义）
- `docs/cpp-engine.md` —— 引擎分层与后端契约

## 环境

- CPU：i9-14900KF，8 P-core + 16 E-core / 32 线程，**有 `avx2`、`fma`、`avxvnni`，无 AVX-512**
- GPU：RTX 4090（本轮任务不碰 CUDA，但不要让它回归）
- g++ 13，CUDA 12.x，cmake ≥ 3.16
- 权重：`output/weights/omnivoice-lm-fp32.ovw`、`omnivoice-lm-int8.ovw`、
  `omnivoice-at-fp32.ovw`、`tokenizer.ovt`（**不要删除或覆盖它们**）
- 参考数据：`output/ref/lm/*`、`output/ref/at_dec/*`、`output/ref/at_enc/*`

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DOV_ENABLE_CUDA=ON
cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure      # 10 项，必须全绿
```

## 当前性能基线（必须原样复现一次再动手）

```bash
# CPU 端到端（int8 LM + fp32 AT）：32 步，RTF 约 2.07
./cpp/build/ovtts output/weights/omnivoice-lm-int8.ovw output/weights/omnivoice-at-fp32.ovw \
  output/weights/tokenizer.ovt "Hello, this is OmniVoice running in C++ on the CPU backend." \
  /tmp/base.wav --lang English --seed 1234 --steps 32

# 分阶段（forward 约 0.206 s @ S=109，其中 91% 是 GEMM，实测 ~1048 GFLOPS）
OV_PROFILE=1 ./cpp/build/ovtts ... --steps 4

# 单 kernel 微基准（GEMM 各形状、silu、snake、rope、gnorm、im2col）
g++ -O3 -fopenmp -std=c++17 -Icpp/include -Icpp/src \
    -o /tmp/cpu_probe tools/perf/cpu_probe.cpp cpp/build/libovcore.a -lgomp
OMP_NUM_THREADS=32 /tmp/cpu_probe
```

int8 数值门禁当前值（记下来，之后对比）：`argmax 470/512 = 91.80%`、
`mean|Δ| = 4.914e-02`、`cos = 0.99999964`。

## 已完成，**不要重做**

1. CUDA attention 已是 fused flash kernel（每层 1 次启动），cuBLAS 版只作回退。
2. 每步的 log_softmax/guidance/argmax 已在设备上跑（`LM::set_cuda_sampler`）。
3. CPU 卷积：分块 im2col + per-thread 复用工作区（`src/backend/cpu/scratch.h`），
   ConvTranspose 已是相位分解 GEMM，Snake 已是行主序 + OpenMP。
4. `snake_f32` / `silu_mul_avx2` / RoPE / 注意力（8-key 扫描 + PV 输出驻留寄存器）都已向量化。
5. **OpenMP 调度已经是 `schedule(dynamic,4)`**（`gemm.cpp`）——这带来 21%，不要改回 static。
6. GEMM 的 `M % 8` 尾部已在 micro-kernel 内用掩码写回处理，没有补齐拷贝了。
7. 8×8 micro-kernel 已经跑到单核 FMA 峰值的 ~87%（123 / 141 GFLOPS），
   **不要再去折腾 6×16 之类的新分块**，那不是瓶颈。

## 你的任务：把 CPU int8 GEMM 改成真正的 AVX-VNNI 点积

### 为什么

当前 int8 路径仍是「每个 k 把 int8 反量化成 fp32 再 FMA」，MAC 吞吐和 fp32 完全一样，
只是省了带宽——所以 int8 只比 fp32 快 1.6×（873 vs 553 GFLOPS）。CPU LM 现在 91% 时间在
GEMM 里，这是 CPU 侧唯一的大杠杆，目标是 **1.5–2.5×**（端到端 RTF 2.07 → 1.0±0.3）。

### 关键约束：VNNI 要求权重换布局

`vpdpbusd` 要求「一个输出列的 k 连续」，而 OVW1 现在存的是 `[K][N]`（N 连续，
见 `docs/weight-format.md` 的 `layout=1`）。所以必须新增一个 **`[N][K]` 的 int8 权重布局**。
好消息：`tools/convert_weights.py` 里的 `quantize_int8(arr)` 本来就是在 `[out, in]` 上算的
（每输出通道一个 scale），现在存的是 `q.T`；改成存 `q` 即可，layout 用 0、per-row scales ——
这正是现在量化 embedding 表用的编码方式（`layout=0` + `n_scales == rows`），写/读机制都已经存在。

### 分三阶段，每阶段必须可独立验收、可回退

**阶段 1：只做布局，不做新内核**（验证格式能往返）

- `tools/convert_weights.py`：加 `--int8-layout {kn,nk}`，默认 `kn`（保持现状，向后兼容）。
  `nk` 时存 `q`（`[N][K]`）、`layout=0`、`quant=1`、`scales=scale`（每输出通道一个）。
  产物写到新文件（例如 `output/weights/omnivoice-lm-int8-nk.ovw`），**不要覆盖旧文件**。
- 加载侧：`cpp/src/model/mat.{h,cpp}` 的 `make_linear` 目前强制要求 `t.transposed`；
  为 `[N][K]` 的 int8 增加一条分支（`Mat` 里加一个 `nk` 形态 + per-N scale 指针），
  `Mat::mul` 分发到一个新的 `linear_s8_nk`。**先写成朴素实现即可**（甚至可以先转置成
  `[K][N]` 再调现有 `linear_s8`）。
- 验收：`./cpp/build/ov_test_lm output/weights/omnivoice-lm-int8-nk.ovw output/ref/lm/b1_s64_full`
  必须复现 `argmax 91.80%`、`mean|Δ| ≈ 4.91e-2`（与 `kn` 文件逐位同量级）。
  这时性能不应有变化——这一步只证明格式与加载正确。

**阶段 2：VNNI 内核**

数据布局（**这是最容易写错的地方，务必按这个来**）：

- 权重预打包成 **4×8 的 k-major tile**：对每个 8 列块 `n0` 和每个 4-k 组 `g`，
  连续存放 32 字节 = `W[4g..4g+3][n0..n0+7]` 的转置（先 n 后 k 无所谓，只要与
  `vpdpbusd` 的 lane 对应关系一致：**lane j 对应列 n0+j，dword 内 4 个字节对应
  4 个连续 k**）。这样一次 `a_bcast = broadcast_dword(A[m][4g..4g+3])` +
  `b = loadu(tile)` + `vpdpbusd` 就推进 **8 列 × 4 k = 32 个 MAC**，
  而且累加器天然就是「每行 8 列」的 8 个 int32 lane，**不需要每列做水平求和**。
- 寄存器块建议 8 行 × 8 列：8 个 ymm 累加器 + 1 个 a 广播 + 1 个 b → 10 个，够用。
- 激活量化：先做 **按行一个 scale**（`u8 = round(A / (max|A_row|/255))`，每行一个 float），
  epilogue 里 `C[m][n] = acc * (ascale[m] * wscale[n])`。这样整个 K 只用一次 flush，最省。
  若 int8 门禁不达标，再退到 llama.cpp Q8_0 那种 **每 32 个 k 一块** 的 scale
  （每 8 个 k-group flush 一次，int32 → fp32 累加）。
- 尾块：M、N、K 都要能处理非 8/4 倍数（本仓库已有掩码写回的先例，照抄思路）。
- 分派：`__builtin_cpu_supports("avxvnni")` + `avx2` + `fma`；
  函数属性 `__attribute__((target("avx2,fma,avxvnni")))`（g++ 13 支持这个属性名；
  若报 target mismatch，就把 VNNI 内核放到单独 TU 用 `-mavxvnni` 编）。
  加一个环境变量 `OV_CPU_NO_VNNI=1` 强制走老路径，用于 A/B 和回归。
- **必须遵守 `ScopedSerialGemm` / `t_serial_gemm`**：attention 会在已并行的区域里串行调用
  GEMM，新内核不能无条件开 OpenMP 区域。

验收：
- `OMP_NUM_THREADS=32 /tmp/cpu_probe` 里 `linear_s8` 在 `M=218 N=3072 K=1024` 上
  从 ~873 GFLOPS 至少提到 **1300 GFLOPS**（理想 2000+）。
- `OV_PROFILE=1 ... --steps 4` 的 forward 从 0.206 s 明显下降。
- `--steps 16` 的 `sample=` 至少降 30%。
- int8 门禁：`argmax ≥ 90%`、`mean|Δ| < 0.15`、`cos > 0.99999`（当前 91.80% / 4.91e-2 / 0.99999964）。
- **fp32 路径与 AT（音频 tokenizer）不受影响**：`ov_test_lm <fp32> ...`、`ov_test_at`、
  `ov_test_at_enc` 必须与改动前同值。

**阶段 3：收尾**

- 更新 `docs/perf-review.md`（§0 结果表、§5 优先级表状态列）与 `README.md`（性能现状），
  写清楚新布局、量化口径（按行还是按 32 块）、以及实测数字。
- 给 `ov_test_gemm` 加一条 `[N][K]` 布局 + VNNI 路径的正确性用例（随机矩阵 vs 双精度参考）。
- 如果最终**没能跑赢**现有内核，也要把代码回退干净、把实测数字和原因写进文档——
  半成品比没有更糟。

## 硬性规则

1. 每一步之后必须 `ctest` 全绿（10 项），并且 `ov_test_lm`（fp32/int8/CUDA）、
   `ov_test_at`、`ov_test_at_enc`、`ov_test_tokenizer` 的数值都不能比现在差。
2. **不要动 CUDA 侧**（`cuda_lm.cu` 的 attention/arena/反量化都已定稿）；它只要求不回归。
3. **不要改 OVW1 的既有语义**：新布局必须是新增的合法组合（`layout=0 + quant=1 + I8`），
   旧文件（`layout=1`）必须继续能加载。默认行为（不传新 flag）不得改变。
4. **不要留半成品**：每个阶段结束都要能编译、能过门禁、能解释清楚。
5. 所有性能结论必须是实测（同一台机器、同一命令、前后对比），不接受估算。
6. 不要用 `-ffast-math` / `-march=native` 这类全局开关去"顺便"提速——数值门禁比这重要。

## 已知陷阱（都是踩过的）

- `Mat`（线性层，`layout=1`，per-column scales）和 `Emb`（查表，`layout=0`，per-row scales）
  是两套语义，别混。新布局借用的是 `Emb` 那种编码，但走的是 `Mat` 的调用点。
- `cuda_lm.cu::weight()` 会读 `t.transposed` / `t.scales` 把 int8 反量化到显存；
  如果你改了缺省布局，CUDA 路径会静默算错。
- `linear_f32_ld(M,N,K,A,W,C,ldc,accumulate)` 被卷积/转置卷积用着，签名和语义别动。
- CPU 上 `linear_f32` 的标量回退路径也要保持可用（无 AVX2 的机器）。
- `tools/perf/` 里的探针不参与构建；改内核后直接重编译探针再测，别用旧二进制。
- 测时用 `OMP_NUM_THREADS=32`，并且**跑两次取第二次**（第一次有 page fault 与频率爬升）。

## 交付

结束时给我：

1. 改动文件清单（含每个文件改了什么）。
2. 一张前后对比表：`linear_s8` GFLOPS（M=218/512）、`--steps 16` 的 sample 时间、
   32 步端到端 RTF、int8 门禁三项数值、`ctest` 结果。
3. 一句结论：VNNI 在本机到底值多少，值不值得作为默认路径。
4. 如果阶段 2 之外还看到别的明显瓶颈，记进 `docs/perf-review.md` 的待办，不要顺手做。

---

## 先做的事（按顺序，别跳）

1. 读 `docs/perf-review.md` + `docs/weight-format.md` + `cpp/src/model/mat.cpp` +
   `cpp/src/backend/cpu/gemm.cpp` + `tools/convert_weights.py` 的 `build_entries`。
2. 跑一遍上面的基线命令，把你的实测数字记下来（与本文的数字对照）。
3. 确认工具链可用。这台机器上已经验证过（g++ 13）：

   ```c
   __attribute__((target("avx2,fma,avxvnni")))
   __m256i f(__m256i a, __m256i b) { return _mm256_dpbusd_epi32(a, b, b); }
   ```
   能编过，`_mm256_dpbusd_epi32` 结果正确（32 个 1×1 → 32）。
   注意 `__builtin_cpu_supports("avxvnni")` 返回的是**位掩码**（这台机器 536870912 = 1<<29），
   判断要写 `!= 0`，不能写 `== 1`。
4. 再开始阶段 1。

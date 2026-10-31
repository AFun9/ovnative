# 性能评审：手写 CPU / CUDA 推理引擎

[English summary](performance.en.md) | 中文完整版

> 实测日期 2026-10-29，机器：i9-14900KF（8 P-core + 16 E-core / 32 线程，AVX2+FMA+AVX-VNNI）
> + RTX 4090 (sm_89)，构建 `-DCMAKE_BUILD_TYPE=Release`（`-O3`）。
> 除明确标注"推算/预期"的项外，所有数字都是本机实测；复现命令见 §7。
> 第一轮结论：**数学运算很快，内存搬运、标量循环和调度才是瓶颈。**
> 第二轮（§0.2、§2 P2-1）：**int8 GEMM 改为真 AVX-VNNI 点积后，CPU 端到端 RTF 2.05 → 1.15。**
> 第三轮（§0.3）：**attention query 分块 + 激活量化池化/并行 + host CFG 并行，RTF 1.15 → 0.89–0.91。**
> 第四轮（§0.4）：**CPU ragged CFG + 全链路融合 + attention 4 行 + sliced qkv/gate-up，RTF 0.89–0.91 → 0.65–0.67。**
> 第五轮（§0.5）：**CUDA 纯 GPU 路径（TF32 + FP16 GEMM + GPU encoder/decoder），auto RTF 0.05–0.06、voice-clone 0.09–0.11。**

---

## 0. 优化结果（已实施）

本文的 P0/P1 项（§5 表里 1–7）已全部落地，`ctest` 8/8 通过（`lm_fp32` argmax 100%、`lm_fp32_cuda` argmax
100%、声码器 `max|Δ|=2.0e-6`、编码器 codes 400/400 精确、BPE 36/36）。

| 组件 | 优化前 | 优化后 | 加速 | 手段 |
|---|---:|---:|---:|---|
| CPU 声码器（86 帧 / 3.44 s） | 0.71 s | **0.15 s** | 4.7× | im2col 分块+复用 / convT 相位 GEMM / snake 换序 |
| CPU 编码器（3.12 s 参考音） | 1.19 s | **0.71 s** | 1.7× | 同上（编码器走 `conv1d_generic_f32`） |
| CPU LM 单次 forward（S=109, int8） | 0.358 s | **0.206 s** | 1.74× | dynamic 调度 / silu 向量化 / 去掉 M%8 补齐 / RoPE 表 / 注意力 |
| CPU 端到端 auto（32 步） | RTF 3.55 | **RTF 2.07** | 1.7× | 上述合计 |
| CPU 端到端 voice-clone（32 步） | RTF ~10.3 | **RTF 4.57** | 2.3× | 上述合计 |
| CUDA 单步（forward + 采样，S=109） | 48.6 ms | **5.8 ms** | 8.4× | fused attention / 设备侧采样 / TF32 + 合并 + int8 路径 FP16 GEMM |
| CUDA lm-int8 载入 | 0.88 s | **0.34 s** | 2.6× | int8 权重设备端反量化（PCIe 4×↓；合并/FP16 副本 +0.01 s） |
| CUDA 端到端 auto（32 步） | RTF 0.67 | **RTF 0.05–0.06** | 12× | 上述合计 + **GPU decoder**（decode 0.15 → 0.01 s） |
| CUDA 端到端 长句（S=325, 11.2 s） | RTF ~1.03 | **RTF 0.17** | 6.0× | 设备侧采样 + decoder |
| CUDA 端到端 voice-clone（32 步） | RTF ~1.4 | **RTF 0.09–0.11** | 13× | 上述合计 + GPU encoder |
| CPU int8 GEMM（M=218,N=3072,K=1024） | 1339 GFLOPS | **2982–3624 GFLOPS**（第三轮复测 3800–4130） | 2.3–3.1× | AVX-VNNI 4×8 tile + 每 32-k 块激活量化 |
| CPU LM forward（int8） | 0.202–0.205 s（218 行） | **0.062–0.070 s（中位 0.065，有效 195 行）** | 3.1× | VNNI GEMM + attention 4 行 + ragged CFG + norm/quant + qk norm/rope + silu/quant + sliced qkv/gate-up |
| CPU 端到端 auto（32 步，16 步） | RTF 2.05–2.08 / 3.53–3.56 s sample | **RTF 0.65–0.67 / 1.06–1.10 s** | 3.1× / -70% | 同上 + host CFG 采样并行 |
| CPU lm-int8-nk 载入 | 0.09 s | **0.30 s** | 一次性 +0.21 s | 加载时重排 440 MB 权重 + 55 MB 块和（重排已并行化，并去掉 N%8==0 且 K%32==0 时多余的 memset：0.43 → 0.30 s） |

> VNNI 的 baseline 是本轮同机复测的旧 int8 内核（1339 GFLOPS），不是本文早前记录的
> 873 GFLOPS；端到端基线 2.05 与早前 2.07 一致，差异来自微基准的构建/频率状态。
> 所有对比均为同一轮、同一 `OMP_NUM_THREADS=32`、第二次运行取值。

质量复核：英文 auto `WER 0%`（Whisper-large-v3）、中文 voice-clone 内容完全正确
（转写为繁体，仅字形/标点差异）。所有硬门禁的数值与优化前同级或更好。

### 0.1 改动的文件

| 文件 | 改动 |
|---|---|
| `cpp/src/backend/cpu/gemm.cpp` | **`schedule(dynamic,4)`**（混合 P/E 核心上静态切块会等最慢的线程：8 步 2.33 s → 1.83 s）；M 尾部改 micro-kernel 掩码写回（删掉补齐+整块 memcpy）；新增 `linear_f32_ld`（`ldc` + 累加）；删掉死掉的 `linear_f32_ref` |
| `cpp/src/backend/cpu/ops_cpu.cpp` | `silu_mul_avx2` 改用 `exp256_ps`；`conv1d_generic_f32` 改分块 im2col + 复用工作区 + tap-major 权重 + 直接写 `ldc=cout`；新增 `rmsnorm_into_f32` / `add_inplace_f32`；RoPE 表按位置缓存 + AVX2 旋转；注意力 8-key 扫描 + PV 输出分块驻留寄存器 |
| `cpp/src/backend/cpu/conv_cpu.cpp` | `Conv1d/ConvT1d::prepare` 改 `[k][cin][cout]`；`conv1d_f32` 分块 im2col；`conv_transpose1d_f32` 改相位分解 GEMM；`snake_f32` 换行主序 + OpenMP |
| `cpp/src/backend/cpu/scratch.h` | 新增：per-thread 增长型 scratch + `panel_rows()` |
| `cpp/src/backend/cuda/cuda_lm.cu` | attention → **fused kernel**（K/V 分块进 shared、在线 softmax、输出行驻寄存器），cuBLAS 版保留为 `D≠128` 回退；**设备侧引导采样 kernel**（log_softmax/guidance/argmax）；RoPE 表一次算好；设备 arena 替掉逐 forward 的 15 次 `cudaMalloc`；残差并入 GEMM（beta=1）；int8 权重设备端反量化 |
| `cpp/src/model/lm.{h,cpp}`、`tts.cpp`、`tools/ovtts.cpp` | `LM::set_cuda_sampler` 钩子：有加速器时每步只回传 `C×T` 个数，不再传 `S×C×V` logits；CPU 路径不变 |
| `cpp/src/model/conv.h` | 权重布局注释与新 GEMM 接口 |
| `cpp/include/ovnative/ov_gemm.h` | `linear_f32_ld` 声明 |
| `tools/perf/*` | 探针（CUDA/CPU 微基准 + 分阶段打点）；CPU 探针新增 `[N,K]` 与 VNNI 路径 |
| `tools/convert_weights.py` | 新增 `--int8-layout {kn,nk}`（默认 `kn` 不变）；`nk` 输出 `*-int8-nk.ovw`，`layout=0`、per-row scale |
| `cpp/src/model/mat.{h,cpp}` | `Mat` 增加 `nk` 形态、加载时 4×8 预打包与 `wsum`；`mul` 分派到 `linear_s8_nk{,_packed}`；`make_linear` 接受合法组合 `layout=0 + quant=1 + I8` |
| `cpp/src/backend/cpu/gemm.cpp` | 新增 `linear_s8_nk` AVX2 回退、`s8_nk_pack`、`quantize_a_u8`、4×8 VNNI 内核、`OV_CPU_NO_VNNI` 分派 |
| `cpp/include/ovnative/ov_gemm.h` | 新接口与 tile/wsum 编码约定 |
| `cpp/tests/test_gemm.cpp` | 新增 `[N][K]` 与 VNNI packed 路径的随机矩阵/双精度参考用例（含 M/N/K 尾块） |
| `cpp/src/model/lm.{h,cpp}`、`cpp/tools/ovtts.cpp`、`cpp/tests/test_lm.cpp` | `has_nk_weights()` + 布局不匹配时拒绝 CUDA 前端，避免静默算错 |
| `cpp/tools/ovdump.cpp` | `layout=0` 的量化标注改为 `int8/row-scale`（embedding/nk 都适用） |

### 0.2 第二轮结果：AVX-VNNI（详见 §2 P2-1）

把 int8 GEMM 从"反量化成 fp32 再 FMA"改成真正的 `vpdpbusd` 点积，权重新增
`[N][K]` 布局（`output/weights/omnivoice-lm-int8-nk.ovw`，旧 `kn` 文件仍是默认并保留）。

| 项目 | 旧 int8 | VNNI | 加速 |
|---|---:|---:|---:|
| CPU int8 GEMM（M=218 / M=512） | 1339 / 1348 GFLOPS | **2982–3624 / 3130–3809** | 2.3–2.8× |
| CPU LM forward（B=2,S=109） | 0.202–0.205 s | **0.104–0.110 s** | 1.9× |
| CPU 32 步 sample / RTF | 6.92–7.01 s / 2.05 | **3.81–3.82 s / 1.15** | -43~45% |
| CPU 16 步 sample | 3.53–3.56 s | **1.86–1.91 s** | -43~48% |
| int8 门禁（argmax / mean / cos） | 91.80% / 4.914e-2 / 0.99999964 | **92.19% / 6.04e-2 / 0.99999942** | 通过 |

`ctest` 仍 10/10；fp32 LM、CUDA LM、声码器、编码器、tokenizer、C API 的数值全部不变
（旧 `[K][N]` int8 文件走原内核，逐位不变）。唯一的一次性代价是载入 0.09 → 0.43 s
（加载时重排 440 MB 权重，RSS +495 MB）。

### 0.3 第三轮结果：attention 分块 + 量化池化/并行 + host CFG 并行（实测复核）

| 指标 | 第二轮（VNNI） | 第三轮 | 变化 |
|---|---:|---:|---:|
| CPU LM forward（B=2,S=109） | 0.104–0.110 s | **0.089–0.093 s**（中位 0.091） | 1.19× |
| CPU 32 步 sample | 3.81–3.82 s | **2.91–2.94 s** | -23~24% |
| CPU 32 步 RTF | 1.15 | **0.89–0.91** | 1.28× |
| CPU 16 步 sample | 1.86–1.91 s | **1.46–1.50 s** | -21~22% |
| CPU lm-int8-nk 载入 | 0.43 s | **0.30 s** | -0.13 s |
| int8 门禁（argmax / mean / cos） | 92.19% / 6.04e-2 / 0.99999942 | 不变 | — |

第三轮的收益不是单一改动，按“单独关掉一项”的 A/B 拆开（32 步 sample，同一轮）：

| 单独关闭的改动 | sample | 相对当前 |
|---|---:|---:|
| 当前（全部开启） | 2.91–2.94 s | — |
| 激活量化改回串行（83 µs/次） | 3.29–3.30 s | +0.37 s |
| host CFG 采样改回串行 | 3.11–3.16 s | +0.20 s |
| 关闭 q/k/v、gate/up 量化池化（7→4 次回退到 7 次） | 3.03–3.05 s | +0.11 s |
| attention query 分块（与上述项有重叠） | — | 约 +0.2 s |

> **数字修正**：第二轮末尾曾把第三轮 forward 写成 `0.074–0.078 s`，该值不可复现。
> 32 步 sample 2.92 s 对应约 91 ms/step，`OV_PROFILE` 32 次的均值是 0.0908 s、中位 0.090 s；
> 0.074 与端到端 sample 自相矛盾。上表已按本轮实测修正；端到端/载入数字不受影响。

第三轮改动文件：

| 文件 | 改动 |
|---|---|
| `cpp/src/backend/cpu/ops_cpu.cpp` | attention `kAttnRB=2`（两行 query 共享 K/V）；修复全屏蔽行对在 `out_stride>D` 时只清零第一行的 bug；scalar 回退补全屏蔽行处理 |
| `cpp/src/backend/cpu/gemm.cpp` | `s8_nk_pack` 按 k-block 并行 + N/K 对齐时跳过整块 memset；`quantize_a_u8` 按行 OpenMP；新增 `s8_nk_quantize` / `linear_s8_nk_packed_q` |
| `cpp/src/model/lm.{h,cpp}` | q/k/v 与 gate/up 共享量化结果（7→4 次/层）；`t_aq/t_asc` thread-local scratch |
| `cpp/src/model/mat.{h,cpp}` | `Mat::vnni_packed()` / `Mat::mul_q()` |
| `cpp/src/model/tts.cpp` | host CFG 的 log_softmax/guidance/argmax 按行 OpenMP（12 ms→~1 ms/step）；删除旧的逐行临时向量 |
| `cpp/src/backend/cuda/cuda_lm.{h,cu}`、`cpp/src/model/lm.h`、`cpp/tools/ovtts.cpp` | 设备采样接口改成 cond/uncond 两个设备指针（为 fused CFG 铺路）；CUDA 采样 kernel 已用 host 参考验证 |
| `cpp/tests/test_gemm.cpp` | 新增量化池化 bit-identical 用例、attention 全屏蔽行回归用例 |

### 0.4 第四轮：CPU ragged CFG 前向（本轮）

CFG 每步是 B=2、S=109 的 forward，但 uncond 行只有前 T=86 行参与采样；原来的
B=2 前向把 uncond 的尾部 23 行也完整算了一遍（每层 GEMM/attention/MLP 都算），
最后丢掉。**这 23/218 = 10.5% 的行是纯浪费**。

本轮把 CPU 的 CFG 输入重排成 `[cond 0..S) + uncond 0..T)` 共 195 行，用一个
`LM::forward_cfg()` 前向：

- GEMM/MLP/head 按 195 行算，权重只从 DRAM 流一遍（不是两次 forward）；
- attention 分成两个 segment：cond 109×109 和 uncond 86×86，尾部行不参与；
- 输出是 token-major `[195, C*V]`，host 采样直接按 `(S-T+t)` / `(S+t)` 取两个平面；
- CUDA 仍走原来的 B=2 + 设备采样（RTF 0.15 不变），可用 `OV_CPU_EXACT_CFG=1`
  在 CPU 上强制回到旧 B=2 路径（**输出 wav 与改动前逐字节相同**）。

| 指标 | 第三轮（B=2） | 第四轮（ragged CFG） | 变化 |
|---|---:|---:|---:|
| CPU forward（有效 195 行） | 0.089–0.093 s（218 行） | **0.062–0.070 s**（中位 0.065，均值 0.065） | ~19% |
| CPU 32 步 sample | 2.91–2.96 s | **2.09–2.14 s** | ~28% |
| CPU 32 步 RTF | 0.89–0.91 | **0.65–0.67** | ~27% |
| CPU 16 步 sample / RTF | 1.46–1.50 s / 0.47–0.48 | **1.06–1.10 s / 0.35–0.37** | ~27% |

数值口径（同一 synthetic CFG 输入 vs 完整 B=2）：

| 路径 | cond max\|Δ\| | uncond mean\|Δ\| | cos | uncond argmax |
|---|---:|---:|---:|---:|
| int8-nk | **0**（逐位） | 3.7e-2 | 0.99999982 | 95.1% |
| fp32 | **0**（逐位） | 1.4e-5 | 1.00000000 | 100% |

cond 平面完全不变；uncond 只差在 attention 对 109 vs 86 个 key 的求和顺序
（尾部 masked key 在数学上是 0，浮点上会留下 ~1 ULP），经 28 层放大到 mean 3.7e-2，
仍明显好于 int8 门禁（mean<0.15、cos>0.99999、argmax≥90%）。`ov_test_lm` 新增了
`cfg-fwd` 用例，对 S=109/T=86 的 synthetic 输入逐项检查这些阈值；`OV_CPU_EXACT_CFG=1`
可回到逐位等价的 B=2 路径。

随后又把 **RMSNorm + 激活量化融合**（`rmsnorm_quantize_f32`）：VNNI 路径不再把
`xn_` 写回全局内存，norm 和量化共用一个 OpenMP 区域。实测 M=195/K=1024 从
分开的 64–91 µs 降到 **~30 µs**，`aq/asc` 与分开路径 **逐字节相同**（新增
`rmsnorm+quant` 回归用例）；int8 门禁、`OV_CPU_NO_VNNI=1` 和旧 int8 文件数值都不变。
这一项把 32 步 sample 再压到 2.62–2.65 s、RTF 0.81–0.82。

最后把 attention 从 **2 行 query** 扩到 **4 行 query**（`attention_block4`：score 2 key × 4 行、
PV 16 维 × 4 行，8 个寄存器累加器；S%4 的尾部用同一 block 别名行、只存有效行）。
单 head 孤立调用反而略慢（0.10–0.14 vs 0.08–0.09 ms），但真实 forward 里 32 个 head
并发、K/V 复用翻倍，端到端更快：32 步 sample **2.62–2.65 → 2.51–2.57 s**、
RTF **0.81–0.82 → 0.78–0.80**；16 步 **1.27–1.30 s / RTF 0.42**。
`OV_ATTN_RB2=1` 可回到两行内核做 A/B；两版输出与旧 B=2 路径均逐字节一致（
`OV_CPU_EXACT_CFG=1` 复验通过）。

另外把 residual add 融进 `o_proj` / `down_proj` 的 VNNI epilogue（`C += P`）：
epilogue 里用 `asm volatile("" : "+x"(o))` 阻止 `mul+add` 被收缩成 FMA，保证与
“先存 proj 再 x+=proj” **逐字节相同**（新增 `s8nk add` 回归用例）。单独 A/B：
32 步 sample 2.58–2.65 → **2.52–2.57 s**，oproj 占比 11.0% → 9.5–10.2%。

最后把 q/k/v 三份 packed 权重拼成一个逻辑 `N=4096` 的矩阵、gate/up 拼成 `N=6144`，
用一个 OpenMP 区域的 **sliced VNNI kernel** 按 64 列 panel 写回三个/两个独立 C buffer
（`linear_s8_nk_packed_q_sliced`）。权重拼接在加载时做一次（swap 掉源 packed，稳态内存不涨），
因此载入 0.29 → 0.32 s；GEMM 数学与逐矩阵完全一致（新增 `s8nk sliced` 逐字节用例）。
32 步 sample 再降到 2.41–2.50 s、RTF 0.75–0.77，qkv 占比 19% → 16.4%。

随后又做了两处**逐字节等价**的融合：

1. **q/k 的 per-head RMSNorm + RoPE** 合成 `qk_norm_rope_f32`：原来 4 趟串行
   （qnorm/knorm/qrope/krope），现在 1 趟、按 row 并行；单层 100 → 35–50 µs，
   `a_prep` 从 5–6 ms 降到 1 ms。
2. **silu(gate)*up + 激活量化** 合成 `silu_mul_quantize_f32`：不物化 silu 输出，
   直接写 aq/asc；单层 183–241 µs → **62 µs**。down 用预量化 `linear_s8_nk_packed_q_add`
   在 epilogue 里加残差。

两项合计把 32 步 sample 从 2.41–2.50 s 压到 **2.09–2.14 s**、RTF **0.65–0.67**，
16 步 **1.06–1.10 s / 0.35–0.37**。

改动：`cpp/src/model/lm.{h,cpp}` 新增 `forward_cfg()` / `AttnSegment` / `embed_cfg()`，
`run_layer` 改为按 segment 跑 attention；`cpp/src/backend/cpu/gemm.cpp` +
`ov_gemm.h` 新增 `rmsnorm_quantize_f32()`（norm+量化融合，逐字节等价）；
`cpp/src/backend/cpu/ops_cpu.cpp` 新增 `attention_block4`（4 行 query、2 key/PV 16 维，
`OV_ATTN_RB2=1` 回退两行）；
`cpp/src/model/tts.cpp` 在 CPU 路径构造 packed ids/mask/position 并维护两套 token 更新；
`ov_test_lm` 新增 `cfg-fwd` 用例、`ov_test_gemm` 新增 `rmsnorm+quant` 用例；CUDA 路径不变。

三个最重要的结论（原始分析，数字为优化前）：


1. **CUDA 的 attention kernel 占了整个 GPU forward 的 70%**（S=109 时 33.6 ms / 47.5 ms，
   S=256 时 148 ms / ~170 ms），而同样的计算用 cuBLAS batched 只需 0.2–0.26 ms。
   改这一个 kernel，GPU 单次 forward 快 2.4×（S=109）到 7.7×（S=256）。
2. **CPU 的后处理（声码器 + 编码器）有 85%–90% 的时间不在做算术**：`conv1d_f32` 的 im2col
   每次调用都重新 `std::vector` 申请几十 MB（实测 110 MB 的申请光是缺页就 26 ms），
   `conv_transpose1d_f32` 是 `cin×t×k` 次标量 axpy，`snake_f32` 是标量 `sin` + 跨步访问。
   这三处合计 663 ms / 723 ms。
3. **CPU LM 的 GEMM 本身已经很能打**（int8 实测 873 GFLOPS @ M=218，32 线程 1179 GFLOPS），
   但一次 forward 里有 37% 的时间不在 GEMM 里：标量 `silu`（35 ms）、`M%8` 补齐路径（约 30 ms）、
   两次整块 `memcpy` + 标量残差相加、`rope` 每层重算三角函数。

而且：**CUDA 越快，CPU 后处理的占比越高**。auto 模式 3.44 s 输出里声码器占 31%，
11.24 s 长句里占 83%；voice clone 还要再叠加 1.19 s 的编码器。
换句话说，GPU 侧的收益现在有一大半被 CPU 侧吃掉。

---

### 0.5 第五轮：CUDA int8 路径

CPU 停在一个稳定点后转到 GPU。int8 checkpoint 仍然在加载时反量化成 fp32，但前向不再用
SGEMM：

1. **TF32**（`CUBLAS_TF32_TENSOR_OP_MATH`）先开；fp32 checkpoint 保持精确 SGEMM，
   因为它的 `ov_test_lm --cuda` 是 near-bit-exact 门禁。
2. 加载时把 q/k/v 拼成 `[H,4096]`、gate/up 拼成 `[H,6144]`；TF32/float 路径
   每层 GEMM 197 → 113 次。
3. int8 路径再把 qkv/gate-up/down/o 的权重和激活转成 **fp16**（fp32 累加），
   用 `cublasGemmEx` tensor core。**但 fp16 路径从合并权重里按 slice 拆回**
   3 次 qkv（N=2048/1024/1024）+ 2 次 gate/up（N=3072）：cuBLAS 对
   `[T,1024]×[1024,4096/6144]` 的 fp16 形状反而选到差算法（实测 11 ms/层 vs 1 ms/层）。
   所以最终 fp16 路径的调用数仍是 197 次，但每次都是小 N 的高效 tensor-core GEMM。
4. 融合 kernel：`rmsnorm_half_kernel`（写 fp16）、`silu_mul_half_kernel`（写 fp16）、
   qkv split；修复了第一版 rmsnorm_half 按“线程=row”写导致的非合并写（曾慢到 11 ms/层）。

实测（`ovtts --backend cuda`，int8 LM + fp32 AT，32 步，3.44 s 音频）：

| 指标 | 上轮 | 本轮 |
|---|---:|---:|
| load | 0.36 s | **0.35 s** |
| sample（32 步） | 0.36 s | **0.19–0.20 s** |
| decode | 0.16 s | **0.01–0.02 s**（GPU） |
| 端到端 RTF | 0.15 | **0.05–0.06** |
| 单步 forward+采样 | 11.3 ms | **5.8 ms** |

单步分阶段（`OV_CUDA_PROFILE=1`，28 层合计）：qkv **1.0 ms**、gate/up **1.2 ms**、
down **0.77 ms**、o_proj **0.70 ms**、attention **1.37 ms**、qk norm+rope **0.31 ms**、
silu **0.16 ms**、final norm+head **0.1 ms**。attention 仍是最大单项（约 24%）；把
`kAttnTPR` 从 8 调到 32（每行 32 线程、每线程 4 维）、`kAttnBK` 16→32 后从
1.9 → 1.37 ms；BM=64/16、BK=64 都没有更好收益，已回退。

对长序列（clone S=187）另加 **batched attention**：K/V 先按 GQA 展开成每 query head
一份，QK/PV 各用一次 `cublasSgemmStridedBatched`（TF32 tensor core），scores
`[B,NH,S,S]` 上做 bias+softmax；默认 `seq>=160` 走 batched、短序列仍走 fused，
`OV_ATTN_FUSED=1` / `OV_ATTN_BATCHED=1` 可强制。scores 的 bias+softmax 从
“block=row、256 线程”改成 **warp=row、4 warp/block、shuffle 归约**，
clone 的 attention 从 ~4.5 ms（fused）→ **~1.44 ms**，端到端 sample
0.32–0.36 → **0.27–0.34 s**、RTF **0.09–0.11**。

**decoder 也搬到了 GPU**：新增 `cuda_decoder.cu`，RVQ gather + project_out/fc2 用 cuBLAS，
conv1d 直接 kernel，ConvTranspose 用 CPU 同款“每个 tap 一次 GEMM + beta=1 + ldc=stride*cout”，
Snake/残差/拼接全是 device kernel；只有 codes 进、PCM 出。与 CPU decoder 对比：
T=25/50/87 的 `max|Δ|` 分别为 1.7e-6 / 2.3e-6 / 3.4e-6，cos=1.0。
decode 0.15 → **0.01–0.02 s**，端到端 RTF **0.05–0.06**。

**voice-clone encoder 也搬到了 GPU**（`cuda_encoder.cu`）：resample、7 层 conv stem、
GroupNorm/LayerNorm/GELU、位置卷积、12 层 transformer（cuBLAS QKV + per-head
cuBLAS attention + softmax）、semantic/acoustic conv 栈、RVQ encode 全部 device。
48 kHz→16 kHz 参考音频、50 帧输出下：CPU encoder **513 ms → GPU 8–9 ms（~60×）**，
RVQ codes 与 CPU **400/400 完全一致**；voice-clone 端到端（79 帧 / 3.16 s）
`sample=0.27–0.34 s, decode=0.01 s, RTF=0.09–0.11`；clone load 从 0.70 s 降到 **0.47–0.52 s**（conv 权重在 GPU 上转置，省掉 host 端 100M 级循环）。encoder 内部：
acoustic 2.9 ms、HuBERT 3.3 ms、stem 0.6 ms、semantic 0.8 ms、RVQ 0.4 ms。
关键改动是 conv1d 从 direct kernel 换成 **im2col + cuBLAS GEMM**（acoustic 144 → 2.9 ms），
以及 RVQ argmin 从“一线程一帧”改为 block-per-frame 归约（17 → 0.4 ms）。

数值：int8 门禁 argmax **92.38%**、mean **4.920e-2**、cos **0.99999964**（比 TF32 的
92.19% 略好）；fp32 CUDA 两个用例仍是 near-bit-exact。回退开关：
`OV_CUDA_FP16=0` → TF32 float GEMM；`OV_CUDA_TF32=0` → 精确 SGEMM；
`OV_CUDA_NO_MERGE=1` → 不合并权重；`OV_CUDA_PROFILE=1` → 分阶段打点。

代价：设备上同时保留 fp32 原始权重、合并副本和 fp16 副本，约 4.4 GB；4090 上不影响。
载入 +0.01 s。

---

## 1. 时间到底去哪了

### 1.1 CUDA 全链路（`ovtts --backend cuda`）

```
prompt: seq=109 target=86 (3.44 s audio)
load=0.90 s   sample=1.60 s (32 步)   decode=0.71 s   → RTF 0.67
```

长句时更极端（seq=325, target=281, 11.24 s 音频, 2 步）：
`sample=0.59 s`（即 **275 ms/forward**）→ 32 步下 LM 要 8.8 s，但 `decode=2.82 s`，
**CPU 声码器占了端到端 3.4 s 里的 83%**。

单步 47.5 ms 的拆解（用逐字复刻 `attention_kernel` 的探针测，见 §7 `probe.cu`）：

| 项目 | 时间 | 占比 | 备注 |
|---|---:|---:|---|
| `attention_kernel` × 28 层 | 33.6 ms | 70 % | 1.2 ms/层 |
| cuBLAS SGEMM × 197 次 | 9.3 ms | 20 % | 196 GFLOP → 21 TFLOPS |
| `cudaMalloc`/`cudaFree` × 15 对 | 1.5–2.1 ms | 4 % | 每次 forward 都重新分配 |
| logits D2H（16.7 MB @S=256） | 1.2–2.7 ms | 4 % | pageable，同步 |
| 其余（mask 上传、permute、host 侧 log_softmax） | ~2 ms | 4 % | |

S=256 时：attention **148 ms** / forward，GEMM 14 ms——比 S=109 时更失衡，
因为 kernel 是 O(S²·D) 的纯串行实现，而 block 数固定为 32。

### 1.2 CPU 解码器（86 帧）

探针直接给 `AcousticDecoder::decode` 的各段打点：

| 阶段 | 时间 | 占比 | 纯算术应该多少 |
|---|---:|---:|---|
| `conv_k7`（im2col + GEMM + bias） | 277 ms | 38 % | ~63 ms |
| `conv_transpose1d` | 242 ms | 33 % | ~7 ms |
| `snake` | 144 ms | 20 % | — |
| `conv_k1` | 36 ms | 5 % | ~9 ms |
| RVQ / fc2 / conv1 / 拷贝 / 残差 | 33 ms | 4 % | |

整个解码器 66.7 GFLOP，按实测 fp32 GEMM 553 GFLOPS 应该 **~120 ms**，实际 723 ms——
**只有 17% 的时间在算乘法**。

### 1.3 CPU 编码器（voice clone 用）

```
encoder 1190 ms (3.12 s 参考音, 78 帧)
   hubert 塔（stem+proj+posconv+12 层+mean） 337 ms
     └ stem 152 ms（其中 group_norm 43 ms、gelu(erf) 12 ms）
     └ 12 层 transformer 163 ms   ← 这部分是健康的（117 GFLOP → 718 GFLOPS）
   semantic conv stack   44 ms
   DAC 声学编码器       833 ms   ← 70%，同样死于 im2col
   RVQ encode            10 ms
```

`acoustic_encode` 的算术量只有 ~10–15 GFLOP，833 ms 意味着 **15 GFLOPS，1.7% 峰值**。

### 1.4 CPU LM（int8，S=109）

内建 `OV_PROFILE=1` 输出：

```
total=0.358 s | embed 0.1% qkv 21.5% attn 5.6% oproj 11.1% mlp 58.3% head 3.5%
```

GEMM 部分 = 196 GFLOP / 224 ms ≈ 875 GFLOPS（与微基准 873 一致，很好）。
剩下 **134 ms（37%）是 GEMM 以外**：`silu` 35 ms、`M%8` 补齐 ~30 ms、
每层两次 `memcpy(x→xn)` 约 10 ms、标量残差相加 5 ms、rmsnorm/rope/attention 约 55 ms。

> 第二轮 VNNI 后同一形状变成 `total=0.104–0.110 s`：GEMM ≈ 82%
> （qkv 20.8% + oproj 12.0% + mlp 46.0% + head 3.3%），attention 17%，
> embed 0.3%。上面的 0.358 s/875 GFLOPS 是 VNNI 之前的原始分析，保留作对照。

---

## 2. 逐项改进建议

### P0-1　重写 CUDA attention（收益 28 ms/forward，GPU 端 2.4×）

`cuda_lm.cu:119` 的问题（每一个都足以拖慢一个数量级）：

- **只有 32 个 block**（`batch*NH`），4090 有 128 个 SM → 75% 的 SM 空转；
  每 block 128 线程 = 1/8 的最大占用。
- **QK 内积访问完全不合并**：`for (d) acc += qrow[d] * krow[d]`，同一 warp 的 32 个线程
  读的是 32 个不同 K 行（间隔 `KVD*4 = 4096 B`）→ 每个 d 触发 32 条独立的 32 B 事务，
  L1 无法利用。
- **PV 只有 128 个线程干活**（`if (tid < D)`），其余线程 `__syncthreads()` 干等。
- 每个输出行做一次 5 级 `__syncthreads()` 的树形归约；`sc[j] * inv` 在 PV 里被重复乘。
- S=256 时 score 矩阵用 `smem` 固定 1024 个 float + `red[256]`，S>1024 直接越界。

**修法 A（一天，稳妥）**：整体换成 cuBLAS strided-batched GEMM + 独立 softmax kernel。
把同一 KV head 的 `group = NH/NKV = 2` 个 Q head 合到一个 batch 里，
用 `cublasSgemmStridedBatched` 一次算 8 个 KV head 的 score，再算 PV。
探针里 batched 版本单层 1.20 → 0.20 ms（纯 GPU 时间），但**整机实测只到 10 ms/forward**
（33.6 → 10）：每次 forward 有 48 次 cuBLAS 调用/层 × 28 层 = 1344 次，
每次 ~4-7 µs 的调用与启动开销把收益吃掉了。把 score/PV 合批到 3 段（每层 33 次调用）
实测反而更慢（29.3 ms/step），说明瓶颈是"每层几十个微小 kernel"本身。
**因此真正要做的不是换 API，而是写一个 fused attention kernel**（每层 1 次启动），
这一点本次没有完成，见 §5 状态列。

**修法 B（已采用）**：flash-attention 风格的 fused kernel，每层 1 次启动：

- grid = (⌈S/32⌉, B·NH)，block = 256 线程（32 行 × 8 线程/行）；
- K/V 按 16 个 key 一块进 shared，q/k 用奇数 stride 129 避免 bank conflict，
  v 用 132 保持 float4 对齐；
- 在线 softmax（running max/sum + 输出行重缩放），score 矩阵从不落到 global；
- 每线程持有 16 个输出 dim（4 个 float4 寄存器累加器）；
- QK 内积用 4 条独立累加链避开 FMA 4 cycle 延迟（block 只有 8 个 warp 可藏）。

实测：**33.6 → 2.0 ms/forward**（S=109，17×）；S=325 时约 10 ms（vs cuBLAS 版 12.7 ms）。
cuBLAS 版本保留为 `head_dim != 128` 的回退，并可用 `OV_ATTN_CUBLAS=1` 强制走它做 A/B。

顺带要修的：
- `rope_kernel` 每个元素都算一次 `powf/cosf/sinf`（T×heads×half = 22 万次/层/次），
  应该**一次算好 `seq × half` 的 cos/sin 表**放常量显存（CPU 侧的 `rope_f32` 已经这么做了）。
- `mask_to_bias_kernel` 每次 forward 都重跑：`attn_mask` 在 32 步里完全不变，应该只在
  变化时上传一次。
- `embed_kernel` 里 `__syncthreads()` 在 codebook 循环内，8 个 codebook 串行 + 全 block 同步，
  改成每个 codebook 一个 warp 或直接做一次 gather-sum。

### P0-2　干掉 im2col 的每次分配（解码器 -277 ms，编码器 -900 ms）

`conv_cpu.cpp:67` 和 `ops_cpu.cpp:424`：

```cpp
std::vector<float> col(static_cast<size_t>(t_out) * CK);   // 每次调用都申请
```

最大的一次是 82560×224×4 = **74 MB**，实测光是 `vector<float>(122880×224)` 的
mmap + 缺页 = **26 ms**（`[im2col] ... 110.1 MB per conv call: 26.3 ms`）。
解码器一次调用要过 ~406 MB，编码器的声学编码要过 ~700 MB，全是纯浪费。
而且 im2col 的填充循环是标量 + 每个元素一个边界判断。

**修法（按性价比排序）**：

1. **stride==1 的卷积改成"按 tap 累加的 GEMM"**，彻底不要 im2col：
   ```
   y[t, :] += W_kk · x[t - pad + kk·dil, :]      kk = 0..k-1
   ```
   权重按 `[k][cin][cout]` 排（`Conv1d::prepare` 现在排的是 `[cin][k][cout]`，改一行），
   然后每个 tap 一次 `gemm_acc(T, cout, cin, x + off*cin, W_kk, y + off*cout, /*ldc=*/cout, beta=1)`。
   优点：x 顺序读、y 原地累加、零分配、零边界判断。
   缺点：y 被读写 k 次（最大 7×10.6 MB），但比 im2col 的 write+read 少一半，且没有缺页。
2. **带 stride 的卷积**保留 im2col，但用**复用的 256 行分块缓冲**（`thread_local` 或
   workspace arena），block 内做 GEMM。256×CK×4 = 1.2 MB（L2 内），缺页只发生一次。
3. GEMM 接口加 `ldc` 和 `beta`（见 P0-3），否则第 1 条做不干净。

这一条同时改善解码器和编码器，是**整个仓库单点收益最大的一条**（约 -1.1 s）。

### P0-3　`conv_transpose1d_f32` 改成"相位分解 GEMM"（242 → ~15 ms）

现在的实现是三层循环里调 `axpy`（`conv_cpu.cpp:88`）：

```cpp
for (i) for (t) for (kk) axpy(y + to*cout, wi + kk*cout, xv, cout);
```

对 block 0（cin=1024, k=16）就是 **140 万次**、每次 512 float 的 `axpy` 调用，纯标量 +
140 万次函数调用开销，实测 242 ms，而算术量只有 4 GFLOP。

转置卷积的每个 tap 落在固定的输出相位上：`t_out = t*stride - pad + kk`，所以按
`p = kk mod stride` 分组：

```
for p in 0..stride-1:
  for j in 0..k/stride-1:
      kk = p + j*stride
      Y[(t+j)*stride + p - pad, :] += X[t, :] · W[kk]        // 一次 GEMM，ldc = stride*cout
```

一共 `k` 次 GEMM（16 次 [86,1024]×[1024,512]），权重按 `[k][cin][cout]` 排即可直接切片。
`ldc = stride*cout` 让结果直接写进目标位置，**不需要 im2col、不需要临时缓冲、不需要 scatter**。
预计 15 ms 以内（4 GFLOP @ 300 GFLOPS）。

> 注意 `ConvT1d::prepare` 现在是 `[cin, k, cout]`，需要改成 `[k, cin, cout]`。

### P0-4　`snake_f32` 向量化（144 → 11~25 ms）

```cpp
for (int c = 0; c < channels; ++c)
  for (int i = 0; i < t; ++i) { float& v = x[i*channels + c]; ... std::sin(a*v) ... }
```

两个问题：外层按 channel、内层跨步 `channels*4` 字节（每次都是新 cache line），
以及每元素一次标量 `sinf`。

**安全版（不改变数值，6 行改动）**：交换循环 + OpenMP。实测 **4.2–11×**：

```cpp
void snake_f32(float* x, const float* alpha, int channels, int t) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < t; ++i) {
    float* row = x + static_cast<size_t>(i) * channels;
    for (int c = 0; c < channels; ++c) {
      const float a = alpha[c];
      const float s = std::sin(a * row[c]);
      row[c] += s * s / a;
    }
  }
}
```

**极限版（AVX2 一次 8 通道 + 多项式 sin）**：`sin²(ax) = (1-cos 2ax)/2` 或直接
`s = sin(ax)`（Cody-Waite 归约到 [-π/2, π/2] + 11 次多项式），实测 **11–16×**，
但和 glibc `sinf` 有 `max|Δ| = 1.9e-6` 的差异——需要重跑 §4 的波形门禁（现门限 3.5e-6）。
建议先上安全版（免费、零风险），极限版作为可选编译路径。

### P1-1　CPU LM 的"非 GEMM 37%"

逐条，按收益：

| 问题 | 位置 | 实测 | 修法 |
|---|---|---|---|
| `silu_mul_avx2` 每 8 个元素回到标量 `std::exp` | `ops_cpu.cpp:49` | 1.25 ms/层 → **35 ms/forward** | 直接用同文件已有的 `exp256_ps`：1.25 → **0.17 ms**（7.2×，`max|Δ|=0`） |
| `M%8 != 0` 走补齐路径：整块 memcpy A + memcpy C | `gemm.cpp:227` | M=218 时 1.536 ms vs M=224 的 1.298 ms（**+18%**），×224 次 ≈ **30 ms** | micro-kernel 尾部按行写回（`for r < M-i0` 才 store），不要整块补 |
| 每层 `memcpy(x→xn)` 两次 + 标量 `x[i] += proj[i]` | `lm.cpp:134,174,178,184` | ~15 ms/forward | rmsnorm 入参改成 (dst, src)，残差加进 GEMM epilogue 或用 AVX2 axpy |
| `rope_f32` 每层重算 cos/sin 表 | `ops_cpu.cpp:297` | 4 ms/forward | 按 (rows, pos[]) 缓存，28 层共用；旋转部分可 8 宽向量化 |
| attention 每 head 重拷 K/V，PV 反复 load/store | `ops_cpu.cpp:157` | 20 ms/forward（profile attn 5.6%） | 同 KV head 的两个 Q head 合批；PV 用 32/64 宽的寄存器累加块 |

### P1-2　CUDA 侧的资源管理（约 -5 ms/forward + 更少抖动）

- `forward()` 里 **15 次 `cudaMalloc` + 15 次 `cudaFree`**（`cuda_lm.cu:329-343`，`release()` 在
  :423）。`cudaMalloc` 会隐式同步设备，实测 1.5–2.1 ms/forward。
  → 构造时按 `(batch, seq)` 上限一次性分配 workspace，forward 只做指针切片。
- `d_ids / d_mask / d_pos / d_amask_raw` 每次 forward 都从 host 重传；32 步里只有
  `d_ids` 会变（每步填 k 个位置），mask/pos **一次都不变** → 只传 delta 或缓存。
- logits 回传是 **16.7 MB 同步 pageable copy + host 侧 permute + host 侧 log_softmax**。
  32 步 = 535 MB PCIe。→ 把 `log_softmax + guidance mix + argmax + top-k 选择` 做成
  一个 device kernel，每步只回传 `C*T*2` 个数（约 5 KB），顺便省掉 host 上 2M 次 `exp`。
- 没有 stream、没有 `cudaGraph`：一次 forward 约 410 次 kernel/GEMM 启动（28 层 × ~15），
  32 步 = 1.3 万次。`cudaGraph` 能把启动开销从 ~3 µs/次压到 ~0.5 µs/次。
- `Available()` 用 `cudaGetDeviceCount` 就够了，但 `LmRunnerCreate` 里的
  `cudaGetDeviceProperties` 在大 batch 首次调用上有 100 ms 级开销，建议懒加载。

### P1-3　GPU 上的 GEMM：先合并，再谈精度

实测（S=109 / S=256，一次 forward 的全部 197 次 GEMM）：

| 路径 | S=109 | S=256 |
|---|---:|---:|
| `cublasSgemm` fp32 | **9.28 ms** | 14.07 ms |
| TF32 (`CUBLAS_TF32_TENSOR_OP_MATH`) | 13.42 ms | **10.54 ms** |
| fp16 权重 + fp16 激活 + fp32 累加 | 14.68 ms | **8.08 ms** |

> **第五轮更新**：上面的 197 次未合并 GEMM 结论已被 §0.5 的超前；合并 qkv/gate-up
> 后，TF32 与 fp16 在 S=109 都明显快于逐矩阵 SGEMM（当前单步 6.5 ms）。下表保留作历史对比。

三个结论：

1. **S 小时 fp32 最快**——因为 M=218 的形状下 GEMM 是 tile/启动受限，不是算力受限，
   tensor core 的 tile 反而更不匹配。所以"无脑开 TF32"在这个 workload 上是错的，
   要**按 M 分派**（M ≥ 512 用 TF32/fp16）。
2. 当前 197 次 GEMM 里，**q/k/v 是三个 [T,1024]×[1024,{2048,1024,1024}]，
   gate/up 是两个 [T,1024]×[1024,3072]**。把它们在**加载时合并成
   `[1024, 4096]` 和 `[1024, 6144]` 各一次 GEMM**（权重本来就是设备驻留，拷贝一次即可），
   启动次数 197 → 113，N 变大后 tile 效率明显更好。CPU 侧同理（少 3×28 次 OpenMP 区域）。
3. 权重 2.45 GB 每次 forward 都要过一遍 HBM（32 步 = 78 GB）。fp16 权重（1.2 GB）
   在 S≥256 时实测 1.7×；int8 权重（0.64 GB）留在设备上 + 反量化在 epilogue 里做，
   理论收益更大（4× 带宽），代价是要处理 per-column scale——`linear_s8` 的语义已经现成。
   `DType::F16` 在 `ov_types.h` / `convert_weights.py` 里**已经定义但从未被使用**，
   `convert_weights.py --mode` 只有 `fp32|int8|both`，加一个 `fp16` 只要十几行。

### P1-4　混合架构 CPU 的线程调度

实测 `linear_s8`（M=512, N=3072, K=1024）随线程数变化：

```
 1 线程  120 GFLOPS      8 线程  906
 2       230            12 线程  491   ← 断崖
 4       492            16      594
                        32     1179
```

8 P-core + 16 E-core + `#pragma omp parallel for ... schedule(static)`（`gemm.cpp:174`）：
静态切块让 E-core 线程拿到同样多的 tile，而它慢 2.5×，整条 region 等最慢的那个。
12 线程时正好 4 个线程落到 E-core，于是比 8 线程还慢 1.8×。
端到端 `ovtts` 上 8→32 线程只有 1.7× 加速（2.60 s → 1.52 s），也印证了这一点。

修法（按顺序）：
1. `schedule(dynamic, 1)` 或 `schedule(guided)`，或直接用一个 atomic 计数器取 tile；
2. tile 粒度别太大：现在 `kMr=8 × kJb=64` 的 tile 在 M=218 时只有 27×48 = 1296 个，
   32 线程够用；但 small-N 的 attention/conv 调用（N=32/64）会退化成 1 列块；
3. 小 GEMM（M·N·K < 1e8）限制线程数（`omp parallel for num_threads(min(nthreads, tiles))`），
   避免 32 线程抢 8 个 tile；
4. 可选：`OMP_PLACES=cores OMP_PROC_BIND=close` + 逻辑 P/E 分组给不同 tile 大小。

### P2-1　AVX-VNNI 点积（已实施）

`/proc/cpuinfo` 有 `avx_vnni`（Raptor Lake 的 256-bit VNNI）。旧 int8 路径是
**每 k 步 `vcvtepi8_epi32` 再 `vfmadd`**，即"反量化成 float 再算"，MAC 吞吐和 fp32 一样，
只是省了带宽。本轮改成 `vpdpbusd`（u8×s8 → s32，一条指令 4 个 MAC/lane）之后：

| 项目 | 旧 int8 | VNNI | 变化 |
|---|---:|---:|---:|
| int8 GEMM M=218 N=3072 K=1024 | 1339 GFLOPS | **2982–3624 GFLOPS** | 2.3–2.7× |
| 同上 M=512 | 1348 GFLOPS | **3130–3809 GFLOPS** | 2.3–2.8× |
| LM forward B=2 S=109 | 0.202–0.205 s | **0.104–0.110 s** | 1.9× |
| 32 步 auto（sample） | 6.92–7.01 s | **3.81–3.82 s** | -43~45% |
| 32 步 auto（RTF） | 2.05–2.08 | **1.15** | 1.7–1.8× |
| 16 步 auto（sample） | 3.53–3.56 s | **1.86–1.91 s** | -43~48% |
| int8 门禁 | argmax 91.80%、mean 4.914e-2、cos 0.99999964 | **argmax 92.19%、mean 6.04e-2、cos 0.99999942** | 通过 |

> 说明：旧内核本轮同机复测为 1339 GFLOPS，本文早前记录的是 873 GFLOPS。端到端基线
> （2.05 vs 2.07）一致，微基准的差异可能来自当时的构建/频率状态；上表所有数字都是
> 同一轮、`OMP_NUM_THREADS=32`、跑两次取第二次。

**实现要点**（细节见 `weight-format.md` §1a 与 `ov_gemm.h`）：

1. **权重布局**：`tools/convert_weights.py --int8-layout nk` 把线性层存成
   `[N][K]`、`layout=0`、per-row scale（`*-int8-nk.ovw`）。默认仍是 `kn`
   （`layout=1`），旧文件继续能加载，CUDA 路径不受影响。
2. **加载时预打包**：每个 8 列块 × 4-k 组存 32 字节，
   `tile[j*4+l] = W[n0+j][4g+l]`（lane j = 列，dword 内 4 字节 = 连续 k）。
   这样一次 `loadu(tile)` + 8 次 `vpbroadcastd(A)` + 8 次 `vpdpbusd`
   完成 8 行 × 8 列 × 4 k = 256 MAC，累加器天然是 8 个 int32 lane，不需要水平求和。
   预打包在 `make_linear` 里做一次，`pa + wsum` 占 440 MB + 55 MB。
3. **激活量化**：每行每 32 个 k 一个 scale（`amax/127`），
   `a_u8 = clamp(round(A/ascale), -127, 127) + 128`。`+128` 只是把有符号激活搬到
   `vpdpbusd` 的无符号域；预存每块每列 `wsum = Σq`，epilogue 里先
   `acc - 128*wsum`，再乘 `ascale_block * wscale_n`。per-32 是实测必须的：
   最先试的 per-row 单 scale 只有 argmax 76.4%、cos 0.9999944，过不了门禁。
4. **micro-kernel 取 4 行 × 8 列，而不是 8×8**：8×8 版本每个 32-k 块都要把
   int32 累加器转 float 并读改写 C（8 行 × load/add/store），实测只有 2290 GFLOPS；
   4×8 用 4 个 int32 + 4 个 float 累加器（10 个 ymm 以内），float 累加跨块保留、
   只在 K 末尾乘一次列 scale，实测 2982–3624 GFLOPS。代价只是 B tile 的 L1 读翻倍。
5. **尾块**：M 用行别名 + 掩码写回；N 用 `maskload/maskstore`（打包时把 N 补到 8、
   K 补到 32）；K 的零填充与 `wsum` 只统计有效 k，三者都有随机矩阵用例。
6. **分派**：`__builtin_cpu_supports("avxvnni")`（位掩码，判断 `!= 0`）+ avx2/fma；
   `__attribute__((target("avx2,fma,avxvnni")))`。`OV_CPU_NO_VNNI=1` 在加载时关闭
   预打包并回落到 `linear_s8_nk` 的 AVX2 逐列点积（同一份 `[N,K]` 权重，数值与旧
   int8 路径一致：argmax 91.80%、mean 4.914e-2）。
7. **不碰 CUDA**：新文件是 CPU/VNNI 专用；`ovtts --backend cuda` / `ov_test_lm --cuda`
   遇到 `nk` 权重会直接报错，提示改用 `omnivoice-lm-int8.ovw`。

**代价 / 未做**：
- 载入从 0.09 s 变成 0.41–0.45 s（一次性重排 ~440 MB）；RSS 多 ~495 MB。
  端到端 RTF 不算 load，但短请求的服务化场景值得后续把打包并行化或缓存。
- 激活量化仍是 lossy：int8 logits 的 mean|Δ| 从 4.914e-2 升到 6.04e-2，
  argmax 反而从 91.80% 提到 92.19%（per-32 scale 抵消了部分噪声）；门禁仍有余量。
- 单次 forward 里 GEMM 占比从 91% 降到 ~82%，**attention 变成 17%**（约 18 ms），
  是下一步 CPU 侧的大头；本轮按约束没有动它。

### P2-2　CPU GEMM micro-kernel（1.5–2× 上限）

当前 8×8 kernel（`gemm.cpp:65`）每个 k 迭代：1 次 W load + 8 次 `broadcast_ss` + 8 FMA
= 9 次 load / 8 FMA，**load 端口比 FMA 端口先饱和**（2 load/cycle → 4.5 cycle，FMA 只要 4）。
换成 6×16（6 行 × 2 个 ymm 列块）：每个 k = 2 次 W load + 6 次 broadcast + 12 FMA
= 8 load / 12 FMA（4 cycle load vs 6 cycle FMA，变成计算受限），寄存器 6+2+2+... 够用。
单线程实测 123 GFLOPS / FMA 峰值 141 GFLOPS = 68%，说明 8×8 已经榨得不错，
**天花板大约 1.5×**，优先级低于上面所有项。

### P2-3　编码器（voice clone）的零碎项

| 问题 | 位置 | 实测 | 修法 |
|---|---|---|---|
| `group_norm_channels_f32` 跨步 + 双精度 + 3 遍 | `ops_cpu.cpp:393` | **43 ms**（stem 的 28%） | 通道维向量化 + 两遍 Welford；或转置成 SoA 处理 |
| `gelu_f32` 用 `std::erf` | `ops_cpu.cpp:388` | **12.4 ms** | tanh 近似（误差 3e-4 需评估）或 AVX2 erf 多项式；`-ffast-math` 下 GCC 会自动向量化 libmvec |
| 两套卷积实现并存：`Conv1d::prepare` 缓存了转置权重，但编码器走 `conv1d_generic_f32` **每次调用重新转置权重** | `ops_cpu.cpp:416` | 编码器里 ~60 次调用 | 删掉 generic 版本，统一用 `Conv1d` + prepare |
| pos-conv `groups=16` 被拆成 16 次 N=48 的 GEMM | `ops_cpu.cpp:427` | 22 ms | N=48 的 GEMM 效率极低；改成 `[T, 48]×[48, 768]` 的分块对角形式，或直接按 group 拼成一个 batched GEMM |
| RVQ 最近邻：`T×1024×64` 三重标量循环 + 每个 quantizer 重算 `norm2` | `at_encoder.cpp:582` | 10 ms | 用 `linear_f32(T, 1024, 64, x64, embedT, dots)`（码本转置一次缓存），再向量化 argmax |
| 每次 `synthesize` 都重跑编码器 | `tts.cpp:124` | 1.19 s/次 | 按 (ref_audio 路径 + 文本) 缓存 codes；服务化场景必做 |

---

## 3. 算法层（比 kernel 更大，但要动语义）

现在 **32 步 × B=2 = 64 次完整 forward**，这是端到端成本的主体。

1. **前缀 KV cache**（省 ~10%）：cond 行的前缀（style + text + ref codes，位置 0..S-T-1）
   在 32 步里完全不变，且全注意力 mask 下它的 hidden state 只依赖自己。
   把 28 层的 prefix K/V 缓存下来，后续步只跑 target 段 → 前缀部分的 qkv/mlp 全省。
   实现上等价于"chunked prefill"，是 vLLM/SGLang 的标准做法。
2. **uncond 行只算 T 个位置**（省 ~10%）：uncond 的 mask 只在前 T×T 块内为真
   （`tts.cpp:194-198`），S-T..S 那些行只有对角，结果全被丢弃，却和 cond 行一起算了。
   拆成两次 forward（cond: S，uncond: T）即可，或让 GEMM 的 M 分段。
3. **audio head 只算 target 位置**（S=109 时省 2~4%）：`logits_flat_` 是 `[T, 8200]`，
   只有 target 段被采样用；head 是 3.7 GFLOP/forward。
4. **采样搬到 device**（见 P1-2）：per step 2M 次 `exp` + 16.7 MB 回传。
5. **步数**：`num_steps=32` 是质量/速度的主要旋钮；如果模型支持，step distillation
   或 guide 只在前若干步开（CFG 只在早期步有效）能直接砍一半 forward。
6. **CFG 的行可以合并**：cond/uncond 共用前缀 → 只有 target 段需要两次前向；
   配合 1、2 条可以把 64 次 forward 降到 ~30 次等效。

---

## 4. 正确性 / 工程性（顺带发现）

- **attention 全屏蔽 query 行**：`kAttnRB=2` 的快速路径原来用一次
  `kAttnRB*D` 的 `memset`，假设两行在内存里连续；LM 的 `out_stride=QD > D` 时
  它只清零第一行、把第二行留在未初始化状态（还踩了行间 padding）。当前 tts 的 mask
  没有全屏蔽行，所以门禁未暴露；已改成逐行 memset，并在 `ov_test_gemm` 里加了
  `out_stride=2*D`、连续两行全屏蔽的回归用例（`attention fully-masked`）。
- **VNNI 的 int8 门禁**：`argmax 472/512 = 92.19%`、`mean|Δ| = 6.04e-2`、`cos = 0.99999942`
  （旧 int8：91.80% / 4.914e-2 / 0.99999964）。`ov_test_gemm` 新增 `[N][K]` 与
  packed VNNI 路径用例（含 M/N/K 尾块），VNNI 用例用"每 32-k 块最大值为 63.5、量化步长
  恰好 0.5"的输入，使激活量化无损失，任何偏差都归因于 tile/zero-point/scale 逻辑。
- **数值门禁要跟着改**：`snake` 的多项式版、`silu` 的向量化版、VNNI 版的 int8 都要重跑
  `ctest` + 波形门禁（`max|Δ| ≤ 3.5e-6`）。我在报告里给的 `silu` 版实测 `max|Δ| = 0`，
  `snake` 安全版数值完全不变，`snake` 多项式版 `1.9e-6`。
- `attn_mask` 语义：`tts.cpp:182` 的 cond 行是全 1（含前缀看 target 的部分），
  和 `docs/cpp-engine.md` 里"加性 bias、-1e30 屏蔽"一致；但 `attention_head_avx2` 里
  `brow` 是按 `bias + s*S` 取的，没问题——只是提醒：换成 batched GEMM 后 bias 需要转置，
  这是最容易写错的地方（我的探针里就踩了）。
- `LM::workspace_bytes()` 定义了但没人用；旧的 `q_head_/v_head_/kt_/scores_` 已删除。
- `linear_f32/linear_s8/linear_s8_nk` 在 `M<=0` 时静默返回，调用方以为算完了。
- nk 权重在加载时预打包（`make_linear`），RSS 多 ~495 MB；`Mat` 现在持有 `std::vector`，
  如果有人**拷贝** `Mat`（目前没有）会深拷贝打包权重，应保持 move/引用语义。
- `attention_head_avx2` 每次调用都 `resize` thread_local 缓冲（32 线程 × 2 × S×D×4 B），
  S=1024 时是 32 MB，值得改成固定 workspace。
- CUDA 侧 `LmRunnerCreate` 在构造里做 `cudaGetDeviceProperties` + 逐 tensor 反量化上传
  （int8 权重会被在 host 上展开成 fp32，峰值多占 2.45 GB 内存，load 0.87 s）。
  保持 int8 设备驻留可以同时省内存、省 PCIe、省 0.87 s 启动。
- `enable_flush_to_zero()` 放在 `attention_head_avx2` 里，每个 head 每次调用都做一次
  MXCSR 读改写（1792 次/forward）。无害但应该提到 `forward()` 入口（已经有一次
  `#pragma omp parallel { enable_flush_to_zero(); }`），删掉内层那次。

---

## 5. 优先级表

| # | 改动 | 预计收益 | 工作量 | 风险 | 状态 |
|---|---|---|---|---|---|
| 1 | CUDA attention → **fused kernel** | GPU forward **2.4×@S=109 / 7.7×@S=256** | 1–3 天 | 低（有 ONNX 参考可对） | ✅ 已做：33.6 → **2.0 ms/forward**（17×），cuBLAS 版降级为回退路径 |
| 2 | 卷积按 tap / 分块 im2col，去掉每次 `vector` 申请 | 解码器 **-277 ms**、编码器 **-900 ms** | 2–3 天 | 低 | ✅ 已做（分块 im2col） |
| 3 | `conv_transpose1d` 相位分解 GEMM | 解码器 **-230 ms** | 1 天 | 低（需 GEMM 加 ldc/beta） | ✅ 已做 |
| 4 | `snake_f32` 换循环序 + OpenMP | 解码器 **-120 ms**（安全版） | 1 小时 | 零 | ✅ 已做（数值不变） |
| 5 | `silu_mul` 用 `exp256_ps` | CPU LM **-30 ms/forward** | 1 小时 | 零（实测 max\|Δ\|=0） | ✅ 已做 |
| 6 | GEMM 去掉 `M%8` 补齐路径 | CPU LM **-30 ms/forward** | 半天 | 低 | ✅ 已做 |
| 7 | CUDA workspace 复用 + 设备端反量化 | GPU forward **-2 ms**、载入 **-0.55 s** | 1 天 | 低 | ✅ 已做（device 侧采样未做） |
| 8 | qkv / gate-up 权重合并成单个 GEMM | GPU -1~2 ms、CPU -5% | 半天 | 低 | ✅ 已做：TF32 路径 197→113 次；FP16 路径按 slice 拆回 197 次（大 N half 反而慢），但 tensor core 更快；CPU 见 #18 |
| 9 | CPU 线程调度改 dynamic | CPU LM +10~30% | 半天 | 低 | ✅ 已做：`schedule(dynamic,4)`，实测 +21% |
| 10 | 前缀 KV cache + uncond 行瘦身 | LM **-20%**（算法） | 3–5 天 | 中（要重跑 WER/CER） | 🟡 uncond 行瘦身 **已做**（ragged CFG，见 #17）；前缀 KV cache 仍未做 |
| 11 | int8 → AVX-VNNI 点积 | CPU LM 再 **1.5–2×** | 3–5 天 | 中（数值门禁） | ✅ 已做：GEMM **1339 → 2982–3624 GFLOPS**、forward **0.205 → 0.104–0.110 s**、32 步 RTF **2.05 → 1.15**；per-32 激活量化，argmax 92.19% |
| 12 | 权重 fp16 化（`.ovw` + cublasGemmEx） | 大 S 时 GPU GEMM 1.7× | 2 天 | 中 | ✅ int8 CUDA 路径已做（qkv/gate-up/down/o，fp32 累加）；fp32 checkpoint 仍走精确 SGEMM |
| 13 | GEMM 6×16 micro-kernel | CPU 1.3–1.5× | 3 天 | 中 | ⬜ 未做（int8 已由 VNNI 4×8 取代；只剩 fp32，收益低于第 14 条） |
| 14 | CPU attention query 分块（`kAttnRB=2`） | 聚合 223 → 271 GFLOPS；forward -~0.2 s/32 步 | 1 天 | 低 | ✅ 已做（2 行共享 K/V；后升级为 4 行，见 #20；全屏蔽行 bug 已修 + 回归用例） |
| 15 | 复用 q/k/v 与 gate/up 的激活量化 + 量化并行 | forward -~14 ms（池化 -3.4 ms、并行 -11.6 ms） | 半天 | 低 | ✅ 已做（7→4 次/层，83→40 µs；池化路径 bit-identical） |
| 16 | VNNI 权重预打包并行化 / 去掉多余 memset | 载入 0.43 → 0.30 s | 半天 | 低 | ✅ 已做（k-block 并行，N/K 对齐时不再 memset）；仍可缓存到 ~0.15 s |
| 17 | cond/uncond 单次 packed 前向（ragged batch） | 行数 **218→195**、权重只流一遍 | 2–3 天 | 中（采样布局/数值要验证） | ✅ 已做：32 步 sample **2.91–2.96 → 2.62–2.65 s**、RTF **0.89–0.91 → 0.81–0.82**；cond 逐位不变，uncond mean 3.7e-2 / cos 0.99999982 / argmax 95.1%；`OV_CPU_EXACT_CFG=1` 可逐位回退 |
| 18 | qkv / gate-up 合并成单个 VNNI GEMM | CPU forward -1~3 ms（少 84 次 region + 更好的 tile 数） | 2–3 天 | 中（输出布局/视图） | ✅ 已做：sliced kernel，逻辑 N=4096/6144，写回独立 C；sample 2.51–2.57 → 2.41–2.50 s；`s8nk sliced` 逐字节用例；载入 +0.03 s |
| 19 | 融合 RMSNorm + 激活量化 | forward -1~2 ms（省掉 xn 物化与一遍读写） | 1–2 天 | 低 | ✅ 已做：`rmsnorm_quantize_f32`，30 µs vs 64–91 µs，aq/asc 逐字节相同 |
| 20 | attention 再优化：4 行 query（`kAttnRB=4`）/ 2 head 共享 K/V / 去掉 K/V 拷贝 | 当前 attn ~13 ms（15–17%） | 2–3 天 | 中（寄存器/布局） | ✅ 4 行已做：sample -~0.1 s/32 步、RTF 0.81–0.82 → 0.78–0.80；`OV_ATTN_RB2=1` 可回退。**试过直接读 K/V（去拷贝）反而略慢，已回退**；2 head 共享仍未做 |
| 21 | VNNI K-block 64 / 预取 / dynamic chunk 调参 | GEMM -3~8% | 1–2 天 | 中（精度要重跑门禁） | 🟡 已试：K-block 64 **更慢且 argmax 降到 90.82%**、W 预取更慢、dynamic 1/2/8 无明显收益；均回退，保持 32-block + dynamic(4) |
| 22 | audio head 只算 target 行 | head ~3.5 ms，按 172/195 行算可省 ~0.4 ms | 半天 | 低 | 🟡 已试：拆成两段 M=86 各读一遍 33.6 MB 权重，实测 2.08–2.15 s，无稳定收益，回退 |
| 24 | residual add 融进 o_proj/down VNNI epilogue | sample -~1%，oproj 11.0→9.5–10.2% | 半天 | 低 | ✅ 已做（bit-identical；`s8nk add` 用例） |
| 27 | CUDA decoder（RVQ/conv1d/ConvT/Snake 全 device） | decode 0.15 → **0.01–0.02 s**，RTF 0.09–0.11 → **0.05–0.06** | 1 天 | 中（convT 长度/布局） | ✅ 已做；与 CPU decoder max\|Δ\|=1.7–3.4e-6 |
| 28 | CUDA encoder（HuBERT/semantic/acoustic/RVQ 全 device） | encoder 513 → **8–9 ms**，voice-clone RTF 0.65 → **0.09–0.11** | 2 天 | 高（12 层 transformer + conv） | ✅ 已做；RVQ codes 与 CPU **400/400** 完全一致；conv 走 im2col+cuBLAS，权重 prep 在 GPU |
| 29 | 长序列 batched attention + warp softmax | clone attention 4.5 → **1.44 ms**，sample 0.32–0.36 → **0.27–0.34 s** | 1 天 | 中 | ✅ 已做；`seq>=160` 启用，短序列仍 fused |
| 30 | GPU 融合 per-head RMSNorm + RoPE | qk prep 0.4 → **0.31 ms** | 半天 | 低 | ✅ 已做；单 kernel 完成归一化+旋转，数值不变 |
| 25 | 融合 q/k RMSNorm + RoPE（1 趟、按 row 并行） | a_prep 5–6 → 1 ms，sample -~0.2 s | 半天 | 低 | ✅ 已做（单层 100→35–50 µs；`qk norm+rope` 逐字节用例） |
| 26 | 融合 silu(gate)*up + 激活量化 | 单层 183–241 → 62 µs；sample -~0.2 s | 半天 | 低 | ✅ 已做（`silu+quant` 逐字节用例；down 用预量化 add） |


如果只做三件事：**1（CUDA attention）、2+3（卷积重写）、4+5（snake/silu）**。
第 1 条让 GPU 路径从"能跑"变成"能演示"，第 2+3+4 让 CPU 解码器从 723 ms 掉到 ~150 ms，
而第 5 条只要一行。

---

## 6. 端到端：实测（auto，86 帧 / 3.44 s 输出，32 步）

```
CUDA（int8 LM + GPU decoder）   load=0.35  sample=0.19–0.20  decode=0.01–0.02      RTF 0.05–0.06
CPU 旧 int8（[K,N]）            load=0.09  sample=6.66–7.01  decode=0.13–0.16 RTF 1.98–2.08
CPU VNNI（第三轮，B=2）         load=0.30  sample=2.91–2.96  decode=0.13–0.16 RTF 0.89–0.91
CPU VNNI（第四轮，ragged CFG）  load=0.32  sample=2.09–2.14  decode=0.15–0.17 RTF 0.65–0.67
CPU VNNI 16 步（ragged CFG）    load=0.32  sample=1.06–1.10  decode=0.15      RTF 0.35–0.37
CPU exact 回退（OV_CPU_EXACT_CFG=1） sample=2.34–2.61                       RTF 0.72–0.81
```

CPU ragged CFG 单次 forward（195 行，`OV_PROFILE=1` 32 次统计）：

```
total = 0.062–0.070 s（中位 0.065，均值 0.065）
  qkv   19.5%   o_proj 12.5%   mlp 49.0%   head 4.5%   → GEMM ≈ 82%
  attn  14.0%   （其中 RoPE/归一化 prep 约 1 ms）
  embed  0.3%
```

第四轮先改“行数”：把 CFG 的 cond [0,S) 和 uncond 有效段 [S,S+T) 拼成 195 行，
跳过尾部 23 行；权重仍只流一遍。随后把 RMSNorm 和激活量化融合，VNNI 路径不再物化
`xn_`，norm+quant 30 µs vs 分开 64–91 µs（`aq/asc` 逐字节相同）。32 步 sample
2.91–2.96 → **2.62–2.65 s**，再把 attention 升到 4 行 query、residual add 融合、
qkv/gate-up 合并 → 2.41–2.50 s，最后 qk norm+rope 与 silu+quant 融合 →
**2.09–2.14 s**，RTF 0.89–0.91 → **0.65–0.67**。
`OV_CPU_EXACT_CFG=1` 回到 B=2 时输出 wav 与改动前逐字节相同。

**现在的瓶颈**（按 forward 占比）：

1. **MLP 49.0%**（约 31 ms）：gate/up/down 三个 GEMM 已接近 4 TFLOPS，
   silu+quant 已融合到 62 µs/层；
2. **qkv 19.5%**（约 12.5 ms，合并后）；
3. **attention 14.0%**（约 9 ms）：4 行 query 分块后仍是最大的非 GEMM 项；
4. **o_proj 12.5%**（约 8 ms）、**head 4.5%**（约 3 ms）。

下一步（按实测收益/风险排序）：

1. **attention 的下一步**（#20 剩余）：2 个 Q head 共享 K/V；直接读 K/V 已试过、更慢。
   4 行内核已经把每 K/V 向量服务 4 行，2-head 的增量主要在 K/V 拷贝/缓存，预计收益有限。
2. **audio head 只算 target 行**（#22），约 -0.4 ms。
3. **GPU**：当前 attention ~1.9 ms/forward（约 30%），下一步可试 FP16 attention；
   qkv/gate-up 合并、int8 路径 FP16 已完成。前缀 KV cache 对长句仍有价值。
4. **CPU 前缀 KV cache**（#10 剩余）：对长 prompt 有意义，短 prompt 收益小。

**已确认不可行的方向**：

- 把 CFG 拆成 cond B=1/S + uncond B=1/T 两次 forward，实测 **慢 7.5%**
  （92 → 99 ms/forward），因为权重各流一遍；ragged batch（#17）才是正确解法。
  naive 拆分还会改变 uncond 的 attention 求和顺序，数值需单独验证。
- attention 去掉每 head 的 K/V 拷贝、直接按 `kv_stride` 读：数值正确，但端到端
  sample 2.71–2.76 vs 2.67–2.72 s，略慢，已回退。
- VNNI K-block 32→64：端到端 2.96–2.98 s（明显更慢），int8 argmax 92.19% → 90.82%，
  已回退。
- W tile 预取、`dynamic` 粒度 1/2/8：均无正收益，保持 32-block + `dynamic,4`。
- VNNI 2 行 ×16 列 tile（2×8 dpbusd/组）：probe 3.0–3.4 TFLOPS（4×8 为 3.8–4.1），
  端到端 2.15–2.19 s，已回退。
- audio head 只算 target 行：拆两段 GEMM，权重读两遍，实测无稳定收益，已回退。

CUDA 侧：设备采样接口是 cond/uncond 两个设备指针；int8 路径走 TF32 + 合并
qkv/gate-up + FP16 GEMM（`OV_CUDA_FP16=0` 回退 TF32，`OV_CUDA_TF32=0` 回退 SGEMM）。
decoder 也已 device 化（`cuda_decoder.cu`），`ovtts --backend cuda` 在 auto 模式是
**纯 GPU**（codes 进、PCM 出），RTF **0.05–0.06**。采样/decoder 数值均用 host 参考验证
（`pred` 136/136、score `max|Δ|=9.5e-7`；decoder T=25/50/87 cos=1.0，max|Δ|≤3.4e-6）。

## 7. 复现

```bash
# 0) 生成 VNNI 专用权重（不覆盖旧 int8 文件；默认仍然是 kn）
python3 tools/convert_weights.py --mode int8 --int8-layout nk --only lm
# -> output/weights/omnivoice-lm-int8-nk.ovw
python3 tools/convert_weights.py --mode int8 --int8-layout nk --only lm --check

# 1) 分阶段 profile（CPU LM，VNNI）
OV_PROFILE=1 ./cpp/build/ovtts output/weights/omnivoice-lm-int8-nk.ovw \
  output/weights/omnivoice-at-fp32.ovw output/weights/tokenizer.ovt \
  "Hello, this is OmniVoice running in C++ on the CPU backend." /tmp/a.wav \
  --lang English --seed 1234 --steps 4

# 2) 端到端 A/B（同一条命令，只换 LM 权重 / CFG 路径）
for w in omnivoice-lm-int8.ovw omnivoice-lm-int8-nk.ovw; do
  OMP_NUM_THREADS=32 ./cpp/build/ovtts output/weights/$w \
    output/weights/omnivoice-at-fp32.ovw output/weights/tokenizer.ovt \
    "Hello, this is OmniVoice running in C++ on the CPU backend." /tmp/$w.wav \
    --lang English --seed 1234 --steps 32
done
# 逐位回退到旧 B=2 前向：
OV_CPU_EXACT_CFG=1 OMP_NUM_THREADS=32 ./cpp/build/ovtts ... /tmp/exact.wav \
  --lang English --seed 1234 --steps 32

# 3) 数值门禁
ctest --test-dir cpp/build --output-on-failure          # 10 项
OMP_NUM_THREADS=32 ./cpp/build/ov_test_lm output/weights/omnivoice-lm-int8-nk.ovw \
  output/ref/lm/b1_s64_full
# 期望：argmax 472/512 = 92.19%，mean|Δ| 6.04e-2，cos 0.99999942
# 以及 cfg-fwd 行：cond 逐位、uncond cos>0.99999 / mean<0.15 / argmax>=90%
# 强制旧 AVX2 回退（同一份 nk 权重）：
OV_CPU_NO_VNNI=1 OMP_NUM_THREADS=32 ./cpp/build/ov_test_lm \
  output/weights/omnivoice-lm-int8-nk.ovw output/ref/lm/b1_s64_full
# 期望：argmax 470/512 = 91.80%，mean|Δ| 4.914e-2，cos 0.99999964

# 4) 单 kernel 微基准（不参与构建；改内核后必须重编译探针）
g++ -O3 -fopenmp -std=c++17 -Icpp/include -Icpp/src \
    -o /tmp/cpu_probe tools/perf/cpu_probe.cpp cpp/build/libovcore.a -lgomp
OMP_NUM_THREADS=32 /tmp/cpu_probe output/weights/omnivoice-at-fp32.ovw
# 输出含 fp32 / s8[K,N] / s8[N,K] / s8[N,K]+VNNI 四行；
# M=218 时 VNNI 应在 3000 GFLOPS 以上（旧 int8 约 1340）。

# 5) CUDA 不受影响（旧 kn 文件）
ctest --test-dir cpp/build -R 'cuda' --output-on-failure
```

探针源码在 `tools/perf/`（`probe.cu`、`cpu_probe.cpp`，以及给 `decode()` / `encode()`
打点用的 `dec_probe.cpp`、`enc_probe2.cpp`），编译命令和打点位置见
[`tools/perf/README.md`](../tools/perf/README.md)。

> 说明：单次 GEMM 微基准在不同时刻会有 ±15% 的抖动（P/E core 调度 + 后台负载），
> 本文引用的是同一轮测量里的相对比较；解码器/编码器的打点版是源码**副本**，原文件未被修改。

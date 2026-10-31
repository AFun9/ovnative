# C++ 引擎架构

[English](architecture.en.md) | 中文

目标：**零推理依赖**（不链接 ONNX Runtime、不链接 libtorch），权重来自 `OVW1`
二进制文件（见 `weight-format.md`），CPU 与 CUDA 后端在目录层面隔离。

## 分层

```
include/ovnative/          公共头（C ABI + 内部类型）
src/core/                   OVW1 loader（mmap）、后续：线程池、分配器
src/model/                  模型结构：lm.{h,cpp}、后续 audio_tokenizer.{h,cpp}
src/backend/cpu/            CPU kernel：gemm.cpp、ops_cpu.cpp
src/backend/cuda/           CUDA kernel（隔离目录，同签名替换）
tools/                      ovdump、后续 ovtts
tests/                      gemm / lm 数值门禁
```

**依赖方向是单向的**：`model → backend`，`backend → core`，`core` 不依赖任何模型细节。
模型代码里没有 `#ifdef`，切换后端只改链接目标。

## 后端契约

模型只调用三类入口，CPU 与 CUDA 必须实现同一组符号：

| 符号 | 语义 |
|---|---|
| `linear_f32(M,N,K,A,W,C)` | `C[M,N] = A[M,K] · W[K,N]`（W 连续在 N 维，因权重预转置） |
| `linear_s8(M,N,K,A,Wq,col_scale,C)` | 同上，W 为 `[K,N]` int8，累加后乘每列 scale |
| `linear_s8_nk(M,N,K,A,Wq,row_scale,C)` | `C[m,n] = row_scale[n] · Σ A[m,k]·Wq[n,k]`，W 为 `[N,K]` int8（VNNI 布局） |
| `linear_s8_nk_packed(...)` | 上述权重的加载时 4×8 预打包形式，AVX-VNNI 专用 |
| `rmsnorm/silu_mul/rope/softmax_bias` | 逐元素/逐行算子（`src/model/ops.h`） |

`Mat` / `Emb` 是权重视图：加载时按 dtype/layout 绑定，前向热路径没有格式判断。
线性层有两种合法 INT8 编码（见 `weight-format.md`）：默认 `[K,N]`+per-column scale
（`layout=1`，旧文件和 CUDA 都用它），以及 `[N,K]`+per-row scale（`layout=0`，
`--int8-layout nk`，CPU VNNI 用）。VNNI 的另一半是激活量化：`linear_s8_nk_packed`
在每次调用前把 A 按行、每 32 个 k 一块量化成 u8（zero point 128），
epilogue 用预存的块和 `wsum` 减掉偏移。`OV_CPU_NO_VNNI=1` 时不做预打包，
回落到 `linear_s8_nk` 的 AVX2 逐列点积。

## LM 数据流（P2 已实现）

```
input_ids [B,8,S] ──┐
audio_mask [B,S] ───┴─► embed:  audio? Σ_c audio_emb[id + c·1025]  :  token_emb[id[0]]
                                   │
                       28 × Qwen3 block:
                         x += o_proj(attn(rmsnorm(x, input_ln)))
                         x += down(silu(gate(rmsnorm(x, post_ln))) * up(...))
                                   │        （q/k 先做 per-head RMSNorm，再 RoPE）
                         rmsnorm(x, final_norm)
                                   │
                         audio_heads: [B*S,1024] → [B*S,8200]
                                   │
                         reshape → logits [B, 8, S, 1025]
```

注意力按 KV head 分组（16 Q / 8 KV），per-head 的 `Kᵀ` 用一次性转置 + 两次
GEMM 完成；掩码是加性 bias（`-1e30` 表示屏蔽），与 PyTorch 的 4-D bool mask 同构。

## 数值门禁

| 测试 | 数据 | 门限 |
|---|---|---|
| `ov_test_gemm` | 随机矩阵 vs 双精度参考 | `rel ≤ 1e-5`，含尾部/边界路径 |
| `ov_test_lm` fp32 | ONNX fp32 logits（`tools/gen_lm_reference.py`） | `rel ≤ 1e-4`、`cos > 0.99999`、无 `|Δ|>0.05` |
| `ov_test_lm` int8（旧 `kn`） | 同上 | `cos > 0.99999`、`mean|Δ| < 0.15`、argmax ≥ 80% |
| `ov_test_lm` int8（`nk`+VNNI） | 同上 | 更严的验收：argmax ≥ 90%、`mean|Δ| < 0.15`、`cos > 0.99999` |

实测（i9-14900KF，32 线程）：

```
B=1 S=64   fp32  max|Δ|=3.7e-4  cos=1.00000000  argmax 100.00%   0.51 s
B=2 S=96   fp32  max|Δ|=3.8e-4  cos=1.00000000  argmax 100.00%   0.55 s
B=1 S=64   int8  mean|Δ|=4.9e-2 cos=0.99999964  argmax  91.80%   0.26 s
B=2 S=256  int8  mean|Δ|=5.2e-2 cos=0.99999928  argmax  93.02%   1.44 s
B=1 S=64   int8-nk+VNNI  mean|Δ|=6.04e-2 cos=0.99999942 argmax 92.19% 0.29 s
```

`int8-nk` 的激活量化会引入额外噪声，但 per-32-k block scale 把它限制在局部；
argmax 反而略好于只量化权重的旧路径。`OV_CPU_NO_VNNI=1` 用同一份 nk 权重回退，
结果与旧路径逐项一致（91.80% / 4.914e-2 / 0.99999964）。

FP32 的残差（`3.7e-4`，相对 `2.7e-6`）来自累加顺序差异，不是语义偏差——argmax 完全一致。

## 声码器解码（P3a）

```
codes [8,T] --RVQ: Σ_q project_out(embed[codes[q]])--> [T,1024]
            --fc2(1024→256)-->
            --conv1(256→1024,k7)-->
            5 × [ Snake → ConvTranspose(stride 8/5/4/2/3, out_pad=stride%2)
                  → 3 × ResUnit(Snake→conv(k7,dil 1/3/9)→Snake→conv(k1), +residual) ]
            --Snake--> conv2(32→1,k7) --> waveform [T·960]
```

与 PyTorch 的两处易错点（都踩过）：**Snake 在 ConvTranspose 之前**，且 `conv1` 之后
没有 Snake（`acoustic_decoder.snake1` 是最后那个）；`output_padding = stride % 2` 由
`_adjust_dac_decoder` 改写，且最终 `tanh` 被替换成了 Identity。

实测（B=1）：

| frames | 时长 | max\|Δ\| vs ONNX | cos | 耗时 |
|---:|---:|---:|---:|---:|
| 25 | 1.00 s | 2.1e-6 | 1.00000000 | 0.2 s |
| 50 | 2.00 s | 1.9e-6 | 1.00000000 | 0.4 s |
| 87 | 3.48 s | 3.5e-6 | 1.00000000 | 0.8 s |
| 225（真实语音 codes） | 9.00 s | — | — | 2.2 s（RTF 0.25） |

真实语音闭环：ONNX encoder 从 `demo_auto.wav` 抽 codes → C++ `ovdecode` 解码 →
Whisper-large-v3 转写 **WER 0.0%**（与原文逐词一致）。

## 文本到语音（P4）

```
text ─Tokenizer─► style tokens + text tokens + T×<audio_mask> ─┐
                                                               ├─ 32 步扩散采样
      cond  = 完整 prompt                                       │   (CFG: cond/uncond 双行)
      uncond = 仅 target 段（[:u_len,:u_len] 全可见）            │
                                                               ▼
                                      codes [8,T] ─AcousticDecoder─► wav
```

三处必须与上游逐字对齐的细节：

1. **uncond 行的注意力块**：`batch_attention_mask[B+i, :, :u_len, :u_len] = True` 是整块，
   不是对角线。写错这一处，CFG 会退化、输出完全不可懂（实测 WER 173% → 6.7%）。
2. **CFG 是"双 softmax"**：先 `log_softmax(c)`、`log_softmax(u)`，再对
   `c_lp + g·(c_lp − u_lp)` 做一次 `log_softmax`，最后 argmax。
3. **位置选择带 Gumbel 噪声**（position_temperature=5），因此同一 seed 只能保证
   本引擎可复现，不与 PyTorch 逐 token 相同；类别采样默认贪心（class_temperature=0）。

实测（i9-14900KF，int8 LM + fp32 decoder，32 步）：

| 用例 | 音频 | 采样 | 解码 | RTF | Whisper 误差 |
|---|---:|---:|---:|---:|---:|
| 英文 auto | 5.12 s | 26.2 s | 1.2 s | 5.35 | WER 6.7%（ONNX→Onyx） |
| 英文 voice design | 4.88 s | 25.5 s | 1.1 s | 5.46 | **WER 0.0%** |
| 中文 | 5.16 s | 26.1 s | 1.2 s | 5.29 | 内容正确（英文词/同音字误识） |

RTF 仍高于 ONNX 路径（2.8）：32 步 × 全序列前向且无 KV cache，加上逐 head 小 GEMM 的
调度开销。这是 P5b 的优化目标。

## 声码器编码（P3b，进行中）

编码器的第一段已经打通并逐值对齐：

```
24 kHz ─重采样(3:2, sinc_interp_hann)─► 16 kHz ─pad(160)─► 7 层 conv(512)
   ─► FeatureProjection(LN + 512→768) ─► pos_conv(groups=16, k=128, weight-norm)
   ─► 12 层 post-norm Transformer ─► 13 个 hidden state 取平均 ─► 2× 下采样
```

其后是语义/声学两条支路：

```
e_semantic  = encoder_semantic(hubert)          # 768ch, 2 × (2×ResUnit + conv3), stride 1
e_acoustic  = DAC encoder(wav)                  # 1→64→(5 个下采样块 8/5/4/2/3)→2048→256
embeddings  = fc(cat[e_acoustic, e_semantic])   # 1024 → 1024
codes       = RVQ(embeddings, 8 quantizers)     # project_in → 最近邻 → 残差迭代
```

实测（2 s 参考音频）：

| 阶段 | max\|Δ\| | cos |
|---|---:|---:|
| 重采样 24k→16k | 1.2e-7 | 1.00000000 |
| HuBERT 语义特征 | 2.2e-5 | 1.00000000 |
| e_semantic | 5.5e-5 | 1.00000000 |
| e_acoustic | 1.1e-4 | 1.00000000 |
| embeddings | 2.2e-4 | 1.00000000 |
| **codes** | — | **精确匹配 400/400 = 100%** |
| 参考音频预处理（RMS 归一化 + 静音裁剪） | **0.0**（逐字节一致） | 1.0 |

预处理必须复刻 pydub 的语义：float→int16 后用 **int16 RMS** 在
`min_silence_len=200 ms` 窗口、`seek_step=10 ms`、`-50 dBFS` 门限上判定静音，
长静音切除时保留 200 ms 上下文，最后按 `lead=100 / trail=200 ms` 修剪两端。
差一个参数，参考 codes 就会变，克隆音色随之改变。

三个必须踩准的细节（都是对着 PyTorch 逐段 dump 找出来的）：

1. **`resample_16k` 的 `width` 必须是 static**：它只在静态 kernel 初始化时被写入，
   若用普通局部变量，第二次调用时 width=0，滤波退化成 3 抽头（表现为 cos 0.94）。
2. **pos_conv 之后还有 GELU**（`HubertPositionalConvEmbedding` 里有个 `activation`
   子模块），漏掉会得到 cos 0.81。
3. **hidden_states 元组不含 projection 输出**：它是
   `[layer0 输入, …, layer11 输入, 最终输出]` 共 13 项；把 projection 算进去会让
   均值出现 ~13% 的相对偏差。
4. 另：`_adjust_dac_decoder` 之外，`weight_norm` 是 **dim=2**（`w = g·v/‖v‖_(0,1)`），
   不是默认的 dim=0。

## 性能优化（P5b，进行中）

### 已落地

| 改动 | 效果 |
|---|---|
| GEMM 8×8 微内核（原 4×8）+ 列块外层循环 | gate/up 364→**473** GFLOPS，down 191→**298**，q 328→**413** |
| `M % 8` 零填充替代标量尾部 | S=158 时单次前向 **2.28s → 0.93s**（尾部 4 行曾走全标量） |
| SIMD `exp`（Cephes 多项式，误差 ~1e-7）+ 向量化 softmax | softmax 从 ~76 ms/forward 降到个位数 ms |
| RoPE 的 cos/sin 按位置预计算（原来每个 head 重算） | prep 46 ms → **14 ms**/forward |
| 串行 GEMM 不再进入 OpenMP 运行时 | 避免 1792 次/forward 的空并行区 |
| attention 按 (batch, KV head, query head) 并行、内层 GEMM 串行 | 短句 S=92：0.81 → **0.50 s/forward** |
| 融合 attention kernel（K/V 常驻 L1、逐行 score→softmax→PV） | 单 head 262 µs（孤立）；`argmax` 99.95% |
| **FTZ/DAZ（flush-to-zero、denormals-are-zero）** | attention 单层 **18 ms → 0.75 ms**，单次前向 **0.886 → 0.593 s**，端到端 RTF **6.10 → 3.74** |

### 还剩什么

### 次正规数：这次优化里最大的坑

attention 的掩码用 `-1e30` 表示"不可见"。softmax 之后这些位置变成
`exp(-1e30) ≈ 1e-38`，落在 **float32 的次正规区间**。x86 对次正规数走微码辅助
（microcode assist），单条指令可以慢 20–100 倍；而且这些值会写进 `attn_`，
污染后续所有 GEMM 的累加。

现象极具误导性：融合 kernel 孤立跑 262 µs/head，放进真实前向却要 5 ms/head
（19 倍），整层 18 ms；而线程数、wait policy、cgroup 配额、计时归因逐一排除后，
"同一段代码孤立快、集成慢 20 倍" 的唯一合理解释就是数据本身触发了慢路径。

修法是每线程设一次 MXCSR：`FTZ(bit15) | DAZ(bit6)`。效果：

```
attention 单层   18.0 ms  ->  0.75 ms   (24x)
单次前向          0.886 s ->  0.593 s
端到端 (32 步)    30.1 s  ->  18.0 s
RTF               6.10    ->  3.74
```

**移植提示**：任何用 `-inf` / `-1e30` 做掩码、又在掩码区域上继续做浮点运算的
推理引擎，都必须打开 FTZ/DAZ，否则性能会莫名塌陷。

### 现在的瓶颈

attention 降到 6% 后，时间重新回到 GEMM：MLP 59%、QKV 21%、o_proj 11%。
实测 ~480 GFLOPS，已经接近这台机器 AVX2 fp32 的实际上限（8 个 P-core × 16 flop/cycle
× ~4.4 GHz ≈ 560 GFLOPS）。继续提速只有三条路：

而孤立回放同样的 896 个 head 只要 23 ms（≈250 GFLOPS）。差距来自层间缓存置换：
每个 MLP 层要流式读 36 MB 权重，把 q/k/v 和每线程 scratch（32×340 KB）全部挤出
L3，attention 的转置与 GEMM 每次都在冷缓存上启动。

下一步（按收益排序）：

1. **融合 attention kernel**：把 K 转置、QKᵀ、softmax、PV 合成一个 pass，中间结果
   留在 L1/L2，不再落回全局内存；
2. **改变 K/V 布局**：`k_proj` 直接输出 `[B, KV, D, S]`（GEMM 的 N 维就是 S），
   彻底消掉转置；
3. 把 `attn_` 的 scatter 换成 stride 写，或让 `o_proj` 直接消费 `[B, H, S, D]`。

## 性能现状

单次 LM 前向（B=2, S=256，int8）：**1.44 s**；ONNX Runtime 同规模约 0.78 s。
差距主要在两处，属于 P5 的优化清单：

1. 注意力按 head 拆成 896 次小 GEMM，OpenMP 启停开销占比高 → 合并成按 KV head 的批量 GEMM；
2. softmax 未向量化（每层 16×256×256 次 exp）→ SIMD exp 近似 + 行并行。

GEMM 微基准本身已达 460 GFLOPS（256×1024×1024），接近这台机器 AVX2 fp32 的实际上限
（无 AVX-512），所以瓶颈不在 GEMM 主体。

### P8：CPU int8 → AVX-VNNI（最新）

上面的 1.44 s 是 P5 阶段的数字。P8 把 int8 权重改成 `[N,K]`（`--int8-layout nk`）、
加载时预打包成 4×8 tile，激活按行每 32 个 k 一块量化成 u8，内层用 `vpdpbusd`。
在 B=2/S=109 的真实采样形状上：单次 forward 0.205 → 0.104–0.110 → 0.089–0.093
→ **0.062–0.070 s（有效 195 行，中位 0.065）**，32 步端到端 **RTF 2.05 → 1.14–1.19
→ 0.89–0.91 → 0.65–0.67**；GEMM 微基准 1.34 → **3.0–4.1 TFLOPS**。
后续又把 attention 改成两行 query 共享 K/V、激活量化池化/并行（q/k/v 一次、
gate/up 一次，7→4）、host CFG 的 log_softmax/argmax 按行并行。最新一轮新增
`LM::forward_cfg()`：把 cond（109 行）和 uncond 有效段（86 行）拼成 195 行一次前向，
跳过尾部 23 行（权重仍只流一遍）；`rmsnorm_quantize_f32()` 把 RMSNorm 和激活量化融合
（aq/asc 逐字节相同，30 µs vs 分开 64–91 µs），VNNI 路径不再物化 xn；
attention 又从 2 行 query 升到 4 行（`attention_block4`，`OV_ATTN_RB2=1` 可回退）；
`o_proj`/`down_proj` 的 residual add 也融进 VNNI epilogue（bit-identical）；
q/k/v 与 gate/up 的 packed 权重在加载时拼成逻辑矩阵，用 sliced kernel 一次 OpenMP
区域写回独立 buffer（数学逐字节一致）；q/k 的 RMSNorm+RoPE、silu+激活量化也各融合成
一趟（单层 100→40 µs、183–241→62 µs）。cond 逐位不变，uncond 因 attention 求和顺序有 ~1 ULP 差异
（int8 mean 3.7e-2 / cos 0.99999982 / argmax 95.1%）。
`OV_CPU_EXACT_CFG=1` 可回到旧 B=2 逐位路径。
旧 `[K,N]` int8 文件继续走原 AVX2 内核，CUDA 路径不变；`OV_CPU_NO_VNNI=1`
可强制回退。GEMM 占比 ~82% 后，**attention（~17%，14 ms）仍是 CPU 侧最大非 GEMM 项**。
细节与门禁见 `perf-review.md` §0.3/§0.4/§2 P2-1。

## CUDA 后端（P6）

`src/backend/cuda/`，默认不编译（`-DOV_ENABLE_CUDA=ON` 才加进来），CPU 构建仍然零 CUDA 依赖。

**设计**：LM 前向整体驻留显存——只有 `input_ids / mask / position_ids` 进、logits 出。
权重在首次使用时**反量化成 fp32 再上传**（int8 → fp32，缓存按主机指针索引）：
4090 有 48 GB，把 LM 存两份都够。fp32 checkpoint 走精确 SGEMM；int8 checkpoint
在第五轮加了 TF32、加载时合并 qkv/gate-up，以及 fp16 权重/激活（fp32 累加）的
`cublasGemmEx` 路径，见 `perf-review.md §0.5`。decoder 由 `cuda_decoder.cu` 全 device
执行（RVQ gather/linear、conv1d、ConvTranspose phase GEMM、Snake、残差），只有 codes 进、
PCM 出；encoder 由 `cuda_encoder.cu` 全 device 执行（resample、HuBERT 12 层 transformer、
semantic/acoustic conv、RVQ encode），conv 走 im2col+cuBLAS，48 kHz→50 帧
**513→8–9 ms（~60×）**，codes 与 CPU 400/400 一致。

**接入方式**：`LM::set_cuda_backend(ctx, fn)` 是一个函数指针钩子，`ovcore` 依旧不链接
CUDA；`ovtts --backend cuda` 和 `ov_test_lm --cuda` 负责创建 runner。

实测（4090，int8 权重反量化，32 步）：

| 路径 | 采样 | 解码 | RTF | 备注 |
|---|---:|---:|---:|---|
| CPU（AVX2，3.44 s 音频） | 2.09–2.14 s | 0.15 s | 0.65–0.67 | VNNI |
| **CUDA（LM + GPU decoder，3.44 s）** | **0.19–0.20 s** | **0.01–0.02 s** | **0.05–0.06** | TF32 + 合并 + fp16 GEMM + device decoder |
| 历史：CUDA（5.12 s 音频） | 3.0 s | 1.1 s | 0.82 | 第五轮前 |
| **CUDA voice-clone（79 帧 / 3.16 s）** | **0.27–0.34 s**（含 encoder） | **0.01 s** | **0.09–0.11** | GPU encoder + LM + decoder |

数值：fp32 权重下 logits `max|Δ|=1.8e-4`、argmax **100%**（比 CPU 路径的 3.7e-4 还准，
因为 cuBLAS 的累加顺序更接近参考实现）；int8 权重反量化后与 CPU int8 路径结果一致
（argmax 91.8%）。

三个踩过的坑：

1. **cuBLAS 是列主序**。`C_col[m + n*ldc] = Σ_k op(A)[m,k]·op(B)[k,n]`——把想要的
   行主序 `s[i*S+j] = Σ_d Q[i,d]K[j,d]` 映射成 `m=j, n=i` 后，调用是
   `(N,N)` 且 **A=K、B=Q**（不是 A=Q）；PV 同理由 (N,N)/A=V,B=P。这一处错了三次才对。
2. **logits 要 permute**：kernel 写的是 `[T, C*V]`，接口契约是 `[B, C, S, V]`。
3. 注意力最终改成**手写 kernel**（一个 block 一个 (b,h)），比 cuBLAS 的 1792 次小 GEMM
   更省心也更快；CPU 端的融合 kernel 可以直接照搬思路。

## 后续阶段

- **P3** `audio_tokenizer.{h,cpp}`：HuBERT 语义编码（7 层 conv + 12 层 transformer）、
  DAC 声学编解码（Conv1d/ConvTranspose1d + Snake + ResUnit）、RVQ 最近邻。
- **P4** BPE（读 `tokenizer.json` 的 merges，或用离线编译进二进制）、chat template、
  CFG 扩散采样循环（top-k + schedule）。
- **P5** `ovtts` CLI + 完整 C API，Whisper-large-v3 ASR 验收（对标 ONNX 的 WER 8.7% / 0%）。
- **P6** CUDA 后端：同签名替换 `linear_s8` / 注意力，权重首次使用时上传并 `madvise` 释放映射。

## 移植注意

- 只支持小端（x86/ARM64 均满足）；`OVW1` 头里的偏移一律 u64，文件 > 2 GB 无需特殊处理。
- AVX2 用 `__attribute__((target("avx2,fma")))` + `__builtin_cpu_supports` 运行时分派，
  未命中时自动回落标量实现，不需要为老 CPU 单独编译。
- 所有 `-1e30` 掩码行在 softmax 里被检测为全屏蔽并输出 0，避免 NaN 传播。

# OVW1 — OmniVoice 纯二进制权重格式

目标：**零依赖**（不需要 ONNX / PyTorch / protobuf / JSON 解析库），单文件自描述，
`mmap` 直接映射后张量指针即权重指针，加载 3.2 GB 权重的时间 ≈ 一次 `mmap`。

## 文件布局

```
┌──────────────┬─────────────┬──────────────┬───────────────┬──────────────┐
│ Header 128 B │ Index n×64B │ Names blob   │ Tensor data   │ Aux (scales) │
└──────────────┴─────────────┴──────────────┴───────────────┴──────────────┘
                                                  每块 64 B 对齐
```

所有整数小端序（x86/ARM64 通用），`u32/u64` 无符号。

## Header（128 字节）

| off | type | field | 说明 |
|----:|------|-------|------|
| 0 | char[8] | `magic` | `"OMNIVOIC"` |
| 8 | u32 | `version` | 1 |
| 12 | u32 | `dtype_mode` | 0 = 全 FP32，1 = 含 INT8 量化张量 |
| 16 | u32 | `n_tensors` | 张量个数 |
| 20 | u32 | `header_bytes` | 128 |
| 24 | u64 | `index_off` | 索引区绝对偏移 |
| 32 | u64 | `names_off` | 名称区绝对偏移 |
| 40 | u64 | `data_off` | 数据区绝对偏移 |
| 48 | u64 | `file_size` | 文件总字节 |
| 56 | u64 | `flags` | bit0: 含转置权重；bit1: 含 INT8 |
| 64 | u64 | `n_params` | 原始参数量（校验用） |
| 72 | u8[56] | reserved | 0 |

## TensorDesc（64 字节 × n）

| off | type | field | 说明 |
|----:|------|-------|------|
| 0 | u32 | `name_off` | 相对 names_off |
| 4 | u32 | `name_len` | 不含结尾 NUL |
| 8 | u8 | `dtype` | 0=FP32, 1=INT8, 2=FP16, 3=INT64 |
| 9 | u8 | `ndim` | 0..4 |
| 10 | u8 | `layout` | 0 = 行主序原样；1 = 已转置（2-D 线性层权重） |
| 11 | u8 | `quant` | 0 = 无；1 = 每输出通道对称 INT8 |
| 12 | u32[4] | `dims` | 存储布局下的形状 |
| 28 | u32 | `aux_n` | scale 个数（INT8 时 = 输出通道数） |
| 32 | u64 | `hash` | 载荷 CRC32（低 32 位有效） |
| 40 | u64 | `data_off` | 绝对偏移 |
| 48 | u64 | `nbytes` | 载荷字节数 |
| 56 | u64 | `aux_off` | scale 绝对偏移（无则 0） |

## 关键约定

### 1. 线性层权重：默认预转置；INT8 另有 `[N][K]` 变体

PyTorch 的 `nn.Linear.weight` 是 `[out, in]`。ONNX 里每次 `MatMul` 都要处理
`transB`，而 C++ 侧希望在 N 维上连续做 SIMD。因此**默认**（`--int8-layout kn`）
把所有 2-D 线性层权重存成 `[in, out]`（`layout=1`，`dims=[in, out]`），运行时
GEMM 签名统一为：

```
C[M, N] = A[M, K] · Wt[K, N]          （Wt 连续，N 维向量化）
```

例外（保持行主序，`layout=0`）：
- `llm.embed_tokens.weight`、`audio_embeddings.weight` —— Gather 用，按行取；
- 所有 1-D 范数/偏置、3-D 卷积核。

#### 1a. 新增合法组合：output-major INT8（AVX-VNNI 用）

`tools/convert_weights.py --int8-layout nk`（默认仍是 `kn`，旧文件不受影响，产物
写到 `omnivoice-<model>-int8-nk.ovw`，不覆盖旧文件）会把量化后的线性层存成：

| 字段 | 值 |
|---|---|
| `dtype` | `1` (INT8) |
| `layout` | `0`（行主序原样） |
| `quant` | `1` |
| `dims` | `[out, in]`（与 PyTorch `nn.Linear.weight` 同形） |
| `aux_n` | `out`（每个输出通道一个 scale，落在存储矩阵的**行**上） |
| 载荷 | `q[out, in]`（不再转置） |

这正是 embedding 表一直使用的编码（`layout=0 + quant=1 + per-row scale`），
只是这种张量由 `make_linear`（`Mat`）消费，而不是 `Emb::gather`。两种布局都只
是合法编码，读取端按 `layout` 自动分派；旧 `layout=1` 文件继续可用。

**为什么需要它**：AVX-VNNI 的 `vpdpbusd` 要求「一个输出列的 4 个连续 k」在
一个 dword 里。`[N][K]` 让同一输出的 K 连续，加载时再重排成 4×8 tile
（细节见 `docs/perf-review.md` §2 P2-1 与 `cpp/include/ovnative/ov_gemm.h`）。
`[K][N]` 布局无法高效喂给 `vpdpbusd`。

**CUDA 注意**：CUDA runner 只理解 `layout=1` 的反量化路径，`-int8-nk.ovw`
是 **CPU/VNNI 专用**。`ovtts --backend cuda` 与 `ov_test_lm --cuda` 会直接报错
并提示改用 `omnivoice-lm-int8.ovw`，不会静默算错。

### 2. INT8 是「每输出通道对称」量化

对原始 `[out, in]` 权重按行求 `scale = max|row| / 127`，`q = round(row/scale)`，
裁剪到 `[-127, 127]`（不用 -128，避免对称性破坏）。存储时：

- `layout=1`（默认 `kn`）：载荷 `q.T` = `[in, out]`，aux 是 `scale[out]`，每列一个；
- `layout=0`（`nk` 或 embedding）：载荷 `q` = `[out, in]`，aux 是 `scale[out]`，每行一个。

运行时的 fp32 反量化是「零成本」的——GEMM 累加完再乘每列/每行 scale：

```
layout=1:  C[m, n] = scale[n] · Σ_k A[m, k] · q[k, n]
layout=0:  C[m, n] = scale[n] · Σ_k A[m, k] · q[n, k]
```

不需要在任何地方物化 fp32 权重副本，内存占用就是 int8 本身。

VNNI 路径会额外把**激活**量化成 u8，但不会改写 OVW1 的语义：权重仍是上面的
对称 int8，量化只发生在运行时（详见 `perf-review.md`）：

```
每 32 个 k 一块：ascale = max|A_block| / 127
                a_u8   = clamp(round(A / ascale), -127, 127) + 128
每块：C[m,n] += (Σ a_u8·q[n,k] - 128·Σ q[n,k]) · ascale · wscale[n]
```

`+128` 只是把有符号激活平移到 `vpdpbusd` 要求的无符号域，epilogue 用预存的
每块权重和把偏移精确减掉。

默认保护 `audio_heads`（LM 输出头）不量化，理由见 ONNX 工具链的 int8hq 实验：
该层直接产生采样 logits，量化噪声会一对一转成 argmax 翻转。

### 3. 为什么要 hash

每个张量带 CRC32 载荷校验，`ovdump --verify` 可以在不依赖 Python 的情况下
证明「C++ 读到的字节 == 转换器写下的字节」，也让增量更新/缓存失效有据可依。

## 与 safetensors 的关系

转换器 `tools/convert_weights.py` **只依赖 numpy**：手动解析 safetensors 的 8 字节
长度 + JSON 头，`mmap` 后按 `data_offsets` 构造 `np.ndarray` 视图，不经过 torch。
因此整条链路（safetensors → OVW1 → C++）没有任何框架依赖。

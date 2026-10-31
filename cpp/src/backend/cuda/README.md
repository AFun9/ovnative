# CUDA backend

Everything CUDA-specific lives in this directory. The CPU build never links
CUDA; `-DOV_ENABLE_CUDA=ON` adds this directory to the build.

## Layout

```
src/backend/cuda/
├── CMakeLists.txt
├── cuda_lm.{h,cu}        LM forward, fused/batched attention, device sampler
├── cuda_encoder.{h,cu}   voice-clone encoder (HuBERT + semantic/acoustic + RVQ)
├── cuda_decoder.{h,cu}   acoustic decoder (codes -> PCM)
└── README.md
```

## Design

- **Weight residency**: the `.ovw` file is mmap-ed; each tensor is dequantised
  once on first use and kept in device memory (fp32 original, merged and fp16
  copies for the int8 path).
- **LM**: int8 checkpoint uses TF32 / FP16 cuBLAS GEMMs with fp32 accumulation;
  fp32 checkpoint keeps the exact SGEMM path. Attention is fused for short
  sequences and strided-batched TF32 GEMM for long sequences (`seq >= 160`).
- **Encoder**: resample, conv stem, 12-layer transformer, semantic/acoustic
  conv stacks and RVQ encode all run on the device. Conv1d uses im2col +
  cuBLAS; weights are transposed on the device at load time.
- **Decoder**: RVQ gather/project, fc2, conv1d, ConvTranspose (per-tap GEMM),
  Snake and residuals all run on the device.
- **Sampler**: guidance/argmax and the final code selection run on the device;
  only codes/PCM cross the PCIe boundary.

## Contract

`LM`, `AudioEncoder` and `AcousticDecoder` expose function-pointer hooks in
`ovcore`; the CUDA runners are installed by `ovtts` (and by tests through the
same public headers). `ovcore` itself never includes a CUDA header.

## Build

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DOV_ENABLE_CUDA=ON
cmake --build cpp/build -j
```

Requires CUDA Toolkit + cuBLAS. Performance numbers and the stage breakdown are
in `docs/perf-review.md`.

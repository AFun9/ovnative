// 1-D convolution kernels on time-major activations ([T, C] instead of
// PyTorch's [C, T]).  Time-major keeps the channel dimension contiguous, which
// is exactly what the GEMM-row layout and the ConvTranspose tap scatter need.
#pragma once

#include <cstdint>
#include <vector>

namespace ov {

struct Conv1d {
  int cin = 0, cout = 0, k = 0, stride = 1, pad = 0, dil = 1;
  const float* b = nullptr;   // [cout] or null
  std::vector<float> wt;      // [k, cin, cout]: tap-major, GEMM-ready

  // w is the PyTorch layout [cout, cin, k].
  void prepare(const float* w);
  int out_len(int t_in) const { return (t_in + 2 * pad - dil * (k - 1) - 1) / stride + 1; }
};

struct ConvT1d {
  int cin = 0, cout = 0, k = 0, stride = 1, pad = 0, out_pad = 0;
  const float* b = nullptr;   // [cout] or null
  std::vector<float> wt;      // [k, cin, cout]: one contiguous [cin, cout] per tap

  // w is the PyTorch layout [cin, cout, k].
  void prepare(const float* w);
  int out_len(int t_in) const { return (t_in - 1) * stride - 2 * pad + k + out_pad; }
};

// x [t_in, cin] -> y [out_len(t_in), cout]
void conv1d_f32(const Conv1d& c, const float* x, int t_in, float* y);
void conv_transpose1d_f32(const ConvT1d& c, const float* x, int t_in, float* y);

// Snake: x + sin²(alpha·x)/alpha, per channel, in place.
void snake_f32(float* x, const float* alpha, int channels, int t);

}  // namespace ov

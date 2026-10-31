// Per-thread scratch shared by the CPU kernels.
//
// The convolution kernels used to allocate their im2col panel with a fresh
// `std::vector` on every call.  For the deepest decoder/encoder blocks that is a
// 70-110 MB mmap whose first touch is pure page faults (measured: 26 ms for one
// 110 MB panel), repeated for every convolution.  The buffers here grow once and
// are reused, so the fault cost is paid a single time per thread.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ov {
namespace cpu_detail {

// slot 0: im2col panel, slot 1: transposed weight copy.  Grow-only.
std::vector<float>& scratch(int slot);

// Rows per im2col panel: a few MB, so the panel stays L3-resident while the GEMM
// walks it, while keeping the number of GEMM calls per convolution small.
inline int64_t panel_rows(int64_t ck) {
  int64_t tb = (int64_t{1} << 20) / std::max<int64_t>(ck, 1);  // 4 MB panels
  return std::min<int64_t>(std::max<int64_t>(tb, 16), 8192);
}

}  // namespace cpu_detail
}  // namespace ov

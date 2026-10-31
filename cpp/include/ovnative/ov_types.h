// Core value types shared by every backend (CPU / CUDA) and by the model code.
// Deliberately dependency-free: no ONNX, no PyTorch, no protobuf.
#pragma once

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ov {

enum class DType : uint8_t { F32 = 0, I8 = 1, F16 = 2, I64 = 3 };

inline const char* dtype_name(DType t) {
  switch (t) {
    case DType::F32: return "f32";
    case DType::I8:  return "i8";
    case DType::F16: return "f16";
    case DType::I64: return "i64";
  }
  return "?";
}

inline size_t dtype_size(DType t) {
  switch (t) {
    case DType::F32: return 4;
    case DType::I8:  return 1;
    case DType::F16: return 2;
    case DType::I64: return 8;
  }
  return 0;
}

// Max rank 4 covers every tensor in OmniVoice (1-D norms, 2-D matmuls, 3-D convs).
struct Shape {
  int32_t dims[4] = {0, 0, 0, 0};
  int32_t ndim = 0;

  Shape() = default;
  Shape(std::initializer_list<int64_t> d) {
    if (d.size() > 4) throw std::runtime_error("Shape rank > 4");
    ndim = static_cast<int32_t>(d.size());
    int32_t i = 0;
    for (int64_t v : d) dims[i++] = static_cast<int32_t>(v);
  }

  int64_t operator[](int i) const { return dims[i]; }
  int64_t numel() const {
    int64_t n = 1;
    for (int32_t i = 0; i < ndim; ++i) n *= dims[i];
    return n;
  }
  std::string str() const {
    std::string s = "[";
    for (int32_t i = 0; i < ndim; ++i) {
      if (i) s += ", ";
      s += std::to_string(dims[i]);
    }
    return s + "]";
  }
};

// A tensor as stored in an OVW1 file. ``data`` points straight into the mmap.
struct Tensor {
  std::string name;
  DType dtype = DType::F32;
  Shape shape;
  bool transposed = false;   // layout==1: 2-D linear weight stored as [in, out]
  bool quantized = false;    // per-output-channel INT8 + fp32 scales
  const void* data = nullptr;
  int64_t nbytes = 0;
  const float* scales = nullptr;
  int64_t n_scales = 0;
  uint32_t crc = 0;          // CRC32 of the payload, as written by the converter

  const float* f32() const {
    if (dtype != DType::F32) throw std::runtime_error("tensor '" + name + "' is not f32");
    return static_cast<const float*>(data);
  }
  const int8_t* i8() const {
    if (dtype != DType::I8) throw std::runtime_error("tensor '" + name + "' is not i8");
    return static_cast<const int8_t*>(data);
  }
  const int64_t* i64() const {
    if (dtype != DType::I64) throw std::runtime_error("tensor '" + name + "' is not i64");
    return static_cast<const int64_t*>(data);
  }
};

}  // namespace ov

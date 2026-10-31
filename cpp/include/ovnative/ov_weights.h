// OVW1 weight-file loader: mmap the file, expose tensors as borrowed views.
//
// Loading 3.2 GB costs one mmap(2) — no copy, no parsing allocation beyond the
// 840-entry tensor table.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ovnative/ov_types.h"

namespace ov {

class WeightFile {
 public:
  WeightFile() = default;
  ~WeightFile();
  WeightFile(const WeightFile&) = delete;
  WeightFile& operator=(const WeightFile&) = delete;
  WeightFile(WeightFile&& other) noexcept;
  WeightFile& operator=(WeightFile&& other) noexcept;

  // Throws std::runtime_error with a readable message on any format problem.
  static WeightFile Open(const std::string& path);

  bool loaded() const { return base_ != nullptr; }
  uint64_t file_size() const { return size_; }
  uint64_t n_params() const { return n_params_; }
  int dtype_mode() const { return dtype_mode_; }
  const std::string& path() const { return path_; }

  const std::vector<Tensor>& tensors() const { return tensors_; }
  // nullptr when absent — callers decide whether that is fatal.
  const Tensor* Find(std::string_view name) const;
  // Throws when absent.
  const Tensor& Get(const std::string& name) const;

  // Format-level CRC32 of one tensor payload, recomputed from the mapping.
  uint32_t ComputeCrc(const Tensor& t) const;
  // Stored CRC32 for the same tensor (from the descriptor).
  uint32_t StoredCrc(const Tensor& t) const;

 private:
  void Release();

  void* base_ = nullptr;
  size_t size_ = 0;
  std::string path_;
  uint64_t n_params_ = 0;
  int dtype_mode_ = 0;
  std::vector<Tensor> tensors_;
  std::vector<uint32_t> crc_;
  std::unordered_map<std::string, int32_t> index_;
};

}  // namespace ov

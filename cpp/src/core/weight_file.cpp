#include "ovnative/ov_weights.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "core/utf8.h"

namespace ov {
namespace {

#pragma pack(push, 1)
struct RawHeader {
  char magic[8];
  uint32_t version;
  uint32_t dtype_mode;
  uint32_t n_tensors;
  uint32_t header_bytes;
  uint64_t index_off;
  uint64_t names_off;
  uint64_t data_off;
  uint64_t file_size;
  uint64_t flags;
  uint64_t n_params;
};

struct RawDesc {
  uint32_t name_off;
  uint32_t name_len;
  uint8_t dtype;
  uint8_t ndim;
  uint8_t layout;
  uint8_t quant;
  uint32_t dims[4];
  uint32_t aux_n;
  uint64_t hash;
  uint64_t data_off;
  uint64_t nbytes;
  uint64_t aux_off;
};
#pragma pack(pop)

static_assert(sizeof(RawHeader) == 72, "OVW1 header must be 72 bytes");
static_assert(sizeof(RawDesc) == 64, "OVW1 descriptor must be 64 bytes");

constexpr char kMagic[8] = {'O', 'M', 'N', 'I', 'V', 'O', 'I', 'C'};
constexpr uint32_t kVersion = 1;

uint32_t crc32_bytes(const void* data, size_t n) {
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      table[i] = c;
    }
    init = true;
  }
  const uint8_t* p = static_cast<const uint8_t*>(data);
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
}

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("OVW1: " + msg); }

}  // namespace

WeightFile::~WeightFile() { Release(); }

void WeightFile::Release() {
  if (base_) {
#ifdef _WIN32
    ::UnmapViewOfFile(base_);
    if (map_) {
      ::CloseHandle(static_cast<HANDLE>(map_));
      map_ = nullptr;
    }
#else
    ::munmap(base_, size_);
#endif
    base_ = nullptr;
    size_ = 0;
  }
  tensors_.clear();
  crc_.clear();
  index_.clear();
}

WeightFile::WeightFile(WeightFile&& other) noexcept { *this = std::move(other); }

WeightFile& WeightFile::operator=(WeightFile&& other) noexcept {
  if (this != &other) {
    Release();
    base_ = other.base_;
    map_ = other.map_;
    size_ = other.size_;
    path_ = std::move(other.path_);
    n_params_ = other.n_params_;
    dtype_mode_ = other.dtype_mode_;
    tensors_ = std::move(other.tensors_);
    crc_ = std::move(other.crc_);
    index_ = std::move(other.index_);
    other.base_ = nullptr;
    other.map_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

WeightFile WeightFile::Open(const std::string& path) {
#ifdef _WIN32
  const std::wstring wpath = utf8_to_wide(path);
  HANDLE file = ::CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) fail("cannot open '" + path + "'");
  LARGE_INTEGER li{};
  if (!::GetFileSizeEx(file, &li)) {
    ::CloseHandle(file);
    fail("cannot stat '" + path + "'");
  }
  const size_t size = static_cast<size_t>(li.QuadPart);
  if (size < sizeof(RawHeader)) {
    ::CloseHandle(file);
    fail("'" + path + "' is smaller than a header");
  }
  HANDLE map = ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!map) {
    ::CloseHandle(file);
    fail("cannot map '" + path + "'");
  }
  void* base = ::MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
  ::CloseHandle(file);
  if (!base) {
    ::CloseHandle(map);
    fail("MapViewOfFile failed for '" + path + "'");
  }
#else
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) fail("cannot open '" + path + "': " + std::strerror(errno));

  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    fail("cannot stat '" + path + "'");
  }
  const size_t size = static_cast<size_t>(st.st_size);
  if (size < sizeof(RawHeader)) {
    ::close(fd);
    fail("'" + path + "' is smaller than a header");
  }

  void* base = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (base == MAP_FAILED) fail("mmap failed for '" + path + "'");
#endif

  WeightFile wf;
  wf.base_ = base;
#ifdef _WIN32
  wf.map_ = map;
#endif
  wf.size_ = size;
  wf.path_ = path;

  const auto* h = static_cast<const RawHeader*>(base);
  if (std::memcmp(h->magic, kMagic, 8) != 0) fail("bad magic (not an OVW1 file)");
  if (h->version != kVersion) fail("unsupported version " + std::to_string(h->version));
  if (h->file_size != size) {
    fail("file size mismatch: header says " + std::to_string(h->file_size) + ", actual " +
         std::to_string(size));
  }
  const uint64_t index_end = h->index_off + static_cast<uint64_t>(h->n_tensors) * sizeof(RawDesc);
  if (h->index_off < sizeof(RawHeader) || index_end > size) fail("index out of bounds");

  wf.n_params_ = h->n_params;
  wf.dtype_mode_ = static_cast<int>(h->dtype_mode);

  const auto* descs = reinterpret_cast<const RawDesc*>(static_cast<const char*>(base) + h->index_off);
  wf.tensors_.reserve(h->n_tensors);
  wf.crc_.reserve(h->n_tensors);

  for (uint32_t i = 0; i < h->n_tensors; ++i) {
    const RawDesc& d = descs[i];
    const uint64_t name_begin = h->names_off + d.name_off;
    if (h->names_off > size || name_begin + d.name_len > size) fail("tensor name out of bounds");
    if (d.data_off + d.nbytes > size) fail("tensor payload out of bounds");

    Tensor t;
    t.name.assign(static_cast<const char*>(base) + name_begin, d.name_len);
    t.dtype = static_cast<DType>(d.dtype);
    t.transposed = d.layout == 1;
    t.quantized = d.quant == 1;
    t.shape.ndim = d.ndim;
    for (int k = 0; k < 4; ++k) t.shape.dims[k] = static_cast<int32_t>(d.dims[k]);
    t.data = static_cast<const char*>(base) + d.data_off;
    t.nbytes = static_cast<int64_t>(d.nbytes);
    t.crc = static_cast<uint32_t>(d.hash & 0xFFFFFFFFu);
    if (t.quantized) {
      if (d.aux_off + static_cast<uint64_t>(d.aux_n) * 4 > size) fail("scales out of bounds");
      t.scales = reinterpret_cast<const float*>(static_cast<const char*>(base) + d.aux_off);
      t.n_scales = d.aux_n;
    }
    const int64_t expect = t.shape.numel() * static_cast<int64_t>(dtype_size(t.dtype));
    if (expect != t.nbytes) {
      fail("size mismatch for '" + t.name + "': shape " + t.shape.str() + " needs " +
           std::to_string(expect) + " bytes, descriptor says " + std::to_string(t.nbytes));
    }
    wf.index_.emplace(t.name, static_cast<int32_t>(wf.tensors_.size()));
    wf.crc_.push_back(t.crc);
    wf.tensors_.push_back(std::move(t));
  }
  return wf;
}

const Tensor* WeightFile::Find(std::string_view name) const {
  auto it = index_.find(std::string(name));
  if (it == index_.end()) return nullptr;
  return &tensors_[static_cast<size_t>(it->second)];
}

const Tensor& WeightFile::Get(const std::string& name) const {
  const Tensor* t = Find(name);
  if (!t) fail("tensor not found: '" + name + "'");
  return *t;
}

uint32_t WeightFile::ComputeCrc(const Tensor& t) const {
  return crc32_bytes(t.data, static_cast<size_t>(t.nbytes));
}

uint32_t WeightFile::StoredCrc(const Tensor& t) const { return t.crc; }

}  // namespace ov

// ovdump — inspect / verify an OVW1 weight file without any Python.
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "ovnative/ov_gemm.h"
#include "ovnative/ov_weights.h"

namespace {

void usage() {
  std::printf(
      "usage: ovdump <file.ovw> [--list N] [--tensor NAME] [--verify]\n"
      "  --list N      print the first N tensors (default 20, 0 = all)\n"
      "  --tensor NAME print one tensor in detail\n"
      "  --verify      recompute every payload CRC32 and compare with the file\n");
}

std::string human(uint64_t bytes) {
  const char* u[] = {"B", "KB", "MB", "GB", "TB"};
  double v = static_cast<double>(bytes);
  int i = 0;
  while (v >= 1024.0 && i < 4) { v /= 1024.0; ++i; }
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.2f %s", v, u[i]);
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { usage(); return 2; }
  std::string path = argv[1];
  int list_n = 20;
  std::string one;
  bool verify = false;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--list") && i + 1 < argc) list_n = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--tensor") && i + 1 < argc) one = argv[++i];
    else if (!std::strcmp(argv[i], "--verify")) verify = true;
    else { usage(); return 2; }
  }

  ov::WeightFile wf;
  try {
    wf = ov::WeightFile::Open(path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }

  std::printf("file      : %s\n", wf.path().c_str());
  std::printf("size      : %s (%llu bytes)\n", human(wf.file_size()).c_str(),
              static_cast<unsigned long long>(wf.file_size()));
  std::printf("tensors   : %zu\n", wf.tensors().size());
  std::printf("params    : %.2f M\n", static_cast<double>(wf.n_params()) / 1e6);
  std::printf("dtype mode: %s\n", wf.dtype_mode() ? "fp32+int8" : "fp32");
  std::printf("cpu kernel: %s\n", ov::cpu_kernel_name());

  std::map<std::string, int> hist;
  uint64_t payload = 0, scales = 0;
  for (const auto& t : wf.tensors()) {
    hist[ov::dtype_name(t.dtype)]++;
    payload += static_cast<uint64_t>(t.nbytes);
    if (t.scales) scales += static_cast<uint64_t>(t.n_scales) * 4;
  }
  std::printf("payload   : %s  (scales %s)\n", human(payload).c_str(), human(scales).c_str());
  std::printf("dtypes    :");
  for (const auto& kv : hist) std::printf(" %s=%d", kv.first.c_str(), kv.second);
  std::printf("\n");

  if (!one.empty()) {
    try {
      const ov::Tensor& t = wf.Get(one);
      std::printf("\ntensor %s\n  dtype=%s shape=%s layout=%s quant=%s bytes=%lld\n",
                  t.name.c_str(), ov::dtype_name(t.dtype), t.shape.str().c_str(),
                  t.transposed ? "transposed" : "row-major",
                  t.quantized ? (t.transposed ? "int8/col-scale" : "int8/row-scale") : "none",
                  static_cast<long long>(t.nbytes));
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 1;
    }
  }

  if (list_n > 0) {
    std::printf("\n%-58s %-4s %-14s %-11s %s\n", "name", "type", "shape", "layout", "bytes");
    int shown = 0;
    for (const auto& t : wf.tensors()) {
      if (shown++ >= list_n) { std::printf("... (%zu total)\n", wf.tensors().size()); break; }
      std::printf("%-58s %-4s %-14s %-11s %lld\n", t.name.c_str(), ov::dtype_name(t.dtype),
                  t.shape.str().c_str(), t.transposed ? "transposed" : "row-major",
                  static_cast<long long>(t.nbytes));
    }
  }

  if (verify) {
    std::printf("\nverifying %zu payload CRCs ...\n", wf.tensors().size());
    size_t bad = 0;
    for (const auto& t : wf.tensors()) {
      if (wf.ComputeCrc(t) != wf.StoredCrc(t)) {
        std::printf("  !! CRC mismatch: %s\n", t.name.c_str());
        if (++bad > 20) { std::printf("  ... stopping after 20\n"); break; }
      }
    }
    if (bad == 0) std::printf("all payload CRCs OK\n");
    return bad ? 1 : 0;
  }
  return 0;
}

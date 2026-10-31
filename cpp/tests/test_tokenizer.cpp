// Tokenizer parity test against the Python (HF) tokenizer.
//
//   ov_test_tokenizer <tokenizer.ovt> <cases.bin>
//
// cases.bin layout: u32 n, then per case: u32 text_len, bytes, u32 n_ids, i32 ids.
#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tokenizer/bpe.h"

namespace {

std::vector<uint8_t> read_file(const std::string& p) {
  std::FILE* f = std::fopen(p.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", p.c_str());
    std::exit(2);
  }
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> v(static_cast<size_t>(n));
  if (std::fread(v.data(), 1, v.size(), f) != v.size()) std::exit(2);
  std::fclose(f);
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: ov_test_tokenizer <tokenizer.ovt> <cases.bin>\n");
    return 2;
  }
  ov::Tokenizer tok(argv[1]);
  const std::vector<uint8_t> blob = read_file(argv[2]);
  size_t pos = 0;
  auto rd_u32 = [&]() {
    uint32_t v;
    std::memcpy(&v, blob.data() + pos, 4);
    pos += 4;
    return v;
  };

  const uint32_t n_cases = rd_u32();
  size_t ok = 0, bad = 0, total_ids = 0;
  for (uint32_t c = 0; c < n_cases; ++c) {
    const uint32_t tlen = rd_u32();
    const std::string text(reinterpret_cast<const char*>(blob.data() + pos), tlen);
    pos += tlen;
    const uint32_t n_ids = rd_u32();
    std::vector<int32_t> want(n_ids);
    std::memcpy(want.data(), blob.data() + pos, n_ids * 4);
    pos += n_ids * 4;
    total_ids += n_ids;

    const std::vector<int32_t> got = tok.encode(text);
    if (got == want) {
      ++ok;
      continue;
    }
    ++bad;
    std::printf("MISMATCH: %s\n", text.c_str());
    std::printf("  want (%zu):", want.size());
    for (size_t i = 0; i < want.size() && i < 24; ++i) std::printf(" %d", want[i]);
    std::printf("\n  got  (%zu):", got.size());
    for (size_t i = 0; i < got.size() && i < 24; ++i) std::printf(" %d", got[i]);
    std::printf("\n  roundtrip: %s\n", tok.decode(got) == text ? "ok" : "DIFFERS");
  }

  std::printf("cases: %zu ok, %zu mismatched (%zu tokens, vocab=%d)\n", ok, bad, total_ids,
              tok.num_tokens());
  return bad == 0 ? 0 : 1;
}

// Qwen2 byte-level BPE, loaded from the OVTOK1 binary.
//
// No JSON parser, no ICU, no regex engine: the pre-tokeniser is a hand-written
// scanner for the exact Qwen2 split pattern, and every Unicode classification
// comes from tables compiled offline (see tools/convert_tokenizer.py).
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ov {

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& ovt_path);

  // Full pipeline: added-token split -> NFC(nop) -> Qwen2 pre-tokenise -> BPE.
  std::vector<int32_t> encode(const std::string& text) const;

  // Inverse mapping for debugging (byte-level decoder, no NFC re-composition).
  std::string decode(const std::vector<int32_t>& ids) const;

  // RuleDurationEstimator: frames needed for `text` at the pace of `ref_text`
  // spoken in `ref_frames` frames (25 Hz). Mirrors omnivoice.utils.duration.
  double estimate_frames(const std::string& text, const std::string& ref_text,
                         double ref_frames) const;

  std::vector<std::string> pretokenize(const std::string& text) const;

  int32_t num_tokens() const { return n_tokens_; }

 private:
  struct Range {
    uint32_t start, end;
  };
  struct CatRange {
    uint32_t start, end;
    uint8_t cls;
  };

  static bool in_ranges(const std::vector<Range>& r, uint32_t cp);
  bool is_letter(uint32_t cp) const { return in_ranges(letter_, cp); }
  bool is_number(uint32_t cp) const { return in_ranges(number_, cp); }
  bool is_space(uint32_t cp) const { return in_ranges(space_, cp); }
  uint8_t category_class(uint32_t cp) const;

  std::vector<uint32_t> bpe(std::string_view piece) const;
  int32_t added_token_at(std::string_view text, size_t pos, size_t* length) const;

  int32_t n_tokens_ = 0;
  std::vector<uint8_t> blob_;
  std::string_view vocab_strings_;
  std::vector<std::pair<uint32_t, uint32_t>> vocab_index_;   // (off, len)
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> merges_;  // (a,b) -> (rank, id)
  struct Added {
    std::string text;
    int32_t id;
  };
  std::vector<Added> added_;
  std::vector<uint32_t> byte_ids_;  // 256
  std::vector<Range> letter_, number_, space_;
  std::vector<CatRange> cats_;
  std::vector<std::pair<uint32_t, float>> script_;  // (end_cp, weight)
  float dur_[7] = {1, 0.2f, 0.5f, 3.5f, 0.f, 3.f, 1.f};  // latin space punct digit mark cjk default
};

}  // namespace ov

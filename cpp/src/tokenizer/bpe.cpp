#include "tokenizer/bpe.h"

#include "core/utf8.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ov {
namespace {

constexpr char kMagic[8] = {'O', 'V', 'T', 'O', 'K', '1', 0, 0};

uint32_t rd_u32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
uint64_t rd_u64(const uint8_t* p) {
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}
float rd_f32(const uint8_t* p) {
  float v;
  std::memcpy(&v, p, 4);
  return v;
}

struct Cp {
  uint32_t cp;
  uint32_t off;  // byte offset of this codepoint
};

std::vector<Cp> decode_utf8(std::string_view s) {
  std::vector<Cp> out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    const uint8_t c = static_cast<uint8_t>(s[i]);
    uint32_t cp = c;
    size_t extra = 0;
    if (c >= 0xF0) { cp = c & 0x07; extra = 3; }
    else if (c >= 0xE0) { cp = c & 0x0F; extra = 2; }
    else if (c >= 0xC0) { cp = c & 0x1F; extra = 1; }
    if (i + extra < s.size() + 1 && extra > 0) {
      for (size_t k = 1; k <= extra && i + k < s.size(); ++k) {
        cp = (cp << 6) | (static_cast<uint8_t>(s[i + k]) & 0x3F);
      }
      out.push_back({cp, static_cast<uint32_t>(i)});
      i += extra + 1;
    } else {
      out.push_back({cp, static_cast<uint32_t>(i)});
      i += 1;
    }
  }
  return out;
}

inline bool is_newline(uint32_t cp) { return cp == '\n' || cp == '\r'; }

// GPT-2 byte-level alphabet: which codepoints stand in for raw bytes.
std::array<uint32_t, 256> byte_level_codepoints() {
  std::array<uint32_t, 256> map{};
  std::vector<int> bs, cs;
  for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
  for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
  for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
  cs = bs;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      ++n;
    }
  }
  for (size_t i = 0; i < bs.size(); ++i) map[bs[i]] = static_cast<uint32_t>(cs[i]);
  return map;
}

}  // namespace

bool Tokenizer::in_ranges(const std::vector<Range>& r, uint32_t cp) {
  size_t lo = 0, hi = r.size();
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (cp < r[mid].start) hi = mid;
    else if (cp > r[mid].end) lo = mid + 1;
    else return true;
  }
  return false;
}

uint8_t Tokenizer::category_class(uint32_t cp) const {
  size_t lo = 0, hi = cats_.size();
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (cp < cats_[mid].start) hi = mid;
    else if (cp > cats_[mid].end) lo = mid + 1;
    else return cats_[mid].cls;
  }
  return 0;
}

Tokenizer::Tokenizer(const std::string& ovt_path) {
  std::FILE* f = fopen_utf8(ovt_path, "rb");
  if (!f) throw std::runtime_error("cannot open tokenizer file: " + ovt_path);
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n < 0) { std::fclose(f); throw std::runtime_error("cannot stat tokenizer file: " + ovt_path); }
  blob_.resize(static_cast<size_t>(n));
  if (n > 0 && std::fread(blob_.data(), 1, static_cast<size_t>(n), f) != static_cast<size_t>(n)) {
    std::fclose(f);
    throw std::runtime_error("cannot read tokenizer file: " + ovt_path);
  }
  std::fclose(f);
  if (blob_.size() < 16) throw std::runtime_error("tokenizer file too small");
  if (std::memcmp(blob_.data(), kMagic, 8) != 0) throw std::runtime_error("not an OVTOK1 file");
  const uint32_t version = rd_u32(blob_.data() + 8);
  const uint32_t n_sections = rd_u32(blob_.data() + 12);
  if (version != 1) throw std::runtime_error("unsupported OVTOK1 version");

  struct Sec {
    uint64_t off, len;
  };
  std::unordered_map<std::string, Sec> secs;
  const size_t dir_off = 16;
  const size_t names_off = dir_off + n_sections * 24;
  if (names_off > blob_.size()) throw std::runtime_error("OVTOK1 section table truncated");
  for (uint32_t i = 0; i < n_sections; ++i) {
    const uint8_t* e = blob_.data() + dir_off + i * 24;
    const uint32_t name_off = rd_u32(e);
    const uint32_t name_len = rd_u32(e + 4);
    const uint64_t off = rd_u64(e + 8);
    const uint64_t len = rd_u64(e + 16);
    if (names_off + name_off + name_len > blob_.size() || off + len > blob_.size()) {
      throw std::runtime_error("OVTOK1 section out of bounds");
    }
    secs.emplace(std::string(reinterpret_cast<const char*>(blob_.data() + names_off + name_off), name_len),
                 Sec{off, len});
  }
  auto need = [&](const char* name) {
    auto it = secs.find(name);
    if (it == secs.end()) throw std::runtime_error(std::string("OVTOK1 missing section: ") + name);
    return it->second;
  };

  // ---- vocabulary --------------------------------------------------------
  const Sec vs = need("vocab_strings");
  vocab_strings_ = std::string_view(reinterpret_cast<const char*>(blob_.data() + vs.off), vs.len);
  const Sec vi = need("vocab_index");
  n_tokens_ = static_cast<int32_t>(vi.len / 8);
  vocab_index_.resize(static_cast<size_t>(n_tokens_));
  for (int32_t i = 0; i < n_tokens_; ++i) {
    vocab_index_[static_cast<size_t>(i)] = {rd_u32(blob_.data() + vi.off + i * 8),
                                            rd_u32(blob_.data() + vi.off + i * 8 + 4)};
  }

  // ---- merges ------------------------------------------------------------
  const Sec mg = need("merges");
  const size_t n_merges = mg.len / 12;
  merges_.reserve(n_merges * 2);
  for (size_t i = 0; i < n_merges; ++i) {
    const uint8_t* p = blob_.data() + mg.off + i * 12;
    const uint32_t a = rd_u32(p), b = rd_u32(p + 4), m = rd_u32(p + 8);
    merges_.emplace((static_cast<uint64_t>(a) << 32) | b,
                    std::make_pair(static_cast<uint32_t>(i), m));
  }

  // ---- added tokens ------------------------------------------------------
  const Sec ad = need("added");
  const size_t n_added = ad.len / 12;
  for (size_t i = 0; i < n_added; ++i) {
    const uint8_t* p = blob_.data() + ad.off + i * 12;
    const uint32_t id = rd_u32(p), off = rd_u32(p + 4), len = rd_u32(p + 8);
    added_.push_back({std::string(vocab_strings_.substr(off, len)), static_cast<int32_t>(id)});
  }
  std::sort(added_.begin(), added_.end(),
            [](const Added& a, const Added& b) { return a.text.size() > b.text.size(); });

  // ---- byte map ----------------------------------------------------------
  const Sec bid = need("byte_ids");
  if (bid.len != 256 * 4) throw std::runtime_error("OVTOK1 byte_ids has wrong size");
  byte_ids_.resize(256);
  for (int i = 0; i < 256; ++i) byte_ids_[static_cast<size_t>(i)] = rd_u32(blob_.data() + bid.off + i * 4);

  // ---- unicode tables ----------------------------------------------------
  auto read_ranges = [&](const char* name) {
    const Sec s = need(name);
    std::vector<Range> r(s.len / 8);
    for (size_t i = 0; i < r.size(); ++i) {
      r[i] = {rd_u32(blob_.data() + s.off + i * 8), rd_u32(blob_.data() + s.off + i * 8 + 4)};
    }
    return r;
  };
  letter_ = read_ranges("unicode_letter");
  number_ = read_ranges("unicode_number");
  space_ = read_ranges("unicode_space");

  const Sec cat = need("unicode_category");
  cats_.resize(cat.len / 12);
  for (size_t i = 0; i < cats_.size(); ++i) {
    const uint8_t* p = blob_.data() + cat.off + i * 12;
    cats_[i] = {rd_u32(p), rd_u32(p + 4), static_cast<uint8_t>(rd_u32(p + 8))};
  }

  const Sec sc = need("duration_script");
  script_.resize(sc.len / 8);
  for (size_t i = 0; i < script_.size(); ++i) {
    script_[i] = {rd_u32(blob_.data() + sc.off + i * 8), rd_f32(blob_.data() + sc.off + i * 8 + 4)};
  }
  const Sec dc = need("duration_consts");
  for (int i = 0; i < 7; ++i) dur_[i] = rd_f32(blob_.data() + dc.off + i * 4);
}

int32_t Tokenizer::added_token_at(std::string_view text, size_t pos, size_t* length) const {
  for (const Added& a : added_) {
    if (pos + a.text.size() <= text.size() &&
        std::memcmp(text.data() + pos, a.text.data(), a.text.size()) == 0) {
      *length = a.text.size();
      return a.id;
    }
  }
  return -1;
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
  const std::vector<Cp> cps = decode_utf8(text);
  const size_t n = cps.size();
  std::vector<std::string> out;
  size_t i = 0;
  while (i < n) {
    size_t j = i;
    const uint32_t c = cps[i].cp;
    bool matched = false;

    // 1) (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (c == '\'' && i + 1 < n) {
      const uint32_t a = cps[i + 1].cp | 0x20;
      if (a == 's' || a == 't' || a == 'm' || a == 'd') { j = i + 2; matched = true; }
      else if ((a == 'r' || a == 'v') && i + 2 < n && (cps[i + 2].cp | 0x20) == 'e') {
        j = i + 3;
        matched = true;
      } else if (a == 'l' && i + 2 < n && (cps[i + 2].cp | 0x20) == 'l') {
        j = i + 3;
        matched = true;
      }
    }

    // 2) [^\r\n\p{L}\p{N}]?\p{L}+
    if (!matched) {
      if (is_letter(c)) {
        j = i;
        while (j < n && is_letter(cps[j].cp)) ++j;
        matched = true;
      } else if (i + 1 < n && is_letter(cps[i + 1].cp) && !is_newline(c) && !is_number(c)) {
        j = i + 1;
        while (j < n && is_letter(cps[j].cp)) ++j;
        matched = true;
      }
    }

    // 3) \p{N}
    if (!matched && is_number(c)) { j = i + 1; matched = true; }

    // 4) ` ?[^\s\p{L}\p{N}]+[\r\n]*`
    if (!matched) {
      size_t k = i;
      if (cps[k].cp == ' ') ++k;
      if (k < n && !is_space(cps[k].cp) && !is_letter(cps[k].cp) && !is_number(cps[k].cp)) {
        j = k;
        while (j < n && !is_space(cps[j].cp) && !is_letter(cps[j].cp) && !is_number(cps[j].cp)) ++j;
        while (j < n && is_newline(cps[j].cp)) ++j;
        matched = true;
      }
    }

    // 5) \s*[\r\n]+   -> ends right after the last newline in the run
    if (!matched && is_space(c)) {
      size_t m = i, last_nl = std::string::npos;
      while (m < n && is_space(cps[m].cp)) {
        if (is_newline(cps[m].cp)) last_nl = m;
        ++m;
      }
      if (last_nl != std::string::npos) {
        j = last_nl + 1;
        matched = true;
      } else if (m == n) {
        // 6) \s+(?!\S): the run reaches the end of the text.
        j = n;
        matched = true;
      } else if (m - i >= 2) {
        // 6) \s+(?!\S) with backtracking: the greedy run is followed by a
        // non-space, so the regex engine gives back one character and the
        // lookahead then sees a space. Net effect: match R-1 spaces.
        j = m - 1;
        matched = true;
      } else {
        // 7) \s+ (single space before a non-space)
        j = m;
        matched = true;
      }
    }

    if (!matched) j = i + 1;
    out.emplace_back(text.substr(cps[i].off,
                                 (j < n ? cps[j].off : text.size()) - cps[i].off));
    i = j;
  }
  return out;
}

std::vector<uint32_t> Tokenizer::bpe(std::string_view piece) const {
  std::vector<uint32_t> syms;
  syms.reserve(piece.size());
  for (char ch : piece) syms.push_back(byte_ids_[static_cast<uint8_t>(ch)]);

  while (syms.size() > 1) {
    uint32_t best_rank = UINT32_MAX;
    size_t best_i = 0;
    uint32_t best_id = 0;
    bool found = false;
    for (size_t i = 0; i + 1 < syms.size(); ++i) {
      const uint64_t key = (static_cast<uint64_t>(syms[i]) << 32) | syms[i + 1];
      auto it = merges_.find(key);
      if (it != merges_.end() && it->second.first < best_rank) {
        best_rank = it->second.first;
        best_i = i;
        best_id = it->second.second;
        found = true;
      }
    }
    if (!found) break;
    syms[best_i] = best_id;
    syms.erase(syms.begin() + static_cast<long>(best_i) + 1);
  }
  return syms;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
  std::vector<int32_t> ids;
  std::string pending;
  size_t i = 0;

  auto flush = [&]() {
    if (pending.empty()) return;
    for (const std::string& piece : pretokenize(pending)) {
      for (uint32_t id : bpe(piece)) ids.push_back(static_cast<int32_t>(id));
    }
    pending.clear();
  };

  while (i < text.size()) {
    size_t len = 0;
    const int32_t id = added_token_at(text, i, &len);
    if (id >= 0) {
      flush();
      ids.push_back(id);
      i += len;
    } else {
      pending.push_back(text[i]);
      ++i;
    }
  }
  flush();
  return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
  const std::array<uint32_t, 256> table = byte_level_codepoints();
  std::unordered_map<uint32_t, uint8_t> rev;
  rev.reserve(256);
  for (int b = 0; b < 256; ++b) rev[table[static_cast<size_t>(b)]] = static_cast<uint8_t>(b);

  std::string out;
  for (int32_t id : ids) {
    if (id < 0 || id >= n_tokens_) throw std::runtime_error("token id out of range");
    const auto [off, len] = vocab_index_[static_cast<size_t>(id)];
    const std::string_view tok = vocab_strings_.substr(off, len);
    for (const Cp& c : decode_utf8(tok)) {
      auto it = rev.find(c.cp);
      if (it != rev.end()) out.push_back(static_cast<char>(it->second));
      else out.append(tok.substr(c.off, c.off + 1 <= tok.size() ? 1 : 1));
    }
  }
  return out;
}

double Tokenizer::estimate_frames(const std::string& text, const std::string& ref_text,
                                  double ref_frames) const {
  if (ref_frames <= 0.0 || ref_text.empty()) return 0.0;

  auto weight_of = [&](uint32_t cp) -> double {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return dur_[0];
    if (cp == ' ') return dur_[1];
    if (cp == 0x0640) return dur_[4];
    const uint8_t cls = category_class(cp);
    if (cls == 3) return dur_[4];             // M
    if (cls == 4 || cls == 5) return dur_[2];  // P, S
    if (cls == 6) return dur_[1];              // Z
    if (cls == 2) return dur_[3];              // N
    size_t lo = 0, hi = script_.size();
    while (lo < hi) {
      const size_t mid = (lo + hi) / 2;
      if (cp <= script_[mid].first) hi = mid;
      else lo = mid + 1;
    }
    if (lo < script_.size()) return script_[lo].second;
    if (cp > 0x20000) return dur_[5];
    return dur_[6];
  };

  auto total = [&](const std::string& s) {
    double w = 0.0;
    for (const Cp& c : decode_utf8(s)) w += weight_of(c.cp);
    return w;
  };

  const double ref_weight = total(ref_text);
  if (ref_weight == 0.0) return 0.0;
  const double speed = ref_weight / ref_frames;
  const double est = total(text) / speed;
  const double low_threshold = 50.0, boost = 3.0;
  if (est < low_threshold) return low_threshold * std::pow(est / low_threshold, 1.0 / boost);
  return est;
}

}  // namespace ov

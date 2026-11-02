// UTF-8 path/argv helpers. On POSIX these are thin wrappers; on Windows the
// narrow CRT APIs use the active ANSI code page, so paths and CLI text must be
// converted to UTF-16 before hitting CreateFileW/_wfopen.
#pragma once

#include <cstdio>
#include <string>

#ifdef _WIN32
#include <windows.h>

namespace ov {
inline std::wstring utf8_to_wide(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(),
                                    static_cast<int>(s.size()), nullptr, 0);
  if (n <= 0) return std::wstring(s.begin(), s.end());
  std::wstring w(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()),
                      w.data(), n);
  return w;
}

inline std::FILE* fopen_utf8(const std::string& path, const char* mode) {
  const std::wstring wp = utf8_to_wide(path);
  std::wstring wm;
  for (const char* p = mode; *p; ++p) wm.push_back(static_cast<wchar_t>(*p));
  return _wfopen(wp.c_str(), wm.c_str());
}
}  // namespace ov
#else
namespace ov {
inline std::FILE* fopen_utf8(const std::string& path, const char* mode) {
  return std::fopen(path.c_str(), mode);
}
}  // namespace ov
#endif

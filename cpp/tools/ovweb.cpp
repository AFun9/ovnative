// ovweb — tiny local HTTP server + web UI for OVNative.
//
//   ovweb <lm.ovw> <at.ovw> <tokenizer.ovt>
//         [--host 127.0.0.1] [--port 8080]
//         [--web-root web] [--ref-dir web_refs] [--backend cpu|cuda]
//
// No external HTTP/JSON library. API:
//   GET  /api/health                -> {"ok":true,"cuda":false,...}
//   POST /api/ref                   -> raw WAV body, returns {"id":"..."}
//   POST /api/tts                   -> form-urlencoded fields, returns audio/wav
//   GET  /...                       -> static files from --web-root
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <shellapi.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "core/utf8.h"
#include "core/wav.h"
#include "model/tts.h"
#include "ovnative/ov_weights.h"
#ifdef OV_ENABLE_CUDA
#include <memory>
#include "backend/cuda/cuda_decoder.h"
#include "backend/cuda/cuda_encoder.h"
#include "backend/cuda/cuda_lm.h"
#endif

namespace {

#ifdef _WIN32
using socket_t = SOCKET;
constexpr socket_t kInvalidSocket = INVALID_SOCKET;
#else
using socket_t = int;
constexpr socket_t kInvalidSocket = -1;
#endif

void close_socket(socket_t s) {
#ifdef _WIN32
  ::closesocket(s);
#else
  ::close(s);
#endif
}

bool send_all(socket_t s, const void* data, size_t n) {
  const char* p = static_cast<const char*>(data);
  while (n > 0) {
    const int chunk = static_cast<int>(std::min<size_t>(n, 1u << 20));
#ifdef _WIN32
    const int r = ::send(s, p, chunk, 0);
#else
    const int r = static_cast<int>(::send(s, p, static_cast<size_t>(chunk), MSG_NOSIGNAL));
#endif
    if (r <= 0) return false;
    p += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

struct Request {
  std::string method;
  std::string path;
  std::string body;
  std::unordered_map<std::string, std::string> headers;
};

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

std::string trim(std::string s) {
  size_t a = 0, b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r')) ++a;
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) --b;
  return s.substr(a, b - a);
}

int hex_digit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

std::string url_decode(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '+') out.push_back(' ');
    else if (s[i] == '%' && i + 2 < s.size()) {
      const int a = hex_digit(s[i + 1]), b = hex_digit(s[i + 2]);
      if (a >= 0 && b >= 0) {
        out.push_back(static_cast<char>((a << 4) | b));
        i += 2;
      } else {
        out.push_back(s[i]);
      }
    } else {
      out.push_back(s[i]);
    }
  }
  return out;
}

std::unordered_map<std::string, std::string> parse_form(const std::string& body) {
  std::unordered_map<std::string, std::string> out;
  size_t pos = 0;
  while (pos <= body.size()) {
    const size_t amp = body.find('&', pos);
    const std::string item = body.substr(pos, amp == std::string::npos ? std::string::npos : amp - pos);
    const size_t eq = item.find('=');
    if (eq != std::string::npos) {
      out[url_decode(item.substr(0, eq))] = url_decode(item.substr(eq + 1));
    } else if (!item.empty()) {
      out[url_decode(item)] = "";
    }
    if (amp == std::string::npos) break;
    pos = amp + 1;
  }
  return out;
}

std::string field(const std::unordered_map<std::string, std::string>& f, const char* key,
                  const std::string& def = "") {
  auto it = f.find(key);
  return it == f.end() ? def : it->second;
}

int field_int(const std::unordered_map<std::string, std::string>& f, const char* key, int def) {
  auto it = f.find(key);
  if (it == f.end() || it->second.empty()) return def;
  char* end = nullptr;
  const long v = std::strtol(it->second.c_str(), &end, 10);
  return (end && *end == '\0') ? static_cast<int>(v) : def;
}

float field_float(const std::unordered_map<std::string, std::string>& f, const char* key,
                  float def) {
  auto it = f.find(key);
  if (it == f.end() || it->second.empty()) return def;
  char* end = nullptr;
  const float v = std::strtof(it->second.c_str(), &end);
  return (end && *end == '\0') ? v : def;
}

std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned char>(c));
          o += buf;
        } else {
          o.push_back(c);
        }
    }
  }
  return o;
}

std::string mime_of(const std::string& path) {
  auto ends = [&](const char* ext) {
    const size_t n = std::strlen(ext);
    return path.size() >= n && path.compare(path.size() - n, n, ext) == 0;
  };
  if (ends(".html")) return "text/html; charset=utf-8";
  if (ends(".css")) return "text/css; charset=utf-8";
  if (ends(".js")) return "application/javascript; charset=utf-8";
  if (ends(".json")) return "application/json; charset=utf-8";
  if (ends(".svg")) return "image/svg+xml";
  if (ends(".png")) return "image/png";
  if (ends(".jpg") || ends(".jpeg")) return "image/jpeg";
  if (ends(".wav")) return "audio/wav";
  if (ends(".ico")) return "image/x-icon";
  return "application/octet-stream";
}

bool read_file_bytes(const std::string& path, std::vector<uint8_t>& out) {
  std::FILE* f = ov::fopen_utf8(path, "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n < 0) {
    std::fclose(f);
    return false;
  }
  out.resize(static_cast<size_t>(n));
  const size_t got = n ? std::fread(out.data(), 1, static_cast<size_t>(n), f) : 0;
  std::fclose(f);
  return got == static_cast<size_t>(n);
}

class WebServer {
 public:
  WebServer(ov::Pipeline& pipe, bool cuda, std::string web_root, std::string ref_dir, int port,
            std::string host)
      : pipe_(pipe), cuda_(cuda), web_root_(std::move(web_root)), ref_dir_(std::move(ref_dir)),
        port_(port), host_(std::move(host)) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::u8path(ref_dir_), ec);
  }

  int run() {
#ifdef _WIN32
    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
      std::fprintf(stderr, "WSAStartup failed\n");
      return 1;
    }
#endif
    socket_t srv = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv == kInvalidSocket) {
      std::fprintf(stderr, "cannot create socket\n");
      return 1;
    }
    int yes = 1;
    ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
      std::fprintf(stderr, "invalid host: %s\n", host_.c_str());
      close_socket(srv);
      return 1;
    }
    if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      std::fprintf(stderr, "cannot bind %s:%d\n", host_.c_str(), port_);
      close_socket(srv);
      return 1;
    }
    if (::listen(srv, 16) != 0) {
      std::fprintf(stderr, "listen failed\n");
      close_socket(srv);
      return 1;
    }
    std::fprintf(stderr, "OVNative web: http://%s:%d  (cuda=%s, web=%s)\n", host_.c_str(), port_,
                 cuda_ ? "yes" : "no", web_root_.c_str());
    for (;;) {
      socket_t c = ::accept(srv, nullptr, nullptr);
      if (c == kInvalidSocket) continue;
      std::thread([this, c] {
        handle(c);
        close_socket(c);
      }).detach();
    }
    close_socket(srv);
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
  }

 private:
  bool read_request(socket_t s, Request& req) {
    std::string data;
    data.reserve(8192);
    char buf[8192];
    size_t header_end = std::string::npos;
    while ((header_end = data.find("\r\n\r\n")) == std::string::npos) {
      const int r = ::recv(s, buf, sizeof buf, 0);
      if (r <= 0) return false;
      data.append(buf, static_cast<size_t>(r));
      if (data.size() > 64 * 1024) return false;
    }
    const std::string head = data.substr(0, header_end);
    size_t line_end = head.find("\r\n");
    const std::string first = head.substr(0, line_end);
    const size_t sp1 = first.find(' ');
    const size_t sp2 = first.rfind(' ');
    if (sp1 == std::string::npos || sp2 == std::string::npos || sp2 <= sp1) return false;
    req.method = first.substr(0, sp1);
    std::string target = first.substr(sp1 + 1, sp2 - sp1 - 1);
    const size_t q = target.find('?');
    req.path = url_decode(target.substr(0, q));
    size_t pos = line_end == std::string::npos ? head.size() : line_end + 2;
    while (pos < head.size()) {
      const size_t e = head.find("\r\n", pos);
      const std::string line = head.substr(pos, e == std::string::npos ? std::string::npos : e - pos);
      const size_t colon = line.find(':');
      if (colon != std::string::npos) {
        req.headers[lower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
      }
      if (e == std::string::npos) break;
      pos = e + 2;
    }
    size_t content_length = 0;
    auto it = req.headers.find("content-length");
    if (it != req.headers.end()) {
      char* end = nullptr;
      const unsigned long long v = std::strtoull(it->second.c_str(), &end, 10);
      if (!end || *end != '\0' || v > 64ull * 1024 * 1024) return false;
      content_length = static_cast<size_t>(v);
    }
    req.body = data.substr(header_end + 4);
    while (req.body.size() < content_length) {
      const int r = ::recv(s, buf, sizeof buf, 0);
      if (r <= 0) return false;
      req.body.append(buf, static_cast<size_t>(r));
    }
    if (req.body.size() > content_length) req.body.resize(content_length);
    return true;
  }

  void respond(socket_t s, int status, const std::string& type, const void* body, size_t n,
               const std::string& extra = "") {
    const char* reason = status == 200 ? "OK" : status == 204 ? "No Content" : status == 400
                             ? "Bad Request" : status == 404 ? "Not Found" : status == 413
                             ? "Payload Too Large" : "Internal Server Error";
    char head[512];
    std::snprintf(head, sizeof head,
                  "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                  "Connection: close\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\n%s\r\n",
                  status, reason, type.c_str(), n, extra.c_str());
    if (!send_all(s, head, std::strlen(head))) return;
    if (n) send_all(s, body, n);
  }

  void respond_json(socket_t s, int status, const std::string& json) {
    respond(s, status, "application/json; charset=utf-8", json.data(), json.size());
  }

  void respond_error(socket_t s, int status, const std::string& msg) {
    respond_json(s, status, "{\"error\":\"" + json_escape(msg) + "\"}");
  }

  std::string ref_path(const std::string& id) const {
    if (id.empty() || id.size() > 64) return {};
    for (char c : id) {
      if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) return {};
    }
    return ref_dir_ + "/" + id + ".wav";
  }

  void handle(socket_t s) {
    Request req;
    if (!read_request(s, req)) return;
    if (req.method == "OPTIONS") {
      respond(s, 204, "text/plain", nullptr, 0, "Access-Control-Allow-Methods: GET,POST,OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\n");
      return;
    }
    if (req.method == "GET" && req.path == "/api/health") {
      respond_json(s, 200, std::string("{\"ok\":true,\"cuda\":") + (cuda_ ? "true" : "false") +
                                ",\"sample_rate\":24000}");
      return;
    }
    if (req.method == "POST" && req.path == "/api/ref") {
      if (req.body.size() < 44 || req.body.compare(0, 4, "RIFF") != 0) {
        respond_error(s, 400, "expected a RIFF/WAVE body");
        return;
      }
      if (req.body.size() > 32u * 1024 * 1024) {
        respond_error(s, 413, "reference audio is too large");
        return;
      }
      const uint64_t n = ++ref_counter_;
      const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now().time_since_epoch()).count();
      const std::string id = "r" + std::to_string(ms) + "_" + std::to_string(n);
      const std::string path = ref_dir_ + "/" + id + ".wav";
      std::FILE* f = ov::fopen_utf8(path, "wb");
      if (!f) {
        respond_error(s, 500, "cannot store reference audio");
        return;
      }
      const size_t wrote = std::fwrite(req.body.data(), 1, req.body.size(), f);
      std::fclose(f);
      if (wrote != req.body.size()) {
        respond_error(s, 500, "short write");
        return;
      }
      respond_json(s, 200, "{\"id\":\"" + id + "\"}");
      return;
    }
    if (req.method == "POST" && req.path == "/api/tts") {
      if (req.body.size() > 4u * 1024 * 1024) {
        respond_error(s, 413, "request body is too large");
        return;
      }
      const auto f = parse_form(req.body);
      const std::string text = field(f, "text");
      if (trim(text).empty()) {
        respond_error(s, 400, "text is required");
        return;
      }
      ov::TtsOptions opt;
      opt.language = field(f, "language", "Chinese");
      opt.instruct = field(f, "instruct");
      opt.num_steps = std::clamp(field_int(f, "steps", 32), 1, 200);
      opt.guidance_scale = std::clamp(field_float(f, "cfg", 2.0f), 0.0f, 10.0f);
      opt.seed = static_cast<uint32_t>(std::max(0, field_int(f, "seed", 1234)));
      opt.trim_silence = field_int(f, "trim", 1) != 0;
      opt.max_internal_silence_ms = std::clamp(field_int(f, "max_pause_ms", 300), 0, 5000);
      opt.keep_internal_silence_ms = std::clamp(field_int(f, "keep_pause_ms", 180), 0, 5000);
      const std::string ref_id = field(f, "ref_id");
      if (!ref_id.empty()) {
        const std::string rp = ref_path(ref_id);
        if (rp.empty() || !std::filesystem::exists(std::filesystem::u8path(rp))) {
          respond_error(s, 400, "unknown reference audio id");
          return;
        }
        opt.ref_audio = rp;
        opt.ref_text = field(f, "ref_text");
      }
      try {
        std::lock_guard<std::mutex> lock(synth_mu_);
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> audio = pipe_.synthesize(text, opt);
        const auto t1 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[tts] %zu samples in %.2fs\n", audio.size(),
                     std::chrono::duration<double>(t1 - t0).count());
        const std::vector<uint8_t> wav = ov::wav_bytes_mono16(audio.data(), audio.size(), 24000);
        respond(s, 200, "audio/wav", wav.data(), wav.size(),
                "Content-Disposition: inline; filename=\"ovnative.wav\"\r\n");
      } catch (const std::exception& e) {
        respond_error(s, 500, e.what());
      }
      return;
    }
    if (req.method == "GET") {
      serve_static(s, req.path);
      return;
    }
    respond_error(s, 404, "not found");
  }

  void serve_static(socket_t s, const std::string& url_path) {
    std::string rel = url_path;
    while (!rel.empty() && rel.front() == '/') rel.erase(rel.begin());
    if (rel.empty()) rel = "index.html";
    if (rel.find("..") != std::string::npos || rel.find('\\') != std::string::npos) {
      respond_error(s, 404, "not found");
      return;
    }
    std::vector<uint8_t> data;
    if (!read_file_bytes(web_root_ + "/" + rel, data)) {
      respond_error(s, 404, "not found");
      return;
    }
    respond(s, 200, mime_of(rel), data.data(), data.size());
  }

  ov::Pipeline& pipe_;
  bool cuda_;
  std::string web_root_;
  std::string ref_dir_;
  int port_;
  std::string host_;
  std::mutex synth_mu_;
  std::atomic<uint64_t> ref_counter_{0};
};

std::vector<std::string> utf8_args(int argc, char** argv) {
  std::vector<std::string> out;
#ifdef _WIN32
  int wargc = 0;
  LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
  if (wargv) {
    out.reserve(static_cast<size_t>(wargc));
    for (int i = 0; i < wargc; ++i) {
      const int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
      std::string a(n > 0 ? static_cast<size_t>(n) : 0, '\0');
      if (n > 0) WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, a.data(), n, nullptr, nullptr);
      if (n > 0) a.resize(static_cast<size_t>(n - 1));
      out.push_back(std::move(a));
    }
    LocalFree(wargv);
    return out;
  }
#endif
  out.reserve(static_cast<size_t>(argc));
  for (int i = 0; i < argc; ++i) out.emplace_back(argv[i] ? argv[i] : "");
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args = utf8_args(argc, argv);
  if (args.size() < 4) {
    std::fprintf(stderr,
                 "usage: ovweb <lm.ovw> <at.ovw> <tokenizer.ovt>\n"
                 "             [--host 127.0.0.1] [--port 8080]\n"
                 "             [--web-root web] [--ref-dir web_refs]\n"
                 "             [--backend cpu|cuda]\n");
    return 2;
  }
  const std::string lm_path = args[1], at_path = args[2], tok_path = args[3];
  std::string host = "127.0.0.1", web_root = "web", ref_dir = "web_refs", backend = "cpu";
  int port = 8080;
  for (size_t i = 4; i < args.size(); ++i) {
    auto need = [&](const char* what) -> std::string {
      if (i + 1 >= args.size()) {
        std::fprintf(stderr, "%s needs a value\n", what);
        std::exit(2);
      }
      return args[++i];
    };
    if (args[i] == "--host") host = need("--host");
    else if (args[i] == "--port") port = std::atoi(need("--port").c_str());
    else if (args[i] == "--web-root") web_root = need("--web-root");
    else if (args[i] == "--ref-dir") ref_dir = need("--ref-dir");
    else if (args[i] == "--backend") backend = need("--backend");
    else {
      std::fprintf(stderr, "unknown option: %s\n", args[i].c_str());
      return 2;
    }
  }
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif
#ifndef _WIN32
  std::signal(SIGPIPE, SIG_IGN);
#endif

  try {
    ov::WeightFile lm_wf = ov::WeightFile::Open(lm_path);
    ov::WeightFile at_wf = ov::WeightFile::Open(at_path);
    ov::Pipeline pipe(lm_wf, at_wf, tok_path);
    bool cuda = false;
    if (backend == "cuda" && pipe.lm().has_nk_weights()) {
      std::fprintf(stderr,
                   "%s uses [N,K] int8 weights; use the pre-transposed omnivoice-lm-int8.ovw "
                   "for --backend cuda\n",
                   lm_path.c_str());
      return 2;
    }
#ifdef OV_ENABLE_CUDA
    ov::cuda::LmRunnerPtr runner{nullptr, &ov::cuda::LmRunnerDelete};
    ov::cuda::DecoderRunnerPtr dec_runner{nullptr, &ov::cuda::DecoderRunnerDelete};
    ov::cuda::EncoderRunnerPtr enc_runner{nullptr, &ov::cuda::EncoderRunnerDelete};
    if (backend == "cuda") {
      runner = ov::cuda::LmRunnerCreate(lm_wf, pipe.lm().config());
      if (runner) {
        pipe.lm().set_cuda_backend(runner.get(), &ov::cuda::LmRunnerForwardThunk,
                                   &ov::cuda::LmRunnerLastLogitsThunk);
        pipe.lm().set_cuda_sampler(runner.get(), &ov::cuda::LmRunnerSampleThunk);
        ov::cuda::LmRunnerSetDeviceSampling(*runner, true);
        enc_runner = ov::cuda::EncoderRunnerCreate(pipe.encoder());
        if (enc_runner) {
          pipe.encoder().set_cuda_encode(enc_runner.get(), &ov::cuda::EncoderRunnerEncodeThunk);
        }
        dec_runner = ov::cuda::DecoderRunnerCreate(pipe.decoder());
        if (dec_runner) {
          pipe.set_cuda_decoder(dec_runner.get(), &ov::cuda::DecoderRunnerDecodeThunk);
        }
        cuda = true;
      } else {
        std::fprintf(stderr, "CUDA backend unavailable, falling back to CPU\n");
      }
    }
#else
    if (backend == "cuda") std::fprintf(stderr, "built without CUDA, falling back to CPU\n");
#endif
    WebServer server(pipe, cuda, web_root, ref_dir, port, host);
    return server.run();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}

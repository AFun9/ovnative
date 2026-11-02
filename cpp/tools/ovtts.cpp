// ovtts — text to speech with zero Python / zero ONNX Runtime.
//
//   ovtts <lm.ovw> <at.ovw> <tokenizer.ovt> "<text>" <out.wav>
//         [--lang English] [--instruct "male, british accent"]
//         [--ref-audio ref.wav --ref-text "transcript"]
//         [--steps 32] [--cfg 2.0] [--seed 1234] [--codes codes.i64]
//         [--no-trim-silence] [--max-pause-ms N] [--keep-pause-ms N]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

#include "core/utf8.h"
#include "core/wav.h"
#include "model/audio_io.h"
#ifdef OV_ENABLE_CUDA
#include <memory>
#include "backend/cuda/cuda_lm.h"
#include "backend/cuda/cuda_decoder.h"
#include "backend/cuda/cuda_encoder.h"
#endif
#include "model/tts.h"
#include "ovnative/ov_weights.h"

int main(int argc, char** argv) {
#ifdef _WIN32
  // The CRT hands `main` ANSI/ACP strings; OVNative's tokenizer expects UTF-8.
  // Rebuild argv from the wide command line (covers Chinese text and paths).
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
  int wargc = 0;
  LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
  std::vector<std::string> utf8_args;
  std::vector<char*> utf8_argv;
  if (wargv) {
    utf8_args.reserve(static_cast<size_t>(wargc));
    for (int i = 0; i < wargc; ++i) {
      const int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
      std::string a(n > 0 ? static_cast<size_t>(n) : 0, '\0');
      if (n > 0) WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, a.data(), n, nullptr, nullptr);
      if (n > 0) a.resize(static_cast<size_t>(n - 1));
      utf8_args.push_back(std::move(a));
    }
    LocalFree(wargv);
    utf8_argv.reserve(utf8_args.size());
    for (auto& a : utf8_args) utf8_argv.push_back(a.data());
    argc = wargc;
    argv = utf8_argv.data();
  }
#endif
  if (argc < 6) {
    std::fprintf(stderr,
                 "usage: ovtts <lm.ovw> <at.ovw> <tokenizer.ovt> \"<text>\" <out.wav>\n"
                 "             [--lang NAME] [--instruct STR] [--steps N] [--cfg F]\n"
                 "             [--ref-audio WAV] [--ref-text TEXT]\n"
                 "             [--backend cpu|cuda]\n"
                 "             [--seed N] [--codes path]\n"
                 "             [--no-trim-silence] [--max-pause-ms N] [--keep-pause-ms N]\n");
    return 2;
  }

  const std::string lm_path = argv[1];
  const std::string at_path = argv[2];
  const std::string tok_path = argv[3];
  const std::string text = argv[4];
  const std::string out_path = argv[5];

  ov::TtsOptions opt;
  std::string codes_out;
  std::string backend = "cpu";
  for (int i = 6; i < argc; ++i) {
    auto need = [&](const char* what) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", what);
        std::exit(2);
      }
      return std::string(argv[++i]);
    };
    if (!std::strcmp(argv[i], "--lang")) opt.language = need("--lang");
    else if (!std::strcmp(argv[i], "--instruct")) opt.instruct = need("--instruct");
    else if (!std::strcmp(argv[i], "--steps")) opt.num_steps = std::atoi(need("--steps").c_str());
    else if (!std::strcmp(argv[i], "--cfg")) opt.guidance_scale = std::stof(need("--cfg"));
    else if (!std::strcmp(argv[i], "--seed")) opt.seed = static_cast<uint32_t>(std::stoul(need("--seed")));
    else if (!std::strcmp(argv[i], "--ref-audio")) opt.ref_audio = need("--ref-audio");
    else if (!std::strcmp(argv[i], "--ref-text")) opt.ref_text = need("--ref-text");
    else if (!std::strcmp(argv[i], "--codes")) codes_out = need("--codes");
    else if (!std::strcmp(argv[i], "--backend")) backend = need("--backend");
    else if (!std::strcmp(argv[i], "--no-trim-silence")) opt.trim_silence = false;
    else if (!std::strcmp(argv[i], "--max-pause-ms")) opt.max_internal_silence_ms = std::atoi(need("--max-pause-ms").c_str());
    else if (!std::strcmp(argv[i], "--keep-pause-ms")) opt.keep_internal_silence_ms = std::atoi(need("--keep-pause-ms").c_str());
    else {
      std::fprintf(stderr, "unknown option %s\n", argv[i]);
      return 2;
    }
  }

  try {
    const auto t0 = std::chrono::steady_clock::now();
    ov::WeightFile lm_wf = ov::WeightFile::Open(lm_path);
    ov::WeightFile at_wf = ov::WeightFile::Open(at_path);
    ov::Pipeline pipe(lm_wf, at_wf, tok_path);
    if (backend == "cuda" && pipe.lm().has_nk_weights()) {
      std::fprintf(stderr,
                   "%s uses [N,K] int8 weights (CPU/VNNI layout); use the pre-transposed "
                   "omnivoice-lm-int8.ovw for --backend cuda\n",
                   lm_path.c_str());
      return 2;
    }
#ifdef OV_ENABLE_CUDA
    ov::cuda::LmRunnerPtr runner{nullptr, &ov::cuda::LmRunnerDelete};
    ov::cuda::DecoderRunnerPtr dec_runner{nullptr, &ov::cuda::DecoderRunnerDelete};
    ov::cuda::EncoderRunnerPtr enc_runner{nullptr, &ov::cuda::EncoderRunnerDelete};
    if (backend == "cuda") {
      runner = ov::cuda::LmRunnerCreate(lm_wf, pipe.lm().config());
      if (!runner) {
        std::fprintf(stderr, "CUDA backend unavailable, falling back to CPU\n");
      } else {
        pipe.lm().set_cuda_backend(runner.get(), &ov::cuda::LmRunnerForwardThunk,
                                   &ov::cuda::LmRunnerLastLogitsThunk);
        // Guidance + argmax on the device: skips the S*C*V logits round trip.
        pipe.lm().set_cuda_sampler(runner.get(), &ov::cuda::LmRunnerSampleThunk);
        ov::cuda::LmRunnerSetDeviceSampling(*runner, true);
        if (!opt.ref_audio.empty() && pipe.has_encoder()) {
          enc_runner = ov::cuda::EncoderRunnerCreate(pipe.encoder());
          if (enc_runner) {
            pipe.encoder().set_cuda_encode(enc_runner.get(), &ov::cuda::EncoderRunnerEncodeThunk);
            std::fprintf(stderr, "encoder: cuda\n");
          } else {
            std::fprintf(stderr, "encoder: CPU (CUDA runner unavailable)\n");
          }
        }
        dec_runner = ov::cuda::DecoderRunnerCreate(pipe.decoder());
        if (dec_runner) {
          pipe.set_cuda_decoder(dec_runner.get(), &ov::cuda::DecoderRunnerDecodeThunk);
          std::fprintf(stderr, "backend: cuda (LM + decoder)\n");
        } else {
          std::fprintf(stderr, "backend: cuda (LM); decoder CPU\n");
        }
      }
    }
#else
    if (backend == "cuda") {
      std::fprintf(stderr, "built without CUDA, falling back to CPU\n");
    }
#endif
    const auto t1 = std::chrono::steady_clock::now();

    auto prompt = pipe.build_prompt(text, opt);
    std::fprintf(stderr, "prompt: seq=%d ref_frames=%d target_frames=%d (%.2fs audio)\n",
                 prompt.seq, prompt.ref_frames, prompt.target_len, prompt.target_len / 25.0);

    std::vector<int64_t> codes = pipe.run_sampling(prompt, opt);
    const auto t2 = std::chrono::steady_clock::now();

    const int frames = prompt.target_len;
    std::vector<float> audio(static_cast<size_t>(frames) * 960);
    // Decode through the same object the pipeline owns by re-running synthesis
    // would duplicate the sampling; instead decode here via the pipeline.
    // (Pipeline::synthesize does sampling+decoding; we already have codes, so
    //  expose the decoder through a small helper.)
    std::vector<float> wav = pipe.decode_codes(codes, frames);
    audio.swap(wav);
    if (opt.trim_silence) {
      ov::trim_output_silence(audio, 24000, opt.trim_lead_ms, opt.trim_trail_ms,
                              opt.max_internal_silence_ms, opt.keep_internal_silence_ms);
    }
    const auto t3 = std::chrono::steady_clock::now();

    if (!codes_out.empty()) {
      std::FILE* f = ov::fopen_utf8(codes_out, "wb");
      if (f) {
        std::fwrite(codes.data(), sizeof(int64_t), codes.size(), f);
        std::fclose(f);
      }
    }
    if (!ov::write_wav_mono16(out_path, audio.data(), audio.size(), 24000)) {
      std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
      return 2;
    }

    const double dur = audio.size() / 24000.0;
    const double load_s = std::chrono::duration<double>(t1 - t0).count();
    const double gen_s = std::chrono::duration<double>(t2 - t1).count();
    const double dec_s = std::chrono::duration<double>(t3 - t2).count();
    std::printf("wrote %s  audio=%.2fs  load=%.2fs  sample=%.2fs  decode=%.2fs  RTF=%.2f\n",
                out_path.c_str(), dur, load_s, gen_s, dec_s, (gen_s + dec_s) / dur);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}

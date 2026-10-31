// WAV I/O and the reference-audio preprocessing that `create_voice_clone_prompt`
// applies before encoding.
//
// The preprocessing must match `omnivoice.utils.audio.remove_silence`, which is
// built on pydub: samples are converted to int16, silence is detected on
// `min_silence_len` windows every `seek_step` ms against a -50 dBFS threshold
// (computed on int16 RMS), long interior silences are cut with `keep_silence`
// ms of context, and the edges are trimmed to `lead_sil` / `trail_sil` ms.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ov {

// Reads a PCM WAV file and returns mono float32 at `target_rate` (24 kHz).
// Only 16/24/32-bit PCM and IEEE-float WAVs are supported; anything else
// throws. Resampling (when the file is not 24 kHz) is a simple windowed-sinc.
std::vector<float> read_wav_mono(const std::string& path, int target_rate = 24000);

// RMS normalisation: 0 < rms < 0.1 -> scale so that rms == 0.1.
void normalize_ref_rms(std::vector<float>& wav);

// pydub-equivalent silence removal (mid=200ms, lead=100ms, trail=200ms).
void remove_silence(std::vector<float>& wav, int sample_rate, int mid_sil = 200,
                    int lead_sil = 100, int trail_sil = 200);

}  // namespace ov

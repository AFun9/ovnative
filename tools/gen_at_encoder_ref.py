#!/usr/bin/env python3
"""Dump every intermediate of the OmniVoice audio-tokenizer *encoder*.

The C++ encoder is built stage by stage against these tensors, so each stage can
be validated in isolation instead of only at the final codes.

    python tools/gen_at_encoder_ref.py --seconds 2
    # -> output/ref/at_enc/*.f32 (+ meta.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import HiggsAudioV2TokenizerModel

REPO = Path(__file__).resolve().parents[1]


def dump(out: Path, name: str, t) -> None:
    a = t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t, dtype=np.float32)
    a.astype(np.float32).tofile(out / f"{name}.f32")
    shapes[name] = list(a.shape)


shapes: dict[str, list[int]] = {}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(REPO / "OmniVoice" / "audio_tokenizer"))
    p.add_argument("--audio", default=str(REPO / "assets" / "zero_shot_prompt.wav"))
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--out", default=str(REPO / "output" / "ref" / "at_enc"))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = HiggsAudioV2TokenizerModel.from_pretrained(args.model)
    tok.eval()

    from omnivoice.utils.audio import load_audio, remove_silence  # voice-clone preprocessing

    wav, sr = torchaudio.load(args.audio)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    hop = tok.config.hop_length
    n = int(args.seconds * 24000)
    n = (n // hop) * hop
    wav = wav[:, :n].unsqueeze(0)  # [1, 1, N]
    dump(out, "input_24k", wav[0, 0])

    # reference-audio preprocessing used by create_voice_clone_prompt
    ref = load_audio(args.audio, 24000)
    rms = float(np.sqrt((ref ** 2).mean()))
    if 0 < rms < 0.1:
        ref = ref * (0.1 / rms)          # create_voice_clone_prompt's RMS normalisation
    ref_trim = remove_silence(ref.copy(), 24000, mid_sil=200, lead_sil=100, trail_sil=200)
    dump(out, "ref_preprocessed", ref_trim[0])
    print(f"ref preprocess: {ref.shape[-1]} -> {ref_trim.shape[-1]} samples")

    with torch.inference_mode():
        # ---- resample 24k -> 16k ----
        wav16 = torchaudio.functional.resample(wav, 24000, tok.config.semantic_sample_rate)
        dump(out, "input_16k", wav16[0, 0])

        # ---- HuBERT ----
        padded = torch.nn.functional.pad(wav16[0], (160, 160))
        dump(out, "hubert_padded", padded)
        feats = tok.semantic_model.feature_extractor(padded)  # [B, T] -> [B, C, T]
        dump(out, "hubert_feature_extractor", feats[0])
        proj = tok.semantic_model.feature_projection(feats.transpose(1, 2))
        dump(out, "hubert_projection", proj[0])
        # exactly what OmniVoice._extract_semantic_features does: the *model's*
        # hidden-state tuple (projection output + every layer input + final)
        model_out = tok.semantic_model(padded, output_hidden_states=True)
        hs = model_out.hidden_states
        print("model hidden_states:", len(hs), "shapes:", [tuple(h.shape) for h in hs][:2], "...")
        stack = torch.stack([h for h in hs], dim=1).mean(dim=1)
        dump(out, "hubert_mean_hidden", stack[0])
        sem = stack[:, ::2, :]
        dump(out, "semantic_downsampled", sem[0])

        # ---- semantic encoder ----
        e_sem = tok.encoder_semantic(sem.transpose(1, 2))
        dump(out, "e_semantic", e_sem[0])

        # ---- acoustic encoder ----
        # mirror HiggsAudioV2TokenizerModel.encode: only pad when the padded and
        # unpadded frame counts would differ from the semantic frame count
        need_pad = (
            tok._get_conv1d_output_lengths(wav.shape[2], tok.acoustic_encoder) != e_sem.shape[2]
        )
        acoustic_in = torch.nn.functional.pad(wav, (tok.pad, tok.pad)) if need_pad else wav
        print("acoustic pad:", need_pad, "frames:", acoustic_in.shape[2] // tok.config.hop_length)
        dump(out, "acoustic_input", acoustic_in[0, 0])
        e_ac = tok.acoustic_encoder(acoustic_in)
        dump(out, "e_acoustic", e_ac[0])

        emb = torch.cat([e_ac, e_sem], dim=1)
        emb = tok.fc(emb.transpose(1, 2)).transpose(1, 2)
        dump(out, "embeddings", emb[0])

        codes = tok.quantizer.encode(emb, tok.config.target_bandwidths[-1])
        dump(out, "codes", codes.float())
        print("codes:", tuple(codes.shape), "range", int(codes.min()), int(codes.max()))

    (out / "meta.json").write_text(json.dumps({"shapes": shapes, "seconds": args.seconds}, indent=1))
    print(f"wrote {len(shapes)} tensors to {out}")
    for k, v in shapes.items():
        print(f"  {k:28s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

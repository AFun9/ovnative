#!/usr/bin/env python3
"""Reference for the C++ acoustic decoder: random codes -> ONNX decoder -> wav.

The C++ side (`ov_test_at`) must reproduce the waveform.  This is the P3a gate:
RVQ lookup + fc2 + DAC decoder (Snake before ConvTranspose, output_padding =
stride % 2, no tanh).

    python tools/gen_at_reference.py --frames 50
    python tools/gen_at_reference.py --frames 87 --batch 1 --seed 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", default=str(REPO / "output" / "audio_tokenizer_decoder" / "model.onnx"))
    p.add_argument("--out", default=str(REPO / "output" / "ref" / "at_dec"))
    p.add_argument("--frames", type=int, default=50, help="codec frames (25 Hz); 50 = 2 s")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--provider", default="CPUExecutionProvider")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    codes = rng.integers(0, 1024, size=(1, 8, args.frames)).astype(np.int64)

    sess = ort.InferenceSession(args.onnx, providers=[args.provider])
    audio = sess.run(["audio"], {"audio_codes": codes})[0]
    assert audio.shape == (1, 1, args.frames * 960), audio.shape

    out = Path(args.out) / f"f{args.frames}"
    out.mkdir(parents=True, exist_ok=True)
    codes.tofile(out / "codes.i64")
    audio.astype(np.float32).tofile(out / "audio.f32")
    (out / "meta.json").write_text(json.dumps({
        "frames": args.frames, "seed": args.seed, "samples": int(audio.size),
        "audio_max": float(np.abs(audio).max()), "audio_rms": float(np.sqrt((audio ** 2).mean())),
    }, indent=2))
    print(f"frames={args.frames} -> {out}  audio={audio.shape} "
          f"max|.|={np.abs(audio).max():.4f} rms={np.sqrt((audio**2).mean()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

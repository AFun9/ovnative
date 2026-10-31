#!/usr/bin/env python3
"""Generate a deterministic LM reference (inputs + logits) from the FP32 ONNX model.

The C++ engine's acceptance gate for phase P2 is: given byte-identical inputs,
`ov::LM::forward` must reproduce these logits.  Nothing "looks roughly right" —
max|Δ| and cosine are printed by `ov_test_lm` and it exits non-zero on drift.

    python tools/gen_lm_reference.py --seq 64 --mask full
    python tools/gen_lm_reference.py --seq 96 --mask block --batch 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[1]


def block_mask(batch: int, seq: int, text_len: int) -> np.ndarray:
    """Mimic `_generate_iterative`: full attention inside the real prefix,
    diagonal-only for the padded tail."""
    m = np.zeros((batch, 1, seq, seq), dtype=bool)
    for b in range(batch):
        valid = max(1, seq - b * 8)
        m[b, 0, :valid, :valid] = True
        if valid < seq:
            d = np.arange(valid, seq)
            m[b, 0, d, d] = True
    return m


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", default=str(REPO / "output" / "omnivoice_lm" / "model.onnx"))
    p.add_argument("--out", default=str(REPO / "output" / "ref" / "lm"))
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--text-len", type=int, default=0, help="default: seq // 2")
    p.add_argument("--mask", default="full", choices=["full", "block"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--provider", default="CPUExecutionProvider")
    args = p.parse_args()

    B, S = args.batch, args.seq
    text_len = args.text_len or S // 2
    rng = np.random.default_rng(args.seed)

    # layer 0 carries text ids on text positions and audio ids on audio positions
    input_ids = rng.integers(0, 1025, size=(B, 8, S), dtype=np.int64)
    input_ids[:, 0, :text_len] = rng.integers(0, 151676, size=(B, text_len))

    audio_mask = np.zeros((B, S), dtype=bool)
    audio_mask[:, text_len:] = True

    attn = (np.ones((B, 1, S, S), dtype=bool) if args.mask == "full"
            else block_mask(B, S, text_len))
    position_ids = np.tile(np.arange(S, dtype=np.int64), (B, 1))

    sess = ort.InferenceSession(args.onnx, providers=[args.provider])
    feeds = {
        "input_ids": input_ids,
        "audio_mask": audio_mask,
        "attention_mask": attn,
        "position_ids": position_ids,
    }
    logits = sess.run(["logits"], feeds)[0]
    assert logits.shape == (B, 8, S, 1025), logits.shape

    out = Path(args.out) / f"b{B}_s{S}_{args.mask}"
    out.mkdir(parents=True, exist_ok=True)
    input_ids.tofile(out / "input_ids.i64")
    audio_mask.astype(np.uint8).tofile(out / "audio_mask.u8")
    attn.astype(np.uint8).tofile(out / "attention_mask.u8")
    position_ids.tofile(out / "position_ids.i64")
    logits.astype(np.float32).tofile(out / "logits.f32")
    (out / "meta.json").write_text(json.dumps({
        "batch": B, "seq": S, "text_len": text_len, "mask": args.mask,
        "seed": args.seed, "onnx": str(args.onnx), "provider": args.provider,
        "logits_shape": list(logits.shape),
        "logits_max": float(np.abs(logits).max()),
        "logits_mean": float(logits.mean()),
    }, indent=2))

    print(f"[{args.mask}] B={B} S={S} text_len={text_len} -> {out}")
    print(f"  logits {logits.shape} max|.|={np.abs(logits).max():.4f} mean={logits.mean():+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

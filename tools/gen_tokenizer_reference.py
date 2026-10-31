#!/usr/bin/env python3
"""Reference token ids for the C++ BPE implementation.

The corpus deliberately covers the cases where a hand-written pre-tokeniser
usually diverges from the Qwen2 regex: contractions, digits, punctuation runs,
newlines, trailing whitespace, CJK punctuation, emoji and the OmniVoice control
tokens (which are added tokens, not BPE).

    python tools/gen_tokenizer_reference.py
    # -> output/ref/tokenizer/cases.tsv   (text<TAB>id,id,id)
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]

CASES = [
    "Hello, this is OmniVoice running entirely on ONNX Runtime.",
    "你好，欢迎来到 OmniVoice。这是一段使用语音克隆模式生成的中文语音。",
    "<|denoise|><|lang_start|>English<|lang_end|><|instruct_start|>male, british accent, low pitch, middle-aged<|instruct_end|>",
    "<|text_start|>Hello, world!<|text_end|>",
    "don't we're I'll I've he'd they'll",
    "12345 3.14 0.5% $100",
    "  leading and trailing   ",
    "line1\nline2\r\nline3",
    "emoji 🙂 test café naïve",
    "a-b_c/d\\e (paren) [bracket] {brace}",
    "   ",
    "（括号）和，。！？；：",
    "Mixed 中英文 text with 123 numbers.",
    "\t\t tabs\tand  spaces ",
    "UPPER lower MiXeD",
    "supercalifragilisticexpialidocious",
    "https://example.com/path?q=1",
    "<|text_start|>Nice to meet you. Hello.<|text_end|>",
    "'s 't 're 've 'm 'll 'd",
    "a  b   c    d",
    "x\t\n\ty",
    "  \n  ",
    "123abc456",
    "\u00dcn\u00efc\u00f6d\u00e9 \u00e0\u00e7cents",
    "\ud55c\uad6d\uc5b4 \u65e5\u672c\u8a9e \u0e44\u0e17\u0e22",
    "Hello...!!! ???",
    "word'word",
    "a'b'c",
    "  \t \n ",
    "e.g. i.e. U.S.A.",
    "\u4e2d\u6587English\u6df7\u6392 123",
    "trailing space ",
    " leading space",
    "\n\n\n",
    "\r\n\r\n",
    "line1\nline2\n\nline3",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(REPO / "OmniVoice"))
    p.add_argument("--out", default=str(REPO / "output" / "ref" / "tokenizer"))
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lines = []
    blob = bytearray(struct.pack("<I", len(CASES)))
    for text in CASES:
        ids = tok(text, add_special_tokens=False).input_ids
        lines.append(json.dumps({"text": text, "ids": ids}, ensure_ascii=False))
        raw = text.encode("utf-8")
        blob += struct.pack("<I", len(raw)) + raw
        blob += struct.pack("<I", len(ids)) + struct.pack(f"<{len(ids)}i", *ids)

    (out / "cases.jsonl").write_text("\n".join(lines) + "\n")
    (out / "cases.bin").write_bytes(bytes(blob))
    total = sum(len(json.loads(x)["ids"]) for x in lines)
    print(f"wrote {out/'cases.bin'}  cases={len(CASES)} tokens={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

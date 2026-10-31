#!/usr/bin/env python3
"""Compile tokenizer.json into OVTOK1 — a self-describing binary for the C++ engine.

Why not parse JSON at run time?  Because it would drag a JSON parser (and a
Unicode database) into the engine.  Instead everything the C++ side needs is
precomputed here, with CPython's own `unicodedata` as the source of truth:

  * vocabulary strings + index (id -> bytes)
  * BPE merges, already resolved to (id_a, id_b) -> id_merged, in rank order
  * added/special tokens (>= 151643) for greedy matching
  * byte -> initial token id (the GPT-2 byte-level alphabet)
  * Unicode tables: letter / number / space ranges, full category ranges
  * the duration estimator's script table + weight constants

Output: output/weights/tokenizer.ovt

    python tools/convert_tokenizer.py
    python tools/convert_tokenizer.py --check     # re-parse and cross-check
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAGIC = b"OVTOK1\x00\x00"
VERSION = 1


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


class Sections:
    """name -> bytes; serialised as a directory + blobs."""

    def __init__(self) -> None:
        self.items: list[tuple[str, bytes]] = []

    def add(self, name: str, blob: bytes) -> None:
        self.items.append((name, blob))

    def write(self, path: Path) -> int:
        names = b"".join(n.encode() + b"\0" for n, _ in self.items)
        n = len(self.items)
        dir_off = 8 + 8
        names_off = dir_off + n * 24
        blob_off = names_off + len(names)
        # align blobs to 64
        blob_off = (blob_off + 63) // 64 * 64

        entries = []
        cursor = blob_off
        name_cursor = 0
        for name, blob in self.items:
            entries.append((name_cursor, len(name), cursor, len(blob)))
            name_cursor += len(name) + 1
            cursor = (cursor + len(blob) + 63) // 64 * 64

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.truncate(cursor)
            f.seek(0)
            f.write(MAGIC)
            f.write(struct.pack("<II", VERSION, n))
            f.seek(dir_off)
            for name_off, name_len, off, length in entries:
                f.write(struct.pack("<IIQQ", name_off, name_len, off, length))
            f.seek(names_off)
            f.write(names)
            for (name_off, name_len, off, length), (_, blob) in zip(entries, self.items):
                f.seek(off)
                f.write(blob)
        return cursor


# ---------------------------------------------------------------------------
# Unicode tables
# ---------------------------------------------------------------------------


def _ranges(pred) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for cp in range(0x110000):
        ok = pred(cp)
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def _cat(cp: int) -> str:
    try:
        return unicodedata.category(chr(cp))
    except ValueError:
        return "Cn"


CLS = {"L": 1, "N": 2, "M": 3, "P": 4, "S": 5, "Z": 6, "C": 7}


def unicode_sections() -> list[tuple[str, bytes]]:
    letter = _ranges(lambda cp: _cat(cp).startswith("L"))
    number = _ranges(lambda cp: _cat(cp).startswith("N"))
    space = _ranges(lambda cp: chr(cp).isspace() or _cat(cp).startswith("Z"))

    def pack_pairs(rs):
        return b"".join(struct.pack("<II", a, b) for a, b in rs)

    # full category table (only the classes the duration model distinguishes)
    cats: list[tuple[int, int, int]] = []
    start = 0
    cur = CLS.get(_cat(0)[0], 0)
    for cp in range(1, 0x110000):
        c = CLS.get(_cat(cp)[0], 0)
        if c != cur:
            cats.append((start, cp - 1, cur))
            start, cur = cp, c
    cats.append((start, 0x10FFFF, cur))
    cat_blob = b"".join(struct.pack("<III", a, b, c) for a, b, c in cats)

    return [
        ("unicode_letter", pack_pairs(letter)),
        ("unicode_number", pack_pairs(number)),
        ("unicode_space", pack_pairs(space)),
        ("unicode_category", cat_blob),
    ]


# ---------------------------------------------------------------------------
# Duration estimator tables (mirrors omnivoice.utils.duration.RuleDurationEstimator)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "cjk": 3.0, "hangul": 2.5, "kana": 2.2, "ethiopic": 3.0, "yi": 3.0,
    "indic": 1.8, "thai_lao": 1.5, "khmer_myanmar": 1.8, "arabic": 1.5,
    "hebrew": 1.5, "latin": 1.0, "cyrillic": 1.0, "greek": 1.0,
    "armenian": 1.0, "georgian": 1.0, "punctuation": 0.5, "space": 0.2,
    "digit": 3.5, "mark": 0.0, "default": 1.0,
}
WEIGHT_ORDER = ["latin", "space", "punctuation", "digit", "mark", "cjk", "default"]


def duration_sections() -> list[tuple[str, bytes]]:
    sys.path.insert(0, str(REPO / "onnx"))
    from omnivoice.utils.duration import RuleDurationEstimator  # noqa: PLC0415

    est = RuleDurationEstimator()
    script = b"".join(struct.pack("<If", end, WEIGHTS[key]) for end, key in est.ranges)
    consts = b"".join(struct.pack("<f", WEIGHTS[k]) for k in WEIGHT_ORDER)
    return [("duration_script", script), ("duration_consts", consts)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build(tok_json: Path, out: Path) -> dict:
    t = json.loads(tok_json.read_text())
    model = t["model"]
    vocab: dict[str, int] = model["vocab"]
    merges = model["merges"]
    added = t.get("added_tokens", [])

    sec = Sections()

    # --- vocabulary ---------------------------------------------------------
    n_tokens = max(vocab.values()) + 1
    strings = bytearray()
    index = bytearray()
    offs = [0] * n_tokens
    lens = [0] * n_tokens
    for s, i in vocab.items():
        b = s.encode("utf-8")
        offs[i] = len(strings)
        lens[i] = len(b)
        strings += b
    for i in range(n_tokens):
        index += struct.pack("<II", offs[i], lens[i])
    sec.add("vocab_strings", bytes(strings))
    sec.add("vocab_index", bytes(index))

    # --- merges: (id_a, id_b) -> merged id, in rank order -------------------
    mb = bytearray()
    for a, b in merges:
        ia = vocab.get(a)
        ib = vocab.get(b)
        im = vocab.get(a + b)
        if ia is None or ib is None or im is None:
            raise SystemExit(f"merge references unknown token: {a!r} {b!r}")
        mb += struct.pack("<III", ia, ib, im)
    sec.add("merges", bytes(mb))

    # --- added tokens -------------------------------------------------------
    ab = bytearray()
    for a in sorted(added, key=lambda x: (x["id"])):
        s = a["content"].encode("utf-8")
        ab += struct.pack("<III", a["id"], len(strings), len(s))
        strings += s
    # rewrite vocab_strings blob now that added strings are appended
    sec.items[0] = ("vocab_strings", bytes(strings))
    sec.add("added", bytes(ab))

    # --- byte -> token id ---------------------------------------------------
    # GPT-2 byte-level alphabet: printable ASCII maps to itself, the rest are
    # shifted into the 256+ range. We only need the forward map here.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_ids = bytearray()
    missing = []
    for b in range(256):
        ch = chr(cs[bs.index(b)])
        tid = vocab.get(ch)
        if tid is None:
            missing.append(ch)
            tid = 0
        byte_ids += struct.pack("<I", tid)
    if missing:
        raise SystemExit(f"byte-level alphabet missing from vocab: {missing}")
    sec.add("byte_ids", bytes(byte_ids))

    # --- unicode + duration -------------------------------------------------
    for name, blob in unicode_sections():
        sec.add(name, blob)
    for name, blob in duration_sections():
        sec.add(name, blob)

    size = sec.write(out)
    return {
        "tokens": n_tokens,
        "merges": len(merges),
        "added": len(added),
        "bytes": size,
        "sections": [n for n, _ in sec.items],
    }


def read_sections(path: Path) -> dict[str, bytes]:
    blob = path.read_bytes()
    magic, version, n = struct.unpack_from("<8sII", blob, 0)
    assert magic == MAGIC, magic
    assert version == VERSION
    out = {}
    for i in range(n):
        name_off, name_len, off, length = struct.unpack_from("<IIQQ", blob, 16 + i * 24)
        name = blob[16 + n * 24 + name_off : 16 + n * 24 + name_off + name_len].decode()
        out[name] = blob[off : off + length]
    return out


def check(tok_json: Path, ovt: Path) -> bool:
    t = json.loads(tok_json.read_text())
    vocab = t["model"]["vocab"]
    secs = read_sections(ovt)
    strings = secs["vocab_strings"]
    index = secs["vocab_index"]
    ok = True
    for s, i in vocab.items():
        off, ln = struct.unpack_from("<II", index, i * 8)
        got = strings[off : off + ln].decode()
        if got != s:
            print(f"  !! id {i}: {got!r} != {s!r}")
            ok = False
            break
    added_ok = len(secs["added"]) // 12 == len(t.get("added_tokens", []))
    print(f"  vocab={len(vocab)} merges={len(secs['merges']) // 12} "
          f"added={len(secs['added']) // 12} sections={len(secs)} "
          f"-> {'OK' if ok and added_ok else 'FAILED'}")
    return ok and added_ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", default=str(REPO / "OmniVoice" / "tokenizer.json"))
    p.add_argument("--out", default=str(REPO / "output" / "weights" / "tokenizer.ovt"))
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    tok = Path(args.tokenizer)
    out = Path(args.out)

    if args.check:
        return 0 if check(tok, out) else 1

    info = build(tok, out)
    print(f"wrote {out}  {info['bytes'] / 1e6:.2f} MB")
    print(f"  tokens={info['tokens']} merges={info['merges']} added={info['added']}")
    print(f"  sections: {', '.join(info['sections'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

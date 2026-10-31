#!/usr/bin/env python3
"""Pack OmniVoice safetensors checkpoints into OVW1 — the pure binary format.

Design goals
------------
* **No framework dependency**: safetensors is parsed by hand (8-byte length +
  JSON header + raw payload) and tensors are `mmap`-ed into numpy views.  No
  torch, no onnx, no protobuf.
* **Zero-copy at run time**: the C++ loader just `mmap`s the file, so loading a
  3.2 GB checkpoint costs one page-table setup.
* **Linear layers pre-transposed** (`layout=1`): PyTorch `Linear.weight` is
  `[out, in]`, we store `[in, out]` so the engine's GEMM can vectorise over the
  contiguous output dimension without a runtime transpose.
* **INT8 = per-output-channel symmetric**, stored transposed with a per-column
  scale array; the engine dequantises for free after accumulation.

See ``docs/weight-format.md`` for the byte-level spec.

Usage
-----
    python tools/convert_weights.py --mode both
    python tools/convert_weights.py --mode int8 --int8-keep audio_heads
    python tools/convert_weights.py --mode int8 --int8-layout nk   # VNNI, writes *-int8-nk.ovw
    python tools/convert_weights.py --check          # re-read and verify CRCs
"""

from __future__ import annotations

import argparse
import json
import mmap
import struct
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

MAGIC = b"OMNIVOIC"
VERSION = 1
HEADER_BYTES = 128
DESC_BYTES = 64
ALIGN = 64

DT_F32, DT_I8, DT_F16, DT_I64 = 0, 1, 2, 3
_DT_BY_NAME = {"F32": DT_F32, "I8": DT_I8, "F16": DT_F16, "I64": DT_I64}
_NP_BY_DT = {DT_F32: np.float32, DT_I8: np.int8, DT_F16: np.float16, DT_I64: np.int64}
_DT_NAMES = {DT_F32: "f32", DT_I8: "i8", DT_F16: "f16", DT_I64: "i64"}

# kept row-major: consumed by Gather (row = token id), never by GEMM
# kept row-major: Gather/lookup tables (row = index), never GEMM operands
ROW_MAJOR_2D = ("embed_tokens.weight", "audio_embeddings.weight", "codebook.embed", "embed_avg")

_HEADER = struct.Struct("<8sIIIIQQQQQQ")
_DESC = struct.Struct("<IIBBBB4IIQQQQ")


def _align(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


# ---------------------------------------------------------------------------
# Minimal safetensors reader (numpy only)
# ---------------------------------------------------------------------------


class SafetensorsFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        (hlen,) = struct.unpack("<Q", self._mm[:8])
        self.header = json.loads(self._mm[8:8 + hlen])
        self.header.pop("__metadata__", None)
        self.data_start = 8 + hlen

    def close(self) -> None:
        # numpy views may still pin the mapping (entries keep tensors alive);
        # dropping the mapping is best-effort — the OS unmaps on process exit.
        try:
            self._mm.close()
        except BufferError:
            pass
        self._fh.close()

    def names(self) -> list[str]:
        return sorted(self.header)

    def array(self, name: str) -> np.ndarray:
        meta = self.header[name]
        begin, end = meta["data_offsets"]
        dtype = _NP_BY_DT[_DT_BY_NAME[meta["dtype"]]]
        flat = np.frombuffer(self._mm, dtype=dtype, count=(end - begin) // np.dtype(dtype).itemsize,
                             offset=self.data_start + begin)
        return flat.reshape(meta["shape"])

    def n_params(self) -> int:
        return sum(int(np.prod(m["shape"])) for m in self.header.values())


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


@dataclass
class Entry:
    name: str
    dtype: int
    layout: int
    quant: int
    array: np.ndarray                 # stored layout, contiguous
    scales: np.ndarray | None = None  # per output channel, fp32
    # filled during layout
    name_off: int = 0
    data_off: int = 0
    aux_off: int = 0
    crc: int = field(default=0)


def _is_row_major_2d(name: str) -> bool:
    return any(name.endswith(s) for s in ROW_MAJOR_2D)


def quantize_int8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-channel symmetric INT8 for a `[out, in]` matrix."""
    absmax = np.abs(w).max(axis=1, keepdims=True)
    scale = np.maximum(absmax / 127.0, 1e-12).astype(np.float32)
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    return q, scale.reshape(-1).astype(np.float32)


def build_entries(path: Path, int8: bool, int8_keep: tuple[str, ...],
                  int8_layout: str = "kn") -> tuple[list[Entry], dict]:
    """Pack one checkpoint.

    ``int8_layout`` only affects quantised *linear* (MatMul) tensors:

    * ``kn`` (default, backward compatible): payload is ``q.T`` = ``[in, out]``,
      ``layout=1``, one scale per output column;
    * ``nk``: payload is ``q`` = ``[out, in]``, ``layout=0``, one scale per
      output row.  This is the layout AVX-VNNI needs (k contiguous inside one
      output column); embeddings already use exactly this encoding.
    """
    if int8_layout not in ("kn", "nk"):
        raise ValueError(f"unknown int8 layout {int8_layout!r}")
    st = SafetensorsFile(path)
    entries: list[Entry] = []
    n_quant = 0
    for name in st.names():
        arr = st.array(name)
        is_ebd = _is_row_major_2d(name) and arr.ndim == 2
        protected = any(p and p in name for p in int8_keep)
        # Quantise every 2-D fp32 matrix big enough to matter, including the
        # embedding tables (Gather + per-row dequant is cheap and saves 470 MB).
        if int8 and arr.dtype == np.float32 and arr.ndim == 2 and arr.size >= 1_000_000 and not protected:
            q, scale = quantize_int8(arr)                  # [rows, cols]
            if is_ebd:
                stored = q
            elif int8_layout == "nk":
                stored = q
            else:
                stored = np.ascontiguousarray(q.T)
            entries.append(Entry(name, DT_I8, layout=0 if is_ebd or int8_layout == "nk" else 1,
                                 quant=1, array=np.ascontiguousarray(stored), scales=scale))
            n_quant += 1
            continue

        is_linear = (arr.dtype == np.float32 and arr.ndim == 2 and not _is_row_major_2d(name))
        stored = np.ascontiguousarray(arr.T) if is_linear else np.ascontiguousarray(arr)
        entries.append(Entry(name, _DT_BY_NAME[_st_dtype(arr)], layout=1 if is_linear else 0,
                             quant=0, array=stored))

    stats = {
        "source": str(path),
        "tensors": len(entries),
        "params": st.n_params(),
        "fp32_bytes": sum(e.array.nbytes for e in entries),
        "quantized": n_quant,
        "int8_layout": int8_layout,
    }
    st.close()
    return entries, stats


def _st_dtype(arr: np.ndarray) -> str:
    return {np.float32: "F32", np.int8: "I8", np.float16: "F16", np.int64: "I64"}[arr.dtype.type]


def write_ovw(out_path: Path, entries: list[Entry], stats: dict, int8_mode: bool) -> dict:
    entries = sorted(entries, key=lambda e: e.name)

    names_off = _align(HEADER_BYTES + DESC_BYTES * len(entries))
    names_blob = bytearray()
    for e in entries:
        e.name_off = len(names_blob)
        names_blob += e.name.encode("utf-8") + b"\0"
    data_off = _align(names_off + len(names_blob))

    cursor = data_off
    for e in entries:
        e.data_off = _align(cursor)
        cursor = e.data_off + e.array.nbytes
        if e.scales is not None:
            e.aux_off = _align(cursor)
            cursor = e.aux_off + e.scales.nbytes
    file_size = _align(cursor)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.truncate(file_size)          # sparse; every byte below is written
        # --- descriptors ---------------------------------------------------
        f.seek(HEADER_BYTES)
        for e in entries:
            e.crc = zlib.crc32(memoryview(e.array).cast("B")) & 0xFFFFFFFF
            dims = list(e.array.shape) + [0] * (4 - e.array.ndim)
            f.write(_DESC.pack(
                e.name_off, len(e.name), e.dtype, e.array.ndim, e.layout, e.quant,
                *dims, int(e.scales.size) if e.scales is not None else 0,
                e.crc, e.data_off, e.array.nbytes, e.aux_off,
            ))
        # --- names ---------------------------------------------------------
        f.seek(names_off)
        f.write(bytes(names_blob))
        # --- payloads ------------------------------------------------------
        for e in entries:
            f.seek(e.data_off)
            f.write(memoryview(e.array).cast("B"))
            if e.scales is not None:
                f.seek(e.aux_off)
                f.write(memoryview(e.scales).cast("B"))
        # --- header --------------------------------------------------------
        flags = (1 if int8_mode else 0) | 2
        f.seek(0)
        f.write(_HEADER.pack(MAGIC, VERSION, 1 if int8_mode else 0, len(entries),
                             HEADER_BYTES, HEADER_BYTES, names_off, data_off,
                             file_size, flags, stats["params"]))

    return {**stats, "path": str(out_path), "bytes": file_size, "tensors": len(entries)}


# ---------------------------------------------------------------------------
# Verification: re-read the file the same way the C++ loader does
# ---------------------------------------------------------------------------


def read_ovw(path: Path) -> tuple[dict, list[dict]]:
    with open(path, "rb") as f:
        head = f.read(HEADER_BYTES)
        magic, ver, mode, n, hb, idx_off, names_off, data_off, fsize, flags, params = \
            _HEADER.unpack_from(head, 0)
        assert magic == MAGIC, f"bad magic {magic!r}"
        assert ver == VERSION, f"bad version {ver}"
        f.seek(idx_off)
        raw = f.read(DESC_BYTES * n)
    descs = []
    for i in range(n):
        (name_off, name_len, dtype, ndim, layout, quant, d0, d1, d2, d3, aux_n,
         crc, off, nbytes, aux_off) = _DESC.unpack_from(raw, i * DESC_BYTES)
        with open(path, "rb") as f:
            f.seek(names_off + name_off)
            name = f.read(name_len).decode()
        descs.append(dict(name=name, dtype=dtype, ndim=ndim, layout=layout, quant=quant,
                          dims=[x for x in (d0, d1, d2, d3)][:ndim], aux_n=aux_n,
                          crc=crc, off=off, nbytes=nbytes, aux_off=aux_off))
    return dict(magic=magic, version=ver, dtype_mode=mode, n=n, file_size=fsize,
                flags=flags, n_params=params), descs


def check_ovw(path: Path, source: Path) -> bool:
    """Verify CRCs against a freshly re-read source checkpoint."""
    hdr, descs = read_ovw(path)
    st = SafetensorsFile(source)
    ok = True
    seen = 0
    with open(path, "rb") as f:
        for d in descs:
            if d["name"] not in st.header:
                print(f"  !! {d['name']}: not in source"); ok = False; continue
            src = st.array(d["name"])
            if d["quant"]:
                # A quantised payload cannot be reproduced bit-exactly from the
                # source, so validate the dequantised error bound instead.
                f.seek(d["off"]); q = np.frombuffer(f.read(d["nbytes"]), np.int8).reshape(d["dims"])
                f.seek(d["aux_off"]); sc = np.frombuffer(f.read(d["aux_n"] * 4), np.float32)
                # layout=1 -> stored [in, out], scales per output column
                # layout=0 -> stored [rows, cols] (embedding or --int8-layout nk
                #              linear), scales per stored row
                ref = src.T if d["layout"] else src
                w = q.astype(np.float32) * (sc[None, :] if d["layout"] else sc[:, None])
                err = float(np.abs(w - ref).max())
                if err > float(np.abs(ref).max()) / 127 * 1.5:
                    print(f"  !! {d['name']}: quant error {err:.4g} too large"); ok = False
            else:
                f.seek(d["off"])
                got = f.read(d["nbytes"])
                want = np.ascontiguousarray(src.T if d["layout"] else src).tobytes()
                if got != want or zlib.crc32(got) & 0xFFFFFFFF != d["crc"]:
                    print(f"  !! {d['name']}: payload mismatch"); ok = False
            seen += 1
    st.close()
    print(f"  checked {seen} tensors, header n_params={hdr['n_params']}, "
          f"file={hdr['file_size'] / 1e9:.2f} GB -> {'OK' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=str(repo / "OmniVoice"),
                   help="checkpoint dir (model.safetensors + audio_tokenizer/)")
    p.add_argument("--out", default=str(repo / "output" / "weights"), help="output dir")
    p.add_argument("--mode", default="both", choices=["fp32", "int8", "both"])
    p.add_argument("--int8-layout", default="kn", choices=["kn", "nk"],
                   help="kn = [in,out] layout=1 (old default); nk = [out,in] layout=0 (AVX-VNNI)")
    p.add_argument("--int8-keep", default="audio_heads",
                   help="comma list of name substrings kept FP32 in int8 mode")
    p.add_argument("--check", action="store_true", help="verify an existing int8/fp32 file")
    p.add_argument("--only", default="lm,at", help="comma list: lm,at")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out)
    sources = {
        "lm": model_dir / "model.safetensors",
        "at": model_dir / "audio_tokenizer" / "model.safetensors",
    }
    wanted = [s.strip() for s in args.only.split(",") if s.strip()]
    keep = tuple(s.strip() for s in args.int8_keep.split(",") if s.strip())

    rc = 0
    for key in wanted:
        src = sources[key]
        if not src.exists():
            print(f"!! missing checkpoint {src}", file=sys.stderr); rc = 2; continue

        if args.check:
            for mode in ("fp32", "int8"):
                f = out_dir / f"omnivoice-{key}-{mode}.ovw"
                if f.exists():
                    print(f"[{key}/{mode}] checking {f.name}")
                    rc |= 0 if check_ovw(f, src) else 1
            if args.int8_layout == "nk":
                f = out_dir / f"omnivoice-{key}-int8-nk.ovw"
                if f.exists():
                    print(f"[{key}/int8-nk] checking {f.name}")
                    rc |= 0 if check_ovw(f, src) else 1
            continue

        modes = ["fp32", "int8"] if args.mode == "both" else [args.mode]
        for mode in modes:
            print(f"[{key}/{mode}] packing {src.name} ...")
            entries, stats = build_entries(src, int8=(mode == "int8"), int8_keep=keep,
                                           int8_layout=args.int8_layout)
            suffix = "int8-nk" if (mode == "int8" and args.int8_layout == "nk") else mode
            dst = out_dir / f"omnivoice-{key}-{suffix}.ovw"
            info = write_ovw(dst, entries, stats, int8_mode=(mode == "int8"))
            print(f"  -> {dst}")
            print(f"     tensors={info['tensors']} (quantized {info['quantized']})  "
                  f"params={info['params'] / 1e6:.1f}M  bytes={info['bytes'] / 1e9:.2f} GB"
                  + (f"  int8_layout={info['int8_layout']}" if mode == "int8" else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())

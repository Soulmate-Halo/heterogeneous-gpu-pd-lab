"""tools/mtp_fetch.py - plan v0.3, P0.3/P6: the MTP block from the BF16 checkpoint, without the checkpoint.

The GSQ-RCO GGUF ships no MTP head. The BF16 checkpoint (Qwen/Qwen3.8-Flash-Next, 360 GB in 131 shards) does:
31 `mtp.*` tensors scattered over 28 shards. Safetensors puts a JSON header (name -> dtype, shape, byte range)
at the start of each shard, so HTTP range requests can read the headers and then only the MTP tensors.

    python tools/mtp_fetch.py inventory --out DIR          # headers only (a few KB per shard)
    python tools/mtp_fetch.py fetch --out DIR [--only SUBSTR]  # the MTP tensors themselves, resumable

`fetch` writes one raw file per tensor plus `mtp-manifest.json` (dtype, shape, source shard, byte range,
sha256). It never downloads anything but the ranges named in the headers. Nothing here runs a model.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time
import urllib.request

REPO = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I64": 8, "I32": 4}


def get(url, start=None, end=None, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "strata-mtp-fetch"})
            if start is not None:
                req.add_header("Range", "bytes=%d-%d" % (start, end))
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if start is not None and len(data) != end - start + 1:
                raise IOError("short range read: %d of %d" % (len(data), end - start + 1))
            return data
        except Exception as e:  # network errors are retried, then surfaced
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            print("retry %s: %s" % (url, e), file=sys.stderr)


def shard_header(shard):
    url = REPO + shard
    n = struct.unpack("<Q", get(url, 0, 7))[0]
    header = json.loads(get(url, 8, 8 + n - 1))
    return 8 + n, header


def inventory(out):
    index = json.loads(get(REPO + "model.safetensors.index.json"))["weight_map"]
    mtp = {k: v for k, v in index.items() if k.startswith("mtp.")}
    rows, total = [], 0
    for shard in sorted(set(mtp.values())):
        base, header = shard_header(shard)
        for name, meta in header.items():
            if name in mtp and mtp[name] == shard:
                a, b = meta["data_offsets"]
                rows.append(dict(name=name, shard=shard, dtype=meta["dtype"], shape=meta["shape"],
                                 start=base + a, end=base + b - 1, bytes=b - a))
                total += b - a
    missing = sorted(set(mtp) - set(r["name"] for r in rows))
    if missing:
        sys.exit("tensors named in the index but absent from their shard headers: %s" % missing)
    rows.sort(key=lambda r: r["name"])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "mtp-inventory.json"), "w", encoding="utf-8") as f:
        json.dump(dict(repo=REPO, total_bytes=total, tensors=rows), f, indent=1)
    lines = ["# MTP block in the BF16 checkpoint", "", "%d tensors, %.3f GB, in %d shards." % (len(rows), total / 1e9, len(set(r["shard"] for r in rows))), "",
             "| tensor | dtype | shape | MB |", "|---|---|---|---:|"]
    for r in rows:
        lines.append("| `%s` | %s | %s | %.1f |" % (r["name"], r["dtype"], "x".join(map(str, r["shape"])), r["bytes"] / 1e6))
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "mtp-inventory.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return rows


def fetch(out, only):
    inv_path = os.path.join(out, "mtp-inventory.json")
    rows = json.load(open(inv_path))["tensors"] if os.path.exists(inv_path) else inventory(out)
    tdir = os.path.join(out, "tensors")
    os.makedirs(tdir, exist_ok=True)
    manifest = []
    chunk = 64 << 20
    for r in rows:
        if only and only not in r["name"]:
            continue
        path = os.path.join(tdir, r["name"] + ".bin")
        have = os.path.getsize(path) if os.path.exists(path) else 0
        with open(path, "ab") as f:
            pos = r["start"] + have
            while pos <= r["end"]:
                end = min(pos + chunk - 1, r["end"])
                f.write(get(REPO + r["shard"], pos, end))
                pos = end + 1
                print("%s %.0f%%" % (r["name"], 100 * (pos - r["start"]) / r["bytes"]), file=sys.stderr)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        if os.path.getsize(path) != r["bytes"]:
            sys.exit("%s: size %d != %d" % (path, os.path.getsize(path), r["bytes"]))
        manifest.append(dict(r, file=os.path.relpath(path, out), sha256=h.hexdigest()))
    with open(os.path.join(out, "mtp-manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inventory", "fetch"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--only")
    a = ap.parse_args()
    inventory(a.out) if a.cmd == "inventory" else fetch(a.out, a.only)


if __name__ == "__main__":
    main()

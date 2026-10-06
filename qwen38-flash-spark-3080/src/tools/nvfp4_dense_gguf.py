"""P4a: build the dense-weight GGUF for the NVFP4 stack (runs on Spark, CPU only).

Strategy (decided 2026-10-02, main-model takeover after the delegated run died):
- The reference Q2_0-00001.gguf holds 1223 dense tensors. Every tensor stored
  BF16/F32/F16 there is an UNQUANTIZED copy of the same BF16 base the NVFP4
  checkpoint ships (modelopt `ignore` list) -> payload is copied BYTE-WISE from
  the reference. This removes all HF-name mapping and all A_log/dt_bias
  semantics risk for hc_*/ssm_*/norm/router/ple_value tensors.
- Tensors QUANTIZED in the reference (token_embd, output.weight, attn_*,
  ssm_out, ffn_*_shexp, ple_key) are BF16 in the NVFP4 checkpoint. They are
  re-encoded from the safetensors source:
    * Q8_0 (ggml type 8, block 32, fp16 scale) for everything the engine serves
      through native_mmvq / the head -- BF16 (type 30) is NOT in
      native_mmvq_supported, so keeping BF16 is not an option there.
    * Q2_0 (type 42) for blk.1.ple_key.weight ONLY: ple.cu fail-closes unless
      key_native_type is 42/18/23, and Q2_0 is what production serves today.
- Metadata KV is cloned from the reference with exact value types preserved
  (a bespoke header parse; gguf_reader drops type ids).

Verify mode (--verify) re-parses the product, byte-compares pass-through
payloads against the reference, and numerically checks re-encoded tensors
against the safetensors source.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gguf_reader import GGUF_MAGIC, GGUF_META, GGML_TYPES, BLOCK_GEOMETRY, GGUFFile  # noqa: E402

ALIGN = 32
PASS_THROUGH = {"F32", "F16", "BF16"}
TYPE_IDS = {"F32": 0, "F16": 1, "Q8_0": 8, "BF16": 30, "Q2_0": 42}


# ----------------------------------------------------------------- metadata parse (types preserved)
def parse_header(path):
    """Return (metadata list of (key, type_name, value), tensors, data_start)."""
    md = []
    tensors = []
    with open(path, "rb") as fh:
        magic, ver, n_t, n_kv = struct.unpack("<IIQQ", fh.read(24))
        assert magic == GGUF_MAGIC and ver == 3, f"{path}: not a GGUF v3"

        def rstr():
            (n,) = struct.unpack("<Q", fh.read(8))
            return fh.read(n).decode("utf-8")

        def scalar(tname):
            fmt = {"u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I", "i32": "<i",
                   "f32": "<f", "bool": "<?", "u64": "<Q", "i64": "<q", "f64": "<d"}[tname]
            return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]

        for _ in range(n_kv):
            key = rstr()
            (t,) = struct.unpack("<I", fh.read(4))
            tname = GGUF_META[t][0]
            if tname == "string":
                md.append((key, "string", rstr()))
            elif tname == "array":
                (et,) = struct.unpack("<I", fh.read(4))
                ename = GGUF_META[et][0]
                (cnt,) = struct.unpack("<Q", fh.read(8))
                if ename == "string":
                    md.append((key, "array:string", [rstr() for _ in range(cnt)]))
                else:
                    md.append((key, "array:" + ename, [scalar(ename) for _ in range(cnt)]))
            else:
                md.append((key, tname, scalar(tname)))
        for _ in range(n_t):
            name = rstr()
            (nd,) = struct.unpack("<I", fh.read(4))
            shape = list(struct.unpack(f"<{nd}Q", fh.read(8 * nd)))
            tid, off = struct.unpack("<IQ", fh.read(12))
            tensors.append({"name": name, "shape": shape, "type_id": tid,
                            "type": GGML_TYPES.get(tid, f"type{tid}"), "offset": off})
        pos = fh.tell()
        data_start = (pos + ALIGN - 1) // ALIGN * ALIGN
    return md, tensors, data_start


def kv_bytes(key, tname, value):
    out = bytearray()
    out += struct.pack("<Q", len(key)) + key.encode()
    ids = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6,
           "bool": 7, "string": 8, "array": 9, "u64": 10, "i64": 11, "f64": 12}

    def scalar(tn, v):
        fmt = {"u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I", "i32": "<i",
               "f32": "<f", "bool": "<?", "u64": "<Q", "i64": "<q", "f64": "<d"}[tn]
        return struct.pack(fmt, v)

    if tname.startswith("array:"):
        en = tname.split(":", 1)[1]
        out += struct.pack("<IIQ", ids["array"], ids[en], len(value))
        for v in value:
            if en == "string":
                b = v.encode()
                out += struct.pack("<Q", len(b)) + b
            else:
                out += scalar(en, v)
        return bytes(out)
    out += struct.pack("<I", ids[tname])
    if tname == "string":
        b = value.encode()
        out += struct.pack("<Q", len(b)) + b
    else:
        out += scalar(tname, value)
    return bytes(out)


# ----------------------------------------------------------------- quantizers (vectorized)
def quantize_q8_0(w):
    """(n,) float32 -> Q8_0 bytes (block 32, fp16 d = amax/127, qs = roundf(w/d) half-away)."""
    b = np.asarray(w, dtype=np.float32).reshape(-1, 32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    safe = np.where(d > 0, d, 1.0)
    inv = np.where(d > 0, 1.0 / safe, 0.0)
    q = np.sign(b) * np.floor(np.abs(b) * inv[:, None] + 0.5)
    q = np.clip(q, -127, 127).astype(np.int8)
    dh = d.astype(np.float16)
    nb = b.shape[0]
    out = np.zeros((nb, 34), dtype=np.uint8)
    out[:, 0:2] = dh.view(np.uint8).reshape(nb, 2)
    out[:, 2:] = q.view(np.uint8).reshape(nb, 32)
    return out.tobytes()


def dequantize_q8_0(raw):
    raw = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    q = raw[:, 2:].view(np.int8).astype(np.float32)
    return (q * d[:, None]).reshape(-1)


def quantize_q2_0_vec(w):
    """docs/q2_0-contract.md decode: (code-1) * d, grid {-1, 0, +1, +2}.
    Encoder: per-block MSE-optimal d over sign x fraction candidates (the doc's
    naive d=amax encoder crushes small positives to 0 -- measured rel_l2 0.72 on
    ple_key vs the reference encoder's 0.35; this search lands at/below that)."""
    b = np.asarray(w, dtype=np.float32).reshape(-1, 64)
    nb = b.shape[0]
    amax = np.abs(b).max(axis=1)
    best_err = np.full(nb, np.inf)
    best_d = np.zeros(nb, dtype=np.float32)
    best_code = np.zeros((nb, 64), dtype=np.uint8)
    for sign in (1.0, -1.0):
        for frac in np.linspace(0.25, 1.0, 31):
            d = sign * frac * amax
            safe = np.where(d != 0, d, 1.0)
            code = np.clip(np.rint(b / safe[:, None]).astype(np.int32) + 1, 0, 3)
            deq = (code - 1) * d[:, None]
            err = ((deq - b) ** 2).sum(axis=1)
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_d = np.where(better, d, best_d)
            best_code[better] = code[better].astype(np.uint8)
    code = best_code.reshape(nb, 64)
    packed = (code[:, 0::4] | (code[:, 1::4] << 2) | (code[:, 2::4] << 4) | (code[:, 3::4] << 6))
    out = np.zeros((nb, 18), dtype=np.uint8)
    out[:, 0:2] = best_d.astype(np.float16).view(np.uint8).reshape(nb, 2)
    out[:, 2:] = packed.astype(np.uint8)
    return out.tobytes()


def dequantize_q2_0_vec(raw):
    raw = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    qs = raw[:, 2:]
    code = np.stack([(qs >> 0) & 3, (qs >> 2) & 3, (qs >> 4) & 3, (qs >> 6) & 3], axis=2)
    code = code.reshape(qs.shape[0], 64).astype(np.float32)
    return ((code - 1.0) * d[:, None]).reshape(-1)


# ----------------------------------------------------------------- HF name map (quantized tensors only)
def hf_name(gguf_name):
    import re
    if gguf_name == "token_embd.weight":
        return "model.language_model.embed_tokens.weight"
    if gguf_name == "output.weight":
        return "lm_head.weight"
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", gguf_name)
    if not m:
        return None
    n, kind = int(m.group(1)), m.group(2)
    p = f"model.language_model.layers.{n}."
    table = {
        "attn_gate": p + "linear_attn.in_proj_z.weight",
        "attn_qkv": p + "linear_attn.in_proj_qkv.weight",
        "ssm_out": p + "linear_attn.out_proj.weight",
        "attn_q": p + "self_attn.q_proj.weight",
        "attn_k": p + "self_attn.k_proj.weight",
        "attn_v": p + "self_attn.v_proj.weight",
        "attn_output": p + "self_attn.o_proj.weight",
        "ffn_gate_shexp": p + "mlp.shared_expert.gate_proj.weight",
        "ffn_up_shexp": p + "mlp.shared_expert.up_proj.weight",
        "ffn_down_shexp": p + "mlp.shared_expert.down_proj.weight",
        "ple_key": p + "ple.key_proj.weight",
    }
    return table.get(kind)


# ----------------------------------------------------------------- safetensors source
class SafetensorsSource:
    def __init__(self, root):
        from safetensors import safe_open
        self._open = safe_open
        idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
        self.wm = idx["weight_map"]
        self.root = root
        self.handles = {}

    def get_f32(self, name):
        shard = self.wm[name]
        if shard not in self.handles:
            self.handles[shard] = self._open(os.path.join(self.root, shard), framework="pt")
        t = self.handles[shard].get_tensor(name)
        return t.to(__import__("torch").float32).numpy(), tuple(t.shape)


# ----------------------------------------------------------------- head regroup
# llama.cpp's Qwen3-Next GGUF conversion regroups the 48 v/gate/y heads from HF's
# k-major order ([k-group g][v-index j], 3 v-heads per k-head) into j-major order
# ([j][g]): GGUF block b holds HF head (b % 16) * 3 + b // 16.  The first dense.gguf
# (2026-10-02) skipped this and the engine produced garbage tokens; the permutation
# was characterized empirically against Q2_0-00001.gguf (ops/_p4d_perm_full.py):
# q/k sections are IDENTITY, v / in_proj_z / ssm_out(input) all follow this one map.
HEAD48 = [(b % 16) * 3 + b // 16 for b in range(48)]


def permute_heads(w, kind):
    if kind == "ssm_out":  # HF (n_embd, 6144): permute the input blocks along axis 1
        wr = w.reshape(w.shape[0], 48, 128)
        return wr[:, HEAD48, :].reshape(w.shape)
    if kind == "attn_qkv":  # HF (10240, n_embd): permute only the v rows (4096..10239)
        head, tail = w[:4096], w[4096:].reshape(48, 128, w.shape[1])
        return np.concatenate([head, tail[HEAD48].reshape(-1, w.shape[1])], axis=0)
    # attn_gate: HF (6144, n_embd), all rows are z heads
    return w.reshape(48, 128, w.shape[1])[HEAD48].reshape(w.shape)


def head_kind(gguf_name):
    for k in ("attn_qkv", "attn_gate", "ssm_out"):
        if gguf_name.endswith(k + ".weight"):
            return k
    return None


# ----------------------------------------------------------------- build
def build(args):
    md, tensors, data_start = parse_header(args.ref)
    src = SafetensorsSource(args.src)
    ref_fh = open(args.ref, "rb")

    plan = []
    skipped_experts = 0
    for t in tensors:
        entry = dict(t)
        # Routed-expert tensors (ffn_*_exps, 144 Q2_0 tensors, ~34 GB) are NOT dense:
        # the NVFP4 stack serves experts from pack-hot/pack-cold blobs, so they are
        # excluded from this GGUF entirely.
        if "_exps." in t["name"]:
            skipped_experts += 1
            continue
        if t["type"] in PASS_THROUGH:
            entry["mode"] = "copy"
        else:
            hf = hf_name(t["name"])
            if hf is None:
                raise SystemExit(f"NO HF MAP for quantized ref tensor {t['name']} ({t['type']})")
            if hf not in src.wm:
                raise SystemExit(f"HF tensor missing in checkpoint: {hf} (for {t['name']})")
            entry["mode"] = "q2_0" if t["name"].endswith("ple_key.weight") else "q8_0"
            entry["hf"] = hf
        plan.append(entry)

    counts = {}
    for e in plan:
        counts[e["mode"]] = counts.get(e["mode"], 0) + 1
    print(f"plan: {counts} of {len(plan)} tensors (skipped {skipped_experts} expert tensors)")

    # header + directory (offsets from payload sizes)
    payloads = []
    off = 0
    infos = bytearray()
    for e in plan:
        elems = int(np.prod(e["shape"]))
        if e["mode"] == "copy":
            geom = BLOCK_GEOMETRY[e["type"]]
            nbytes = elems // geom[0] * geom[1]
            out_type = e["type_id"]
        elif e["mode"] == "q8_0":
            nbytes = elems // 32 * 34
            out_type = TYPE_IDS["Q8_0"]
        else:
            nbytes = elems // 64 * 18
            out_type = TYPE_IDS["Q2_0"]
        payloads.append((e, nbytes))
        infos += struct.pack("<Q", len(e["name"])) + e["name"].encode()
        infos += struct.pack("<I", len(e["shape"]))
        infos += struct.pack(f"<{len(e['shape'])}Q", *e["shape"])
        infos += struct.pack("<I", out_type)
        infos += struct.pack("<Q", off)
        off = (off + nbytes + ALIGN - 1) // ALIGN * ALIGN

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total_gb = off / 1e9
    print(f"output size estimate: {total_gb:.2f} GB -> {args.out}")
    with open(args.out, "wb") as out:
        head = bytearray()
        head += struct.pack("<IIQQ", GGUF_MAGIC, 3, len(plan), len(md))
        for k, tn, v in md:
            head += kv_bytes(k, tn, v)
        head += infos
        head += b"\0" * ((-len(head)) % ALIGN)
        out.write(head)

        done_gb = 0.0
        for i, (e, nbytes) in enumerate(payloads):
            if e["mode"] == "copy":
                ref_fh.seek(data_start + e["offset"])
                remaining = nbytes
                while remaining:
                    chunk = ref_fh.read(min(remaining, 1 << 24))
                    out.write(chunk)
                    remaining -= len(chunk)
            else:
                w, shape = src.get_f32(e["hf"])
                assert shape == (e["shape"][1], e["shape"][0]) or int(np.prod(shape)) == int(np.prod(e["shape"])), \
                    f"{e['name']}: HF shape {shape} vs gguf {e['shape']}"
                kind = head_kind(e["name"])
                if kind is not None:
                    w = permute_heads(w, kind)
                w = w.reshape(-1)
                raw = quantize_q8_0(w) if e["mode"] == "q8_0" else quantize_q2_0_vec(w)
                assert len(raw) == nbytes, f"{e['name']}: {len(raw)} != {nbytes}"
                out.write(raw)
                del w, raw
            pad = (-nbytes) % ALIGN
            if pad:
                out.write(b"\0" * pad)
            done_gb += nbytes / 1e9
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(plan)} tensors, {done_gb:.2f} GB written", flush=True)
    ref_fh.close()
    print(f"BUILD_DONE {args.out} {os.path.getsize(args.out)} bytes")


# ----------------------------------------------------------------- verify
def verify(args):
    md_ref, ref_tensors_all, ref_data = parse_header(args.ref)
    ref_tensors = [t for t in ref_tensors_all if "_exps." not in t["name"]]
    g = GGUFFile(pathlib.Path(args.out))
    problems = []
    if len(g.tensors) != len(ref_tensors):
        problems.append(f"tensor count {len(g.tensors)} != {len(ref_tensors)}")
    for a, b in zip(g.tensors, ref_tensors):
        if a.name != b["name"] or a.shape != b["shape"]:
            problems.append(f"table mismatch {a.name} vs {b['name']}")
            break
    if len(g.metadata) != len(md_ref):
        problems.append(f"metadata count {len(g.metadata)} != {len(md_ref)}")
    for key in ["general.architecture", "qwen4exp.block_count", "qwen4exp.embedding_length",
                "qwen4exp.expert_count", "split.tensors.count"]:
        if g.metadata.get(key) != dict((k, v) for k, _, v in md_ref).get(key):
            problems.append(f"metadata {key} mismatch: {g.metadata.get(key)!r}")
    print(f"header: tensors={len(g.tensors)} md={len(g.metadata)} problems={len(problems)}")

    src = SafetensorsSource(args.src)
    out_fh = open(args.out, "rb")
    ref_fh = open(args.ref, "rb")
    by_name = {t.name: t for t in g.tensors}
    ref_by_name = {t["name"]: t for t in ref_tensors}

    # 1) pass-through payloads: full byte compare (streamed)
    n_copy = 0
    bad_copy = []
    for t in ref_tensors:
        if t["type"] not in PASS_THROUGH:
            continue
        o = by_name[t["name"]]
        geom = BLOCK_GEOMETRY[t["type"]]
        nbytes = int(np.prod(t["shape"])) // geom[0] * geom[1]
        out_fh.seek(g.data_start + o.offset)
        ref_fh.seek(ref_data + t["offset"])
        remaining = nbytes
        same = True
        while remaining:
            n = min(remaining, 1 << 24)
            if out_fh.read(n) != ref_fh.read(n):
                same = False
                break
            remaining -= n
        n_copy += 1
        if not same:
            bad_copy.append(t["name"])
    print(f"copy compare: {n_copy} tensors, bad={bad_copy[:5]}")
    if bad_copy:
        problems.append(f"copy mismatch {bad_copy}")

    # 2) re-encoded tensors: numeric check vs safetensors (all of them, vectorized)
    rng = [t for t in ref_tensors if t["type"] not in PASS_THROUGH]
    worst = 0.0
    worst_name = None
    per_tensor = {}
    for t in rng:
        o = by_name[t["name"]]
        out_fh.seek(g.data_start + o.offset)
        if t["name"].endswith("ple_key.weight"):
            raw = out_fh.read(int(np.prod(t["shape"])) // 64 * 18)
            deq = dequantize_q2_0_vec(raw)
        else:
            raw = out_fh.read(int(np.prod(t["shape"])) // 32 * 34)
            deq = dequantize_q8_0(raw)
        w, _ = src.get_f32(hf_name(t["name"]))
        kind = head_kind(t["name"])
        if kind is not None:
            w = permute_heads(w, kind)
        w = w.reshape(-1)
        denom = float(np.linalg.norm(w)) or 1.0
        rel = float(np.linalg.norm(deq - w) / denom)
        per_tensor[t["name"]] = rel
        if rel > worst:
            worst, worst_name = rel, t["name"]
        limit = 0.45 if t["name"].endswith("ple_key.weight") else 0.015
        if rel > limit:
            problems.append(f"{t['name']} rel_l2 {rel:.5f} > {limit}")
    print(f"reencode: {len(rng)} tensors checked, worst rel_l2 {worst:.6f} ({worst_name})")

    verdict = "PASS" if not problems else "FAIL"
    report = {
        "task": "P4a dense GGUF (NVFP4 stack)",
        "ref": args.ref, "src": args.src, "out": args.out,
        "out_bytes": os.path.getsize(args.out),
        "copy_tensors": n_copy, "reencoded_tensors": len(rng),
        "worst_rel_l2": worst, "worst_tensor": worst_name,
        "problems": problems, "verdict": verdict,
    }
    print("REPORT_JSON " + json.dumps(report))
    print("P4A_VERIFY_" + verdict)
    return 0 if verdict == "PASS" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.environ.get("STRATA_REF_GGUF", ""), help="matching dense/Q2 GGUF reference")
    ap.add_argument("--src", default=os.environ.get("STRATA_MODEL_DIR", ""), help="matching source model directory")
    ap.add_argument("--out", default=os.environ.get("STRATA_DENSE_GGUF_OUT", "dense.gguf"), help="output dense.gguf")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        return verify(args)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

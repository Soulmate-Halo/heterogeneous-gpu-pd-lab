#!/usr/bin/env python3
"""Derive a valid --expert-cache value for the 3080 host from measured VRAM.

The frozen operating point uses 4608 slots. The measured ladder in
evidence/expert-cache-scaling-20260930.json shows why the top of the ladder is
not automatically the best choice:

    cache 3072 (pool off)   decode  57.45 tok/s  - the cache never engages
    cache 8192 (pool on)    decode  52.81 tok/s  hit rate 0.9041, best hit path
    cache 11264             decode   6.64 tok/s  <1 GiB left, graphs squeezed
    cache 12288             refused: needs 15.82 GiB, only 15.28 GiB free

So the usable answer is the largest value that still leaves headroom for CUDA
graphs and scratch, not the largest value that fits. This helper prints the
decision and the arithmetic behind it; it never writes engine configuration.

Run:

    python bundle/tools/calibrate_expert_cache.py --free-gib 19.58 --base-gib 4.31
"""
from __future__ import annotations

import argparse
import json

SLOT_BYTES = 2_775_208          # bytes per expert slot row (see batch-scaling probe)
SLOT_MIB = SLOT_BYTES / (1024 * 1024)
DEFAULT_CEILING = 11264         # measured: unusable above this
HEADROOM_GIB = 1.5              # measured: the 11264 rung had <1 GiB and collapsed

# rungs from the measured ladder, kept for the report
MEASURED = [
    {"cache": 8192, "verdict": "best measured decode 52.81 tok/s, hit rate 0.9041"},
    {"cache": 11264, "verdict": "rejected: decode 6.64 tok/s, layer time 0.394 -> 3.139 ms"},
    {"cache": 12288, "verdict": "refused: needs 15.82 GiB, only 15.28 GiB free"},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-gib", type=float, required=True, help="VRAM free on the 3080 before the expert cache")
    ap.add_argument("--base-gib", type=float, required=True, help="baseline engine occupancy with the cache off")
    ap.add_argument("--headroom-gib", type=float, default=HEADROOM_GIB)
    ap.add_argument("--frozen", type=int, default=4608)
    args = ap.parse_args()

    available = args.free_gib - args.base_gib - args.headroom_gib
    if available <= 0:
        raise SystemExit("no room for an expert cache: free=%.2f base=%.2f headroom=%.2f"
                         % (args.free_gib, args.base_gib, args.headroom_gib))
    slots = int(available * 1024 * 1024 / SLOT_MIB)
    recommended = min(slots, DEFAULT_CEILING)
    # round down to a multiple of 512: the arena is allocated in whole blocks
    recommended -= recommended % 512

    report = {
        "slot_bytes": SLOT_BYTES,
        "free_gib": args.free_gib,
        "base_gib": args.base_gib,
        "headroom_gib": args.headroom_gib,
        "usable_gib": round(available, 2),
        "slots_that_fit": slots,
        "measured_ceiling": DEFAULT_CEILING,
        "recommended_expert_cache": recommended,
        "frozen_operating_point": args.frozen,
        "use_frozen": recommended >= args.frozen,
        "note": "pick the largest value that keeps graph/scratch headroom; the frozen point is "
                "what the published V2 numbers were taken with",
        "measured_ladder": MEASURED,
    }
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

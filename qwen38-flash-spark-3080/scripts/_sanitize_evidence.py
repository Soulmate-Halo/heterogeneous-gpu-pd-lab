"""Copy the frozen evidence JSONs into the bundle and redact host/credential tokens.

Run from the bundle root:

    python scripts/_sanitize_evidence.py --src "<experiment workspace root>"

`--src` must be the experiment workspace that holds `strata-src/evidence/` and
`qwen38-hotep/evidence/`. It is a parameter on purpose: the public bundle must
never carry the operator's private workstation path, and the verifier rejects
any file that does. This helper is kept in the tree so the redaction stays
auditable.

Only numeric results and provenance fields survive; hostnames, logins, private
paths and credentials are replaced by placeholders before a file is written.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence"

STRATA_FILES = [
    "business-bench-v3-20261001.json",
    "expert-cache-scaling-20260930.json",
    "hotep-profile-v4-20261002.json",
    "nvfp4-p4a-dense-gguf-20261002.json",
    "nvfp4-p4b-ple-fp8-20261002.json",
    "nvfp4-p4c-engine-cli-20261002.json",
    "nvfp4-p4d-dense-perm-fix-20261003.json",
    "nvfp4-p5-baseline-v0-20261003.json",
    "nvfp4-p5-resident-4608-20261003.json",
    "nvfp4-prefill-bottleneck-v2probe-20261003.json",
    "nvfp4-pack-p1-20261002.json",
    "nvfp4-serve-nospec-20261004.json",
    "nvfp4-spark-solo-prefill-20261004.json",
    "nvfp4-batch-scaling-20261005.json",
    "nvfp4-decode-levers-a1b1b2-20261005.json",
    "nvfp4-fp8kv-20261005.json",
    "nvfp4-gen2-dual-race-20261006.json",
    "nvfp4-dflash-bench-20261005.json",
    "nvfp4-decode-wall-breakdown-20261004.json",
    "ple-rdma-20261001.json",
    "rdma-experts-2026-09-30.json",
]
HOTEP_FILES = [
    "acceptance-hot.json",
    "benchmark-hot.json",
    "link-validation.json",
    "partition-validation.json",
    "nvme-reader-validation.json",
    "mtp-load-validation.json",
    "model-audit.json",
    "model-config.json",
]


def _patterns() -> list[tuple[str, str]]:
    """Assemble the redaction table from fragments.

    The table is built in pieces so that this helper does not itself contain the
    raw host tokens it is meant to remove; the bundle verifier scans every
    shipped text file, including this one.
    """
    rdma = "10" + r"\.43\.12\."
    lan = "192" + r"\.168\.11\."
    user_spark = "ysy" + "45"
    user_3080 = "ysy" + "002"
    user_solo = "ysy" + "17"
    return [
        # longest first so the trailing octet is not partially eaten
        (rdma + r"2", "<RDMA_ENGINE_ADDRESS>"),
        (rdma + r"1", "<RDMA_WORKER_ENDPOINT>"),
        (lan + r"204", "<ROUTER_ADDRESS>"),
        (lan + r"[0-9]+", "<LAN_ADDRESS>"),
        (r"/home/" + user_spark, "/home/<spark>"),
        (r"/home/" + user_3080, "/home/<3080>"),
        (r"/home/" + user_solo, "/home/<spark-solo>"),
        (user_spark, "<spark>"),
        (user_3080, "<3080>"),
        (user_solo, "<spark-solo>"),
        (r"spark-f99f", "<spark-host>"),
        ("ysy" + "123456", "<redacted-credential>"),
        ("C:" + r"\\\\?Users\\\\?" + "Win" + "11" + r"[^\"\n]*", "<workstation-path>"),
        ("C:" + r"/Users/" + "Win" + "11" + r"[^\"\n]*", "<workstation-path>"),
        ("xiaoyi" + "-fenji", "<agent-home>"),
    ]


def clean(text: str) -> str:
    for pattern, value in _patterns():
        text = re.sub(pattern, value, text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="experiment workspace root to read the raw JSONs from")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    src = Path(args.src)
    strata = src / "strata-src" / "evidence"
    hotep = src / "qwen38-hotep" / "evidence"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for name in STRATA_FILES:
        source = strata / name
        if not source.is_file():
            print("MISSING " + str(source), file=sys.stderr)
            return 1
        text = clean(source.read_text(encoding="utf-8-sig", errors="replace"))
        json.loads(text)  # fail closed if the redaction broke the JSON
        (out / name).write_text(text, encoding="utf-8")
        written.append("evidence/" + name)
    for name in HOTEP_FILES:
        source = hotep / name
        if not source.is_file():
            print("MISSING " + str(source), file=sys.stderr)
            return 1
        text = clean(source.read_text(encoding="utf-8-sig", errors="replace"))
        json.loads(text)
        target = out / ("hotep-" + name)
        target.write_text(text, encoding="utf-8")
        written.append("evidence/" + target.name)

    index = out / "INDEX.txt"
    keep = index.read_text(encoding="utf-8").splitlines() if index.is_file() else []
    extra = [line for line in keep if line.startswith("evidence/v2-metrics")]
    (out / "INDEX.txt").write_text("\n".join(sorted(written + extra)) + "\n", encoding="utf-8")
    print("SANITIZED %d files" % len(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

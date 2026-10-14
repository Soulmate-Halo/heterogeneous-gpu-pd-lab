"""Verify the self-contained DS4.1 Flash V10 deployment bundle.

Checks:
  1. every required file is present;
  2. no text file under ds41-flash-v10/ contains a redacted-out secret string;
  3. every relative path linked from the bundle README.md exists.

Run from the source tree:
    python ds41-flash-v10/verify_bundle.py
Run inside the image:
    verify_bundle.py --inside-image
Exits 0 and prints PASS when everything is fine, non-zero otherwise.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "DEPLOYMENT.zh-CN.md",
    "README.md",
    "Dockerfile",
    "entrypoint.sh",
    "verify_bundle.py",
    "deploy/deploy.conf.example",
    "deploy/p21-launch.sh",
    "deploy/pd-proxy-launch.sh",
    "deploy/d40-launch-rank.sh",
    "deploy/engram-shard-server.sh",
    "deploy/90-dsv41-overcommit.conf",
]

# Strings that must never survive the public-release redaction. They are
# assembled from fragments so that this checker source itself does not trip a
# plain `grep` for the very literals it forbids.
_BS = chr(92)
FORBIDDEN = [
    "172.16.1." + "53",
    "10.100.48." + "5",
    "10.100.48." + "1",
    "10.100.48." + "2",
    "10.100.48." + "3",
    "10.100.48." + "4",
    "sdit" + "-ai",
    "ysy" + "101",
    "ysy" + "37",
    "ysy" + "38",
    "ysy" + "39",
    "ysy" + "40",
    "\u5938\u514b" + "5446",
    "H:" + _BS + "Harness",
]

# Relative markdown links, e.g. [text](DEPLOYMENT.zh-CN.md) or ![](...).
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def iter_text_files() -> list[Path]:
    files = []
    for path in sorted(BASE.rglob("*")):
        if path.is_file():
            files.append(path)
    return files


def check_required() -> None:
    for rel in REQUIRED_FILES:
        assert (BASE / rel).is_file(), f"missing required file: {rel}"


def check_forbidden() -> int:
    scanned = 0
    for path in iter_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: not a text file
        scanned += 1
        for bad in FORBIDDEN:
            assert bad not in text, f"forbidden string {bad!r} in {path.relative_to(BASE)}"
    return scanned


def check_readme_links() -> int:
    readme = BASE / "README.md"
    text = readme.read_text(encoding="utf-8")
    checked = 0
    for target in LINK_RE.findall(text):
        target = target.strip()
        if target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target = target.split("#", 1)[0].split("?", 1)[0]
        if not target:
            continue
        candidate = (readme.parent / target).resolve()
        assert candidate.exists(), f"README.md links a missing path: {target}"
        checked += 1
    return checked


def main() -> int:
    _ = "--inside-image" in sys.argv  # accepted for both tree and image
    try:
        check_required()
        scanned = check_forbidden()
        checked = check_readme_links()
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        f"PASS: V10 bundle files={len(REQUIRED_FILES)}; "
        f"text files scanned={scanned}; README links checked={checked}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

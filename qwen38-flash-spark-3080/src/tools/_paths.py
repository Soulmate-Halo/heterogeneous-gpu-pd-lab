"""tools/_paths.py - where the tools find llama.cpp's gguf-py (for the MTP packer and the fixtures).

Order: $STRATA_GGUF_PY, then <engine>/third_party/llama.cpp/gguf-py (what setup.py clones), then the development
tree's ../../.ref/llama.cpp/gguf-py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]


def gguf_py() -> str:
    cands = [os.environ.get("STRATA_GGUF_PY"), ENGINE / "third_party" / "llama.cpp" / "gguf-py",
             ENGINE.parents[min(1, len(ENGINE.parents) - 1)] / ".ref" / "llama.cpp" / "gguf-py"]   # the dev tree; a shallow install (G:\Strata) has no grandparent
    for c in cands:
        if c and (Path(c) / "gguf").is_dir():
            return str(c)
    sys.exit("llama.cpp's gguf-py was not found: run setup.py (it clones llama.cpp into third_party/), "
             "or set STRATA_GGUF_PY")


def add_gguf_py() -> None:
    p = gguf_py()
    if p not in sys.path:
        sys.path.insert(0, p)

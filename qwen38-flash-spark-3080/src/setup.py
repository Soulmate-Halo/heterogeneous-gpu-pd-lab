#!/usr/bin/env python3
"""Strata one-click setup and start (Windows and Linux, NVIDIA GPUs).

    START-HERE.bat  (Windows)   /   ./setup.sh  (Linux)      - they install Python if needed and run this file

The first time it asks four questions - which model (the original Qwen3.8-Flash-Next or the Swift 1.5 fine-tune),
which size, how much context, and whether the model should also read images - then installs everything and starts the model on http://127.0.0.1:8080 (OpenAI- and Anthropic-compatible
API; a small page there shows that it runs). Every later start skips straight to running the model: nothing that
is already downloaded, installed or prepared is done again.

What the first run does (each step is skipped when it is already done):

  1. checks your PC: NVIDIA GPU and driver, RAM, CPU, free disk space
  2. asks the questions
  3. installs the Python packages it needs into .venv (numpy, jinja2, ..., and NVIDIA's CUDA libraries)
  4. gets the Strata engine: a ready-made build for RTX 30/40/50 cards (no compiler needed); if none fits your PC,
     it installs the build tools (asks first) and compiles the engine for your GPU
  5. downloads the model from Hugging Face (resumable), and the vision encoder if you want images
  6. prepares the model for Strata and fetches the MTP draft layer (~5 GB, from the original Qwen checkpoint)
  7. writes run-<model>.bat / run-<model>.sh and starts the model

Options: --family qwen|swift, --model Q2_0|IQ2_XS|IQ3_XXS|IQ3_S, --context 32768, --vision yes|no|gpu|cpu, --port 8080, --yes (recommended
answers, no questions), --setup (install another model / change settings instead of starting), --no-start,
--host 0.0.0.0 --api-key KEY (reach it from other devices on your network), --experimental-speed-projection on|off
(EXPERIMENTAL, off by default),
--models-dir DIR, --gguf-dir DIR (use GGUF files you already have), --build (compile instead of the ready-made
engine), --check (only check this PC).
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN = os.name == "nt"
HF = "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/"
LLAMA_CPP_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
LLAMA_CPP_ZIP = f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_CPP_COMMIT}.zip"

# The ready-made engine: <PREBUILT_URL><asset>, a zip with strata(.exe), strata-vision(.exe) and BUILD.json, built
# by tools/make_release.py.  Set this to the GitHub release download folder when publishing, e.g.
# "https://github.com/<you>/Strata/releases/latest/download/" (or pass --prebuilt / set STRATA_PREBUILT_URL).
PREBUILT_URL = "https://github.com/Niko1221/Strata/releases/latest/download/"
PREBUILT_ASSET = "strata-windows-x64.zip" if WIN else "strata-linux-x64.zip"
# the CUDA libraries the ready-made engine loads (the same CUDA 13.0 it is built with), from NVIDIA's pip packages
CUDA_WHEELS = ["nvidia-cublas==13.0.2.14", "nvidia-cuda-runtime==13.0.96"]
MIN_DRIVER = 580                       # CUDA 13.0
MIN_ENGINE = (0, 1, 25)                # v0.1.25: faster prompts (grouping off the copy engine, fused hyper-connection kernels), AMD HIP backend, --kv k8v4; v0.1.24: long prompts faster (QSA select on tensor cores); v0.1.23: image requests honor sampling, 8 GB cards start, batched verify window; v0.1.22: faster prompts (tensor-core attention), multi-GPU across images/steering/KV streaming; v0.1.21: multi-GPU layer split (--gpus); v0.1.20: system-prompt checkpoint, PCIe probe, hit rate; v0.1.19: penalties
PY_PACKAGES = ["numpy", "jinja2", "regex", "pyyaml", "tqdm", "requests", "cmake", "ninja", "pillow", "psutil"]

MODELS = {
    "Q2_0": {"about": "2-bit, the fastest", "download_gb": 66.4, "ram_gb": 48, "arena_gb": 34.0},
    "IQ2_XS": {"about": "2-bit i-quant, a little better quality, close in speed", "download_gb": 68.0, "ram_gb": 48,
               "arena_gb": 35.5},
    "IQ3_XXS": {"about": "3-bit i-quant, better quality, slower (more CPU work per token)", "download_gb": 75.8,
                "ram_gb": 60, "arena_gb": 42.9},
    # the original model only (Swift 1.5 has no IQ3_S): matches the full BF16 model on the published benchmarks
    "IQ3_S": {"about": "3.5-bit i-quant, the best quality (matches the full model), the slowest; needs a 64 GB PC "
                       "with little else running", "download_gb": 83.6, "ram_gb": 62, "arena_gb": 50.3,
              "families": ("qwen",)},
    # the Coder release: 256 of the 512 experts kept (the ones code, tools and vision use), IQ2_S-IQ4_XS like IQ3_S
    "IQ1_M": {"about": "the Coder's only size: half the experts, stored like IQ3_S (3.5 bits)", "download_gb": 58.4,
              "ram_gb": 32, "arena_gb": 23.4, "families": ("coder",)},
}
CONTEXTS = [8192, 32768, 65536, 131072, 262144]
# The model families: the same architecture, weights in the same three GSQ-RCO sizes, different files.
FAMILIES = {
    "qwen": {"title": "Qwen3.8-Flash-Next", "by": "Qwen; GSQ-RCO quants by ISTA-DASLab",
             "about": "the original model",
             "hf": "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/{q}/",
             "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "",
             "mmproj_hf": "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/",
             "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next"},
    "swift": {"title": "Swift 1.5", "by": "UkisAI's fine-tune of Qwen3.8-Flash-Next",
              "about": "thinks much shorter (-63% thinking tokens, 1.8x sooner answers by its authors' numbers)",
              "hf": "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/",
              "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "swift-",
              "mmproj_hf": "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/",
              "mmproj": "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf", "name": "swift-1.5",
              "license": "Swift Open License 1.0: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"},
    # ISTA-DASLab's expert-pruned release: half of each layer's experts removed, chosen for code, agentic tool use and
    # vision; its shard 2 (the n-gram table) and vision encoder are the original's files, shared with it
    "coder": {"title": "Qwen3.8-Flash-Next Coder", "by": "ISTA-DASLab's coding version",
              "about": "half the experts (code, tools, images kept): needs ~32 GB of RAM, faster; weaker outside coding",
              "hf": "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/resolve/main/{q}/",
              "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "coder-",
              "mmproj_hf": "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/resolve/main/",
              "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next-coder",
              "profile": "expert-profile-coder.bin"},
}
MMPROJ = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
# EXPERIMENTAL, off by default (setup asks): a control vector shipped with the repository, see its README
ESP_VECTOR = ROOT / "data" / "experimental-speed-projection" / "Qwen3.8-Flash-Next-experimental-speed-projection.gguf"
# the image encoder on the GPU (~1.2 GB at 1024 image tokens) warms up before the engine starts, so the engine
# sizes its expert slots around it and the default reserve (700 MiB) is enough; engines before 0.1.2 need more
VISION = {"gpu": {"max_tokens": 1024, "reserve_mib": 700},
          "cpu": {"max_tokens": 300, "reserve_mib": 700}}
EXE = "strata.exe" if WIN else "strata"
VEXE = "strata-vision.exe" if WIN else "strata-vision"


# ------------------------------------------------------------------------------------------------ output
def say(msg=""):
    print(msg, flush=True)


def step(n, title):
    say()
    say(f"=== Step {n}: {title} ===")


def ok(msg):
    say(f"  [ok] {msg}")


def warn(msg):
    say(f"  [!]  {msg}")


def fail(msg, hint=None):
    say(f"\n  [X]  {msg}")
    if hint:
        say(f"       {hint}")
    say("\nSetup stopped. Fix the item above and run it again - everything already done is kept and skipped.")
    sys.exit(1)


def ask(question, choices, default, yes):
    if yes:
        return default
    while True:
        try:
            a = input(f"{question} [{default}]: ").strip()
        except EOFError:
            return default
        if not a:
            return default
        if a.lower() in [c.lower() for c in choices]:
            return next(c for c in choices if c.lower() == a.lower())
        say(f"  please answer one of: {', '.join(choices)}")


def run(cmd, cwd=None, env=None, check=True, quiet=False):
    say("  > " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env,
                       stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None,
                       text=True)
    if check and r.returncode != 0:
        if quiet and r.stdout:
            say(r.stdout[-4000:])
        fail(f"command failed (exit {r.returncode}): {Path(str(cmd[0])).name}")
    return r


def out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def done(path: Path) -> bool:
    """A step's finish mark: <path>.done exists (written only after the step completed)."""
    return path.with_name(path.name + ".done").exists()


def mark(path: Path, text=""):
    path.with_name(path.name + ".done").write_text(text or time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ the PC
def _memory_status():
    """Windows' GlobalMemoryStatusEx: RAM, and the commit limit (ullTotalPageFile = RAM + page file)."""
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m


def ram_gb():
    if WIN:
        return _memory_status().ullTotalPhys / 2**30
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            return int(line.split()[1]) * 1024 / 2**30
    return 0.0


def page_file_gb():
    """The page file's current size (GB) on Windows, None elsewhere.  The graphics card's memory needs room there
    too: under Windows' driver model every allocation on the card is also charged to the commit (RAM + page file),
    so with the page file off or tiny the engine cannot use the free VRAM (issue #60)."""
    if not WIN:
        return None
    m = _memory_status()
    return max(0.0, (m.ullTotalPageFile - m.ullTotalPhys) / 2**30)


def cpu_info():
    """(name, avx2, avx512): avx512 means everything Strata's fast AVX-512 kernels use (F, BW, VL, VNNI, VBMI),
    the same test the engine makes (cpu_avx512_ok), not just AVX-512F."""
    name, avx2, avx512 = platform.processor() or "unknown CPU", False, False
    if WIN:
        pf = ctypes.windll.kernel32.IsProcessorFeaturePresent
        avx2 = bool(pf(40))                       # PF_AVX2_INSTRUCTIONS_AVAILABLE
        n = out(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]).strip()
        name = n or name
        avx512 = bool(pf(41)) and _cpuid_avx512_full()
    else:
        try:
            txt = open("/proc/cpuinfo").read()
            flags = set(re.search(r"^flags\s*:\s*(.*)$", txt, re.M).group(1).split())
            avx2 = "avx2" in flags
            avx512 = {"avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512vbmi"} <= flags
            m = re.search(r"^model name\s*:\s*(.*)$", txt, re.M)
            name = m.group(1) if m else name
        except OSError:
            pass
    return name, avx2, avx512


def _cpuid_avx512_full() -> bool:
    """Windows has no feature bit for VNNI / VBMI: ask the CPU (CPUID leaf 7) through a tiny machine-code stub."""
    try:
        code = bytes([0x53, 0x49, 0x89, 0xC8, 0xB8, 0x07, 0x00, 0x00, 0x00, 0x31, 0xC9, 0x0F, 0xA2,   # push rbx; r8=rcx; cpuid(7,0)
                      0x41, 0x89, 0x18, 0x41, 0x89, 0x48, 0x04, 0x5B, 0xC3])                   # [r8]=ebx,[r8+4]=ecx; pop rbx
        k32 = ctypes.windll.kernel32
        k32.VirtualAlloc.restype = ctypes.c_void_p
        buf = k32.VirtualAlloc(None, len(code), 0x3000, 0x40)
        if not buf:
            return False
        ctypes.memmove(buf, code, len(code))
        regs = (ctypes.c_uint32 * 2)()
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(buf)(ctypes.addressof(regs))
        ebx, ecx = regs[0], regs[1]
        need_ebx = (1 << 16) | (1 << 30) | (1 << 31)                   # F, BW, VL
        need_ecx = (1 << 1) | (1 << 11)                                # VBMI, VNNI
        return (ebx & need_ebx) == need_ebx and (ecx & need_ecx) == need_ecx
    except Exception:
        return False


def gpus():
    """Every NVIDIA GPU, numbered as nvidia-smi numbers them (by PCI bus, the order the engine is told to use)."""
    s = out(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader,nounits"])
    found = []
    for line in s.strip().splitlines():
        try:
            idx, name, mem, cc, drv = [x.strip() for x in line.split(",")]
            found.append({"index": int(idx), "name": name, "vram_gb": float(mem) / 1024.0, "arch": cc.replace(".", ""),
                          "driver": drv})
        except ValueError:
            continue
    return found


GPU_PICK = None                                         # --gpu N (issue #51); None: the card with the most VRAM
SPLIT_MIN_VRAM_GB = 8                                   # a card sharing a model holds the dense weights and its own
                                                        # prompt buffers too (docs/MULTI_GPU.md)


def cc(g) -> str:
    return f"{g['arch'][:-1]}.{g['arch'][-1]}"


def gpu_problem(g, together=False):
    """Why Strata cannot use this card, in plain words (None: it can)."""
    if int(g["arch"]) < 80:
        return (f"not supported - older than the RTX 30 series (compute capability {cc(g)}; Strata needs 8.0 or "
                "newer)")
    if together and g["vram_gb"] < SPLIT_MIN_VRAM_GB - 0.5:
        return (f"not supported together with other GPUs - {g['vram_gb']:.0f} GB of VRAM (a card sharing the model "
                f"needs {SPLIT_MIN_VRAM_GB} GB or more)")
    return None


def gpu_rank(g):
    """The order cards share a model in: the newest generation first (it gets the first layers and most of the
    work), then the most VRAM."""
    return (-int(g["arch"]), -round(g["vram_gb"]), g["index"])


def gpu_name(g) -> str:
    return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB)"


def gpu_table(found) -> None:
    say("  Your NVIDIA GPUs:")
    for g in found:
        p = gpu_problem(g)
        say(f"    GPU {g['index']}: {g['name']}, {g['vram_gb']:.0f} GB VRAM - " + ("can be used" if p is None else p))


def together_ok(found) -> list:
    """The cards that can share one model, in the order they would (empty if fewer than two)."""
    ok_ = sorted([g for g in found if gpu_problem(g, together=True) is None], key=gpu_rank)
    return ok_ if len(ok_) >= 2 else []


def parse_gpus(text, found) -> list:
    """--gpus / --gpu with several: "0,2" or "all" (every card that can share the model)."""
    if str(text).strip().lower() == "all":
        sel = [g["index"] for g in together_ok(found)]
        if not sel:
            gpu_table(found)
            fail("--gpus all: this PC does not have two GPUs Strata can use together")
        return sel
    try:
        sel = [int(x) for x in str(text).split(",") if x.strip()]
    except ValueError:
        fail(f"--gpus takes GPU numbers as nvidia-smi numbers them, e.g. --gpus 0,2 (or --gpus all), not {text!r}")
    if len(sel) < 2 or len(set(sel)) != len(sel):
        fail("--gpus takes two or more different GPUs, e.g. --gpus 0,2 (one GPU: --gpu 0)")
    return sel


def check_gpus(sel, found, what="") -> None:
    """Stops with a plain message when a chosen card is missing or cannot be used, and says what can."""
    together = len(sel) > 1
    for i in sel:
        g = next((x for x in found if x["index"] == i), None)
        p = "not found on this PC" if g is None else gpu_problem(g, together)
        if p is None:
            continue
        say()
        gpu_table(found)
        can = together_ok(found)
        single = [x for x in found if gpu_problem(x) is None]
        ones = " or ".join(f"--gpu {x['index']}" for x in single)
        both = "--gpus " + ",".join(str(x["index"]) for x in can) if can else ""
        hint = ((f"use these together: {both}" + (f" (or one card: {ones})" if not together else "")) if can else
                f"use one card: {ones}" if single else "Strata needs an NVIDIA RTX 30 series or newer card")
        fail(f"GPU {i}{'' if g is None else ' (' + g['name'] + ')'} {what}cannot be used: {p}", hint)


def engine_archs():
    """The GPU generations the installed engine has code for: (archs, ptx), or None when there is none."""
    info = ROOT / "engine" / "BUILD.json"
    try:
        meta = json.loads(info.read_text())
    except (OSError, ValueError):
        return None
    return [int(x) for x in meta.get("archs", [])], bool(meta.get("ptx"))


def engine_runs_on(g) -> bool:
    ea = engine_archs()
    if ea is None or not ea[0]:
        return True
    archs, ptx = ea
    return int(g["arch"]) in archs or (ptx and int(g["arch"]) > max(archs))


def choose_gpus(a, found) -> list:
    """Which cards this install uses: --gpus / --gpu, or asked when two or more can share the model (the two best
    together recommended), else the supported card with the most VRAM.  Returns their numbers, the main one first."""
    if a.gpus:
        sel = parse_gpus(a.gpus, found)
        check_gpus(sel, found)
        return sel
    if a.gpu is not None:
        check_gpus([a.gpu], found)
        return [a.gpu]
    single = sorted([g for g in found if gpu_problem(g) is None], key=lambda x: (-round(x["vram_gb"]), x["index"]))
    if not single:
        gpu_table(found)
        fail("none of your GPUs can run Strata", "it needs an NVIDIA RTX 30 series or newer (compute capability 8.0+)")
    can = together_ok(found)
    if not can:
        return [single[0]["index"]]
    say()
    say(f"  Strata can run the model on one GPU, or share it across {'these' if len(can) > 2 else 'both'}: then each"
        " card holds the")
    say("  experts of its own layers, so together they hold about twice as many, and prompts are read about 20%")
    say("  faster (details: docs/MULTI_GPU.md). A much slower extra card can also make it slower.")
    opts = [can[:2]] + ([can] if len(can) > 2 else []) + [[g] for g in single]
    for i, o in enumerate(opts, 1):
        label = (" + ".join(gpu_name(g) for g in o) + " together") if len(o) > 1 else gpu_name(o[0]) + " only"
        say(f"  {i}) {label}" + ("   (recommended)" if i == 1 else ""))
    for g in found:
        if gpu_problem(g, together=True) is not None:
            say(f"     (GPU {g['index']}, {g['name']}: {gpu_problem(g, together=True)})")
    pick = opts[int(ask("Which GPUs?", [str(i) for i in range(1, len(opts) + 1)], "1", a.yes or a.check)) - 1]
    return [g["index"] for g in pick]


def offer_together(cfg_path: Path, cfg: dict, yes: bool) -> dict:
    """Starting a model set up for one card on a PC with two or more that can share it: asked once (the answer is
    saved in its config)."""
    if isinstance(cfg.get("gpu"), list) or cfg.get("gpus_asked"):
        return cfg
    found = gpus()
    can = together_ok(found)
    if not can:
        return cfg
    pair = can[:2]
    cfg["gpus_asked"] = True
    say()
    say("  This PC has " + " and ".join(gpu_name(g) for g in pair) + ": Strata can share the model across both.")
    say("  Together they hold about twice the model's experts and read prompts about 20% faster (docs/MULTI_GPU.md).")
    missing = [g for g in pair if not engine_runs_on(g)]
    if missing:
        say("  The installed engine has no code for " + ", ".join(g["name"] for g in missing) + ": to use them "
            "together, run START-HERE.bat --setup --gpus " + ",".join(str(g["index"]) for g in pair))
    elif ask("  Use both from now on? (you can change it later: START-HERE.bat --gpu N for one card)",
             ["y", "n"], "y", yes) == "y":
        cfg["gpu"] = [g["index"] for g in pair]
        cfg["layer_split"] = cfg.get("layer_split") or "auto"
        ok("from now on this model runs on " + " + ".join(gpu_name(g) for g in pair))
    else:
        ok("staying on one GPU (START-HERE.bat --gpus " + ",".join(str(g["index"]) for g in pair) + " switches)")
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return cfg


def gpu_info(pick=None):
    """The GPU Strata runs on: `pick` (nvidia-smi's number) if given, else the one with the most VRAM (ties: the
    lower number).  None when there is no NVIDIA GPU.  The dict also says how many there are ("count")."""
    found = gpus()
    if not found:
        return None
    pick = GPU_PICK if pick is None else pick
    if pick is not None:
        g = next((x for x in found if x["index"] == pick), None)
        if g is None:
            fail(f"there is no GPU {pick}: " + ", ".join(f"{x['index']} = {x['name']}" for x in found))
    else:
        g = max(found, key=lambda x: (round(x["vram_gb"]), -x["index"]))
    return {**g, "count": len(found)}


def find_nvcc():
    cands = [shutil.which("nvcc")]
    if os.environ.get("CUDA_PATH"):
        cands.append(str(Path(os.environ["CUDA_PATH"]) / "bin" / ("nvcc.exe" if WIN else "nvcc")))
    if WIN:
        base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if base.exists():
            cands += [str(p / "bin" / "nvcc.exe") for p in sorted(base.iterdir(), reverse=True)]
    else:
        cands += [str(p / "bin" / "nvcc") for p in sorted(Path("/usr/local").glob("cuda*"), reverse=True)]
        cands += [str(p / "bin" / "nvcc") for p in sorted(Path("/opt").glob("cuda*"), reverse=True)]   # Arch (#46)
    best = (None, None)
    for c in dict.fromkeys(cands):                     # every toolkit found; the newest wins
        if c and Path(c).exists():
            v = re.search(r"release (\d+)\.(\d+)", out([c, "--version"]))
            if v and (best[1] is None or (int(v.group(1)), int(v.group(2))) > best[1]):
                best = (c, (int(v.group(1)), int(v.group(2))))
    return best


def find_vcvars():
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.exists():
        return None
    # CUDA 13 accepts Visual Studio 2019 and 2022 only: a newer one (2026 = version 18) installed next to them
    # must not be picked ("unsupported Microsoft Visual Studio version"); with only a newer one there is none
    p = out([str(vswhere), "-latest", "-products", "*", "-version", "[16.0,18.0)", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"]).strip()
    v = Path(p) / "VC/Auxiliary/Build/vcvars64.bat" if p else None
    return v if v and v.exists() else None


def find_tool(name):
    """A tool on PATH, or the one pip installed next to this Python (cmake, ninja)."""
    p = shutil.which(name)
    if p:
        return p
    for d in (Path(sys.executable).parent / "Scripts", Path(sys.executable).parent,
              Path.home() / ".local" / "bin"):
        c = d / (name + (".exe" if WIN else ""))
        if c.exists():
            return str(c)
    return None


def free_gb(path):
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


# ------------------------------------------------------------------------------------------------ downloads
def download(url, dst: Path, what=None):
    """Resumable HTTP(S) download with a progress line; `file://` and plain paths are copied (tests, mirrors).
    A finished file gets a <name>.done mark, so a later run skips it without asking the server."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and done(dst):
        ok(f"{what or dst.name} already downloaded")
        return
    if not url.startswith(("http://", "https://")):
        src = Path(url[7:] if url.startswith("file://") else url)
        if not src.exists():
            fail(f"not found: {src}")
        shutil.copyfile(src, dst)
        mark(dst)
        ok(f"{what or dst.name} copied")
        return
    part = dst.with_name(dst.name + ".part")
    total = 0
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "strata-setup"})
            total = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", 0))
            break
        except OSError as e:
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})", "check your internet connection and run it again")
            time.sleep(5)
    if dst.exists() and total and dst.stat().st_size == total:    # finished by an older setup (no mark yet)
        mark(dst)
        ok(f"{what or dst.name} already downloaded")
        return
    have = part.stat().st_size if part.exists() else 0
    for attempt in range(30):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "strata-setup", "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:                     # the server ignored the range: start over
                    f.seek(0)
                    f.truncate()
                    have = 0
                last = 0.0
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
                    have += len(b)
                    if time.time() - last > 2:
                        last = time.time()
                        size = f"{have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)" if total \
                            else f"{have / 1e6:7.1f} MB"
                        print(f"\r  {what or dst.name}: {size}   ", end="", flush=True)
            print()
            if not total or have >= total:
                break
        except OSError as e:
            print()
            warn(f"download interrupted ({e}); retrying in 10 s ...")
            time.sleep(10)
    if total and part.stat().st_size != total:
        fail(f"could not finish downloading {dst.name}: {part.stat().st_size:,} bytes on disk, the server says {total:,}",
             "check your internet connection and run it again (the download resumes where it stopped)")
    part.replace(dst)
    mark(dst)
    ok(f"{what or dst.name} downloaded")


def check_shards(shards):
    """Every shard present and whole, or setup stops naming the file and the numbers.  Whole means as long as
    its own tensor directory says (the header is read, the data is not): a truncated copy (--gguf-dir, a .part
    renamed by hand, a download finished by an older setup) otherwise passes as a model file and the engine
    fails much later, at the first tensor that runs past the end."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    for s in shards:
        if not s.exists():
            fail(f"missing {s}")
        try:
            g = GGUFFile(s)
        except (ValueError, struct.error) as e:
            fail(f"{s.name} is not a whole GGUF shard ({e})", "delete it and run setup again")
        need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
        have = s.stat().st_size
        if have < need:
            fail(f"{s.name} is short: {have:,} of {need:,} bytes ({need - have:,} missing)",
                 "delete it and run setup again (or copy the whole file into --gguf-dir)")


def get_llama_cpp():
    """llama.cpp at the pinned commit (ggml for the build, gguf-py for the tools, mtmd for images), as a zip: no git."""
    llama = ROOT / "third_party" / "llama.cpp"
    if (llama / "ggml" / "CMakeLists.txt").exists() and (llama / "gguf-py").is_dir():
        return llama
    z = ROOT / "third_party" / f"llama.cpp-{LLAMA_CPP_COMMIT[:7]}.zip"
    download(LLAMA_CPP_ZIP, z, "llama.cpp source")
    tmp = ROOT / "third_party" / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    top = next(tmp.iterdir())
    shutil.rmtree(llama, ignore_errors=True)
    # PR #63: on Windows a rename can fail with PermissionError while an antivirus scanner still holds a file of the
    # fresh unpack; shutil.move falls back to copy-and-delete, and a few retries let the scanner finish.  The target
    # is `llama` itself - moving into its parent would keep the zip's `llama.cpp-<sha>` folder name.
    for attempt in range(5):
        try:
            shutil.move(str(top), str(llama))
            break
        except PermissionError:
            if attempt == 4:
                raise
            shutil.rmtree(llama, ignore_errors=True)   # a partial copy from the failed attempt
            time.sleep(2)
    shutil.rmtree(tmp, ignore_errors=True)
    z.unlink(missing_ok=True)
    z.with_name(z.name + ".done").unlink(missing_ok=True)
    return llama


def pip_install(packages, what):
    """pip install into .venv, skipped when the same list was installed before."""
    stamp = Path(sys.prefix) / ".strata-pip.json"
    have = json.loads(stamp.read_text()) if stamp.exists() else []
    need = [p for p in packages if p not in have]
    if not need:
        ok(f"{what} already installed")
        return
    say(f"  Installing {what} ...")
    run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *need])
    stamp.write_text(json.dumps(sorted(set(have) | set(need)), indent=0))
    ok(f"{what} installed")


def cuda_lib_dirs():
    """Where pip put NVIDIA's CUDA libraries (nvidia/cu13/bin/x86_64 on Windows, nvidia/cu13/lib on Linux)."""
    pattern = "cublas64_13.dll" if WIN else "libcublas.so.13*"
    dirs = []
    for sp in {Path(p) for p in sys.path if p.endswith("site-packages")}:
        for hit in (sp / "nvidia").rglob(pattern) if (sp / "nvidia").is_dir() else []:
            if hit.parent not in dirs:
                dirs.append(hit.parent)
    return [str(d) for d in dirs]


# ------------------------------------------------------------------------------------------------ AMD (experimental)
# The RX 7900 XT / XTX (gfx1100) on Linux, through the HIP backend (docs/AMD_HIP.md).  There is no ready-made AMD
# engine: ROCm comes from AMD's TheRock Python wheels into .venv (no sudo; a system ROCm in /opt/rocm is used when it
# has hipcc and hipBLAS) and the engine is compiled here.  One GPU, no images yet.
ROCM_INDEX = os.environ.get("STRATA_ROCM_INDEX", "https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/")
ROCM_VERSION = os.environ.get("STRATA_ROCM_VERSION", "7.10.0a20251120")   # what Strata's HIP build was tested with
AMD_ARCHS = ("gfx1100",)
AMD_NAMES = {"gfx1100": "AMD Radeon RX 7900 series (gfx1100)"}   # when sysfs has no product name


def amd_gpus():
    """AMD GPUs from the kernel's KFD topology (the amdgpu driver; no ROCm needed), numbered as HIP numbers them:
    the GPU nodes in order, the CPU nodes skipped.  Integrated GPUs are listed too (not supported)."""
    base = Path("/sys/class/kfd/kfd/topology/nodes")
    found = []
    if WIN or not base.is_dir():
        return found
    for node in sorted((p for p in base.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)):
        try:
            props = {}
            for line in (node / "properties").read_text().splitlines():
                k, _, v = line.partition(" ")
                props[k] = v.strip()
            ver = int(props.get("gfx_target_version") or 0)
            if ver == 0 or int(props.get("simd_count") or 0) == 0:
                continue
        except (OSError, ValueError):
            continue
        arch = f"gfx{ver // 10000}{(ver // 100) % 100:x}{ver % 100:x}"
        dev = Path(f"/sys/class/drm/renderD{props.get('drm_render_minor', '')}/device")
        try:
            vram = int((dev / "mem_info_vram_total").read_text()) / 2 ** 30
        except (OSError, ValueError):
            vram = 0.0
        try:
            name = (dev / "product_name").read_text().strip() or f"AMD Radeon ({arch})"
        except OSError:
            name = f"AMD Radeon ({arch})"
        if name == f"AMD Radeon ({arch})" and arch in AMD_NAMES:
            name = AMD_NAMES[arch]
        found.append({"index": len(found), "name": name, "vram_gb": vram, "arch": arch, "driver": "amdgpu",
                      "vendor": "amd"})
    return found


def amd_problem(g):
    if g["arch"] not in AMD_ARCHS:
        return (f"not supported - Strata's AMD backend runs on the RX 7900 XT / XTX ({', '.join(AMD_ARCHS)}) only, "
                f"this is {g['arch']}")
    return None


def rocm_root():
    """ROCm for compiling and running the HIP engine: (root, library folders).  A system ROCm with hipcc and hipBLAS,
    else AMD's TheRock wheels (ROCM_VERSION, from ROCM_INDEX) installed into .venv."""
    sysroot = Path(os.environ.get("ROCM_PATH") or "/opt/rocm")
    if (sysroot / "bin" / "hipcc").exists() and list((sysroot / "lib").glob("libhipblas.so*")):
        return sysroot, [str(sysroot / "lib")]
    stamp = Path(sys.prefix) / ".strata-rocm.json"
    if not stamp.exists() or json.loads(stamp.read_text()).get("version") != ROCM_VERSION:
        say(f"  Installing ROCm {ROCM_VERSION} for AMD GPUs into .venv (AMD's TheRock wheels, ~10 GB, no sudo) ...")
        run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--index-url",
             ROCM_INDEX, f"rocm[libraries,devel]=={ROCM_VERSION}"])
        stamp.write_text(json.dumps({"version": ROCM_VERSION, "index": ROCM_INDEX}))
    sdk = Path(sys.executable).parent / "rocm-sdk"
    root = Path(out([str(sdk), "path", "--root"]).strip())
    if not (root / "llvm" / "bin" / "clang++").exists():
        fail(f"ROCm was installed but its compiler is missing ({root})",
             f"remove {stamp} and run this again; or install ROCm 7 system-wide")
    dirs = [str(root / "lib")]
    for sp in {Path(p) for p in sys.path if p.endswith("site-packages")}:
        dirs += [str(d / "lib") for d in sorted(sp.glob("_rocm_sdk_libraries_*")) if (d / "lib").is_dir()]
    ok(f"ROCm: {root}")
    return root, list(dict.fromkeys(dirs))


def build_engine_hip(gpu, llama) -> Path:
    """Compile the HIP engine for this AMD GPU into engine/ (again only when its source changed: a `git pull`)."""
    eng = ROOT / "engine"
    eng.mkdir(exist_ok=True)
    stamp = eng / "BUILD.json"
    meta = json.loads(stamp.read_text()) if stamp.exists() else {}
    src = source_hash(ENGINE_SOURCES)
    if meta.get("backend") == "hip" and (eng / EXE).exists() and meta.get("src") == src:
        ok("engine already built for this PC")
        return eng
    if not (shutil.which("c++") or shutil.which("g++")) or not shutil.which("git"):
        fail("a C++ compiler and git are needed to compile the AMD engine",
             "Ubuntu/Debian: sudo apt install build-essential git   Fedora: sudo dnf install gcc-c++ git")
    root, dirs = rocm_root()
    libs = [str(Path(d).parent) for d in dirs[1:]]
    bitcode = next((p for p in (root / "lib" / "llvm" / "amdgcn" / "bitcode", root / "amdgcn" / "bitcode") if p.is_dir()),
                   root / "amdgcn" / "bitcode")
    os.environ.update({"HIP_PLATFORM": "amd", "HIP_COMPILER": "clang", "HIP_RUNTIME": "rocclr", "ROCM_PATH": str(root),
                       "HIP_PATH": str(root)})
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + [os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    os.environ["PATH"] = os.pathsep.join([str(root / "bin"), str(root / "llvm" / "bin"), os.environ.get("PATH", "")])
    say("  The engine's source changed: compiling it again (only what changed, a few minutes) ..."
        if meta.get("backend") == "hip" and (eng / EXE).exists()
        else "  Compiling the Strata engine for your AMD GPU (10-20 minutes, once) ...")
    cmake_build(ROOT, ROOT / "build-hip", "strata",
                ["-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_BUILD_TESTS=OFF",
                 "-DSTRATA_PREFILL_MMQ=ON", f"-DCMAKE_HIP_ARCHITECTURES={gpu['arch']}",
                 f"-DCMAKE_HIP_COMPILER={root / 'llvm' / 'bin' / 'clang++'}", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root}",
                 "-DCMAKE_PREFIX_PATH=" + ";".join([str(root), *libs]),
                 f"-DCMAKE_HIP_FLAGS=--rocm-path={root} --rocm-device-lib-path={bitcode}",
                 f"-DSTRATA_GGML_DIR={llama}"], None, "")
    shutil.copy2(ROOT / "build-hip" / EXE, eng / EXE)
    stamp.write_text(json.dumps({"source": "local-hip", "backend": "hip", "version": source_version(),
                                 "archs": [gpu["arch"]], "vision": "none", "lib_dirs": dirs, "src": src}, indent=1))
    ok(f"engine compiled: {eng / EXE}")
    return eng


# ------------------------------------------------------------------------------------------------ the engine
def driver_major(gpu):
    try:
        return int(gpu["driver"].split(".")[0])
    except (ValueError, KeyError):
        return 0


def get_prebuilt(url_base, gpu, vision, updating=False) -> Path | None:
    """The ready-made engine in engine/ (kept between runs), or None when there is none for this PC.
    updating: called to replace an installed engine, which starts instead when this fails (no compile)."""
    eng = ROOT / "engine"
    info = eng / "BUILD.json"
    if info.exists() and (eng / EXE).exists():
        meta = json.loads(info.read_text())
        ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:3] if x.isdigit())
        if meta.get("source") == "local" or ver >= MIN_ENGINE:
            ok("ready-made engine already installed")
            return eng
        say(f"  Updating the ready-made engine ({meta.get('version')} -> {'.'.join(map(str, MIN_ENGINE))} or newer) ...")
        info.unlink()
    if not url_base:
        return None
    z = ROOT / "engine" / PREBUILT_ASSET
    base = url_base if url_base.endswith(("/", "\\")) else url_base + "/"
    if base.startswith(("http://", "https://")):
        try:                                           # not published (yet), or no internet: compile instead
            req = urllib.request.Request(base + PREBUILT_ASSET, method="HEAD", headers={"User-Agent": "strata-setup"})
            urllib.request.urlopen(req, timeout=60).close()
        except OSError as e:
            warn(f"no ready-made engine at {base} ({e})" + ("" if updating else ": compiling instead"))
            return None
    say("  Downloading the ready-made Strata engine ...")
    download(base + PREBUILT_ASSET, z, "Strata engine")
    tmp = ROOT / "engine" / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    meta = json.loads((tmp / "BUILD.json").read_text())
    if tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:3] if x.isdigit()) < MIN_ENGINE:
        need = ".".join(map(str, MIN_ENGINE))
        if updating:                                   # these files are newer than the published release (#58)
            warn(f"engine {need} is not published yet (the release may still be uploading): run this again "
                 f"in a few minutes to update it")
        else:
            warn(f"the ready-made engine at {base} is version {meta.get('version')}; this setup needs "
                 f"{need}: compiling instead")
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    archs = [int(a) for a in meta.get("archs", [])]
    miss = [int(x) for x in gpu.get("archs", [gpu["arch"]])
            if int(x) not in archs and not (meta.get("ptx") and int(x) > max(archs))]
    if miss:
        warn(f"the ready-made engine is built for {', '.join(str(a) for a in archs)}; your GPU is "
             f"{', '.join(str(x) for x in miss)}" + ("" if updating else ": compiling instead"))
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    for p in tmp.iterdir():
        dst = eng / p.name
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        p.replace(dst)
    shutil.rmtree(tmp, ignore_errors=True)
    z.unlink(missing_ok=True)
    z.with_name(z.name + ".done").unlink(missing_ok=True)
    if not (eng / EXE).exists():
        fail("the ready-made engine archive has no " + EXE)
    if not WIN:
        for x in (EXE, VEXE):
            if (eng / x).exists():
                (eng / x).chmod(0o755)
    ok(f"ready-made engine {meta.get('version', '')} for {', '.join('sm_' + str(a) for a in archs)} (CUDA "
       f"{meta.get('cuda', '?')})")
    return eng


def update_installed_engine(url_base) -> None:
    """An installed ready-made engine older than MIN_ENGINE is replaced before the model starts, so a plain
    START-HERE.bat on an existing install picks up a new release.  If that cannot happen (no internet, the model
    still running, no ready-made engine for this GPU) the installed engine is kept and starts as before."""
    eng = ROOT / "engine"
    info = eng / "BUILD.json"
    if not info.exists() or not (eng / EXE).exists():
        return
    meta_text = info.read_text()
    meta = json.loads(meta_text)
    if meta.get("backend") == "hip":                   # AMD: compiled here, again when its source changed
        if meta.get("src") != source_hash(ENGINE_SOURCES):
            try:
                g = next((x for x in amd_gpus() if amd_problem(x) is None), None)
                if g is None:
                    raise RuntimeError("no supported AMD GPU found")
                build_engine_hip(g, get_llama_cpp())
            except (Exception, SystemExit) as e:
                warn(f"could not compile the updated engine{'' if isinstance(e, SystemExit) else f' ({e})'}: "
                     "starting the installed one")
        return
    ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:3] if x.isdigit())
    local = meta.get("source") == "local"
    vision = meta.get("vision") or "none"
    if local:                                          # compiled here: is it older than the source (a git pull)?
        if meta.get("src") == source_hash(ENGINE_SOURCES) and \
                (vision == "none" or meta.get("vision_src") == source_hash(VISION_SOURCES)):
            return
    elif ver >= MIN_ENGINE:
        return
    try:                                               # a running engine cannot be replaced (Windows keeps it locked)
        for x in (EXE, VEXE):
            if (eng / x).exists():
                with open(eng / x, "r+b"):
                    pass
    except OSError:
        warn(f"engine {meta.get('version') or ''} is in use: close the model window and run this again to update it")
        return
    gpu = gpu_info()
    if local:
        try:                                           # a failed compile must not stop the model from starting
            if gpu is None:
                raise RuntimeError("no NVIDIA GPU found")
            gpu = {**gpu, "archs": sorted({int(gpu["arch"]), *(int(x) for x in meta.get("archs", []))})}
            build_engine(gpu, vision, False, get_llama_cpp())
        except (Exception, SystemExit) as e:
            warn(f"could not compile the updated engine{'' if isinstance(e, SystemExit) else f' ({e})'}: starting the installed one")
        return
    new = None
    if gpu is not None:
        try:
            new = get_prebuilt(url_base, gpu, "gpu", updating=True)
        except Exception as e:                         # a failed download must not stop the model from starting
            warn(f"updating the engine failed ({e})")
    if new is None:
        if not info.exists():
            info.write_text(meta_text)                 # get_prebuilt drops it before downloading: put it back
        warn(f"could not update the engine: starting the installed {meta.get('version')}")
        return
    pip_install(CUDA_WHEELS, "NVIDIA CUDA libraries (cuBLAS, CUDA runtime; ~0.4 GB)")


def install_build_tools(gpu, yes):
    """The compiler and the CUDA toolkit, installed for the user (asks once).  Returns (nvcc, vcvars)."""
    nvcc, cuda_v = find_nvcc()
    need_cuda = (12, 8) if max(int(x) for x in gpu.get("archs", [gpu["arch"]])) >= 120 else (12, 0)
    vcvars = find_vcvars() if WIN else None
    have_cc = vcvars is not None if WIN else shutil.which("g++") is not None
    missing = []
    if not have_cc:
        missing.append("Visual Studio 2022 Build Tools (C++)" if WIN else "the C++ compiler (build-essential)")
    if nvcc is None or cuda_v < need_cuda:
        missing.append("the NVIDIA CUDA Toolkit 13.0")
    if not missing:
        ok(f"build tools present (CUDA {cuda_v[0]}.{cuda_v[1]})")
        return nvcc, vcvars
    say("  The engine has to be compiled for your PC, which needs: " + " and ".join(missing) + ".")
    say("  They can be installed now (about 8-10 GB, 15-40 minutes" + (", Windows will ask for permission" if WIN else
                                                                        ", sudo will ask for your password") + ").")
    if ask("  Install them now?", ["y", "n"], "y", yes) != "y":
        fail("the build tools are needed", "install them yourself (see README.md) and run it again")
    if WIN:
        if shutil.which("winget") is None:
            fail("winget (Windows package manager) is not available",
                 "install 'App Installer' from the Microsoft Store, or install the tools by hand (README.md)")
        wg = ["winget", "install", "-e", "--source", "winget", "--accept-package-agreements",
              "--accept-source-agreements", "--disable-interactivity"]
        if not have_cc:
            run([*wg, "--id", "Microsoft.VisualStudio.2022.BuildTools", "--override",
                 "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"],
                check=False)
        if nvcc is None or cuda_v < need_cuda:
            run([*wg, "--id", "Nvidia.CUDA", "--version", "13.0"], check=False)
        vcvars = find_vcvars()
    else:
        apt = shutil.which("apt-get")
        if apt is None:
            fail("missing: " + " and ".join(missing) + " (the automatic install is only done on Ubuntu/Debian)",
                 "install them with your distribution's packages (Arch: pacman -S base-devel cuda; nvcc is found on "
                 "PATH, in /usr/local/cuda* and in /opt/cuda*), then run it again")
        if not have_cc:
            run(["sudo", "apt-get", "install", "-y", "build-essential"])
        if nvcc is None or cuda_v < need_cuda:
            osr = dict(line.split("=", 1) for line in open("/etc/os-release").read().splitlines() if "=" in line)
            ver = osr.get("VERSION_ID", "").strip('"').replace(".", "")
            if osr.get("ID") != "ubuntu" or ver not in ("2204", "2404"):
                fail("the CUDA Toolkit can be installed automatically on Ubuntu 22.04 / 24.04 only",
                     "install it from https://developer.nvidia.com/cuda-downloads and run it again")
            deb = Path("/tmp/cuda-keyring.deb")
            download(f"https://developer.download.nvidia.com/compute/cuda/repos/ubuntu{ver}/x86_64/cuda-keyring_1.1-1_all.deb",
                     deb, "CUDA repository key")
            run(["sudo", "dpkg", "-i", str(deb)])
            run(["sudo", "apt-get", "update"])
            run(["sudo", "apt-get", "install", "-y", "cuda-toolkit-13-0"])
    nvcc, cuda_v = find_nvcc()
    if (WIN and find_vcvars() is None) or (not WIN and shutil.which("g++") is None):
        fail("the C++ build tools did not install", "install them by hand (README.md) and run it again")
    if nvcc is None or cuda_v < need_cuda:
        fail("the CUDA Toolkit did not install", "install it from https://developer.nvidia.com/cuda-downloads, then run it again")
    ok(f"build tools installed (CUDA {cuda_v[0]}.{cuda_v[1]})")
    return nvcc, find_vcvars() if WIN else None


def cmake_build(src, bdir, target, defs, vcvars, bat_name):
    cmake, ninja = find_tool("cmake"), find_tool("ninja")
    if cmake is None or ninja is None:
        fail("cmake / ninja not found after installing them", "run: .venv python -m pip install cmake ninja")
    conf = [cmake, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}", "-S", str(src), "-B", str(bdir),
            "-DCMAKE_BUILD_TYPE=Release", *defs]
    build = [cmake, "--build", str(bdir), "--target", target, "-j", str(max(2, (os.cpu_count() or 4) // 2))]
    # A failed build is tried once more: CUDA 13.0's ptxas now and then fails to parse a PTX file it just wrote, and
    # the same command then gets past it (issue #45); a second attempt only compiles what is still missing.
    if WIN:
        bat = ROOT / bat_name
        q = lambda c: " ".join(f'"{x}"' if " " in str(x) else str(x) for x in c)  # noqa: E731
        bat.write_text(f'@echo off\r\ncall "{vcvars}" >nul\r\n{q(conf)} || exit /b 1\r\n{q(build)} && exit /b 0\r\n'
                       f'echo   (the build stopped - trying it once more)\r\n{q(build)} || exit /b 1\r\n',
                       encoding="utf-8")
        run(["cmd", "/c", str(bat)])
    else:
        run(conf)
        if run(build, check=False).returncode != 0:
            say("  (the build stopped - trying it once more)")
            run(build)


ENGINE_SOURCES = ("CMakeLists.txt", "src", "include", "third_party/ggml")
VISION_SOURCES = ("tools/vision",)


def source_hash(parts) -> str:
    """A fingerprint of the files a compiled engine is built from, kept in engine/BUILD.json: when a `git pull`
    changes them, the engine is compiled again (issue #31)."""
    h = hashlib.sha256(LLAMA_CPP_COMMIT.encode())
    for part in parts:
        base = ROOT / part
        for f in [base] if base.is_file() else sorted(x for x in base.rglob("*") if x.is_file()):
            h.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def build_engine(gpu, vision, yes, llama) -> Path:
    """Compile the engine (and, for images, the encoder) for this GPU; the results go to engine/.  A compiled
    engine whose source files changed since (a `git pull`) is compiled again: only the changed files, a few minutes."""
    eng = ROOT / "engine"
    eng.mkdir(exist_ok=True)
    stamp = eng / "BUILD.json"
    meta = json.loads(stamp.read_text()) if stamp.exists() else {}
    want_vision = vision != "none"
    local = meta.get("source") == "local"
    src, vsrc = source_hash(ENGINE_SOURCES), source_hash(VISION_SOURCES)
    engine_ok = local and (eng / EXE).exists() and meta.get("src") == src
    vision_ok = not want_vision or ((eng / VEXE).exists() and (not local or meta.get("vision_src") == vsrc))
    if engine_ok and vision_ok:
        ok("engine already built for this PC")
        return eng
    nvcc, vcvars = install_build_tools(gpu, yes)
    archs = sorted({int(x) for x in gpu.get("archs", [gpu["arch"]])})    # every card the model runs on
    cuda_archs = ";".join(str(x) for x in archs)
    if not engine_ok:
        say("  The engine's source changed: compiling it again (only what changed, a few minutes) ..."
            if local and (eng / EXE).exists() else "  Compiling the Strata engine for your GPU (10-20 minutes, once) ...")
        cmake_build(ROOT, ROOT / "build", "strata",
                    ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
                     f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DSTRATA_GGML_DIR={llama}"], vcvars, "build-strata.bat")
        shutil.copy2(ROOT / "build" / EXE, eng / EXE)
    if not vision_ok:
        say("  Compiling the image encoder" + (" with CUDA (10-20 minutes, once) ..." if vision == "gpu" else " ..."))
        defs = [f"-DLLAMA_DIR={llama}", f"-DSTRATA_VISION_CUDA={'ON' if vision == 'gpu' else 'OFF'}"]
        if vision == "gpu":
            defs += [f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}", f"-DCMAKE_CUDA_COMPILER={nvcc}"]
        cmake_build(ROOT / "tools" / "vision", ROOT / "build-vision", "strata-vision", defs, vcvars, "build-vision.bat")
        shutil.copy2(ROOT / "build-vision" / "bin" / VEXE, eng / VEXE)
    bindir = Path(nvcc).parent                            # the toolkit's own libraries (bin, bin/x64, lib64)
    dirs = [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()]
    stamp.write_text(json.dumps({"source": "local", "version": source_version(), "archs": archs,
                                 "vision": vision,
                                 "cuda_dirs": dirs, "src": src, "vision_src": vsrc if want_vision else None}, indent=1))
    ok(f"engine compiled: {eng / EXE}")
    return eng


# ------------------------------------------------------------------------------------------------ the data folder
# The model files - the GGUFs, the prepared packs and the MTP layer, 70-120 GB - live in a data folder NEXT TO the
# Strata folder (`Strata-data`), not inside it: updating Strata by unzipping a new copy used to give a new, empty
# folder and a full download again.  Where it is, and which Strata folders this user ran, is kept in a small
# per-user file, so every Strata folder on the PC finds the same files.
DATA_ITEMS = ("models", "packs", "mtp")


def settings_path() -> Path:
    if WIN:
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Strata" / "settings.json"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "strata" / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(s: dict) -> None:
    try:
        settings_path().parent.mkdir(parents=True, exist_ok=True)
        settings_path().write_text(json.dumps(s, indent=1), encoding="utf-8")
    except OSError as e:
        warn(f"could not save {settings_path()} ({e})")


def has_data(folder: Path) -> bool:
    for d in DATA_ITEMS:
        try:
            if (folder / d).is_dir() and any((folder / d).iterdir()):
                return True
        except OSError:
            pass
    return False


def other_installs(settings: dict) -> list:
    """Strata folders besides this one that may hold model files: the ones this user ran before, and Strata* folders
    next to this one (a zip unpacked again lands in e.g. `Strata-main (1)\\Strata-main`)."""
    cands = [Path(p) for p in settings.get("installs", [])]
    for base in dict.fromkeys((ROOT.parent, ROOT.parent.parent)):
        try:
            for d in base.iterdir():
                if d.is_dir() and d.name.lower().startswith("strata"):
                    cands.append(d)
                    cands += [c for c in d.iterdir() if c.is_dir() and c.name.lower().startswith("strata")]
        except OSError:
            pass
    found = []
    for d in cands:
        try:
            d = d.resolve()
            if d != ROOT and d not in found and (d / "setup.py").is_file():
                found.append(d)
        except OSError:
            pass
    return found


def same_drive(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def move_into(src: Path, dst: Path) -> None:
    """A rename into the data folder (same drive: instant); a folder merges into one already there, keeping what the
    destination has.  Whatever cannot be moved (a file in use) stays where it is."""
    if not dst.exists():
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
            return
        except OSError:
            if not src.is_dir():
                return
            dst.mkdir(parents=True, exist_ok=True)
    if src.is_dir() and dst.is_dir():
        for c in list(src.iterdir()):
            move_into(c, dst / c.name)
        try:
            src.rmdir()
        except OSError:
            pass


def repoint_config(cfg_file: Path, old: Path, new: Path) -> None:
    """A config whose model files moved from `old` to `new` points at them there (each path only if its file is
    now there and no longer at the old place)."""
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return

    def fix(v):
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, str):
            for d in DATA_ITEMS:
                o = str(old / d)
                nv, no = os.path.normcase(v), os.path.normcase(o)   # Windows: C:\ and c:\ are the same place
                if nv == no or nv.startswith(no + os.sep):
                    n = str(new / d) + v[len(o):]
                    if Path(n).exists() and not Path(v).exists():
                        return n
        return v

    new_cfg = fix(cfg)
    if new_cfg != cfg:
        cfg_file.write_text(json.dumps(new_cfg, indent=1), encoding="utf-8")


def data_folder(requested: str | None) -> tuple:
    """(the data folder, folders on other drives that still hold model files).  Moves the model files of this folder
    and of earlier Strata folders on the same drive into the data folder, and points their configs there."""
    settings = load_settings()
    dest = Path(requested).expanduser().resolve() if requested else \
        Path(settings["data_dir"]) if settings.get("data_dir") else ROOT.parent / "Strata-data"
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:                                # e.g. no write access next to the Strata folder
        warn(f"cannot use {dest} for the model files ({e}): keeping them in {ROOT}")
        dest = ROOT
    elsewhere = []
    for folder in [ROOT, *other_installs(settings)]:
        if folder == dest or not has_data(folder):
            continue
        if not same_drive(folder, dest):
            elsewhere.append(folder)                    # another drive: used where it is (no 70 GB copy)
            continue
        # the downloads merge file by file (the same file wherever it came from); a prepared pack or MTP layer moves
        # whole or not at all, so two copies are never mixed
        if (folder / "models").is_dir():
            move_into(folder / "models", dest / "models")
        for item in [*((folder / "packs").glob("*") if (folder / "packs").is_dir() else []), folder / "mtp"]:
            rel = item.relative_to(folder)
            if item.exists() and not (dest / rel).exists():
                move_into(item, dest / rel)
        for d in ("packs",):
            try:
                (folder / d).rmdir()                    # empty now
            except OSError:
                pass
        for c in folder.glob("strata-*.json"):
            repoint_config(c, folder, dest)
        if has_data(folder):
            elsewhere.append(folder)                    # in use, or a copy the data folder already has
            warn(f"some model files are still in {folder} (in use, or already in {dest})")
        else:
            ok(f"model files from {folder} moved to {dest} (a new copy of Strata finds them there)")
    installs = [str(ROOT)] + [p for p in settings.get("installs", []) if p != str(ROOT) and Path(p).is_dir()]
    save_settings({**settings, "data_dir": str(dest), "installs": installs[:20]})
    return dest, elsewhere


def previous_config(elsewhere_first: list, settings: dict):
    """The most recently used model config of another Strata folder on this PC, for a folder that has none yet."""
    cands = []
    for folder in [*elsewhere_first, *other_installs(settings)]:
        cands += list(folder.glob("strata-*.json"))
    cands = [c for c in dict.fromkeys(cands) if c.is_file()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def choices_from_config(cfg_path: Path) -> dict:
    """The setup answers a config was written with (family, size, context, KV, images, projection, network)."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    tag = cfg_path.stem[len("strata-"):]
    family = next((f for f, d in FAMILIES.items() if d["tag"] and tag.startswith(d["tag"])), "qwen")
    model = tag.split("-")[-1].upper()
    a = cfg.get("args", [])
    val = lambda k: a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else None   # noqa: E731
    vis = cfg.get("vision")
    esp = val("--control-vector-scaled")
    esp_path = esp.rsplit(":", 1)[0] if esp else None
    return {"family": family, "model": model if model in MODELS else None,
            "context": int(val("--max-context")) if val("--max-context") else None,
            "kv": val("--kv") if val("--kv") in ("int8", "q4_0") else None,
            "vision": ("gpu" if vis.get("gpu") else "cpu") if isinstance(vis, dict) else "none",
            "esp": ("on" if Path(esp_path).name == ESP_VECTOR.name else esp_path) if esp_path else "off",
            "host": cfg.get("host"), "api_key": cfg.get("api_key"), "port": cfg.get("port"), "gpu": cfg.get("gpu"),
            "layer_split": cfg.get("layer_split")}


def find_in(roots: list, rel: str):
    """The first of roots/rel that exists."""
    for r in roots:
        if (r / rel).exists():
            return r / rel
    return None


# ------------------------------------------------------------------------------------------------ start
def installed_configs():
    return sorted(ROOT.glob("strata-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def source_version() -> str:
    """The engine version the source tree builds (CMakeLists.txt's project version)."""
    m = re.search(r"project\(strata VERSION ([\d.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))
    return m.group(1) if m else "0"


def engine_version(exe: Path) -> tuple:
    """The version in the engine folder's BUILD.json, or else the one compiled into the binary.  A locally compiled
    engine is not necessarily the source's version: when compiling a `git pull` fails, the previous engine is kept
    (issue #49)."""
    try:
        meta = json.loads((Path(exe).parent / "BUILD.json").read_text())
    except (OSError, ValueError):
        meta = {}
    v = str(meta.get("version") or "")
    if not v:                                          # the version compiled into the binary: 0.1.13 and newer
        try:                                           # carry it, so a binary without it is older
            m = re.search(rb"engine=(\d+\.\d+\.\d+)\n", Path(exe).read_bytes())
            v = m.group(1).decode() if m else "0.1.12"
        except OSError:
            v = "0"
    return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())


def is_wsl() -> bool:
    return sys.platform.startswith("linux") and "microsoft" in platform.uname().release.lower()


def hardware_key(cfg: dict) -> str:
    """What a calibration is valid for: this GPU, CPU and RAM, and the model with its context and images setting
    (the context's KV cache and the image encoder take VRAM from the expert cache)."""
    sel = cfg.get("gpu")
    gl = [gpu_info(i) or {} for i in sel] if isinstance(sel, list) else [gpu_info(sel) or {}]
    g = {"name": " + ".join(x.get("name", "?") for x in gl), "vram_gb": sum(x.get("vram_gb", 0) for x in gl)}
    a = cfg.get("args", [])
    ctx = a[a.index("--max-context") + 1] if "--max-context" in a else "?"
    return "|".join([g.get("name", "?"), f"{g.get('vram_gb', 0):.0f}GB", cpu_info()[0], f"{ram_gb():.0f}GB",
                     cfg.get("model_name", "?"), ctx, "images" if "--vision" in a else "text"])


def calibrate_config(cfg_path: Path) -> bool:
    """Measure the engine's hardware-dependent settings on this PC (tools/calibrate.py), write them into the run
    config and remember them per PC and model in the settings file, so an update or a reinstall keeps them."""
    sys.path.insert(0, str(ROOT / "tools"))
    import calibrate as CAL
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    say()
    say("  Tuning Strata for this PC: the output speed is measured with a few engine settings (the PCIe share, the")
    say("  draft depth, the CPU threads). It takes about 5-10 minutes; the PC is busy meanwhile.")
    try:
        res = CAL.run(cfg, say=say)
    except Exception as e:                             # never stops an install: the defaults stay
        warn(f"the tuning did not finish ({e}): the default settings stay")
        return False
    cfg["args"] = CAL.apply(cfg["args"], res["settings"])
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    st = load_settings()
    st.setdefault("calibration", {})[hardware_key(cfg)] = {"settings": res["settings"], "tok_s": res["report"].get("tok_s"),
                                                           "date": time.strftime("%Y-%m-%d")}
    save_settings(st)
    if res["settings"]:
        ok("tuned for this PC: " + ", ".join(f"{k} {v}" for k, v in res["settings"].items())
           + (f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""))
    else:
        ok("tuned for this PC: the default settings are already the fastest here"
           + (f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""))
    return True


def saved_calibration(cfg: dict) -> dict | None:
    """The settings an earlier calibration found for this PC and model, if any."""
    return (load_settings().get("calibration") or {}).get(hardware_key(cfg))


def upgrade_config(cfg_path: Path, cfg: dict) -> dict:
    """Configs written before v0.1.13 read prompts in fixed 2048-token chunks; the engine now picks the chunk
    itself (`--prefill auto`: up to 8192, as the free VRAM allows - about 2x faster on long prompts).  Under WSL,
    KV streaming is dropped: its RAM copy must be pinned, and the driver pins only about 1 GB there."""
    a = cfg.get("args", [])
    changed = False
    ver = engine_version(cfg["exe"]) if "--prefill" in a else (0, 0, 0)
    if "--prefill" in a and a[a.index("--prefill") + 1] == "2048" and ver >= (0, 1, 13):
        a[a.index("--prefill") + 1] = "auto"
        changed = True
        ok("prompt reading: the engine now picks its chunk size (--prefill auto)")
    elif "--prefill" in a and a[a.index("--prefill") + 1] == "auto" and (0, 0, 0) < ver < (0, 1, 13):
        a[a.index("--prefill") + 1] = "2048"           # an older engine kept after a failed update (issue #49)
        changed = True
        warn(f"the installed engine is {'.'.join(map(str, ver))}: prompts are read in 2048-token chunks until it is updated")
    if is_wsl() and "--kv-resident" in a:
        i = a.index("--kv-resident")
        del a[i:i + 2]
        changed = True
        ok("WSL: KV streaming off (the driver pins only about 1 GB of RAM); the KV cache stays in VRAM")
    if changed:
        cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return cfg


def start(cfg_path: Path, port: int | None, gpu: int | list | None = None, open_browser=True, yes=False,
          layer_split=None) -> int:
    cfg = upgrade_config(cfg_path, json.loads(cfg_path.read_text(encoding="utf-8-sig")))
    missing = [p for p in [cfg["exe"], *[a for a in cfg["args"] if a.endswith(".gguf")]] if not Path(p).exists()]
    if missing:
        fail(f"{cfg_path.name} refers to missing files: {missing[0]}", "run it again with --setup to repair")
    cfg_path.touch()                                     # the most recently used model
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port or cfg.get("port", 8080))]
    if cfg.get("backend") == "hip":                    # AMD: one card, numbered as HIP numbers them
        if isinstance(gpu, list):
            fail("several GPUs sharing one model: NVIDIA only for now", "use one AMD card (--gpu N)")
        if gpu is not None:
            cmd += ["--gpu", str(gpu)]
        found = []
        use = gpu if gpu is not None else cfg.get("gpu")
        g = next((x for x in amd_gpus() if x["index"] == (use if use is not None else x["index"])
                  and amd_problem(x) is None), None)
        if g is not None:
            ok(f"GPU: {g['name']} ({g['vram_gb']:.0f} GB, AMD)")
    else:
        found = gpus()
    if cfg.get("backend") == "hip":
        pass
    elif isinstance(gpu, list):                        # --gpus: saved, this model runs on these cards from now on
        check_gpus(gpu, found)
        cfg["gpu"], cfg["gpus_asked"] = gpu, True
        cfg["layer_split"] = layer_split or cfg.get("layer_split") or "auto"
        cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        gpu = None
    elif gpu is not None:                              # --gpu N: this start only, on that card
        check_gpus([gpu], found)
        cmd += ["--gpu", str(gpu)]
    else:
        cfg = offer_together(cfg_path, cfg, yes)
    use = gpu if gpu is not None else cfg.get("gpu")
    if cfg.get("backend") == "hip":
        pass
    elif isinstance(use, list):
        check_gpus(use, found, "(chosen for this model) ")
        byid = {g["index"]: g for g in found}
        ok("GPUs: " + " + ".join(gpu_name(byid[i]) for i in use) + f" together (layers split {cfg.get('layer_split') or 'auto'})")
    elif found:
        g = next((x for x in found if x["index"] == use), None) if use is not None else max(
            found, key=lambda x: (round(x["vram_gb"]), -x["index"]))
        if g is not None:
            ok("GPU: " + gpu_name(g))
    if open_browser:
        cmd.append("--open")
    gb = 0.0
    if "--native" in cfg["args"]:
        try:
            gb = Path(cfg["args"][cfg["args"].index("--native") + 1]).stat().st_size / 1e9
        except (OSError, IndexError):
            pass
    say()
    say("  " + "-" * 100)
    say(f"  Starting {cfg.get('model_name', 'the model')}: it loads {f'about {gb:.0f} GB' if gb >= 1 else '34-55 GB'} "
        "into RAM and locks part of it for the GPU.")
    say("  While it does, YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES (longer the first time after a")
    say("  restart). That is normal: please wait and don't close this window - the browser opens when it is ready.")
    say("  Later, closing this window stops the model.")
    say("  " + "-" * 100)
    return subprocess.call(cmd)


def write_run_script(model, cfg_path, port):
    serve = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
             "--port", str(port), "--open"]
    if WIN:
        script = ROOT / f"run-{model.lower()}.bat"
        script.write_text("@echo off\r\ntitle Strata " + model + "\r\ncd /d \"" + str(ROOT) + "\"\r\n" +
                          " ".join(f'"{x}"' for x in serve) + "\r\npause\r\n", encoding="utf-8")
    else:
        script = ROOT / f"run-{model.lower()}.sh"
        script.write_text("#!/bin/sh\ncd \"" + str(ROOT) + "\"\nexec " + " ".join(f'"{x}"' for x in serve) + "\n",
                          encoding="utf-8")
        script.chmod(0o755)
    return script


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=list(FAMILIES), help="qwen = Qwen3.8-Flash-Next, swift = Swift 1.5")
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--context", type=int)
    ap.add_argument("--kv", choices=["int8", "q4_0", "k8v4"],
                    help="KV cache precision above 8K context: int8 (default), q4_0 (half the memory, a little less "
                         "precise) or k8v4 (hybrid: INT8 K + 4-bit V, 816 B/cell)")
    ap.add_argument("--vision", choices=["yes", "no", "none", "gpu", "cpu"],
                    help="let the model read images (yes = the encoder on the GPU)")
    ap.add_argument("--experimental-speed-projection", metavar="on|off|GGUF",
                    help="EXPERIMENTAL, off by default: the control vector in data/experimental-speed-projection "
                         "(or another GGUF) as a projection on layers 4-44; see docs/DETAILS.md")
    ap.add_argument("--port", type=int, help="the server's port (default: the one the install was set up with, 8080 for a new one)")
    ap.add_argument("--gpu", help="one GPU, numbered as nvidia-smi numbers them (default: asked when several can be "
                                  "used; with --setup it is saved, when starting it is for that start only)")
    ap.add_argument("--gpus", help="several GPUs sharing one model, as nvidia-smi numbers them: \"0,2\", or \"all\" "
                                   "(every card that can); the first is the main one. Saved, also when starting "
                                   "(see docs/MULTI_GPU.md)")
    ap.add_argument("--layer-split", help="with --gpus: where each later GPU's layers start (\"18\", \"16,32\"); "
                                          "default auto, placed from each GPU's free VRAM")
    ap.add_argument("--host", help="where the server listens: 127.0.0.1 = this PC only (default), 0.0.0.0 = also other "
                                   "devices on your network (issue #26; set --api-key too)")
    ap.add_argument("--api-key", help="require this key from clients (recommended with --host 0.0.0.0)")
    ap.add_argument("--data-dir", help="where the model files go (~70-120 GB): default Strata-data next to this folder, "
                                       "remembered for every Strata folder on this PC")
    ap.add_argument("--models-dir", help="where the GGUF files go (default: <data folder>/models)")
    ap.add_argument("--gguf-dir", help="use GGUF files you already have (a folder with the two shards)")
    ap.add_argument("--yes", action="store_true", help="accept the recommended answers")
    ap.add_argument("--setup", action="store_true", help="install another model or change settings")
    ap.add_argument("--no-start", action="store_true", help="install only, do not start the model")
    ap.add_argument("--build", action="store_true", help="compile the engine instead of using the ready-made one")
    ap.add_argument("--prebuilt", default=os.environ.get("STRATA_PREBUILT_URL", PREBUILT_URL),
                    help="where the ready-made engine is (a URL folder or a local folder)")
    ap.add_argument("--check", action="store_true", help="only check this PC and exit")
    ap.add_argument("--calibrate", action="store_true",
                    help="tune the engine's settings for this PC (about 5-10 minutes), then start the model")
    ap.add_argument("--backend", choices=["cuda", "hip"],
                    help="cuda = NVIDIA (default), hip = AMD RX 7900 XT/XTX on Linux (experimental; chosen by itself "
                         "when the PC has no NVIDIA card Strata can use)")
    ap.add_argument("--skip-build", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.gpu is not None:                              # --gpu 0,2 means --gpus 0,2 (a user tried it: issue report)
        if "," in a.gpu:
            a.gpus, a.gpu = a.gpus or a.gpu, None
        elif a.gpu.strip().isdigit():
            a.gpu = int(a.gpu)
        else:
            ap.error(f"--gpu takes a GPU number as nvidia-smi numbers them, e.g. --gpu 1 (or --gpus 0,2), not {a.gpu!r}")
    say("Strata - Qwen3.8-Flash-Next on a normal PC (a GPU + system RAM + CPU)")
    data, elsewhere = data_folder(a.data_dir)          # the model files: in the data folder, found from any copy
    roots = [data, *elsewhere]
    if a.models_dir is None:
        a.models_dir = str(data / "models")

    # ---- 0. already installed: just start it
    have = installed_configs()
    explicit = a.setup or a.model or a.family or a.check or a.no_start
    if not have and not explicit:                      # a new copy of Strata (an update unzipped elsewhere): set it
        prev = previous_config(elsewhere, load_settings())   # up like the last one, from the files already here
        if prev is not None:
            ch = choices_from_config(prev)
            if ch["model"]:
                say(f"  Found your earlier install in {prev.parent} ({prev.stem[len('strata-'):]}): setting up this "
                    "copy the same way - the model files are reused, nothing big is downloaded.")
                a.family, a.model, a.context = ch["family"], ch["model"], a.context or ch["context"]
                a.kv = a.kv or ch["kv"]
                a.vision = a.vision or ch["vision"]
                a.experimental_speed_projection = a.experimental_speed_projection or ch["esp"]
                a.host, a.api_key = a.host or ch["host"], a.api_key or ch["api_key"]
                a.port = a.port or ch["port"]
                if isinstance(ch.get("gpu"), list):     # a layer split: set up across the same cards again
                    a.gpus = a.gpus or ",".join(str(g) for g in ch["gpu"])
                    a.layer_split = a.layer_split or ch.get("layer_split")
                else:
                    a.gpu = a.gpu if a.gpu is not None else ch.get("gpu")
                a.yes = True
    global GPU_PICK
    # starting an installed model: --gpus 0,2 (or all) saves those cards for it and starts on them (it used to start
    # on the first one alone unless given with --setup), --gpu N runs this start on one card; neither: the saved
    # choice, and asked once when the PC has cards that could share the model
    run_gpu = (parse_gpus(a.gpus, gpus()) if a.gpus else None) or a.gpu
    port = a.port or 8080                              # a new install's port (issue #32: --port for an existing one)
    if have and a.calibrate and not (a.setup or a.model or a.family or a.check):
        if not a.build:
            update_installed_engine(a.prebuilt)
        pick_cfg = have[0]
        if len(have) > 1:
            say()
            for i, c in enumerate(have, 1):
                say(f"  {i}) {json.loads(c.read_text(encoding='utf-8-sig')).get('model_name', c.stem)}")
            pick_cfg = have[int(ask("Tune which one?", [str(i) for i in range(1, len(have) + 1)], "1", a.yes)) - 1]
        calibrate_config(pick_cfg)
        return 0 if a.no_start else start(pick_cfg, a.port, run_gpu, yes=a.yes, layer_split=a.layer_split)
    if have and not (a.setup or a.model or a.family or a.check or a.no_start):
        if not a.build:
            update_installed_engine(a.prebuilt)
        if len(have) == 1:
            return start(have[0], a.port, run_gpu, yes=a.yes, layer_split=a.layer_split)
        say()
        for i, c in enumerate(have, 1):
            say(f"  {i}) {json.loads(c.read_text(encoding='utf-8-sig')).get('model_name', c.stem)}")
        say(f"  {len(have) + 1}) install another model / change settings")
        pick = int(ask("Which one?", [str(i) for i in range(1, len(have) + 2)], "1", a.yes))
        if pick <= len(have):
            return start(have[pick - 1], a.port, run_gpu, yes=a.yes, layer_split=a.layer_split)

    # ---- 1. the PC
    step(1, "checking your PC")
    found = gpus()
    amd = [] if WIN else amd_gpus()
    hip = a.backend == "hip" or (a.backend is None and not any(gpu_problem(g) is None for g in found)
                                 and any(amd_problem(g) is None for g in amd))
    if hip:                                            # AMD (experimental): one card, compiled here
        if WIN:
            fail("Strata's AMD backend runs on Linux only", "use an NVIDIA RTX 30 series or newer card on Windows")
        say("  Your AMD GPUs:" if amd else "  No AMD GPU found (the amdgpu driver's KFD topology is empty).")
        for g in amd:
            say(f"    GPU {g['index']}: {g['name']}, {g['vram_gb']:.0f} GB VRAM - " + (amd_problem(g) or "can be used"))
        usable = [g for g in amd if amd_problem(g) is None]
        if not usable:
            fail("no AMD GPU Strata can use", "the AMD backend runs on the RX 7900 XT / XTX (gfx1100) on Linux")
        if a.gpus:
            fail("several GPUs sharing one model: NVIDIA only for now", "use one AMD card (--gpu N)")
        if a.gpu is not None:
            gpu = next((g for g in usable if g["index"] == a.gpu), None)
            if gpu is None:
                fail(f"AMD GPU {a.gpu} cannot be used", "use one of: " + ", ".join(f"--gpu {g['index']}" for g in usable))
        else:
            gpu = max(usable, key=lambda x: (round(x["vram_gb"]), -x["index"]))
        gpu = {**gpu, "count": len(amd), "archs": [gpu["arch"]]}
        sel, multi, chosen = [gpu["index"]], [], [gpu]
        a.gpu = gpu["index"] if len(amd) > 1 else a.gpu
        ok(f"GPU: {gpu['name']}, {gpu['vram_gb']:.1f} GB VRAM, {gpu['arch']} (AMD, experimental: docs/AMD_HIP.md)")
    else:
        if not found:
            fail("no NVIDIA GPU found (nvidia-smi did not answer)",
                 "install the NVIDIA driver from https://www.nvidia.com/drivers and restart the PC"
                 + ("; an AMD RX 7900 XT/XTX: --backend hip" if amd else ""))
        if len(found) > 1 or gpu_problem(found[0]) is not None:
            gpu_table(found)
        sel = choose_gpus(a, found)                    # asked when two or more cards can share the model
        multi = sel if len(sel) > 1 else []
        a.gpu = sel[0]                                 # the main GPU: the checks and the sizing below are its
        GPU_PICK = a.gpu
        gpu = gpu_info(a.gpu)
        chosen = [gpu_info(i) for i in sel]
        gpu["archs"] = sorted({x["arch"] for x in chosen})  # the engine needs code for every one of them
        if multi:
            ok("GPUs: " + " + ".join(gpu_name(x) for x in chosen) + " together (the model's layers are split across them)")
        ok(f"GPU: {gpu['name']}, {gpu['vram_gb']:.1f} GB VRAM, compute capability {cc(gpu)}, driver {gpu['driver']}")
        if driver_major(gpu) < MIN_DRIVER:
            fail(f"the NVIDIA driver is too old ({gpu['driver']}; {MIN_DRIVER} or newer is needed)",
                 "update it with the NVIDIA App or from https://www.nvidia.com/drivers, restart, and run this again")
    if gpu["vram_gb"] < 11:
        warn("less than 12 GB of VRAM: Strata will run, but most experts stay on the CPU and it will be slow")
    ram = ram_gb()
    cpu, avx2, avx512 = cpu_info()
    need = min(d["ram_gb"] for d in MODELS.values())
    if ram < need - 4 and not a.check:
        # every model keeps ALL its experts in RAM (23+ GB); VRAM only holds a copy of the most-used ones, so a
        # bigger GPU does not lower this
        fail(f"RAM: {ram:.0f} GB - the smallest model (the Coder) needs about {need} GB",
             "Strata keeps all of the model's experts in RAM (23-50 GB, whatever the GPU) and the GPU holds a copy "
             "of the most-used ones: it needs 32 GB of RAM or more (48 GB for the full model)")
    ok(f"RAM: {ram:.0f} GB" if ram >= need - 4 else f"RAM: {ram:.0f} GB (less than the {need} GB the smallest model needs)")
    pf = page_file_gb()
    if pf is not None and pf < 4:
        warn(f"Windows' page file is {pf:.1f} GB: the graphics card's memory needs room there too (issue #60), so "
             "the model may not start or may use less VRAM. Set it to \"System managed\": System > About > "
             "Advanced system settings > Performance > Advanced > Virtual memory")
    ok(f"CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2' if avx2 else 'no AVX2'})")
    if not avx2:
        fail("this CPU has no AVX2; Strata needs at least AVX2")
    if a.check:
        say()
        for m, d in MODELS.items():
            verdict = "fits" if ram >= d["ram_gb"] else "tight" if ram >= d["ram_gb"] - 8 else "does not fit"
            say(f"  {m:8s} needs ~{d['ram_gb']} GB RAM: {verdict}")
        say("\nThis PC can run Strata. Run it again without --check to install.")
        return 0

    # ---- 2. the questions
    step(2, "your choices")
    fams = list(FAMILIES)
    if a.family:
        family = a.family
    else:
        for i, f in enumerate(fams, 1):
            d = FAMILIES[f]
            say(f"  {i}) {d['title']:20s} {d['by']} - {d['about']}")
        family = fams[int(ask("Which model?", [str(i) for i in range(1, len(fams) + 1)], "1", a.yes)) - 1]
    fam = FAMILIES[family]
    ok(f"model: {fam['title']}")
    if fam.get("license"):
        say(f"  Its license: {fam['license']}")
    say()
    names = [m for m in MODELS if family in MODELS[m].get("families", ("qwen", "swift"))]
    if a.model and a.model not in names:
        fail(f"{fam['title']} has no {a.model} model file", "choose one of: " + ", ".join(names))
    for i, m in enumerate(names, 1):
        d = MODELS[m]
        fit = "" if ram >= d["ram_gb"] else f"   <- needs {d['ram_gb']} GB RAM, you have {ram:.0f}"
        say(f"  {i}) {m:8s} {d['about']}; download {d['download_gb']:.0f} GB, uses ~{d['arena_gb']:.0f} GB of RAM{fit}")
    rec = str(names.index("IQ3_XXS") + 1) if ram >= 60 and "IQ3_XXS" in names else "1"
    model = a.model or names[int(ask("Which size?", [str(i) for i in range(1, len(names) + 1)], rec, a.yes)) - 1]
    if ram < MODELS[model]["ram_gb"] - 4:
        # #125: a warning and a question, not a stop: the user may accept paging (asked, "no" by default, so an
        # unattended --yes install still stops here)
        need_gb, arena = MODELS[model]["ram_gb"], MODELS[model]["arena_gb"]
        warn(f"{model} needs about {need_gb} GB of RAM and this PC has {ram:.0f} GB: its experts alone are "
             f"{arena:.0f} GB and must stay in RAM, so Windows/Linux will page part of them from disk. Expect it "
             "to be much slower, and it may not start at all.")
        say("       A smaller size (Q2_0 or IQ2_XS) fits; more RAM fixes it.")
        if ask("  Install it anyway?", ["y", "n"], "n", a.yes) != "y":
            fail(f"{model} needs about {need_gb} GB of RAM; this PC has {ram:.0f} GB",
                 "choose Q2_0 or IQ2_XS, or add RAM")
        warn(f"installing {model} with {ram:.0f} GB of RAM, as you chose")
    ok(f"size: {model}")
    tag = fam["tag"] + model                           # names of the pack, config and start script
    small = min(x["vram_gb"] for x in chosen)         # each card keeps its layers' KV of the whole context
    rec_ctx = 32768 if small < 14 else 65536 if small < 20 else 131072
    if a.context:
        ctx = a.context
    else:
        say()
        say("  Context length = how much text the model can see at once (your chat, files, tool output).")
        say("  Longer needs more VRAM for it, so fewer experts fit on the GPU:")
        for i, c in enumerate(CONTEXTS, 1):
            say(f"  {i}) {c // 1024}K tokens" + ("   (recommended for your GPU)" if c == rec_ctx else ""))
        ctx = CONTEXTS[int(ask("Context?", [str(i) for i in range(1, 6)], str(CONTEXTS.index(rec_ctx) + 1), a.yes)) - 1]
    if model in ("IQ3_XXS", "IQ3_S") and ram < 90 and ctx > 131072:
        warn(f"{model} with a 262K context needs more than 64 GB of RAM ({MODELS[model]['arena_gb']:.0f} GB of experts "
             "+ the context): using 128K")
        ctx = 131072
    ok(f"context: {ctx} tokens")
    # the KV cache (the model's memory of the conversation): 8-bit, or 4-bit after a Hadamard rotation (PR #21)
    kv = "fp16" if ctx <= 8192 else (a.kv or "int8")
    if ctx > 8192 and not a.kv and not a.yes:
        say()
        say("  KV cache precision (the model's memory of the conversation):")
        say("  1) 8-bit   (recommended: what every published number was measured with)")
        say("  2) 4-bit   half the memory (about 4% faster at 128K), but measurably less precise on long")
        say("             documents; long-context lookups (needle tests) still pass")
        kv = ["int8", "q4_0"][int(ask("KV cache?", ["1", "2"], "1", a.yes)) - 1]
    if ctx > 8192:
        ok(f"KV cache: {'8-bit' if kv == 'int8' else '4-bit (Hadamard-rotated)'}")
    if hip:
        vision = "none"
        if a.vision not in (None, "no", "none"):
            warn("images are not available with the AMD backend yet: off")
    elif a.vision:
        vision = {"yes": "gpu", "no": "none"}.get(a.vision, a.vision)
    else:
        say()
        say("  Images: the model can also read pictures (screenshots, photos, scanned pages). This adds a 0.9 GB")
        say("  download and keeps ~1.4 GB of VRAM free for the image encoder, so text is a few % slower.")
        vision = "gpu" if ask("Do you want images?", ["y", "n"], "n", a.yes) == "y" else "none"
    ok("images: " + {"none": "off", "gpu": "on", "cpu": "on (encoder on the CPU)"}[vision])
    # EXPERIMENTAL: the experimental-speed-projection control vector (data/experimental-speed-projection), off unless
    # chosen here; with it loaded, the web app and the API switch it off per request
    esp = None
    esp_choice = (a.experimental_speed_projection or "").strip()
    if family in ("qwen", "coder"):                   # the Coder: the same model's residual stream
        if not esp_choice:
            say()
            say("  EXPERIMENTAL - speed projection: a small control vector applied while the model runs (layers 4-44).")
            say("  It changes how the model answers: its package describes it as a refusal-direction projection (the")
            say("  model declines far fewer requests). Off unless you choose it; when on, the web app can switch it off")
            say("  per chat. Details: data/experimental-speed-projection/README.md")
            esp_choice = "on" if ask("Turn on the experimental speed projection?", ["y", "n"], "n", a.yes) == "y" else "off"
        if esp_choice.lower() not in ("off", "no", "n", "0"):
            esp = ESP_VECTOR if esp_choice.lower() in ("on", "yes", "y", "1") else Path(esp_choice).expanduser().resolve()
            if not esp.is_file():
                fail(f"the experimental speed projection's vector is missing: {esp}")
        ok("experimental speed projection: " + ("ON (experimental)" if esp else "off"))
    elif esp_choice.lower() not in ("", "off", "no", "n", "0"):
        warn("the experimental speed projection is made for the original Qwen3.8-Flash-Next, not Swift 1.5: left off")
    models_dir = Path(a.gguf_dir) if a.gguf_dir else Path(a.models_dir) / tag
    shards = [models_dir / fam["file"].format(q=model, i=i) for i in (1, 2)]
    if not a.gguf_dir and not all(sh.exists() and done(sh) for sh in shards):
        for r in elsewhere:                            # already downloaded in a Strata folder on another drive
            cand = [r / "models" / tag / sh.name for sh in shards]
            if all(c.exists() and done(c) for c in cand):
                models_dir, shards = cand[0].parent, cand
                ok(f"model files found in {models_dir}")
                break
    have_model = all(s.exists() and (done(s) or a.gguf_dir) for s in shards)
    need = (0 if a.gguf_dir or have_model else MODELS[model]["download_gb"]) + 8 + \
        (40 if model == "Q2_0" and avx512 and family == "qwen" else 0) + (1 if vision != "none" else 0)
    if free_gb(models_dir) < need:
        fail(f"not enough free disk space in {models_dir}: need ~{need:.0f} GB", "use --models-dir on a bigger drive")

    # ---- 3. python packages
    step(3, "Python packages")
    pip_install(PY_PACKAGES, "numpy, jinja2, regex, pyyaml, tqdm, requests, cmake, ninja, pillow, psutil")

    # ---- 4. the engine
    step(4, "the Strata engine")
    llama = get_llama_cpp()
    ok(f"llama.cpp {LLAMA_CPP_COMMIT[:7]} (gguf-py, ggml, mtmd)")
    eng = None if a.build or hip else get_prebuilt(a.prebuilt, gpu, vision)
    if eng is not None and json.loads((eng / "BUILD.json").read_text()).get("source") != "local":
        pip_install(CUDA_WHEELS, "NVIDIA CUDA libraries (cuBLAS, CUDA runtime; ~0.4 GB)")
        if vision != "none" and not (eng / VEXE).exists():
            warn("the ready-made engine has no image encoder: compiling it")
            eng = None
    if eng is None:
        eng = build_engine_hip(gpu, llama) if hip else build_engine(gpu, vision, a.yes, llama)
    meta = json.loads((eng / "BUILD.json").read_text())
    lib_dirs = meta.get("lib_dirs") or meta.get("cuda_dirs") or cuda_lib_dirs()
    ok(f"engine: {eng / EXE}")

    # ---- 5. the model files
    step(5, f"downloading {fam['title']} {model}")
    if not a.gguf_dir:
        for s in shards:
            if s.exists() and done(s):
                ok(f"{s.name} already downloaded")
                continue
            # the original's shard 2 is the same file for all its sizes and the Coder: reuse one that is already here
            other = [p for p in Path(a.models_dir).glob("*/Qwen3.8-Flash-Next-GSQ-RCO-*-00002-of-00002.gguf") if done(p)]
            if family in ("qwen", "coder") and s.name.endswith("00002-of-00002.gguf") and other and not s.exists():
                try:
                    os.link(other[0], s)
                    mark(s)
                    ok(f"{s.name} shared with {other[0].parent.name} (identical file)")
                    continue
                except OSError:
                    pass
            download(fam["hf"].format(q=model) + s.name, s)
    check_shards(shards)
    ok("model files present")
    mmproj = Path(a.models_dir) / fam["mmproj"]
    if not mmproj.exists():
        mmproj = find_in(roots, f"models/{fam['mmproj']}") or mmproj
    if vision != "none":
        if not mmproj.exists() and a.gguf_dir and (Path(a.gguf_dir) / fam["mmproj"]).exists():
            mmproj = Path(a.gguf_dir) / fam["mmproj"]
        else:
            download(fam["mmproj_hf"] + fam["mmproj"], mmproj, "vision encoder")
        ok(f"vision encoder: {mmproj}")

    # ---- 6. the pack and the MTP draft layer
    step(6, "preparing the model for Strata")
    pack = find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    env = dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py"))
    if model == "Q2_0" and avx512 and family == "qwen":
        # the Q2_0 experts repacked for the AVX-512 kernel (the measured speed): a one-time ~40 GB conversion
        if not (pack / "index.txt").exists() or not (pack / "experts.bin").exists():   # index.txt is written last
            say("  Converting the Q2_0 experts for the AVX-512 kernel (one time, ~40 GB written, 2-5 min) ...")
            run([sys.executable, str(ROOT / "tools" / "strata_pack.py"), "build", "--gguf", str(shards[0]),
                 "--out", str(pack), "--skip-hash"], env=env)
            run([sys.executable, str(ROOT / "tools" / "pack_index.py"), "--pack", str(pack)], env=env)
        if not (pack / "tokenizer" / "vocab.json").exists():
            run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(shards[0]),
                 "--out", str(pack)], env=env)   # writes <pack>/tokenizer/
    elif not (pack / "native_experts.txt").exists() or not (pack / "tokenizer" / "vocab.json").exists():
        # every tensor as the GGUF stores it; the experts are read from the GGUF at start (seconds to build)
        run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack)], env=env)
    ok(f"model prepared: {pack}")
    mtp = (find_in(roots, "mtp/rt/experts.bin") or data / "mtp/rt/experts.bin").parent.parent
    rt = mtp / "rt"
    if not (rt / "experts.bin").exists():
        say("  The MTP draft layer (speculative decoding, ~2x faster output) comes from the original Qwen checkpoint:")
        say("  only its ~5 GB of MTP tensors are downloaded.")
        run([sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "fetch", "--out", str(mtp)], env=env)
        run([sys.executable, str(ROOT / "tools" / "mtp_pack.py"), "--src", str(mtp), "--experts", "q2_0",
             "--out", str(mtp / "mtp-q2_0.gguf")], env=env)
        run([sys.executable, str(ROOT / "tools" / "mtp_rt.py"), "--gguf", str(mtp / "mtp-q2_0.gguf"), "--out", str(rt)],
            env=env)
    if not (rt / "draft_vocab.bin").exists():
        shutil.copyfile(ROOT / "data" / "draft_vocab.bin", rt / "draft_vocab.bin")
    ok(f"MTP draft layer: {rt}")

    # ---- 7. the start script
    step(7, "writing the start script")
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile                   # the PLE table's shard: shard 2 (original) or 1 (Swift)
    ple = next((s for s in shards if any(t.name == "per_layer_token_embd.weight" for t in GGUFFile(s).tensors)), None)
    if ple is None:
        fail("the model has no per_layer_token_embd tensor (is this a Qwen3.8-Flash-Next GGUF?)")
    args = ["--pack", str(pack), "--native", str(shards[0]), "--ple-gguf", str(ple),
            "--expert-profile", str(ROOT / "data" / fam.get("profile", "expert-profile.bin")), "--expert-cache", "auto",
            "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(rt),
            "--max-context", str(ctx)]
    if ctx > 8192:
        args += ["--kv", kv]
    # KV streaming: from 64K up the whole KV cache lives in RAM and only the part the attention reads (32K positions
    # per layer) stays in VRAM; the VRAM it frees holds more experts (+6% at 128K, +23% at 262K with Q2_0). It
    # costs ~13.7 KB of RAM per context token with 8-bit KV (1.7 GB at 128K), 7.5 KB with 4-bit, so only when it fits.
    kv_ram_gb = ctx * (13 * (576 if kv == "q4_0" else 1056)) / 1e9   # 12 QSA layers + the draft layer
    # Hybrid K8V4 never streams its KV (mode 0 only, layer.hpp), so it is excluded from the WHOLE streaming
    # decision rather than one threshold at a time - a future tier added to this chain cannot reintroduce the
    # combination the engine refuses (PR review).
    if kv == "k8v4":
        if ctx >= 65536:
            ok("KV streaming off: not supported with --kv k8v4; the KV cache stays in VRAM")
    elif is_wsl() and ctx >= 65536:
        ok("WSL: KV streaming off (the driver pins only about 1 GB of RAM); the KV cache stays in VRAM")
    elif ctx >= 65536 and ram >= MODELS[model]["ram_gb"] + kv_ram_gb + 1:
        args += ["--kv-resident", "32768"]
        ok(f"KV streaming on: the context's KV cache lives in RAM ({kv_ram_gb:.1f} GB), more experts fit in VRAM")
    if vision != "none":
        args += ["--vision", "--vram-reserve-mib", str(VISION[vision]["reserve_mib"])]
    if esp is not None:
        # the package's profile, with llama.cpp's flags (the engine takes the same ones)
        args += ["--control-vector-scaled", f"{esp}:1.0", "--control-vector-layer-range", "4", "44",
                 "--cvec-mode", "project", "--cvec-dir", "per-layer"]
    cfg = {"exe": str(eng / EXE), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": f"{fam['name']}-{model.lower()}", "log": str(ROOT / f"strata-{tag.lower()}.log"),
           "lib_dirs": lib_dirs, "port": port}
    if hip:
        cfg["backend"] = "hip"
        # the dense prompt GEMMs through hipBLASLt with kernels measured on this GPU generation (tools/hip; +40-60%
        # prompt speed on the 7900 XTX); the engine refuses a table made for another hipBLASLt version and falls back
        tables = sorted((ROOT / "tools" / "hip").glob(f"{gpu['arch']}-hipblaslt-*.txt"))
        if tables:
            cfg["env"] = {"STRATA_HIPBLASLT_TUNING": str(tables[-1])}
    if gpu["count"] > 1 or a.gpu is not None:
        cfg["gpu"] = gpu["index"]                      # the engine is told this card (issue #51)
        cfg["gpus_asked"] = True                       # chosen at setup: not asked again at start
    if multi:                                          # a layer split across these cards (the server adds the flag)
        cfg["gpu"] = multi
        cfg["layer_split"] = a.layer_split or "auto"
        ok(f"layer split across GPUs {multi} ({cfg['layer_split']})")
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if vision != "none":
        cfg["vision"] = {"exe": str(eng / VEXE), "mmproj": str(mmproj), "model": str(shards[0]),
                         "gpu": vision == "gpu", "max_tokens": VISION[vision]["max_tokens"]}
        if vision == "cpu":
            cfg["vision"]["threads"] = max(1, (os.cpu_count() or 8) // 2)
    cfg_path = ROOT / f"strata-{tag.lower()}.json"
    cal = None if hip else saved_calibration(cfg)     # tools/calibrate.py is NVIDIA-only for now
    if cal is not None:
        sys.path.insert(0, str(ROOT / "tools"))
        import calibrate as CAL
        cfg["args"] = CAL.apply(cfg["args"], cal.get("settings") or {})
        ok("the settings tuned for this PC earlier are used" + (f" ({cal['date']})" if cal.get("date") else ""))
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    script = write_run_script(tag, cfg_path, port)
    # offered only when someone answers: --yes installs and adopted earlier installs are not held up by it
    if cal is None and not hip and not a.no_start and not a.yes and ask(
            "Tune Strata for this PC now? It measures a few engine settings (about 5-10 minutes; the PC is busy "
            "meanwhile; later: START-HERE --calibrate)", ["y", "n"], "y", a.yes) == "y":
        calibrate_config(cfg_path)
    ok(f"start script: {script.name}")

    say()
    say("All set.")
    say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1   (any API key; model name: anything)")
    say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    if a.host and a.host not in ("127.0.0.1", "localhost"):
        say(f"  Other devices:    the server window prints this PC's address (http://<IP>:{port}/)"
            + ("" if a.api_key else " - no API key set: anyone on your network can use it"))
    say(f"  Next time:        just run {'START-HERE.bat' if WIN else './setup.sh'} (or {script.name}) - it starts right away")
    if vision != "none":
        say("  Images:           send them in the chat page, in chat.py (/image <path>) or over the API")
    if a.no_start:
        return 0
    return start(cfg_path, port)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nstopped.")
        sys.exit(1)

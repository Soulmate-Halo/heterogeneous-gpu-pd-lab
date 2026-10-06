"""serve/server.py - plan v0.3 P8: OpenAI and Anthropic endpoints over any engine that maps token ids to tokens.

    python -m serve.server --engine mock --port 8095            (a scripted engine, for clients and tests)
    python -m serve.server --engine strata --config strata.json --port 8080   (the real engine, resident)

Endpoints: POST /v1/chat/completions (OpenAI, stream and non-stream), POST /v1/messages (Anthropic, stream and
non-stream), GET /v1/models, GET /models, GET /props, GET /slots, GET /health, GET /mcp. One sequence at a time behind a FIFO (plan: one resident sequence).
Tools from MCP servers (serve/mcp.py, `"mcp_servers"` in the config or --mcp-config) are offered only to requests that
ask for them with `"strata_mcp": true` - the web app does; other clients see exactly the API they always saw.
Images (optional, when the config has a "vision" entry): OpenAI image_url parts and Anthropic image blocks (base64
data, http(s) URLs or local file paths) go through `strata-vision` (the model's mmproj file) and reach the engine as
embeddings (`GENI`).  JPEG/PNG/BMP/GIF go straight in; WebP, TIFF, AVIF, ... (agents like omp send WebP) are
converted to PNG first with Pillow.
Requests whose prompt plus max tokens exceed the engine's context are REJECTED with 400, never truncated.
An unset (or 0, or -1) max tokens means "unlimited": whatever the prompt leaves of the context.

The engine boundary is `Engine.generate(prompt_ids, max_new, sampling, cancel) -> iterator of token ids`.
`StrataEngine` keeps one `strata --serve` process resident (weights, expert arena and VRAM tier load once) and
talks to it over stdin/stdout; `MockEngine` is a scripted stand-in that makes every API path testable without a GPU.
"""
from __future__ import annotations

import argparse
import collections
import base64
import hashlib
import codecs
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, Protocol
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))   # run as a script (run-<model>.bat) as well as a module
from serve.frontend import (ChatTemplate, Event, OutputParser, anthropic_to_messages,  # noqa: E402
                            images_of, openai_to_messages)
from serve.mcp import McpCancelled, hub_from_config  # noqa: E402

IM_END = "<|im_end|>"
IMAGE_PAD = "<|image_pad|>"
CTX_SLACK = 8               # `strata --serve` rejects prompt + max_new + 8 > context: keep the same margin here
# The live tok/s is a rate over a window, not a mean since the first token: a mean reads ~1/elapsed at the first
# token (the Monitor showed five-digit numbers) and then undershoots for the first second of every answer.
RATE_WINDOW_S = 2.0
RATE_MIN_SPAN_S = 0.25      # younger than this there is no rate yet: the mean so far, with the span floored here


# ------------------------------------------------------------------------------------------------ engines
class Engine(Protocol):
    max_context: int
    def generate(self, ids: list[int], max_new: int, sampling: dict, cancel: threading.Event) -> Iterator[int]: ...


class MockEngine:
    """Replays a scripted completion (text) as token ids, one per step, then the end-of-turn token.  Given a list of
    scripts, each request gets the next one and the last one repeats (a tool call, then the answer after it)."""

    def __init__(self, tokenizer, script: str | list[str], max_context: int = 32768, delay_s: float = 0.0):
        self.tok, self.max_context, self.delay = tokenizer, max_context, delay_s
        end = tokenizer.encode(IM_END, parse_special=True)
        self.scripts = [tokenizer.encode(x, parse_special=True) + end for x in ([script] if isinstance(script, str)
                                                                                 else script)]
        self.script, self.turns = self.scripts[0], 0
        self.last_prompt: list[int] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_prompt = list(ids)
        self.last_embeddings = embeddings
        if len(self.scripts) > 1:
            self.script = self.scripts[min(self.turns, len(self.scripts) - 1)]
            self.turns += 1
        for t in self.script[:max_new]:
            if cancel.is_set():
                return
            if self.delay:
                time.sleep(self.delay)
            yield t


class EngineDied(RuntimeError):
    """The engine process ended in the middle of a request (issue #27: on Linux, the out-of-memory killer)."""


def narrate_start(log_path: str, offset: int, args: list, done: threading.Event, heartbeat=20.0) -> None:
    """While the engine starts, say in the server window what it is doing, from its log: the start reads tens of GB
    into RAM and locks part of it for the GPU, and on many PCs everything is slow or frozen for a minute or more -
    people closed the window thinking it had hung.  The warning comes at that step, not after it."""
    gb = 0.0
    if "--native" in args:                              # about the size of the experts it will read
        try:
            gb = os.path.getsize(args[args.index("--native") + 1]) / 1e9
        except (OSError, IndexError):
            pass
    size = f"about {gb:.0f} GB" if gb >= 1 else "tens of GB"
    t0 = last = time.time()
    said = set()

    def say(key, text):
        nonlocal last
        if key not in said:
            said.add(key)
            last = time.time()
            print(text, flush=True)

    say("weights", "[strata] starting the engine: reading the model's weights ...")
    pos = offset
    while not done.wait(0.5):
        try:
            with open(log_path, "rb") as f:
                f.seek(pos)
                chunk = f.read()
        except OSError:
            chunk = b""
        if chunk.count(b"\n"):
            cut = chunk.rfind(b"\n") + 1
            pos += cut
            for line in chunk[:cut].decode("utf-8", "replace").splitlines():
                if "PLE on" in line or "expert arena:" in line:
                    say("arena", f"[strata] loading the experts into RAM ({size}) and locking part of them for the GPU.\n"
                                 "         YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES NOW - this is normal.\n"
                                 "         Please wait and don't close this window; the browser opens when it is ready.")
                elif " loaded " in line and "GiB at" in line:
                    say("loaded", "[strata] experts loaded: " + line.split(" loaded ", 1)[1].strip() +
                        f" ({time.time() - t0:.0f} s so far)")
                elif "expert cache " in line and " slots, " in line and "auto" not in line:
                    n = line.split("expert cache ", 1)[1].split(";")[0].replace(" slots,", " experts,").strip()
                    say("cache", f"[strata] filling the GPU's expert cache ({n}) ...")
                elif "session is up" in line:
                    say("up", "[strata] almost ready ...")
        if time.time() - last > heartbeat:
            last = time.time()
            print(f"[strata] still starting ({time.time() - t0:.0f} s) - please wait ...", flush=True)


class StrataEngine:
    """The resident engine: `strata --serve` reads `GEN <max_new> <ids>` lines and streams `T <id>` lines, then
    `DONE ...`.  Requests are serialized by the service's FIFO, so one pipe is enough.

    Per-request sampling rides the same line as engine-side keys between max_new and the ids
    (`temperature=F top_p=F top_k=N seed=N`, the engine's own spelling).  An absent temperature keeps the
    engine's default, which is greedy; `temperature=0` means the same thing, so it is not forwarded.
    """

    def __init__(self, exe: str, args: list[str], cwd: str | None = None, log: str | None = None,
                 env: dict | None = None):
        self.spawn = (exe, list(args), cwd, log, env)   # to start it again after it died (issue #27)
        paths = {k: v for k, v in zip(args, args[1:]) if k in ("--native", "--pack")}
        self.model_path = paths.get("--native") or paths.get("--pack", "pack/full")
        self.log_path = log
        self.log = open(log, "a", encoding="utf-8") if log else subprocess.DEVNULL
        loading = threading.Event()                     # set once READY: the narrator below stops
        if log:
            threading.Thread(target=narrate_start, args=(log, os.path.getsize(log), args, loading),
                             daemon=True).start()
        self.proc = subprocess.Popen([exe, "--serve", *args], cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, encoding="utf-8", bufsize=1, env=env)
        self.max_context = 0
        self.can_stop = False            # the engine honours a STOP line mid-request (READY <ctx> stop)
        self.last = {}
        self.info = {}                   # INFO key=value facts (engine 0.1.8+): kv, expert slots, ... (Monitor tab)
        self.progress = None             # (read, total) prompt tokens while a prompt is read, from PP lines
        try:                             # a ready-made engine's BUILD.json says its version
            self.info["version"] = json.loads((Path(exe).parent / "BUILD.json").read_text()).get("version")
        except (OSError, ValueError):
            self.info["version"] = None
        for line in self.proc.stdout:
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = int(v) if v.lstrip("-").isdigit() else v
            if line.startswith("READY"):
                f = line.split()
                self.max_context = int(f[1])
                self.can_stop = "stop" in f[2:]
                break
        loading.set()
        if self.max_context <= 0:
            raise RuntimeError("the engine exited before it was ready" + (f" (see {log})" if log else ""))
        # (from PR #41, midhatn) a locally built engine can sit next to another release's BUILD.json: engines that
        # report their own version (INFO engine=, 0.1.8+) win, the manifest stays the fallback for older ones
        if self.info.get("engine"):
            self.info["version"] = str(self.info["engine"])
        # the engine's stdout on a thread, so a request can wait with a timeout (heartbeats, cancel checks)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.put(line)
        self.ended = True                               # its output closed: it is gone, even before the OS says so
        self.lines.put(None)

    def death_note(self) -> str:
        """Why the engine most likely ended, from the end of its log: its own watchdog (issue #29), else RAM."""
        tail = ""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4096))
                tail = f.read().decode("utf-8", "replace")
        except (OSError, TypeError):
            pass
        for line in reversed(tail.splitlines()):
            if "issue #29" in line:
                return ("The engine stopped itself because it had stopped making progress - a hang it caught. Its log "
                        "line: " + line.strip() + " - please report it at github.com/Niko1221/Strata/issues.")
        return ("The usual cause is running out of RAM: Linux then ends the biggest program (check: sudo dmesg | "
                "grep -i -E 'killed process|out of memory'); Windows slows down instead. Close other programs or use a "
                "smaller model (Q2_0 / IQ2_XS).")

    def alive(self) -> bool:
        return not getattr(self, "ended", False) and self.proc.poll() is None

    def exit_code(self):
        try:
            return self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return None

    def restart(self):
        """Start the engine again (the same command) after it died; the new process has its own line queue."""
        try:
            self.proc.kill()
        except OSError:
            pass
        info = dict(self.info)
        self.ended = False
        self.__init__(*self.spawn)
        self.info = {**info, **self.info}

    def _parse_done(self, line):
        f = line.split()
        self.last = {"generated": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]),
                     "decode_ms": float(f[4]), "finish": f[5]}
        if len(f) >= 9:                                   # the conversation cache's fields (engine 0.1.3+)
            self.last.update(drafts_accepted=int(f[6]), drafts_offered=int(f[7]), reused=int(f[8]))
        if len(f) >= 11:                                  # decode hit rate fields
            self.last.update(hits=int(f[9]), lookups=int(f[10]))

    @staticmethod
    def sampling_keys(sampling: dict) -> str:
        keys = ""
        t = sampling.get("temperature")
        if isinstance(t, (int, float)) and float(t) > 0.0:
            keys += f" temperature={float(t)!r}"
        tp = sampling.get("top_p")
        if isinstance(tp, (int, float)) and float(tp) < 1.0:
            keys += f" top_p={float(tp)!r}"
        tk = sampling.get("top_k")
        if isinstance(tk, int) and not isinstance(tk, bool) and tk >= 0:
            # the engine's sampled path keeps at most 64 candidates: 0 ("off") and wider lists get all 64
            keys += f" top_k={tk if 1 <= tk <= 64 else 64}"
        mp = sampling.get("min_p")
        if isinstance(mp, (int, float)) and 0.0 < float(mp) <= 1.0:
            keys += f" min_p={float(mp)!r}"
        rp = sampling.get("repetition_penalty")
        rp_on = isinstance(rp, (int, float)) and float(rp) != 1.0
        pf = sampling.get("frequency_penalty")
        pf_on = isinstance(pf, (int, float)) and float(pf) != 0.0
        pp = sampling.get("presence_penalty")
        pp_on = isinstance(pp, (int, float)) and float(pp) != 0.0
        if rp_on:
            keys += f" penalty_repeat={float(rp)!r}"
        if pf_on:
            keys += f" penalty_freq={float(pf)!r}"
        if pp_on:
            keys += f" penalty_present={float(pp)!r}"
        if rp_on or pf_on or pp_on:
            # a penalty without a window counts over nothing: the engine's default is the last 64 tokens
            pln = sampling.get("penalty_last_n")
            if isinstance(pln, int) and not isinstance(pln, bool) and pln > 0:
                keys += f" penalty_last_n={pln}"
            else:
                keys += " penalty_last_n=64"
        seed = sampling.get("seed")
        if isinstance(seed, int) and seed > 0:
            keys += f" seed={seed}"
        # setup's calibration (tools/calibrate.py): engine settings for this request only, measured without a restart
        tune = sampling.get("strata_tune")
        if isinstance(tune, dict):
            for k in ("pcie_frac", "spec_min_p"):
                v = tune.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0:
                    keys += f" {k}={float(v)!r}"
        return keys + StrataEngine.projection_key(sampling)

    @staticmethod
    def projection_key(sampling: dict) -> str:
        """`cvec=0|1`: the experimental-speed-projection control vector for this request, when the engine was
        started with one (--control-vector-scaled; an engine without one ignores the key).  Absent = on."""
        on = sampling.get("experimental_speed_projection")
        return f" cvec={int(on)}" if isinstance(on, bool) else ""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        """Yields token ids, and None as a heartbeat every 10 s while the engine is quiet (reading a long prompt):
        the HTTP layer turns it into an SSE comment, which keeps clients' watchdogs calm and notices a client that
        has gone.  A consumer that stops early (or `cancel`) makes the engine STOP, so it does not run to max_new."""
        self.progress = None
        # an image request takes the same sampling keys as text (#75: it used to decode greedily whatever was asked)
        head = f"GENI {int(max_new)}{self.sampling_keys(sampling or {})} {embeddings}" if embeddings else \
            f"GEN {int(max_new)}{self.sampling_keys(sampling or {})}"
        try:
            self.proc.stdin.write(f"{head} {','.join(str(int(t)) for t in ids)}\n")
            self.proc.stdin.flush()
        except OSError:                                  # the pipe is gone: the engine died (not the client)
            raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None
        done = False
        try:
            while True:
                try:
                    line = self.lines.get(timeout=10)
                except queue.Empty:
                    if cancel.is_set():
                        return
                    yield None
                    continue
                if line is None:
                    done = True
                    raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
                if line.startswith("T "):
                    if cancel.is_set():
                        return
                    yield int(line[2:])
                elif line.startswith("PP "):
                    f = line.split()
                    if len(f) >= 3 and f[1].isdigit() and f[2].isdigit():
                        self.progress = (int(f[1]), int(f[2]))             # prompt progress, one per chunk: also a heartbeat (the
                    if cancel.is_set():                   # lines reset the 10 s wait, so without this a long prompt
                        return                            # would send no keep-alives at all)
                    yield None
                elif line.startswith("DONE"):
                    self._parse_done(line)
                    done = True
                    return
                elif line.startswith("ERR"):
                    done = True
                    raise ValueError(line[4:].strip())
        finally:
            if not done:                                  # the consumer stopped early: stop the engine, drain to DONE
                if self.can_stop:
                    try:
                        self.proc.stdin.write("STOP\n")
                        self.proc.stdin.flush()
                    except OSError:
                        pass
                while True:
                    line = self.lines.get()
                    if line is None or line.startswith("ERR"):
                        break
                    if line.startswith("DONE"):
                        self._parse_done(line)
                        break

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


class Vision:
    """The resident image encoder: `strata-vision` (llama.cpp mtmd + the mmproj file) reads `ENC <image> <out>`
    lines and writes each image's embeddings; results are cached by the image's hash, so a conversation that
    sends the same picture again (every turn, with most clients) encodes it once."""

    def __init__(self, cfg: dict, log=None, env: dict | None = None):
        args = [cfg["exe"], "--mmproj", cfg["mmproj"], "--model", cfg["model"]]
        if cfg.get("gpu"):
            args.append("--gpu")
        if cfg.get("threads"):
            args += ["--threads", str(cfg["threads"])]
        if cfg.get("max_tokens"):
            args += ["--max-tokens", str(cfg["max_tokens"])]
        self.dir = Path(tempfile.mkdtemp(prefix="strata-vision-"))
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log or subprocess.DEVNULL,
                                     text=True, encoding="utf-8", bufsize=1, env=env)
        line = self.proc.stdout.readline()
        if not line.startswith("READY"):
            raise RuntimeError("the vision encoder did not start: " + line.strip())
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[Path, int]] = {}

    @staticmethod
    def load(source: str) -> bytes:
        if source.startswith("data:"):
            return base64.b64decode(source.split(",", 1)[1])
        if source.startswith(("http://", "https://")):
            req = urllib.request.Request(source, headers={"User-Agent": "strata"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        path = source[7:] if source.startswith("file://") else source
        if path and os.path.isfile(path):
            return Path(path).read_bytes()
        raise ValueError("an image must be a data: URL, an http(s) URL or a local file path")

    @staticmethod
    def normalize(data: bytes) -> bytes:
        """The formats strata-vision's decoder (stb_image) reads pass through; anything else is converted to PNG."""
        if data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n" or data[:2] == b"BM" or \
                data[:6] in (b"GIF87a", b"GIF89a"):
            return data
        try:
            import io
            from PIL import Image
        except ImportError:
            raise ValueError("this image format needs Pillow (python -m pip install pillow); JPEG, PNG, BMP and "
                             "GIF work without it") from None
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
        except Exception as e:
            raise ValueError(f"the image could not be read ({e})") from None
        if im.mode in ("RGBA", "LA", "P") and "transparency" in im.info or im.mode in ("RGBA", "LA"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))   # transparent areas become white, not black
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()

    def encode(self, source: str) -> tuple[Path, int]:
        """-> (embeddings file, number of image tokens)."""
        data = self.normalize(self.load(source))
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            img, out = self.dir / f"{key}.img", self.dir / f"{key}.sve"
            img.write_bytes(data)
            self.proc.stdin.write(f"ENC {img} {out}\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline().strip()
            img.unlink(missing_ok=True)
            if not line.startswith("OK"):
                raise ValueError("the image could not be read: " + (line[4:] if line.startswith("ERR") else
                                                                     "the vision encoder stopped"))
            self.cache[key] = (out, int(line.split()[1]))
            if len(self.cache) > 64:                                   # oldest first
                old = next(iter(self.cache))
                self.cache.pop(old)[0].unlink(missing_ok=True)
            return self.cache[key]

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def gpu_list(cfg: dict) -> list[int]:
    """The config's "gpu": one card (2), or several for a layer split ([0, 2] or "0,2"), numbered as nvidia-smi
    numbers them; [] when it names none."""
    g = cfg.get("gpu")
    if g is None or g == "":
        return []
    items = g if isinstance(g, (list, tuple)) else str(g).split(",")
    return [int(str(x).strip()) for x in items if str(x).strip() != ""]


def engine_args(cfg: dict) -> list[str]:
    """The engine's arguments: the config's, and with several GPUs the layer split across them ("layer_split" in the
    config: "auto" by default, or the first layer of each later GPU's share, e.g. "18" or "16,32")."""
    args = list(cfg["args"])
    if len(gpu_list(cfg)) > 1 and "--layer-split" not in args:
        args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
    return args


def child_env(cfg: dict) -> dict:
    """The engine's environment: the CUDA libraries setup installed (pip's nvidia packages, or the toolkit that
    compiled it) first on the library search path."""
    env = dict(os.environ)
    if gpu_list(cfg) and cfg.get("backend") == "hip":   # AMD: numbered as HIP numbers them (setup's KFD order)
        env["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_list(cfg))
    elif gpu_list(cfg):                              # issue #51: the GPU(s) to run on, numbered as nvidia-smi does; CUDA's
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # own default order (fastest first) can number the cards otherwise
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_list(cfg))
    for k, v in (cfg.get("env") or {}).items():      # engine settings the config carries (AMD: the GEMM tuning table)
        env[str(k)] = str(v)
    dirs = [d for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
    if dirs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    return env


class ByteTokenizer:
    """Tiny stand-in tokenizer for tests without the pack: one id per UTF-8 byte, specials as ids >= 256."""
    SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]

    def encode(self, text, parse_special=False):
        out, i = [], 0
        while i < len(text):
            for k, s in enumerate(self.SPECIALS):
                if parse_special and text.startswith(s, i):
                    out.append(256 + k)
                    i += len(s)
                    break
            else:
                out.extend(text[i].encode("utf-8"))
                i += 1
        return out

    def decode(self, ids, errors="replace"):
        raw = bytearray()
        for t in ids:
            raw += self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])
        return raw.decode("utf-8", errors=errors)


# ------------------------------------------------------------------------------------------------ core
class Detokenizer:
    """Incremental decode: each token's bytes go through an incremental UTF-8 decoder, which emits the complete
    characters and holds a multi-byte character split across tokens until it is complete (invalid bytes become
    U+FFFD, as a whole decode with errors="replace" makes them).  Constant time per token - the old re-decode of
    every generated id cost 2 ms per token after 8K tokens and 4 ms after 16K (perf-review F-1).  A tokenizer
    without `token_bytes` (the tests' byte tokenizer) keeps the re-decode."""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], 0
        self.inc = codecs.getincrementaldecoder("utf-8")(errors="replace") if hasattr(tok, "token_bytes") else None

    def push(self, t: int) -> str:
        if self.inc is not None:
            return self.inc.decode(self.tok.token_bytes(t))
        self.ids.append(t)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta


class Service:
    def __init__(self, engine: Engine, tokenizer, template: ChatTemplate, model_name: str = "qwen3.8-flash-next",
                 vision: Vision | None = None, sampling_defaults: dict | None = None,
                 fit_max_tokens: bool = False):
        self.engine, self.tok, self.template, self.model, self.vision = engine, tokenizer, template, model_name, vision
        self.fit_max_tokens = fit_max_tokens          # --fit-max-tokens: clamp the output cap instead of 400
        self.sampling_defaults = dict(sampling_defaults or {})   # the run config's `sampling` block
        self.shared = {}                              # the web app's Chat settings for every client (POST /settings)
        self.shared_path = None                       # where they are kept between starts (next to the config)
        self.fifo = threading.Lock()
        self.embeddings = threading.local()           # the current request's image embeddings file (GENI)
        self.api_key = ""                              # when set, /v1/* needs it (Bearer or x-api-key)
        self.status = {"busy": False, "queued": 0}      # GET /status: what the model is doing right now
        self.rate = collections.deque(maxlen=32)        # (time, generated) samples for the live tok/s window
        self.history = collections.deque(maxlen=500)    # the last finished requests, newest last (GET /metrics)
        # since the server started (the Monitor's totals, issue #35)
        self.totals = {"since": time.time(), "requests": 0, "prompt_tokens": 0, "reused": 0, "output_tokens": 0,
                       "prompt_ms": 0.0, "decode_ms": 0.0}
        self.last_timings = None                         # the last finished request's, llama.cpp's names (/v1/status)
        self.last_request_at = None                      # when a request last started or finished
        self.started_at = time.time()
        self.status_lock = threading.Lock()
        self.mcp = None                                  # serve/mcp.py's McpHub when MCP servers are configured
        self.stop_ids = set(tokenizer.encode(IM_END, parse_special=True) +
                            tokenizer.encode("<|endoftext|>", parse_special=True))

    def set_shared(self, defaults) -> dict:
        """The Chat settings every client gets for what it leaves out; {} / None = clients use their own again."""
        self.shared = clean_shared_defaults(defaults)
        if self.shared_path:
            try:
                if self.shared:
                    Path(self.shared_path).write_text(json.dumps(self.shared, indent=1), encoding="utf-8")
                else:
                    Path(self.shared_path).unlink(missing_ok=True)
            except OSError as e:
                print(f"[strata] could not save the shared settings: {e}", flush=True)
        return self.shared

    def with_shared(self, req: dict, api: str) -> dict:
        """The request with the shared thinking level and max tokens filled in where it has none of its own."""
        s = self.shared
        if not s:
            return req
        req = dict(req)
        if "max_tokens" in s and not req.get("max_tokens") and not req.get("max_completion_tokens"):
            req["max_tokens"] = s["max_tokens"]
        effort = s.get("reasoning_effort")
        if effort:
            if api == "openai":
                ctk = req.get("chat_template_kwargs") if isinstance(req.get("chat_template_kwargs"), dict) else {}
                if not req.get("reasoning_effort") and not req.get("reasoning") and \
                        "enable_thinking" not in ctk and "reasoning_effort" not in ctk:
                    req["reasoning_effort"] = effort
            elif not req.get("thinking") and not req.get("output_config"):
                if effort == "none":
                    req["thinking"] = {"type": "disabled"}
                else:
                    req["output_config"] = {"effort": effort}
        return req

    def start_telemetry(self):
        """The hardware sampler behind GET /metrics (serve/telemetry.py), recording this server's tok/s too."""
        if getattr(self, "telemetry", None) is None:
            from serve.telemetry import Telemetry
            self.telemetry = Telemetry(extra=lambda: {"tok_s": self._tok_s(), "tok_s_mean": self._tok_s_mean()},
                                       gpu_index=int(getattr(self, "gpu_index", 0) or 0),
                                       gpu_indices=getattr(self, "gpu_indices", None))

    def _tok_s(self):
        """tok/s over the last RATE_WINDOW_S seconds.  Returns 0.0 while nothing is generating."""
        with self.status_lock:
            s = dict(self.status)
            rate = list(self.rate)
        if not s.get("busy") or not s.get("first_token"):
            return 0.0
        now = time.time()
        newest = rate[-1] if rate else None
        oldest = next(((t, g) for t, g in rate if now - t <= RATE_WINDOW_S), None)
        if newest and oldest and newest[0] - oldest[0] >= RATE_MIN_SPAN_S:
            return max(0.0, (newest[1] - oldest[1]) / (newest[0] - oldest[0]))
        return s["generated"] / max(RATE_MIN_SPAN_S, now - s["first_token"])

    def _tok_s_mean(self):
        """The whole-request mean since the first token (the old formula), kept so the two can be compared."""
        with self.status_lock:
            s = dict(self.status)
        if not s.get("busy") or not s.get("first_token"):
            return 0.0
        return s["generated"] / max(1e-6, time.time() - s["first_token"])

    def metrics(self, all_requests=False) -> dict:
        """GET /metrics: what the Monitor tab shows - the engine's facts, what it is doing, the last requests, and
        the hardware (with a minute of history per series)."""
        with self.status_lock:
            s = dict(self.status)
            hist = list(self.history)
            totals = dict(self.totals)
        now = time.time()
        progress = getattr(self.engine, "progress", None)
        if s.get("busy") and s.get("first_token") is None:
            state = "reading"
        elif s.get("busy"):
            state = "generating"
        else:
            state = "idle"
        live = {"state": state, "queued": s.get("queued", 0), "phase": s.get("phase") if s.get("busy") else None,
                "prompt_tokens": s.get("prompt_tokens") if s.get("busy") else None,
                "prompt_read": None, "prompt_total": None, "generated": s.get("generated") if s.get("busy") else None,
                "max_tokens": s.get("max_tokens") if s.get("busy") else None,
                "elapsed_s": round(now - s["started"], 1) if s.get("busy") and s.get("started") else None,
                "tok_s": round(self._tok_s(), 1) if state == "generating" else None,
                "tok_s_mean": round(self._tok_s_mean(), 1) if state == "generating" else None,
                "tok_s_window_s": RATE_WINDOW_S if state == "generating" else None}
        if state == "reading" and progress:
            live["prompt_read"], live["prompt_total"] = progress
        engine = {"model": self.model, "max_context": self.engine.max_context, "images": self.vision is not None,
                  **dict(getattr(self.engine, "info", {}) or {})}
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "history": {}, "static": {}}
        return {"engine": engine, "live": live, "requests": hist[::-1][:None if all_requests else 12],
                "requests_kept": len(hist), "totals": totals, "hardware": tel["now"],
                "hardware_static":
                tel["static"], "history": tel["history"], "time": now}

    def v1_status(self) -> dict:
        """GET /v1/status: what this server is and does, for a client that would rather ask than guess (a front-end
        that polls its OpenAI-compatible server's status, collabosm's for one): the model and its window, images,
        the APIs, what is running, and the last request's timings in llama.cpp's names.  /metrics has the rest."""
        with self.status_lock:
            s, totals = dict(self.status), dict(self.totals)
            last_t, last_at = (dict(self.last_timings) if self.last_timings else None), self.last_request_at
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "static": {}}
        hw, static = tel.get("now") or {}, tel.get("static") or {}

        def scaled(v, unit, digits=0):
            return round(v / unit, digits) if isinstance(v, (int, float)) else None

        busy, ctx = bool(s.get("busy")), self.engine.max_context
        images = self.vision is not None
        return {
            "service": "strata", "model": self.model,
            "engine": (getattr(self.engine, "info", {}) or {}).get("version"),
            "started": int(self.started_at), "uptime_s": int(time.time() - self.started_at),
            "cache_max_tokens": ctx,
            "context": {"native": ctx, "max_positions": ctx},
            "concurrency": {"serving": 1, "requested": 1},       # one request at a time; more wait their turn
            "dialects": ["/v1/chat/completions", "/v1/messages"],
            "vision": {"enabled": images, "available": images, "error": None},
            "activity": {"requests": totals["requests"] + int(busy), "in_flight": int(busy) + int(s.get("queued") or 0),
                         "last_request_at": int(last_at) if last_at else None},
            "last_timings": last_t,
            "machine": {
                "at": int(time.time()),
                "gpu": {"name": static.get("gpu_name"), "used_mib": scaled(hw.get("gpu_mem_used"), 2 ** 20),
                        "total_mib": scaled(hw.get("gpu_mem_total"), 2 ** 20), "util_pct": hw.get("gpu_util"),
                        "temp_c": hw.get("gpu_temp"), "power_w": hw.get("gpu_power")} if static.get("gpu_name") else None,
                "ram": {"used_gib": scaled(hw.get("ram_used"), 2 ** 30, 1),
                        "total_gib": scaled(hw.get("ram_total"), 2 ** 30, 1)} if hw.get("ram_total") else None}}

    def prepare(self, messages, tools, kwargs, max_new=None):
        """-> (ids, thinking, max_new). An unset or non-positive max_new (some clients send -1) means "unlimited":
        the rest of the context."""
        prompt = self.template.render(messages, tools=tools, **kwargs)
        ids = self.tok.encode(prompt, parse_special=True)
        self.embeddings.path = None
        images = images_of(messages)
        if images:
            if self.vision is None:
                raise ValueError("this server was started without the vision encoder (run setup again and choose "
                                 "'vision'), so it cannot read images")
            pad = self.tok.encode(IMAGE_PAD, parse_special=True)[0]
            # Encode only while the engine is idle: the engine and the image encoder (a separate process) must not
            # run on the GPU at the same time - an encode during a running request left that request stuck at
            # "reading the prompt" with CPU and GPU busy, for good (reproduced).  So encoding takes its turn in the
            # same FIFO as the requests.
            with self.fifo:
                encoded = [self.vision.encode(src) for src in images]
            out, k = [], 0
            for t in ids:                               # one <|image_pad|> per image -> one per image token
                if t == pad and k < len(encoded):
                    out += [pad] * encoded[k][1]
                    k += 1
                else:
                    out.append(t)
            if k != len(encoded):
                raise ValueError("the prompt and its images do not match")
            ids = out
            combined = self.vision.dir / f"req-{uuid.uuid4().hex[:12]}.sve"
            with open(combined, "wb") as f:
                for path, _ in encoded:
                    f.write(path.read_bytes())
            self.embeddings.path = combined
        room = self.engine.max_context - CTX_SLACK - len(ids)
        if max_new is None or max_new <= 0 or (self.fit_max_tokens and room < 1):
            if room < 1:
                raise ValueError(f"prompt ({len(ids)} tokens) leaves no room to answer in the context "
                                 f"({self.engine.max_context}); requests are never truncated")
            max_new = room
        elif max_new > room:
            if not self.fit_max_tokens:
                raise ValueError(f"prompt ({len(ids)} tokens) + max tokens ({max_new}) exceeds the context "
                                 f"({self.engine.max_context}); requests are never truncated")
            max_new = max(1, room)          # --fit-max-tokens: a shorter completion beats a 400
        return ids, kwargs.get("enable_thinking", True) is not False, max_new

    def _note(self, n, evs):
        with self.status_lock:
            s = self.status
            s["generated"] = n
            if s.get("first_token") is None:
                s["first_token"] = time.time()
            self.rate.append((time.time(), n))          # the live rate's window over the last RATE_WINDOW_S
            for ev in evs:
                if ev.kind == "reasoning":
                    s["phase"] = "thinking"
                elif ev.kind == "content":
                    s["phase"] = "answering"
                elif ev.kind == "tool_start":
                    s["phase"], s["tool"] = f"writing a tool call: {ev.call.name}", ev.call.name
                elif ev.kind == "tool_call":
                    s["phase"] = "tool call complete"
                s["tail"] = (s["tail"] + (ev.text or ""))[-600:]

    def _progress(self, last_print, every=1.0):
        """A progress line in the server window every `every` seconds while a request runs."""
        now = time.time()
        if now - last_print < every:
            return last_print
        with self.status_lock:
            s = dict(self.status)
        el = now - s.get("started", now)
        if s.get("first_token") is None:
            pr = getattr(self.engine, "progress", None)   # (position reached, prompt tokens): a reused prefix counts
            done = f"{pr[0]:,} of {pr[1]:,}" if pr and pr[1] else f"{s.get('prompt_tokens', 0):,}"   # as read (#29)
            print(f"[strata] reading the prompt: {done} tokens, {el:.0f} s so far", flush=True)
        else:
            rate = s["generated"] / max(1e-6, now - s["first_token"])
            print(f"[strata] {s['phase']}: {s['generated']} of max {s.get('max_tokens')} tokens, {rate:.1f} tok/s, "
                  f"{el:.0f} s", flush=True)
        return now

    def run(self, ids, thinking, tools, max_new, sampling, cancel) -> Iterator[tuple[str, object]]:
        """Yields ("event", Event) as text arrives, then ("done", {"finish": .., "completion_tokens": ..})."""
        defaults = {**self.sampling_defaults, **self.shared}   # the config's, then the Chat settings shared with apps
        if defaults:                   # the request's own fields win (explicit 0 stays greedy)
            req_values = {k: v for k, v in (sampling or {}).items() if v is not None}
            sampling = {**defaults, **req_values}
        parser = OutputParser(thinking=thinking, tools=tools, stream_tools=True)
        detok, n, finish = Detokenizer(self.tok), 0, "length"
        timings, before = None, None                    # this request's timings; the engine's `last` before it
        raw_ids = []                                    # every generated id (STRATA_DEBUG: dump raw model text)
        emb = getattr(self.embeddings, "path", None)
        # Identity token: only a DONE line replaces engine.last, so a request that died, errored or was
        # disconnected must not have the PREVIOUS request's decode figures recorded as its own.
        engine_last0 = getattr(self.engine, "last", None)
        with self.status_lock:
            self.status["queued"] += 1
        try:
            with self.fifo:
                with self.status_lock:
                    self.status["queued"] -= 1
                if hasattr(self.engine, "alive") and not self.engine.alive():
                    # issue #27: it died in an earlier request - start it again instead of failing every request
                    code = self.engine.exit_code() if hasattr(self.engine, "exit_code") else None
                    print(f"[strata] the engine had stopped (exit code {code}); starting it again "
                          "(a minute or two) ...", flush=True)
                    self.engine.restart()
                    print("[strata] the engine is running again", flush=True)
                with self.status_lock:
                    self.status.update(busy=True, phase="reading the prompt", prompt_tokens=len(ids), generated=0,
                                       started=time.time(), first_token=None, tool=None, tail="", max_tokens=max_new)
                    self.last_request_at = time.time()
                    self.rate.clear()               # the previous request's samples must not leak into this one
                before = getattr(self.engine, "last", None)
                last_print = time.time()
                gen = self.engine.generate(ids, max_new, sampling, cancel, embeddings=emb) if emb else \
                    self.engine.generate(ids, max_new, sampling, cancel)
                try:
                    for t in gen:
                        if t is None:                   # heartbeat while the engine is quiet
                            last_print = self._progress(last_print)
                            yield "ping", None
                            continue
                        n += 1
                        if t in self.stop_ids:
                            finish = "stop"
                            raw_ids.append(t)
                            break
                        raw_ids.append(t)
                        evs = parser.feed(detok.push(t))
                        self._note(n, evs)
                        last_print = self._progress(last_print)
                        for ev in evs:
                            yield "event", ev
                    if cancel.is_set():
                        finish = "cancel"
                except EngineDied as e:
                    finish = "error"
                    note = self.engine.death_note() if hasattr(self.engine, "death_note") else ""
                    print(f"[strata] {e}. {note} The next request starts the engine again."
                          f"{' Its log: ' + self.engine.log_path if getattr(self.engine, 'log_path', None) else ''}",
                          flush=True)
                    raise
                except ValueError as e:                 # the engine's ERR line (it may have ended after it)
                    finish = "error"
                    print(f"[strata] the engine reported an error: {e}", flush=True)
                    raise
                finally:
                    gen.close()                         # STOP+drain to THIS request's DONE while still holding the
                    #                                     fifo, so a stop-token break can't leave the shared engine
                    #                                     queue mid-drain for the next request to read as its own DONE
        except GeneratorExit:                           # the client disconnected mid-stream
            finish = "disconnect"
            raise
        finally:
            if emb:
                Path(emb).unlink(missing_ok=True)
            with self.status_lock:
                if self.status.get("busy"):
                    # only this request's DONE counts: same object means no DONE arrived (death, error, disconnect)
                    last = dict(getattr(self.engine, "last", {}) or {}) \
                        if getattr(self.engine, "last", None) is not engine_last0 else {}
                    started = self.status.get("started", time.time())
                    loaded = str((getattr(self.engine, "info", {}) or {}).get("cvec", 0)) not in ("0", "", "None")
                    hit_rate = round(last["hits"] / last["lookups"], 3) if last.get("lookups") else None
                    self.history.append({
                        "projection": (sampling or {}).get("experimental_speed_projection") is not False
                        if loaded else None,
                        "time": started, "duration_s": round(time.time() - started, 1), "finish": finish,
                        "prompt_tokens": len(ids), "reused": last.get("reused"), "output_tokens": n,
                        "engine_generated": last.get("generated"),
                        "prompt_ms": last.get("prompt_ms"), "decode_ms": last.get("decode_ms"),
                        "decode_tok_s": round(last["generated"] / (last["decode_ms"] / 1000), 1)
                        if n and last.get("generated") and last.get("decode_ms") else None,
                        "hit_rate": hit_rate})
                    t = self.totals
                    t["requests"] += 1
                    t["prompt_tokens"] += len(ids)
                    t["reused"] += last.get("reused") or 0
                    t["output_tokens"] += n
                    t["prompt_ms"] += last.get("prompt_ms") or 0.0
                    t["decode_ms"] += last.get("decode_ms") or 0.0
                    fresh = getattr(self.engine, "last", None)
                    if fresh is not None and fresh is not before:      # the engine's clock for THIS request
                        timings = request_timings(len(ids), n, last)
                        self.last_timings = dict(timings, at=int(time.time())) if timings else None
                    self.last_request_at = time.time()
                    now = time.time()
                    el = now - self.status.get("started", now)
                    ft = self.status.get("first_token")
                    rate = n / max(1e-6, now - ft) if ft else 0.0
                    hit_msg = f", expert cache {hit_rate*100:.1f}% hit" if hit_rate is not None else ""
                    print(f"[strata] done: {n} tokens in {el:.0f} s ({rate:.1f} tok/s) "
                          f"({finish}, cancel={cancel.is_set()}){hit_msg}", flush=True)
                    if os.environ.get("STRATA_DEBUG") and raw_ids:
                        print(f"[strata] raw: {self.tok.decode(raw_ids)!r}", flush=True)
                self.status["busy"] = False
        for ev in parser.finish():
            yield "event", ev
        yield "done", {"finish": finish, "completion_tokens": n, "reused": (timings or {}).get("cache_n", 0),
                       "timings": timings}


def request_timings(prompt_tokens: int, generated: int, last: dict) -> dict | None:
    """One request's `timings` in llama.cpp's names (what its clients show as speed), from the engine's own clock
    (StrataEngine.last): prompt_n is what was read, cache_n what the conversation cache already held.  None when the
    engine keeps no clock (MockEngine)."""
    if last.get("prompt_ms") is None:
        return None
    cache_n = int(last.get("reused") or 0)
    prompt_n, prompt_ms, decode_ms = max(0, prompt_tokens - cache_n), float(last["prompt_ms"]), float(last.get("decode_ms") or 0)
    decoded = int(last.get("generated") or generated)          # the engine's count gives its rate, as /metrics does
    return {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": round(prompt_ms, 1),
            "prompt_per_token_ms": round(prompt_ms / prompt_n, 3) if prompt_n else None,
            "prompt_per_second": round(prompt_n / (prompt_ms / 1000), 1) if prompt_n and prompt_ms > 0 else None,
            "predicted_n": generated, "predicted_ms": round(decode_ms, 1),
            "predicted_per_token_ms": round(decode_ms / decoded, 3) if decoded else None,
            "predicted_per_second": round(decoded / (decode_ms / 1000), 1) if decoded and decode_ms > 0 else None,
            # the speculative drafts, as llama.cpp names them (from PR #83, @mikicvi): only when the engine reported them
            **({"draft_n": int(last["drafts_offered"]), "draft_n_accepted": int(last["drafts_accepted"])}
               if last.get("drafts_offered") is not None else {})}


def _debug_req(api, req, messages, tools, max_new, thinking, prompt_tokens):
    """One compact line per request while diagnosing blank/empty turns. Set STRATA_DEBUG=1 to enable."""
    if not os.environ.get("STRATA_DEBUG"):
        return
    last = messages[-1] if messages else {}
    body = last.get("content")
    if isinstance(body, list):
        body = " ".join(p.get("text", "") for p in body if isinstance(p, dict))
    preview = (str(body or "")[:80]).replace("\n", " ")
    print(f"[strata] req {api}: msgs={len(messages)} tools={len(tools or [])} "
          f"max_tokens_raw={req.get('max_tokens')!r}/{req.get('max_completion_tokens')!r} "
          f"max_new={max_new} thinking={thinking} stream={bool(req.get('stream'))} "
          f"prompt_tokens={prompt_tokens} last={last.get('role')!r}:{preview!r}", flush=True)


# ------------------------------------------------------------------------------------------------ MCP tool loop
def run_with_mcp(svc: Service, hub, messages, tools, kw, ids, thinking, max_new, max_req, sampling, cancel,
                 mcp_names):
    """Service.run with the MCP tools executed here: the model writes a call to an MCP tool, the server runs it, adds
    the call and its result to the conversation and lets the model continue - up to `max_rounds` times.  Yields what
    Service.run yields (text, thinking, the request's own tool calls) plus ("mcp", {...}) for the tool activity, and
    one ("done", ...) at the very end with the output tokens of every round.

    `mcp_names`: the MCP tools this request offered; any other call is one of the request's own tools and ends the
    turn as always (the client answers it).  MCP calls written in the same answer are then not run (their results
    could not reach the model before the client's)."""
    max_rounds = int(hub.settings["max_rounds"])
    total, rounds, done = 0, 0, None
    messages = list(messages)
    while True:
        text, reasoning, calls, own_calls = [], [], [], 0
        for kind, x in svc.run(ids, thinking, tools, max_new, sampling, cancel):
            if kind == "done":
                done = x
                continue
            if kind == "event":
                ev: Event = x
                if ev.call is not None and ev.call.name in mcp_names:
                    if ev.kind == "tool_start":          # the model has started writing a call: say so at once
                        yield "mcp", {"event": "start", "id": ev.call.id, "name": ev.call.name}
                    elif ev.kind == "tool_call":
                        calls.append(ev.call)
                    continue                             # its argument pieces are not streamed to the client
                if ev.kind == "tool_call":
                    own_calls += 1
                elif ev.kind == "content":
                    text.append(ev.text)
                elif ev.kind == "reasoning":
                    reasoning.append(ev.text)
            yield kind, x
        total += done["completion_tokens"]
        run_them = calls and not own_calls and done["finish"] == "stop" and not cancel.is_set()
        if run_them and rounds >= max_rounds:
            yield "mcp", {"event": "limit", "max_rounds": max_rounds}
            run_them = False
        if not run_them:
            for c in calls:                              # announced, never run: close them in the client's view
                yield "mcp", {"event": "result", "id": c.id, "ok": False, "skipped": True, "text": "not run",
                              "chars": 0, "truncated": False, "ms": 0}
            break
        rounds += 1
        results = []
        for c in calls:
            s, tool = hub.routes().get(c.name, (None, c.name))
            yield "mcp", {"event": "call", "id": c.id, "name": c.name, "server": s.name if s else None,
                          "tool": tool, "arguments": c.arguments, "round": rounds}
            # The call runs on a thread while this generator keeps yielding heartbeats: they reach the client as
            # keep-alives, which is how a client that went away (the web app's Stop) is noticed during a slow tool.
            box = {}

            def work(c=c, box=box):
                try:
                    box["r"] = hub.call(c.name, c.arguments, cancel)
                except McpCancelled:
                    box["cancelled"] = True
            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            try:
                while worker.is_alive():
                    worker.join(1.0)
                    if worker.is_alive():
                        yield "ping", None
            except GeneratorExit:
                cancel.set()                             # the client is gone: stop the tool too
                raise
            if "r" not in box:
                break
            r = box["r"]
            print(f"[strata] tool {c.name}: {'ok' if r['ok'] else 'error'}, {r['chars']:,} characters in "
                  f"{r['ms'] / 1000:.1f} s{' (truncated for the model)' if r['truncated'] else ''}", flush=True)
            results.append(r["text"])
            yield "mcp", {"event": "result", "id": c.id, **{k: r[k] for k in ("ok", "text", "chars", "truncated", "ms")}}
        if cancel.is_set() or len(results) < len(calls):
            done = {**done, "finish": "cancel"}
            break
        messages.append({"role": "assistant", "content": "".join(text).strip(),
                         **({"reasoning_content": "".join(reasoning).strip()} if reasoning else {}),
                         "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in calls]})
        messages += [{"role": "tool", "content": r} for r in results]
        ids, thinking, max_new = svc.prepare(messages, tools, kw, max_req)
    yield "done", {**done, "completion_tokens": total, "prompt_tokens": len(ids)}


# ------------------------------------------------------------------------------------------------ OpenAI
def openai_chunks(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, run=None):
    """`run`: the events to send instead of Service.run's (run_with_mcp); its ("mcp", {...}) items become chunks with
    an empty delta and a `strata_mcp` field, which only the web app reads."""
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

    def chunk(delta, finish=None):
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": svc.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield chunk({"role": "assistant", "content": ""})
    calls = 0
    streamed = {}                                  # tool call id -> index, for calls sent piece by piece
    for kind, x in run if run is not None else svc.run(ids, thinking, tools, max_new, req, cancel):
        if kind == "ping":
            yield None
        elif kind == "mcp":
            c = chunk({})
            c["strata_mcp"] = x
            yield c
        elif kind == "event":
            ev: Event = x
            if ev.kind == "reasoning" and ev.text:
                yield chunk({"reasoning_content": ev.text})
            elif ev.kind == "content" and ev.text:
                yield chunk({"content": ev.text})
            elif ev.kind == "tool_start":
                streamed[ev.call.id] = calls
                calls += 1
                yield chunk({"tool_calls": [{"index": streamed[ev.call.id], "id": ev.call.id, "type": "function",
                                             "function": {"name": ev.call.name, "arguments": ""}}]})
            elif ev.kind == "tool_args":
                yield chunk({"tool_calls": [{"index": streamed[ev.call.id], "function": {"arguments": ev.text}}]})
            elif ev.kind == "tool_call" and ev.call.id in streamed:
                continue
            elif ev.kind == "tool_call":
                yield chunk({"tool_calls": [{"index": calls, "id": ev.call.id, "type": "function",
                                             "function": {"name": ev.call.name,
                                                          "arguments": json.dumps(ev.call.arguments, ensure_ascii=False)}}]})
                calls += 1
        else:
            finish = "tool_calls" if calls and x["finish"] == "stop" else {"cancel": "stop"}.get(x["finish"], x["finish"])
            last = chunk({}, finish)
            pt = x.get("prompt_tokens", len(ids))     # after MCP rounds: the last round's prompt
            last["usage"] = {"prompt_tokens": pt, "completion_tokens": x["completion_tokens"],
                             "total_tokens": pt + x["completion_tokens"],
                             # the part of the prompt the conversation cache already held (OpenAI's field)
                             "prompt_tokens_details": {"cached_tokens": x.get("reused") or 0}}
            if x.get("timings"):
                last["timings"] = x["timings"]          # llama.cpp's field: the speed its clients show
            yield last


def openai_collect(chunks) -> dict:
    content, reasoning, by_index, last, mcp = [], [], {}, None, []
    for c in chunks:
        if c is None:                              # a heartbeat
            continue
        if c.get("strata_mcp"):
            mcp.append(c["strata_mcp"])
        d = c["choices"][0]["delta"]
        content.append(d.get("content") or "")
        reasoning.append(d.get("reasoning_content") or "")
        for tc in d.get("tool_calls") or []:       # streamed calls arrive in pieces: merge them by index
            cur = by_index.setdefault(tc.get("index", len(by_index)), {"id": None, "type": "function",
                                                                        "function": {"name": "", "arguments": ""}})
            cur["id"] = tc.get("id") or cur["id"]
            fn = tc.get("function") or {}
            cur["function"]["name"] += fn.get("name") or ""
            cur["function"]["arguments"] += fn.get("arguments") or ""
        last = c
    calls = [by_index[i] for i in sorted(by_index)]
    msg = {"role": "assistant", "content": "".join(content) or None}
    if "".join(reasoning):
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = calls
    if mcp:
        msg["strata_mcp"] = mcp
    out = {"id": last["id"], "object": "chat.completion", "created": last["created"], "model": last["model"],
           "choices": [{"index": 0, "message": msg, "finish_reason": last["choices"][0]["finish_reason"]}],
           "usage": last["usage"]}
    if last.get("timings"):
        out["timings"] = last["timings"]
    return out


# ------------------------------------------------------------------------------------------------ Anthropic
def anthropic_events(svc: Service, req: dict, ids, thinking, tools, max_new, cancel):
    mid = "msg_" + uuid.uuid4().hex[:24]
    yield "message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": svc.model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": len(ids), "output_tokens": 0}}}
    index, open_kind, used_tool = -1, None, False

    def close():
        return ("content_block_stop", {"type": "content_block_stop", "index": index})

    streamed = set()
    for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel):
        if kind == "ping":
            yield None
            continue
        if kind == "event":
            ev: Event = x
            if ev.kind == "tool_args":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "input_json_delta", "partial_json": ev.text}}
                continue
            if ev.kind == "tool_call" and ev.call.id in streamed:
                continue
            want = {"reasoning": "thinking", "content": "text", "tool_call": "tool_use", "tool_start": "tool_use"}[ev.kind]
            if ev.kind not in ("tool_call", "tool_start") and not ev.text:
                continue
            if ev.kind == "tool_start":
                streamed.add(ev.call.id)
                used_tool = True
            if open_kind != want or want == "tool_use":
                if open_kind is not None:
                    yield close()
                index += 1
                open_kind = want
                block = {"thinking": {"type": "thinking", "thinking": "", "signature": ""},
                         "text": {"type": "text", "text": ""},
                         "tool_use": {"type": "tool_use", "id": ev.call.id if ev.call else "", "name":
                                      ev.call.name if ev.call else "", "input": {}}}[want]
                yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": block}
            if want == "thinking":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "thinking_delta", "thinking": ev.text}}
            elif want == "text":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "text_delta", "text": ev.text}}
            elif ev.kind == "tool_start":
                pass                                # its input follows as tool_args pieces
            else:
                used_tool = True
                yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta": {
                    "type": "input_json_delta", "partial_json": json.dumps(ev.call.arguments, ensure_ascii=False)}}
        else:
            if open_kind is not None:
                yield close()
            stop = "tool_use" if used_tool and x["finish"] == "stop" else \
                {"stop": "end_turn", "length": "max_tokens", "cancel": "end_turn"}[x["finish"]]
            # the final counts, Anthropic's way: input_tokens leaves out what the conversation cache already held,
            # which is cache_read_input_tokens (message_start could only say the whole prompt)
            reused = min(x.get("reused") or 0, len(ids))
            yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                    "usage": {"input_tokens": len(ids) - reused, "cache_read_input_tokens": reused,
                                              "output_tokens": x["completion_tokens"]}}
            yield "message_stop", {"type": "message_stop"}


def anthropic_collect(events) -> dict:
    msg, blocks = None, []
    for item in events:
        if item is None:                           # a heartbeat
            continue
        name, e = item
        if name == "message_start":
            msg = e["message"]
        elif name == "content_block_start":
            blocks.append(dict(e["content_block"]))
        elif name == "content_block_delta":
            d, b = e["delta"], blocks[-1]
            if d["type"] == "text_delta":
                b["text"] += d["text"]
            elif d["type"] == "thinking_delta":
                b["thinking"] += d["thinking"]
            else:                                  # input_json_delta pieces: parsed when complete
                b["_json"] = b.get("_json", "") + d["partial_json"]
        elif name == "content_block_stop" and blocks and "_json" in blocks[-1]:
            b = blocks[-1]
            b["input"] = json.loads(b.pop("_json") or "{}")
        elif name == "message_delta":
            msg["stop_reason"] = e["delta"]["stop_reason"]
            msg["usage"].update(e["usage"])
    msg["content"] = blocks
    return msg


# ------------------------------------------------------------------------------------------------ HTTP
def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                       # SSE ends by closing the connection

        def log_message(self, fmt, *args):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not svc.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if given == svc.api_key:
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
            return False

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path.startswith("/fonts/"):
                # the web app's font (Outfit, OFL: serve/web/fonts); the page falls back to the system font
                name = path[len("/fonts/"):]
                f = ROOT / "serve" / "web" / "fonts" / name
                if "/" in name or "\\" in name or not name.endswith(".woff2") or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "font/woff2")
                self.send_header("Cache-Control", "max-age=86400")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith("/web/"):
                # the web app's own files (serve/web): styles, script, icon sprite - same origin, no CDN
                name = path[len("/web/"):]
                types = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                         ".svg": "image/svg+xml"}
                f = ROOT / "serve" / "web" / name
                ext = os.path.splitext(name)[1]
                if "/" in name or "\\" in name or ext not in types or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", types[ext])
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/metrics":
                if self._authorized():
                    # the last 12 requests; `?requests=all` every one kept (the Monitor's "Show all", issue #35)
                    self._json(200, svc.metrics(all_requests="requests=all" in self.path))
                return
            if path == "/settings":
                if self._authorized():
                    self._json(200, {"shared": bool(svc.shared), "defaults": svc.shared})
                return
            if path == "/mcp":
                # the MCP servers, their state and tools (the web app's switch and Monitor card)
                if self._authorized():
                    self._json(200, svc.mcp.status() if svc.mcp else {"servers": [], "tools": 0})
                return
            if path == "":
                body = (ROOT / "serve" / "web" / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/health":
                self._json(200, {"status": "ok", "max_context": svc.engine.max_context, "model": svc.model,
                                 "images": svc.vision is not None, "api_key": bool(svc.api_key)})
            elif path == "/status":
                with svc.status_lock:
                    s = dict(svc.status)
                now = time.time()
                if s.get("busy"):
                    s["elapsed_s"] = round(now - s["started"], 1)
                    if s.get("first_token"):
                        s["tokens_per_s"] = round(svc._tok_s(), 1)
                        s["tokens_per_s_mean"] = round(svc._tok_s_mean(), 1)
                for k in ("started", "first_token"):
                    s.pop(k, None)
                self._json(200, s)
            elif path in ("/v1/models", "/models"):
                if self._authorized():
                    loaded = not hasattr(svc.engine, "alive") or svc.engine.alive()
                    model = {"id": svc.model, "object": "model", "status": {"value": "loaded"},
                             "meta": {"n_ctx": svc.engine.max_context},
                             "architecture": {"input_modalities": ["text", "image"] if svc.vision is not None else ["text"],
                                              "output_modalities": ["text"]}}
                    self._json(200, {"object": "list", "data": [model] if loaded else []})
            elif path == "/props":
                if self._authorized():
                    self._props()
            elif path == "/slots":
                if self._authorized():
                    loaded = not hasattr(svc.engine, "alive") or svc.engine.alive()
                    with svc.status_lock:
                        busy = bool(svc.status.get("busy"))
                    slot = {"id": 0, "n_ctx": svc.engine.max_context, "is_processing": busy}
                    self._json(200, [slot] if loaded else [])
            elif path == "/v1/status":
                if self._authorized():
                    self._json(200, svc.v1_status())
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self._authorized():
                return
            path = self.path.split("?")[0].rstrip("/")   # issue #55: Claude Code posts /v1/messages?beta=true
            if path == "/settings":
                self._settings()
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if path == "/v1/chat/completions":
                    self._openai(req)
                elif path == "/v1/messages":
                    self._anthropic(req)
                else:
                    self._json(404, {"error": {"message": "not found"}})
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
            except EngineDied as e:                          # before the answer started (not streamed)
                self._json(503, {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}})

        def _props(self):
            model = parse_qs(urlsplit(self.path).query).get("model", [svc.model])[0]
            if model != svc.model:
                self._json(404, {"error": {"message": "model not found"}})
                return
            if hasattr(svc.engine, "alive") and not svc.engine.alive():
                self._json(503, {"error": {"message": "the engine is not running"}})
                return
            defaults = {**svc.sampling_defaults, **svc.shared}
            names = {"repetition_penalty": "repeat_penalty", "penalty_last_n": "repeat_last_n"}
            params = {names.get(k, k): v for k, v in defaults.items()
                      if k in ("temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty",
                               "presence_penalty", "frequency_penalty", "penalty_last_n")}
            params["n_predict"] = svc.shared.get("max_tokens", -1)
            props = {"default_generation_settings": {"n_ctx": svc.engine.max_context, "params": params},
                     "total_slots": 1, "model_alias": svc.model, "chat_template": svc.template.source,
                     "modalities": {"vision": svc.vision is not None}, "models_autoload": False,
                     "is_sleeping": False}
            if getattr(svc.engine, "model_path", None):
                props["model_path"] = svc.engine.model_path
            version = getattr(svc.engine, "info", {}).get("version")
            if version:
                props["build_info"] = "Strata " + str(version)
            self._json(200, props)

        def _own_page(self, what) -> bool:
            """Only JSON (a form or a "simple" cross-site request can't send it without a CORS preflight, which this
            server never grants) and no foreign Origin: a web page elsewhere must not change settings or run tools."""
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return False
            origin = self.headers.get("Origin")
            if origin and origin.split("://", 1)[-1] != self.headers.get("Host", ""):
                self._json(403, {"error": {"message": f"{what} only from Strata's own page"}})
                return False
            return True

        def _settings(self):
            # They change what every client gets, so only the app's own page may set them
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if not self._own_page("settings can be changed"):
                return
            try:
                req = json.loads(body or b"{}")
                shared = svc.set_shared(req.get("defaults") if isinstance(req, dict) else None)
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            print("[strata] other apps now use the Chat settings: " + ", ".join(f"{k}={v}" for k, v in shared.items())
                  if shared else "[strata] other apps use their own settings again", flush=True)
            self._json(200, {"shared": bool(shared), "defaults": shared})

        def _sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

        def _openai(self, req):
            req = svc.with_shared(req, "openai")
            messages, tools, kw = openai_to_messages(req)
            max_req = max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 0)   # 0/-1: the rest
            use_mcp = req.get("strata_mcp") is True and svc.mcp is not None      # the web app's opt-in (serve/mcp.py)
            own = {t.get("name") for t in tools or []}
            if use_mcp:
                if not self._own_page("MCP tools can be used"):   # tools run with the user's rights on this PC
                    return
                svc.mcp.wait(10)                                  # servers still starting (only right after start)
                extra = svc.mcp.template_tools(exclude=own)       # the request's own tools win a name clash
                use_mcp = bool(extra)
                tools = (tools or []) + extra or None
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new)
            _debug_req("openai", req, messages, tools, max_new, thinking, len(ids))
            cancel = threading.Event()
            run = run_with_mcp(svc, svc.mcp, messages, tools, kw, ids, thinking, max_new, max_req, req, cancel,
                               {t["name"] for t in extra}) if use_mcp else None
            chunks = openai_chunks(svc, req, ids, thinking, tools, max_new, cancel, run=run)
            if not req.get("stream"):
                return self._json(200, openai_collect(chunks))
            self._sse()
            try:
                for c in chunks:
                    if c is None:
                        self.wfile.write(b": keep-alive\n\n")      # an SSE comment: clients ignore it
                    else:
                        self.wfile.write(b"data: " + json.dumps(c, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            except OSError:
                cancel.set()                                 # client went away: stop the engine
                chunks.close()
            except EngineDied as e:                          # mid-stream: say so, then end the stream properly
                err = {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}}
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")
            except ValueError as e:                          # the engine's ERR after the stream started: the
                err = {"error": {"type": "server_error", "message": str(e)}}   # headers are sent, so no 400 now
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")

        def _anthropic(self, req):
            req = svc.with_shared(req, "anthropic")
            messages, tools, kw = anthropic_to_messages(req)
            max_new = int(req.get("max_tokens") or 0)                  # 0/-1: the rest of the context
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new)
            _debug_req("anthropic", req, messages, tools, max_new, thinking, len(ids))
            cancel = threading.Event()
            events = anthropic_events(svc, req, ids, thinking, tools, max_new, cancel)
            if not req.get("stream"):
                return self._json(200, anthropic_collect(events))
            self._sse()
            try:
                for item in events:
                    if item is None:
                        self.wfile.write(b": keep-alive\n\n")
                    else:
                        name, e = item
                        self.wfile.write(f"event: {name}\n".encode() + b"data: " +
                                         json.dumps(e, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except OSError:
                cancel.set()
                events.close()
            except EngineDied as e:                          # mid-stream: Anthropic's error event
                err = {"type": "error", "error": {"type": "api_error", "message": f"{e}; the next request restarts it"}}
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except ValueError as e:                          # the engine's ERR after the stream started
                err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")

    return Handler


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already serving, and requests then land on
    # either one (a forgotten second start of run-<model>.bat).  Without it the second start fails loudly instead.
    allow_reuse_address = os.name != "nt"


def warn_tight_ram(arena_mib) -> None:
    """The model's experts live in RAM (INFO arena_mib, engine 0.1.10+).  With less than ~6 GB left beside them for the
    system, the engine and this server, Linux ends the engine mid-answer when memory runs out (issue #27) and Windows
    pages to disk; say so at start instead of after a lost answer."""
    if not isinstance(arena_mib, int) or arena_mib <= 0:
        return
    try:
        import psutil
        total = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001 - psutil is optional here
        return
    left = total / 2**30 - arena_mib / 1024
    if left < 6:
        print(f"[strata] WARNING: RAM is tight - the model's experts take {arena_mib / 1024:.1f} GB of this PC's "
              f"{total / 2**30:.0f} GB, leaving {left:.1f} GB for everything else. "
              + ("Linux may stop the engine in the middle of an answer. " if os.name != "nt" else
                 "Windows will slow down (paging to disk). ")
              + "Close other programs, or run START-HERE --setup and pick a smaller size (Q2_0 / IQ2_XS).", flush=True)


def lan_addresses() -> list[str]:
    """This PC's IPv4 addresses on its networks (what another device types in), without loopback/link-local."""
    import socket
    first, ips = None, set()
    try:                                                # the address of the default route; sends nothing (UDP)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            first = s.getsockname()[0]
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    ok = lambda ip: ip and not ip.startswith(("127.", "169.254.", "0."))
    return ([first] if ok(first) else []) + sorted(ip for ip in ips if ok(ip) and ip != first)


def serve(svc: Service, host="127.0.0.1", port=8095) -> ThreadingHTTPServer:
    svc.start_telemetry()
    httpd = Server((host, port), make_handler(svc))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


SHARED_KEYS = ("reasoning_effort", "temperature", "top_p", "top_k", "seed", "max_tokens", "experimental_speed_projection")


def clean_shared_defaults(d) -> dict:
    """The Chat settings other apps get (POST /settings): only known keys, each checked; ValueError names a bad one."""
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError("defaults must be an object")
    out = {}
    for key, value in d.items():
        if value is None or value == "":
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "reasoning_effort":
            if value not in ("none", "low", "medium", "high"):
                raise ValueError("reasoning_effort: none, low, medium or high")
        elif key == "temperature":
            if not number or not 0 <= value <= 2:
                raise ValueError("temperature: 0..2")
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise ValueError("top_p: 0 < top_p <= 1")
        elif key == "top_k":
            if not number or value != int(value) or not 1 <= value <= 64:
                raise ValueError("top_k: an integer 1..64")
            value = int(value)
        elif key in ("seed", "max_tokens"):
            if not number or value != int(value) or value <= 0:
                raise ValueError(f"{key}: a positive integer")
            value = int(value)
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise ValueError("experimental_speed_projection: true or false")
        else:
            raise ValueError(f"unknown setting {key!r}")
        out[key] = float(value) if key in ("temperature", "top_p") else value
    return out


def sampling_defaults_from_config(cfg: dict) -> dict:
    """The run config's optional `sampling` block: defaults for the sampling fields a request leaves out, so
    a plain client gets configured sampling instead of greedy.  Supported: temperature, top_p, top_k, min_p,
    presence_penalty, repetition_penalty, frequency_penalty, penalty_last_n, seed.  The request's own fields
    always win - an explicit temperature=0 still means greedy, a field set to null falls back to the default.
    A bad value refuses to start the server (a typo'd config should not quietly change sampling); unknown keys
    are named at startup and ignored."""
    out = {}
    for key, value in (cfg.get("sampling") or {}).items():
        if value is None:
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "temperature":
            if not number or value < 0:
                raise SystemExit(f"[strata] config sampling.temperature={value!r}: expected a number >= 0 (0 = greedy)")
            out[key] = float(value)
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise SystemExit(f"[strata] config sampling.top_p={value!r}: expected 0 < top_p <= 1")
            out[key] = float(value)
        elif key == "min_p":
            if not number or not 0 <= value <= 1:
                raise SystemExit(f"[strata] config sampling.min_p={value!r}: expected 0 <= min_p <= 1")
            out[key] = float(value)
        elif key == "top_k":
            if not number or value != int(value) or not 1 <= value <= 64:
                raise SystemExit(f"[strata] config sampling.top_k={value!r}: the sampled path takes an integer 1..64")
            out[key] = int(value)
        elif key == "presence_penalty":
            if not number or value < 0:
                raise SystemExit(f"[strata] config sampling.presence_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "frequency_penalty":
            if not number or value < 0:
                raise SystemExit(f"[strata] config sampling.frequency_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "repetition_penalty":
            if not number or value <= 0:
                raise SystemExit(f"[strata] config sampling.repetition_penalty={value!r}: expected a number > 0 (1 = off)")
            out[key] = float(value)
        elif key == "penalty_last_n":
            if not number or value != int(value) or value < 0:
                raise SystemExit(f"[strata] config sampling.penalty_last_n={value!r}: expected a non-negative integer")
            out[key] = int(value)
        elif key == "seed":
            if not number or value != int(value) or value <= 0:
                raise SystemExit(f"[strata] config sampling.seed={value!r}: expected a positive integer")
            out[key] = int(value)
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise SystemExit(f"[strata] config sampling.experimental_speed_projection={value!r}: expected true or "
                                 "false (the default for requests that leave it out, when the engine has the vector)")
            out[key] = value
        else:
            print(f"[strata] config sampling.{key}={value!r}: unknown key, ignored", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["mock", "strata"], default="mock")
    ap.add_argument("--config", help="strata engine config (JSON: exe, args, cwd, tokenizer, model_name), "
                                     "written by setup.py")
    ap.add_argument("--host", default=None,
                    help="the address to listen on: 127.0.0.1 = this PC only (the default), 0.0.0.0 = also other devices "
                         "on your network (set an API key); also \"host\" in the config")
    ap.add_argument("--script", action="append",
                    help="the mock engine's answer (default: a short greeting); given more than once, requests get "
                         "them in turn and the last one repeats")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--gpu", help="the GPU to run on, as nvidia-smi numbers them, or several for a layer split "
                                  "(\"0,2\"; also \"gpu\" in the config)")
    ap.add_argument("--tokenizer", default=str(ROOT / "pack/full/tokenizer"),
                    help="pack tokenizer directory (falls back to a byte tokenizer if absent)")
    ap.add_argument("--open", action="store_true", help="open the local page in the browser once the model is ready")
    ap.add_argument("--fit-max-tokens", action="store_true",
                    help="clamp max_tokens to the remaining context instead of rejecting the request "
                         "(default: reject with 400, like llama.cpp; also \"fit_max_tokens\": true in the config)")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY", ""),
                    help="require this key on /v1/* (Authorization: Bearer ... or x-api-key); also $STRATA_API_KEY")
    ap.add_argument("--mcp-config", help="a JSON file with MCP servers in Claude Desktop's format ({\"mcpServers\": "
                                         "{...}}); the web app's chat can use their tools (also \"mcp_servers\" in "
                                         "the config)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig")) if a.config else {}   # Notepad adds a BOM
    if a.gpu is not None:
        cfg["gpu"] = int(a.gpu) if a.gpu.strip().isdigit() else a.gpu
    a.host = a.host or cfg.get("host") or "127.0.0.1"   # issue #26: the run scripts pass no --host, the config can
    try:                                                # before the minutes of loading: is the port free?
        Server((a.host, a.port), BaseHTTPRequestHandler).server_close()
    except OSError:
        ap.error(f"port {a.port} is already in use - is Strata (or another server) already running? "
                 f"Close it, or start this one with a different --port")
    if cfg.get("tokenizer"):
        a.tokenizer = cfg["tokenizer"]
    tok = ByteTokenizer()
    tpath = Path(a.tokenizer)
    if a.engine == "strata" and not (tpath / "vocab.json").exists():
        ap.error(f"the model's tokenizer is missing ({tpath / 'vocab.json'}); run setup again")
    if (tpath / "vocab.json").exists():
        import strata_tokenizer as ST
        vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((tpath / "token_type.json").read_text())
        tok = ST.Tokenizer(tokens, merges, types)
    hub = hub_from_config(cfg, a.mcp_config)            # before the minutes of loading: a bad entry stops here
    if a.engine == "strata":
        if not cfg:
            ap.error("--engine strata needs --config")
        vision = None
        env = child_env(cfg)
        sampling_defaults = sampling_defaults_from_config(cfg)
        if sampling_defaults:
            pretty = ", ".join(f"{k}={v}" for k, v in sampling_defaults.items())
            print(f"[strata] sampling defaults from the config: {pretty}", flush=True)
        if cfg.get("vision"):
            print("loading the vision encoder ...", flush=True)
            vision = Vision(cfg["vision"], log=open(cfg["log"], "a", encoding="utf-8") if cfg.get("log") else None,
                            env=env)
        print("loading the model (the first start takes a minute or two) ...", flush=True)
        if len(gpu_list(cfg)) > 1:
            print(f"[strata] layer split across GPUs {gpu_list(cfg)} ({cfg.get('layer_split') or 'auto'})", flush=True)
        engine = StrataEngine(cfg["exe"], engine_args(cfg), cwd=cfg.get("cwd"), log=cfg.get("log"), env=env)
        warn_tight_ram(engine.info.get("arena_mib"))
    else:
        engine, vision, sampling_defaults = MockEngine(tok, a.script or [
            "Thinking about it.</think>\n\nHello from the mock engine."]), None, {}
    # the model's own chat template (exported with its tokenizer), else the original model's
    tpl = tpath / "chat_template.jinja"
    svc = Service(engine, tok, ChatTemplate(tpl if tpl.exists() else ROOT / "serve/chat_template.jinja"),
                  model_name=cfg.get("model_name", "qwen3.8-flash-next"), vision=vision,
                  sampling_defaults=sampling_defaults,
                  fit_max_tokens=a.fit_max_tokens or cfg.get("fit_max_tokens") is True)
    svc.api_key = a.api_key or cfg.get("api_key", "")
    svc.gpu_index = (gpu_list(cfg) or [0])[0]           # the Monitor reads the card the engine runs on (issue #51)
    svc.gpu_indices = gpu_list(cfg)                     # ... or every card of a layer split (issue #112)
    if a.config:                                        # the Chat settings shared with other apps, from last time
        svc.shared_path = str(Path(a.config).with_suffix("")) + ".shared-settings.json"
        try:
            svc.shared = clean_shared_defaults(json.loads(Path(svc.shared_path).read_text(encoding="utf-8")))
            if svc.shared:
                print("[strata] other apps use the Chat settings: " +
                      ", ".join(f"{k}={v}" for k, v in svc.shared.items()), flush=True)
        except (OSError, ValueError):
            svc.shared = {}
    if hub is not None:
        import atexit
        svc.mcp = hub
        print(f"[strata] starting {len(hub.servers)} MCP server{'s' * (len(hub.servers) != 1)} for the web app's "
              f"chat: {', '.join(hub.servers)}", flush=True)
        hub.start()
        atexit.register(hub.close)                      # the servers Strata started end with it
    httpd = serve(svc, host=a.host, port=a.port)
    here = "127.0.0.1" if a.host in ("0.0.0.0", "", "::") else a.host
    print(f"ready: http://{here}:{a.port}/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, "
          f"context {engine.max_context} tokens{', images on' if vision else ''}"
          f"{', API key required' if svc.api_key else ''})", flush=True)
    print(f"       open http://{here}:{a.port}/ in a browser to chat; close this window to stop the model", flush=True)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        # issue #26: reachable from other devices - say at which address, and what can still block it
        ips = lan_addresses()
        for ip in ips:
            print(f"       from other devices: http://{ip}:{a.port}/   (API: http://{ip}:{a.port}/v1)", flush=True)
        if not ips:
            print("       from other devices: http://<this PC's IP address>:" + str(a.port) + "/", flush=True)
        if not svc.api_key:
            print("       WARNING: no API key - anyone on your network can use this model. Add \"api_key\": \"...\" "
                  "to the config (clients send it as their API key; the web page asks for it)", flush=True)
        if os.name == "nt":
            print("       nothing arrives? Windows Firewall blocks it until allowed: accept its prompt for Python, or run "
                  "in an admin PowerShell:\n         New-NetFirewallRule -DisplayName \"Strata " + str(a.port) + "\" "
                  "-Direction Inbound -Protocol TCP -LocalPort " + str(a.port) + " -Action Allow -Profile Private\n"
                  "       (and set this network to Private in Windows' network settings)", flush=True)
    if a.open:
        import webbrowser
        webbrowser.open(f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{a.port}/")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        httpd.shutdown()
        if hasattr(engine, "close"):
            engine.close()
        if vision:
            vision.close()
        if hub is not None:
            hub.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

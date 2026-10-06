#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
STATE="${STRATA_STATE_DIR:-$ROOT/state}"
mkdir -p "$STATE"; umask 077
: "${STRATA_PACK_HOT:?set STRATA_PACK_HOT}"; : "${STRATA_DENSE_GGUF:?set STRATA_DENSE_GGUF}"
: "${STRATA_PROFILE:?set STRATA_PROFILE}"; : "${STRATA_REMOTE_HOST:?set STRATA_REMOTE_HOST}"
BUILD="${STRATA_BUILD_DIR:-$ROOT/build-3080}"; EXE="${STRATA_EXE:-$BUILD/strata}"
test -x "$EXE" || { echo "missing executable: $EXE (run build-3080.sh)" >&2; exit 2; }
test -d "$STRATA_PACK_HOT" || { echo "missing pack-hot: $STRATA_PACK_HOT" >&2; exit 2; }
test -f "$STRATA_DENSE_GGUF" || { echo "missing dense.gguf: $STRATA_DENSE_GGUF" >&2; exit 2; }
test -f "$STRATA_PROFILE" || { echo "missing profile.bin: $STRATA_PROFILE" >&2; exit 2; }
PORT="${STRATA_HTTP_PORT:-8100}"; HOST="${STRATA_HTTP_HOST:-0.0.0.0}"
LOG="${STRATA_3080_LOG:-$STATE/3080-http.log}"; CFG="$STATE/3080-serve.json"
if [[ -f "$STATE/3080.pid" ]] && kill -0 "$(cat "$STATE/3080.pid")" 2>/dev/null; then
  echo "already running pid=$(cat "$STATE/3080.pid")"; exit 0
fi
export STRATA_CFG="$CFG" STRATA_ENGINE_EXE="$EXE" STRATA_PACK_HOT STRATA_DENSE_GGUF STRATA_PROFILE
export STRATA_PLE_GGUF="${STRATA_PLE_GGUF:-}" STRATA_PLE_FP8="${STRATA_PLE_FP8:-}" STRATA_PLE_FP8_SCALE="${STRATA_PLE_FP8_SCALE:-0.00019931793212890625}"
export STRATA_REMOTE_HOST STRATA_REMOTE_PORT="${STRATA_REMOTE_PORT:-39580}" STRATA_REMOTE_HCA="${STRATA_REMOTE_HCA:-}"
export STRATA_REMOTE_GID="${STRATA_REMOTE_GID:-0}" STRATA_REMOTE_SLOTS="${STRATA_REMOTE_SLOTS:-4}" STRATA_REMOTE_SLOT_BYTES="${STRATA_REMOTE_SLOT_BYTES:-4194304}"
export STRATA_MAX_CONTEXT="${STRATA_MAX_CONTEXT:-16384}" STRATA_EXPERT_CACHE="${STRATA_EXPERT_CACHE:-4608}" STRATA_3080_LOG="$LOG"
python3 - <<'PY'
import json, os
args = ["--pack", os.environ["STRATA_PACK_HOT"], "--expert-format", "nvfp4",
        "--native", os.environ["STRATA_DENSE_GGUF"], "--expert-profile", os.environ["STRATA_PROFILE"],
        "--expert-cache", os.environ["STRATA_EXPERT_CACHE"], "--prefill", "auto", "--kv", "fp8",
        "--max-context", os.environ["STRATA_MAX_CONTEXT"], "--remote-rdma", os.environ["STRATA_REMOTE_HOST"],
        "--remote-rdma-port", os.environ["STRATA_REMOTE_PORT"], "--remote-rdma-gid", os.environ["STRATA_REMOTE_GID"],
        "--remote-rdma-slots", os.environ["STRATA_REMOTE_SLOTS"], "--remote-rdma-slot-bytes", os.environ["STRATA_REMOTE_SLOT_BYTES"]]
hca = os.environ.get("STRATA_REMOTE_HCA", "").strip()
if hca: args += ["--remote-rdma-device", hca]
ple = os.environ.get("STRATA_PLE_FP8", "").strip()
if ple:
    args += ["--ple-fp8", ple, "--ple-fp8-scale", os.environ["STRATA_PLE_FP8_SCALE"]]
else:
    gguf = os.environ.get("STRATA_PLE_GGUF", "").strip()
    if not gguf: raise SystemExit("set STRATA_PLE_FP8 or STRATA_PLE_GGUF")
    args += ["--ple-gguf", gguf, "--ple-io", "rdma"]
cfg = {"exe": os.environ["STRATA_ENGINE_EXE"], "args": args, "cwd": os.path.dirname(os.environ["STRATA_ENGINE_EXE"],),
       "tokenizer": os.path.join(os.environ["STRATA_PACK_HOT"], "tokenizer"), "host": os.environ.get("STRATA_HTTP_HOST", "0.0.0.0"),
       "model_name": "Qwen3.8-27B-NVFP4", "max_context": int(os.environ["STRATA_MAX_CONTEXT"]), "log": os.environ["STRATA_3080_LOG"]}
with open(os.environ["STRATA_CFG"], "w", encoding="utf-8") as f: json.dump(cfg, f, ensure_ascii=False, indent=2)
PY
nohup python3 "$ROOT/src/serve/server.py" --engine strata --config "$CFG" --host "$HOST" --port "$PORT" >>"$LOG" 2>&1 < /dev/null &
echo $! > "$STATE/3080.pid"
echo "launched role=3080 pid=$(cat "$STATE/3080.pid") http=$HOST:$PORT log=$LOG"


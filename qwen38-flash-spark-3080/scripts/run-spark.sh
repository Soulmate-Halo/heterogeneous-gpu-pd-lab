#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"; STATE="${STRATA_STATE_DIR:-$ROOT/state}"
mkdir -p "$STATE"; umask 077
: "${STRATA_PACK_COLD:?set STRATA_PACK_COLD}"; : "${STRATA_DENSE_GGUF:?set STRATA_DENSE_GGUF}"
: "${STRATA_PLE_FP8:?set STRATA_PLE_FP8}"; : "${STRATA_COLD_HCA:?set STRATA_COLD_HCA (empty means verbs auto-detect)}"
BUILD="${STRATA_BUILD_DIR:-$ROOT/build-spark}"; EXE="${STRATA_SPARK_EXE:-$BUILD/strata}"
test -x "$EXE" || { echo "missing executable: $EXE (run build-spark.sh)" >&2; exit 2; }
test -d "$STRATA_PACK_COLD" || { echo "missing strata-cold-pack: $STRATA_PACK_COLD" >&2; exit 2; }
test -f "$STRATA_DENSE_GGUF" || { echo "missing dense.gguf: $STRATA_DENSE_GGUF" >&2; exit 2; }
test -f "$STRATA_PLE_FP8" || { echo "missing ple-fp8.bin: $STRATA_PLE_FP8" >&2; exit 2; }
PORT="${STRATA_SPARK_HTTP_PORT:-8200}"; HOST="${STRATA_SPARK_HTTP_HOST:-0.0.0.0}"
LOG="${STRATA_SPARK_LOG:-$STATE/spark-http.log}"; CFG="$STATE/spark-serve.json"
if [[ -f "$STATE/spark.pid" ]] && kill -0 "$(cat "$STATE/spark.pid")" 2>/dev/null; then echo "already running pid=$(cat "$STATE/spark.pid")"; exit 0; fi
export STRATA_CFG="$CFG" STRATA_ENGINE_EXE="$EXE" STRATA_PACK_COLD STRATA_DENSE_GGUF STRATA_PLE_FP8 STRATA_PLE_FP8_SCALE="${STRATA_PLE_FP8_SCALE:-0.00019931793212890625}"
export STRATA_COLD_HCA STRATA_COLD_PORT="${STRATA_COLD_PORT:-39580}" STRATA_COLD_GID="${STRATA_COLD_GID:-0}" STRATA_COLD_SLOTS="${STRATA_COLD_SLOTS:-4}" STRATA_COLD_SLOT_BYTES="${STRATA_COLD_SLOT_BYTES:-4194304}"
export STRATA_MAX_CONTEXT="${STRATA_MAX_CONTEXT:-16384}" STRATA_SPARK_LOG="$LOG"
python3 - <<'PY'
import json, os
args = ["--pack", os.environ["STRATA_PACK_COLD"], "--expert-format", "nvfp4", "--native", os.environ["STRATA_DENSE_GGUF"],
        "--ple-fp8", os.environ["STRATA_PLE_FP8"], "--ple-fp8-scale", os.environ["STRATA_PLE_FP8_SCALE"],
        "--expert-resident-all", "--expert-cache", "24576", "--prefill", "auto", "--kv", "fp8", "--max-context", os.environ["STRATA_MAX_CONTEXT"],
        "--cold-expert-inproc", "--cold-expert-port", os.environ["STRATA_COLD_PORT"], "--cold-expert-gid", os.environ["STRATA_COLD_GID"],
        "--cold-expert-slots", os.environ["STRATA_COLD_SLOTS"], "--cold-expert-slot-bytes", os.environ["STRATA_COLD_SLOT_BYTES"]]
hca = os.environ.get("STRATA_COLD_HCA", "").strip()
if hca: args += ["--cold-expert-device", hca]
cfg = {"exe": os.environ["STRATA_ENGINE_EXE"], "args": args, "cwd": os.path.dirname(os.environ["STRATA_ENGINE_EXE"]),
       "tokenizer": os.path.join(os.environ["STRATA_PACK_COLD"], "tokenizer"), "host": os.environ.get("STRATA_SPARK_HTTP_HOST", "0.0.0.0"),
       "model_name": "Qwen3.8-27B-NVFP4", "max_context": int(os.environ["STRATA_MAX_CONTEXT"]), "log": os.environ["STRATA_SPARK_LOG"]}
with open(os.environ["STRATA_CFG"], "w", encoding="utf-8") as f: json.dump(cfg, f, ensure_ascii=False, indent=2)
PY
if [[ "${STRATA_FOREGROUND:-0}" == "1" ]]; then
  exec python3 "$ROOT/src/serve/server.py" --engine strata --config "$CFG" --host "$HOST" --port "$PORT"
fi
nohup python3 "$ROOT/src/serve/server.py" --engine strata --config "$CFG" --host "$HOST" --port "$PORT" >>"$LOG" 2>&1 < /dev/null &
echo $! > "$STATE/spark.pid"
echo "launched role=spark pid=$(cat "$STATE/spark.pid") http=$HOST:$PORT log=$LOG"

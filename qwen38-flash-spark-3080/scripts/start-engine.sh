#!/usr/bin/env bash
# start-engine.sh - start the Strata NVFP4 engine on the RTX 3080 host.
#
#   ./scripts/start-engine.sh                 # one-shot generate over $STRATA_TOKENS_FILE
#   ./scripts/start-engine.sh --serve         # resident engine behind the JSON config
#
# --serve writes $STRATA_RUN_DIR/strata-serve.json and launches serve/server.py
# from the source tree; the HTTP front then talks to the resident engine over
# stdin/stdout. Use scripts/health.sh to poll it.
set -euo pipefail
. "$(dirname "$0")/_lib.sh"

SERVE=0
[ "${1:-}" = "--serve" ] && SERVE=1

require_env STRATA_HOT_PACK STRATA_DENSE_GGUF STRATA_PLE_FP8 STRATA_PEER_ADDR
require_file "$STRATA_PROFILE" "expert profile"
require_file "$STRATA_DENSE_GGUF" "dense GGUF (--native)"
require_file "$STRATA_PLE_FP8" "PLE FP8 table"
prepare_run_dir
[ -x "$STRATA_BUILD_DIR/strata" ] || die "engine not built: run scripts/build-engine.sh first"

HCA_ARGS=()
HCA="$(hca_detect)"
[ -n "$HCA" ] && HCA_ARGS=(--remote-rdma-device "$HCA")

ARGS=(
  --pack "$STRATA_HOT_PACK"
  --expert-format nvfp4
  --native "$STRATA_DENSE_GGUF"
  --ple-fp8 "$STRATA_PLE_FP8"
  --ple-fp8-scale "$STRATA_PLE_FP8_SCALE"
  --expert-profile "$STRATA_PROFILE"
  --expert-cache "$STRATA_EXPERT_CACHE"
  --prefill "$STRATA_PREFILL"
  --max-context "$STRATA_MAX_CONTEXT"
  --kv "$STRATA_KV"
  --remote-rdma "$STRATA_PEER_ADDR"
  --remote-rdma-port "$STRATA_PEER_PORT"
  --remote-rdma-slots "$STRATA_RDMA_SLOTS"
  --remote-rdma-slot-bytes "$STRATA_RDMA_SLOT_BYTES"
)
[ "${#HCA_ARGS[@]}" -gt 0 ] && ARGS+=("${HCA_ARGS[@]}")
# --ple-gguf is required alongside --ple-fp8 whenever the PLE keys are read from
# a GGUF: the native loader rejects --ple-fp8 alone.
if [ -n "$STRATA_Q2_GGUF_00002" ]; then
  require_file "$STRATA_Q2_GGUF_00002" "PLE GGUF (Q2_0 shard 2)"
  ARGS+=(--ple-gguf "$STRATA_Q2_GGUF_00002")
fi

ENV_PREFIX=()
[ "$STRATA_RESIDENT_PREFILL" = "1" ] && ENV_PREFIX+=(STRATA_PF_NVFP4_RESIDENT=1)
[ -n "${STRATA_PREFILL_TIMING:-}" ] && ENV_PREFIX+=(STRATA_PREFILL_TIMING="$STRATA_PREFILL_TIMING")
[ -n "${STRATA_DECODE_TIMING:-}" ] && ENV_PREFIX+=(STRATA_DECODE_TIMING="$STRATA_DECODE_TIMING")

if [ "$SERVE" -eq 0 ]; then
  require_env STRATA_TOKENS_FILE
  require_file "$STRATA_TOKENS_FILE" "token ids file"
  ARGS+=(--tokens-file "$STRATA_TOKENS_FILE" --max-new "${STRATA_MAX_NEW:-1}")
  [ "${STRATA_GREEDY:-1}" = "1" ] && ARGS+=(--greedy)
  start_bg engine engine-nvfp4 env "${ENV_PREFIX[@]}" "$STRATA_BUILD_DIR/strata" "${ARGS[@]}"
else
  ARGS+=(--serve)
  CFG="$STRATA_RUN_DIR/strata-serve.json"
  "$STRATA_PYTHON" - "$CFG" "$STRATA_BUILD_DIR/strata" "$STRATA_TREE" \
                    "${STRATA_TOKENIZER:-}" "${STRATA_MODEL_NAME:-qwen3.8-flash-next}" \
                    "$STRATA_RESIDENT_PREFILL" "${STRATA_LOG_DIR}/engine-nvfp4.log" "${ARGS[@]}" <<'PY'
import json, sys
cfg, exe, cwd, tok, name, resident, log, *args = sys.argv[1:]
env = {}
if resident == "1":
    env["STRATA_PF_NVFP4_RESIDENT"] = "1"
doc = {"exe": exe, "cwd": cwd, "args": args, "model_name": name, "log": log}
if tok:
    doc["tokenizer"] = tok
if env:
    doc["env"] = env
open(cfg, "w", encoding="utf-8").write(json.dumps(doc, indent=1))
print("WROTE " + cfg)
PY
  if [ "${STRATA_FOREGROUND:-0}" = "1" ]; then
    exec "$STRATA_PYTHON" -m serve.server --engine strata --config "$CFG" --host "${STRATA_BIND:-0.0.0.0}" --port "$STRATA_HTTP_PORT"
  fi
  start_bg http http.log "$STRATA_PYTHON" -m serve.server \
    --engine strata --config "$CFG" --host "${STRATA_BIND:-127.0.0.1}" --port "$STRATA_HTTP_PORT"
  note "waiting for the HTTP front on :$STRATA_HTTP_PORT ..."
  wait_http "http://127.0.0.1:$STRATA_HTTP_PORT/v1/models" "${STRATA_HTTP_TIMEOUT:-900}" || {
    tail -30 "$(logfile http)" >&2; die "HTTP front did not come up"; }
  note "engine READY: http://127.0.0.1:$STRATA_HTTP_PORT/v1  (model ${STRATA_MODEL_NAME:-qwen3.8-flash-next})"
fi

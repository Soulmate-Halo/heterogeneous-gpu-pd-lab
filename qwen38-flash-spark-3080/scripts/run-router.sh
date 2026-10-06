#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"; STATE="${STRATA_STATE_DIR:-$ROOT/state}"
mkdir -p "$STATE"; PREFILL="${STRATA_PREFILL_BACKEND:?set STRATA_PREFILL_BACKEND}"; DECODE="${STRATA_DECODE_BACKEND:?set STRATA_DECODE_BACKEND}"
PORT="${STRATA_ROUTER_PORT:-8400}"; HOST="${STRATA_ROUTER_HOST:-0.0.0.0}"; THRESHOLD="${STRATA_ROUTER_THRESHOLD:-4096}"; LOG="${STRATA_ROUTER_LOG:-$STATE/router.log}"
if [[ "${STRATA_FOREGROUND:-0}" == "1" ]]; then
  exec python3 "$ROOT/router/router.py" --host "$HOST" --port "$PORT" --prefill-backend "$PREFILL" --decode-backend "$DECODE" --char-threshold "$THRESHOLD" --timeout "${STRATA_ROUTER_TIMEOUT:-600}"
fi
if [[ -f "$STATE/router.pid" ]] && kill -0 "$(cat "$STATE/router.pid")" 2>/dev/null; then echo "already running pid=$(cat "$STATE/router.pid")"; exit 0; fi
nohup python3 "$ROOT/router/router.py" --host "$HOST" --port "$PORT" --prefill-backend "$PREFILL" --decode-backend "$DECODE" --char-threshold "$THRESHOLD" --timeout "${STRATA_ROUTER_TIMEOUT:-600}" >>"$LOG" 2>&1 < /dev/null &
echo $! > "$STATE/router.pid"
echo "launched role=router pid=$(cat "$STATE/router.pid") http=$HOST:$PORT log=$LOG"

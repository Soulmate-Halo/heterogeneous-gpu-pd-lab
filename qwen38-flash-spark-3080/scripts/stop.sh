#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"; STATE="${STRATA_STATE_DIR:-$ROOT/state}"
for role in router 3080 spark; do
  pidfile="$STATE/$role.pid"
  if [[ -s "$pidfile" ]]; then
    pid="$(cat "$pidfile")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then kill "$pid"; fi
    rm -f "$pidfile"
  fi
done
echo "stopped bundle services (router, 3080, spark)"


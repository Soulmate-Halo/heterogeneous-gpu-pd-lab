#!/usr/bin/env bash
set -euo pipefail
case "${ROLE:-${STRATA_ROLE:-}}" in
    spark|worker) PORT="${STRATA_SPARK_HTTP_PORT:-8200}" ;;
    3080|engine) PORT="${STRATA_HTTP_PORT:-8100}" ;;
    router) PORT="${STRATA_ROUTER_PORT:-8400}" ;;
    *) exit 0 ;;
esac
exec python3 - "$PORT" <<'PY'
import sys, urllib.request
with urllib.request.urlopen('http://127.0.0.1:' + sys.argv[1] + '/health', timeout=4) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROLE="${ROLE:-${STRATA_ROLE:-}}"
if [[ "${1:-}" =~ ^(verify|worker|engine|router|shell|smoke|sanitize)$ ]]; then ROLE="$1"; shift || true; fi
case "${ROLE:-verify}" in
  verify) exec python3 "$ROOT/verify_bundle.py" "$@" ;;
  worker|wrk|spark) exec "$ROOT/scripts/run-spark.sh" "$@" ;;
  engine|eng|3080) exec "$ROOT/scripts/run-3080.sh" "$@" ;;
  router) exec "$ROOT/scripts/run-router.sh" "$@" ;;
  smoke) exec "$ROOT/scripts/run-smoke.sh" "$@" ;;
  sanitize) exec "$ROOT/scripts/sanitize.sh" "$@" ;;
  shell) exec /bin/bash "$@" ;;
  *) echo "usage: ROLE=worker|engine|router|shell|verify $0" >&2; exit 2 ;;
esac

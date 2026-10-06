#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROLE="${ROLE:-${STRATA_ROLE:-help}}"
if [[ $# -gt 0 ]]; then ROLE="$1"; shift; fi
case "$ROLE" in
  verify) exec python3 "$ROOT/verify_bundle.py" "$@" ;;
  help|--help|-h) exec "$ROOT/scripts/container-help.sh" ;;
  doctor) exec python3 "$ROOT/scripts/doctor.py" "$@" ;;
  build) exec "$ROOT/scripts/container-build.sh" "$@" ;;
  worker|wrk|spark) exec "$ROOT/scripts/container-start.sh" spark "$@" ;;
  engine|eng|3080|decode) exec "$ROOT/scripts/container-start.sh" 3080 "$@" ;;
  router) export STRATA_FOREGROUND=1; exec "$ROOT/scripts/run-router.sh" "$@" ;;
  smoke) exec "$ROOT/scripts/container-smoke.sh" ;;
  shell) exec /bin/bash "$@" ;;
  sanitize) exec "$ROOT/scripts/sanitize.sh" "$@" ;;
  *) echo "unknown role: $ROLE (run: help)" >&2; exit 2 ;;
esac

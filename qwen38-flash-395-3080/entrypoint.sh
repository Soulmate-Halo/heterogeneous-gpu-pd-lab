#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROLE="${ROLE:-${STRATA_ROLE:-help}}"
if [[ $# -gt 0 ]]; then ROLE="$1"; shift; fi
case "$ROLE" in
  verify) exec python3 "$ROOT/verify_bundle.py" "$@" ;;
  help|--help|-h) exec "$ROOT/scripts/help.sh" ;;
  doctor) exec python3 "$ROOT/scripts/doctor.py" "$@" ;;
  engine|eng|395|3080|start) exec "$ROOT/scripts/container-start.sh" "$@" ;;
  freeze) exec python3 "$ROOT/bundle/freeze_baseline.py" "$@" ;;
  smoke) exec "$ROOT/scripts/smoke.sh" ;;
  shell) exec /bin/bash "$@" ;;
  *) echo "unknown role: $ROLE (run: help)" >&2; exit 2 ;;
esac

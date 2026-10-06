#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
OVERLAY="${BUNDLE_OVERLAY_DIR:-/opt/strata-overlay}"
if [[ "${1:-}" == "--overlay-dir" ]]; then
  [[ -n "${2:-}" ]] || { echo "--overlay-dir requires a path" >&2; exit 2; }
  OVERLAY="$2"
elif [[ -n "${1:-}" ]]; then OVERLAY="$1"; fi
python3 "$ROOT/verify_bundle.py"
mkdir -p "$OVERLAY"
install -m 0644 "$ROOT/bundle/runtime.env" "$OVERLAY/runtime.env"
install -m 0644 "$ROOT/bundle/config/server.json" "$OVERLAY/server.json"
printf 'APPLIED overlay=%s files=runtime.env,server.json\n' "$OVERLAY"

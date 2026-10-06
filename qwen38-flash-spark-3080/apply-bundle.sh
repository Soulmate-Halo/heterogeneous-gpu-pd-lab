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
shopt -s nullglob
packages=("$ROOT"/bundle/packages/*.tgz)
((${#packages[@]} > 0)) || { echo "no configuration package found" >&2; exit 2; }
for archive in "${packages[@]}"; do tar -xzf "$archive" -C "$OVERLAY"; done
printf 'APPLIED overlay=%s packages=%d\n' "$OVERLAY" "${#packages[@]}"

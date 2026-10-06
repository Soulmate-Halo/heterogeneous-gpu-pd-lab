#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PORT="${STRATA_HTTP_PORT:-8095}"
export STRATA_CACHE_DIR="${STRATA_CACHE_DIR:-/cache}"
export STRATA_STATE_DIR="${STRATA_STATE_DIR:-/state}"
export STRATA_FOREGROUND=1
export STRATA_ENGINE_BIN="${STRATA_ENGINE_BIN:-/models/strata/strata}"
export STRATA_HIPCOLD_LIB="${STRATA_HIPCOLD_LIB:-$STRATA_CACHE_DIR/hipcold/libstrata_hipdecode.so}"
export STRATA_HTTP_BIND="${STRATA_HTTP_BIND:-0.0.0.0:$PORT}"
export STRATA_LOG="${STRATA_LOG:-$STRATA_STATE_DIR/engine.log}"
python3 "$ROOT/scripts/doctor.py" engine
# HIP is intentionally absent from the image: build the gfx1151 cold-expert library on
# first start (hipcc from an optional mounted ROCm tree) and cache it under /cache.
if [[ "${STRATA_HIPDECODE:-1}" != "0" && ! -f "$STRATA_HIPCOLD_LIB" ]]; then
  if command -v hipcc >/dev/null 2>&1 || [[ -x /opt/rocm/bin/hipcc ]]; then
    mkdir -p "$(dirname -- "$STRATA_HIPCOLD_LIB")"
    HIPCC="$(command -v hipcc || echo /opt/rocm/bin/hipcc)"
    echo "building cold-expert library for gfx1151 -> $STRATA_HIPCOLD_LIB"
    "$HIPCC" -O2 --offload-arch=gfx1151 -fPIC -shared \
      -o "$STRATA_HIPCOLD_LIB" "$ROOT/bundle/src/strata_hipcold_lib_r435.cpp" || true
  else
    echo "no hipcc found: set STRATA_HIPDECODE=0 or mount a ROCm toolchain"
  fi
fi
echo "engine bin=$STRATA_ENGINE_BIN http=$STRATA_HTTP_BIND hipdecode=$STRATA_HIPDECODE log=$STRATA_LOG"
exec "$STRATA_ENGINE_BIN" --config "$ROOT/bundle/config/server.json" --http "$STRATA_HTTP_BIND" "$@"

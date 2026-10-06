#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
ROLE="${1:?role required}"; shift
case "$ROLE" in spark|3080) ;; *) echo 'role must be spark or 3080' >&2; exit 2 ;; esac
export STRATA_PACK_HOT="${STRATA_PACK_HOT:-/models/strata-pack-hot}"
export STRATA_PACK_COLD="${STRATA_PACK_COLD:-/models/strata-cold-pack}"
export STRATA_DENSE_GGUF="${STRATA_DENSE_GGUF:-/models/dense.gguf}"
export STRATA_PLE_FP8="${STRATA_PLE_FP8:-/models/ple-fp8.bin}"
export STRATA_PROFILE="${STRATA_PROFILE:-/models/profile.bin}"
export STRATA_BUILD_DIR="${STRATA_CACHE_DIR:-/cache}/build-$ROLE"
export STRATA_STATE_DIR="${STRATA_STATE_DIR:-/state}"
export STRATA_FOREGROUND=1
export STRATA_COLD_HCA="${STRATA_COLD_HCA:-}"
python3 "$ROOT/scripts/doctor.py" "$ROLE"
STAMP="$(python3 - "$ROOT" <<'PY'
import hashlib, pathlib, sys
p=pathlib.Path(sys.argv[1]); h=hashlib.sha256()
for name in ['SOURCE-REVISION.txt', 'SHA256SUMS']: h.update((p/name).read_bytes())
print(h.hexdigest())
PY
)"
if [[ ! -x "$STRATA_BUILD_DIR/strata" ]] || [[ "$(cat "$STRATA_BUILD_DIR/source.sha256" 2>/dev/null || true)" != "$STAMP" ]]; then
    bash "$ROOT/scripts/container-build.sh" "$ROLE"
fi
export STRATA_SPARK_EXE="$STRATA_BUILD_DIR/strata" STRATA_EXE="$STRATA_BUILD_DIR/strata"
export STRATA_SPARK_LOG="$STRATA_STATE_DIR/spark-engine.log" STRATA_3080_LOG="$STRATA_STATE_DIR/3080-engine.log"
if [[ "$ROLE" == spark ]]; then exec bash "$ROOT/scripts/run-spark.sh" "$@"; fi
exec bash "$ROOT/scripts/run-3080.sh" "$@"

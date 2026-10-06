#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
case "${1:-}" in
    spark|worker) ROLE=spark; ARCH=121; ;;
    3080|engine) ROLE=3080; ARCH=86; ;;
    *) echo 'usage: build spark|3080' >&2; exit 2 ;;
esac
export STRATA_BUILD_DIR="${STRATA_CACHE_DIR:-/cache}/build-$ROLE"
export CUDA_ARCHITECTURES="$ARCH"
export STRATA_ENABLE_NVFP4=ON
: "${STRATA_SGL_KERNEL_INCLUDE:?missing SGL headers; use the published CUDA build image}"
: "${STRATA_TVM_FFI_INCLUDE:?missing TVM-FFI headers}"
: "${STRATA_TVM_FFI_LIB:?missing TVM-FFI library}"
mkdir -p "$STRATA_BUILD_DIR"
echo "BUILD_START role=$ROLE arch=sm_$ARCH cache=$STRATA_BUILD_DIR"
bash "$ROOT/scripts/build-$ROLE.sh"
test -x "$STRATA_BUILD_DIR/strata"
python3 - "$ROOT" "$STRATA_BUILD_DIR/source.sha256" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
h = hashlib.sha256()
for name in ['SOURCE-REVISION.txt', 'SHA256SUMS']:
    h.update((root / name).read_bytes())
pathlib.Path(sys.argv[2]).write_text(h.hexdigest() + '\n')
PY
echo "BUILD_OK role=$ROLE"

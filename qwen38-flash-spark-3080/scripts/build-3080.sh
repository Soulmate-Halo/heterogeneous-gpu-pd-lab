#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
SRC="${STRATA_SRC_DIR:-$ROOT/src}"
BUILD="${STRATA_BUILD_DIR:-$ROOT/build-3080}"
: "${STRATA_SGL_KERNEL_INCLUDE:?set STRATA_SGL_KERNEL_INCLUDE to the SGL kernel headers}"
: "${STRATA_TVM_FFI_INCLUDE:?set STRATA_TVM_FFI_INCLUDE to the TVM-FFI headers}"
: "${STRATA_TVM_FFI_LIB:?set STRATA_TVM_FFI_LIB to the TVM-FFI library}"
test -f "$SRC/CMakeLists.txt" || { echo "missing source: $SRC" >&2; exit 2; }
CUDA_ARCH="${CUDA_ARCHITECTURES:-86}"
JOBS="${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_NVFP4=ON -DSTRATA_ENABLE_NVFP4_COLD=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF \
  -DSTRATA_SGL_KERNEL_INCLUDE="$STRATA_SGL_KERNEL_INCLUDE" \
  -DSTRATA_TVM_FFI_INCLUDE="$STRATA_TVM_FFI_INCLUDE" \
  -DSTRATA_TVM_FFI_LIB="$STRATA_TVM_FFI_LIB"
cmake --build "$BUILD" -j "$JOBS" --target strata
echo "BUILD_EXIT=0 target=strata arch=$CUDA_ARCH build=$BUILD"


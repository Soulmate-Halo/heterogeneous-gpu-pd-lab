#!/usr/bin/env bash
# build-engine.sh - build the Strata engine (target `strata`) on the RTX 3080 / x86 host.
#
# Native CUDA 13 build. There is no prebuilt binary in this bundle and no
# private artifact anywhere: this script is the whole build story.
#
#   ./scripts/build-engine.sh [--configure-only]
set -euo pipefail
. "$(dirname "$0")/_lib.sh"

CONFIGURE_ONLY=0
[ "${1:-}" = "--configure-only" ] && CONFIGURE_ONLY=1

require_env STRATA_LLAMACPP_DIR
[ -d "$STRATA_TREE" ] || die "src tree not found: $STRATA_TREE"
[ -d "$STRATA_LLAMACPP_DIR" ] || die "llama.cpp checkout not found: $STRATA_LLAMACPP_DIR"

ARCH="${STRATA_CUDA_ARCH:-86}"
command -v nvcc >/dev/null 2>&1 && note "nvcc: $(nvcc --version | tail -1)"
command -v cmake >/dev/null 2>&1 || die "cmake not found"

# STRATA_ENABLE_NVFP4 pulls in third_party/marlin (in this bundle) plus the
# sgl_kernel + tvm_ffi headers/libs from the machine's own sglang install.
# Those two are third-party build inputs, not redistributable bundle content.
CMAKE_ARGS=(
  -S "$STRATA_TREE"
  -B "$STRATA_BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DSTRATA_ENABLE_CUDA=ON
  -DSTRATA_BUILD_TESTS=OFF
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
  -DSTRATA_GGML_DIR="$STRATA_LLAMACPP_DIR"
  -DSTRATA_ENABLE_NVFP4=ON
)
if [ -n "$STRATA_SGL_KERNEL_INCLUDE" ]; then
  CMAKE_ARGS+=(-DSTRATA_SGL_KERNEL_INCLUDE="$STRATA_SGL_KERNEL_INCLUDE")
fi
if [ -n "$STRATA_TVM_FFI_INCLUDE" ]; then
  CMAKE_ARGS+=(-DSTRATA_TVM_FFI_INCLUDE="$STRATA_TVM_FFI_INCLUDE")
fi
if [ -n "$STRATA_TVM_FFI_LIB" ]; then
  CMAKE_ARGS+=(-DSTRATA_TVM_FFI_LIB="$STRATA_TVM_FFI_LIB")
fi

note "cmake ${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}"

[ "$CONFIGURE_ONLY" -eq 1 ] && { note "configure-only: done"; exit 0; }

# `strata` only. A full build also compiles tools/cold_expert_worker.cpp, which
# needs the CUDA/ARM cold kernels and belongs to the Spark host.
note "building target strata (-j $STRATA_JOBS)"
cmake --build "$STRATA_BUILD_DIR" -j "$STRATA_JOBS" --target strata
[ -x "$STRATA_BUILD_DIR/strata" ] || die "build produced no $STRATA_BUILD_DIR/strata"
note "engine built: $STRATA_BUILD_DIR/strata"

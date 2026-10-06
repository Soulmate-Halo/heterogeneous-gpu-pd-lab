# Strata Engine - top-level build (P1.S1).
#
# Targets (phase-1 Ă‚Â§P1.S1): `strata_artifact` (no CUDA dependency), `tests`.  `strata_plan` and the C++
# `strata-pack` are added when their sources exist - a target with no sources would make the build graph
# claim something the tree does not have, which is worse than an absent target.
#
# CUDA targets arrive in Phase 2.  The kernels need sm_80 or newer (tf32 mma, bf16 math) and the guard
# below refuses anything older: silently building for a pre-Ampere arch would produce a binary that runs
# and is wrong about its own performance, which is the worst of both outcomes.
cmake_minimum_required(VERSION 3.24)
project(strata VERSION 0.1.25 LANGUAGES CXX)
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")
# issue #31: release builds keep a PDB beside the exe (not shipped in it), so a minidump from a stalled engine can
# be read back to the exact source line
if(MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zi>)
  add_link_options(/DEBUG /OPT:REF /OPT:ICF)
endif()

# The prebuilt release (setup.py downloads it, so users need no compiler): ggml's CPU backend for any AVX2 CPU
# instead of this machine's (Strata's own AVX-512 kernels are per-file and chosen at run time either way), and
# the MSVC runtime linked in, so no Visual C++ redistributable is needed.
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_ENABLE_CUDA "Build the CUDA targets (Phase 2+)" OFF)
option(STRATA_ENABLE_HIP "Build the HIP targets for gfx1100 wave32 hardware" OFF)
option(STRATA_PREFILL_MMQ "Build the optional GGML MMQ prefill path for HIP" OFF)
if(STRATA_PREFILL_MMQ AND NOT STRATA_ENABLE_HIP)
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_ENABLE_HIP=ON")
endif()
if(STRATA_ENABLE_CUDA AND STRATA_ENABLE_HIP)
  message(FATAL_ERROR "STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP are separate backends; enable only one")
endif()
if(STRATA_ENABLE_HIP)
  include(cmake/hip_backend.cmake)
endif()
# The test suite and the micro benchmarks live in tests/ and bench/micro/, which the published source leaves out.
set(_strata_tests_default OFF)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro)
  set(_strata_tests_default ON)
endif()
option(STRATA_BUILD_TESTS "Build the test suite" ${_strata_tests_default})
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)

if(STRATA_ENABLE_CUDA)
  enable_language(CUDA)
  # `enable_language(CUDA)` gives the .cu compile rules but NOT the imported targets, so `CUDA::cudart` has to
  # be brought in explicitly for the host-side tests that call cudaMalloc themselves.
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
  endif()
  foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" _base "${_a}")     # 120-real / 120-virtual are still 120
    # Plan v0.3 P6: developed and measured on sm_120 (RTX 50); the kernels need sm_80 or newer (tf32 mma in
    # the attention scorer, bf16 math), so RTX 30 (86) and RTX 40 (89) build too.  Older cards are refused.
    if(_base STREQUAL "native" OR _base STREQUAL "all" OR _base STREQUAL "all-major")
      continue()
    endif()
    if(_base LESS 80 AND NOT STRATA_EXPERIMENTAL_SM75)   # the A/B of a Turing card as a layer-split stage
      message(FATAL_ERROR
        "Strata needs an NVIDIA GPU of compute capability 8.0 or newer (RTX 30 / 40 / 50 series), but "
        "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}'.")
    endif()
  endforeach()
  message(STATUS "Strata: CUDA enabled, arch ${CMAKE_CUDA_ARCHITECTURES}")
  set(_strata_gpu_runtime_target CUDA::cudart)
  set(_strata_gpu_blas_target CUDA::cublas)
  set(_strata_gpu_include_directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
elseif(STRATA_ENABLE_HIP)
  set(_strata_gpu_runtime_target strata_hip_runtime)
  set(_strata_gpu_blas_target roc::hipblas)
  set(_strata_gpu_include_directories ${STRATA_HIP_COMPAT_INCLUDE_DIR})
endif()

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd Catch2, whose -W4 warnings are not this project's to fix and
# whose /WX failures would be indistinguishable from our own.
#
# Every flag is guarded by COMPILE_LANGUAGE:CXX.  Without that guard nvcc receives bare `/W4 /utf-8` and dies
# with "A single input file is required for a non-link phase" - it parses them as inputs rather than host
# flags.  Host flags for CUDA must go through -Xcompiler, which is what COMPILE_LANGUAGE:CUDA does below.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/W4>
  $<$<COMPILE_LANGUAGE:HIP>:-Wall>
  $<$<COMPILE_LANGUAGE:HIP>:-Wextra>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>
    $<$<COMPILE_LANGUAGE:HIP>:-Werror>)
endif()

# ---- strata_artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)

add_executable(strata-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(strata-gguf PRIVATE strata_artifact)

add_executable(strata-dequant src/artifact/dequant.cpp)
target_link_libraries(strata-dequant PRIVATE strata_artifact)
# this file is now nothing but the CLI entry point, and its inherited guard spells that as a "selftest"
target_compile_definitions(strata-dequant PRIVATE STRATA_DEQUANT_SELFTEST)
if(STRATA_BUILD_TESTS)
  # the reader refuses a duplicate tensor name at open, naming it (a synthetic GGUF: no model, no GPU)
  add_executable(gguf_reader_test src/artifact/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE strata_artifact)
  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
endif()

# ---- strata_plan ----------------------------------------------------------
# Also header-only.  A "planner" that hides state would be one whose decision is not inspectable, and the
# whole point of P1.S9 is that the decision is arithmetic you can check by hand.
add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)

add_executable(strata-plan src/plan/plan_main.cpp)
target_link_libraries(strata-plan PRIVATE strata_plan)

# ---- strata_core (GPU runtime) ---------------------------------------------
# Only built when a GPU backend is on, so the Phase 1 targets keep working without a toolkit.
if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
  add_library(strata_core STATIC src/core/device.cu src/core/pinned.cu src/platform/memory.cpp src/core/graph.cpp src/core/weights.cpp
                               src/core/layout.cpp)
  target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # The selected runtime is PUBLIC because graph.cpp is a HOST translation unit that calls cudaGraphLaunch and
  # cudaEventQuery: the .cu sources get cudart through the CUDA language rules, a .cpp does not, and the
  # linker error ("unresolved external symbol cudaGraphInstantiate") names the function rather than the cause.
  target_link_libraries(strata_core PUBLIC strata_plan strata_warnings ${_strata_gpu_runtime_target})

  add_executable(strata-device src/core/device_main.cpp)
  target_link_libraries(strata-device PRIVATE strata_core)

  add_executable(strata-load src/core/load_main.cpp)
  target_link_libraries(strata-load PRIVATE strata_core strata_kernels)

  add_executable(strata-concurrent src/core/concurrent_main.cpp)
  target_include_directories(strata-concurrent PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata-concurrent PRIVATE strata_core strata_kernels ${_strata_gpu_runtime_target})

  add_executable(strata-overlap src/core/overlap_main.cpp)
  # a .cpp that calls cudaMemcpyAsync itself needs the toolkit headers and cudart explicitly
  target_include_directories(strata-overlap PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata-overlap PRIVATE strata_core strata_kernels ${_strata_gpu_runtime_target})

  # The load test needs the 34 GB artifact, so it is registered ONLY where that artifact exists.  A test that
  # silently skips is a test that stops running; an absent registration is visible in the configure output.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pack/full/experts.bin)
    add_test(NAME expert_arena_load
             COMMAND strata-load --file ${CMAKE_CURRENT_SOURCE_DIR}/pack/full/experts.bin --layers 1)
    set_tests_properties(expert_arena_load PROPERTIES PASS_REGULAR_EXPRESSION "per-layer checksums")
  else()
    message(STATUS "strata: pack/full/experts.bin absent - the expert arena load test is NOT registered")
  endif()

  # ---- strata_kernels (P2.S2) ----
  # Naive first, each with a parity test against a reference that was itself validated against ggml.  The
  # kernels link `strata_artifact` for the scalar dequantizers, because using the VALIDATED scalar path as the
  # reference is the whole point - a kernel checked against another copy of itself proves nothing.
  add_library(strata_kernels STATIC src/kernels/cuda/dequant_s2.cu src/kernels/cuda/s2_gemv.cu
                          src/kernels/cuda/s_gemv.cu
                          src/kernels/cuda/s2_gemv_quads.cu
                          src/kernels/cuda/s2_gemv_fast.cu
                          src/kernels/cuda/router_top10.cu
                          src/kernels/cuda/quantize_act.cu
                          src/kernels/cuda/rope.cu
                          src/kernels/cuda/sampler.cu
                          src/kernels/cuda/s2_gemv_q8.cu
    src/kernels/cuda/s2_expert_grouped.cu
                          src/kernels/cuda/shared_expert.cu
                          src/kernels/cuda/gr.cu
                          src/kernels/cuda/gdn.cu
                          src/kernels/cuda/qsa.cu
                          src/kernels/cuda/elementwise.cu
                          src/kernels/cuda/bf16_gemv.cu
                          src/kernels/cuda/native_mmvq.cu
                          src/kernels/cuda/native_bf16.cu
                          src/kernels/cuda/native_gr_norm.cu
                          src/kernels/cuda/native_moe.cu
                          src/kernels/cuda/native_gdn.cu
                          src/kernels/cuda/native_gr_postops.cu
                          src/kernels/cuda/native_router.cu
                          src/kernels/cuda/native_gdn_preprocess.cu
                          src/kernels/cuda/native_qsa.cu
                          src/kernels/cuda/native_rope.cu
                          src/kernels/cuda/native_ple_postops.cu
                          src/kernels/cuda/native_flash_attn.cu
                          src/kernels/cuda/native_qsa_indexer.cu
                          src/kernels/cuda/native_qsa_score.cu
                          src/kernels/cuda/ple.cu
                          src/kernels/cuda/kv_q8.cu
                          src/kernels/cuda/kv_fp8.cu
                          src/kernels/cuda/kv_stream.cu
                          src/kernels/cuda/kv_q4.cu
                          src/kernels/cuda/fused_gr.cu
                          src/kernels/cuda/cvec.cu
                          src/kernels/cuda/qsa_decode_attn.cu
                          src/kernels/cuda/qsa_prompt_attn.cu
                          src/kernels/cuda/fused_gdn.cu
                          src/kernels/cuda/dequant_bf16.cu
                          src/kernels/cuda/qsa_select.cu
                          src/kernels/cuda/verify_kernels.cu
                          src/kernels/cuda/iq_kernels.cu
                          src/kernels/ngram.cpp
                          src/ngram/ple_reader.cpp
                          src/platform/direct_file.cpp)
  target_link_libraries(strata_kernels PUBLIC strata_core strata_artifact)
  # perf-review D-1: the tensor-core prompt attention vs the FP32 kernel and FP64 (GPU, synthetic, no model); with
  # the tests, or alone with -DSTRATA_PARITY_PROMPT_ATTN=ON
  if(STRATA_BUILD_TESTS OR STRATA_PARITY_PROMPT_ATTN)
    add_executable(qsa_prompt_attn_parity src/kernels/qsa_prompt_attn_parity.cpp)
    target_include_directories(qsa_prompt_attn_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qsa_prompt_attn_parity PRIVATE strata_kernels CUDA::cudart)
  endif()
  # plan v0.3 P6: llama.cpp's block layouts and codebook grids for the i-quant formats (MIT, third_party/ggml)
  target_include_directories(strata_kernels PRIVATE third_party/ggml)
  # ---- P2: the NVFP4 marlin hot-expert tier (3080, sm_86).  OFF by default - Q2_0 builds are untouched.
  # The marlin sources live beside the engine checkout at ../third_party/marlin (strata-src layout) and the
  # sgl_kernel/tvm_ffi headers+lib come from the machine's sglang venv (the same paths P2前半's standalone
  # test compiled and linked against, bit-exact with the production JIT).
  option(STRATA_ENABLE_NVFP4 "NVFP4 marlin hot-expert tier (extracted marlin + tvm_ffi)" OFF)
  if(STRATA_ENABLE_NVFP4)
    if(NOT STRATA_ENABLE_CUDA)
      message(FATAL_ERROR "STRATA_ENABLE_NVFP4 needs STRATA_ENABLE_CUDA")
    endif()
    if(CMAKE_CUDA_ARCHITECTURES MATCHES "121")
      set(STRATA_SGL_CUDA_ARCH 1210)   # GB10 (Spark): probe-built and bit-exact vs the 3080 golden, 2026-10-04
    elseif(CMAKE_CUDA_ARCHITECTURES MATCHES "86")
      set(STRATA_SGL_CUDA_ARCH 860)    # 3080
    else()
      message(FATAL_ERROR "STRATA_ENABLE_NVFP4 is validated for sm_86 (SGL_CUDA_ARCH=860) and sm_121 (SGL_CUDA_ARCH=1210); got '${CMAKE_CUDA_ARCHITECTURES}'")
    endif()
    set(STRATA_SGL_KERNEL_INCLUDE "" CACHE PATH "sglang JIT include dir (the sgl_kernel headers)")
    set(STRATA_TVM_FFI_INCLUDE "" CACHE PATH "tvm_ffi include dir")
    set(STRATA_TVM_FFI_LIB "" CACHE FILEPATH "libtvm_ffi.so from the sglang venv")
    if(NOT STRATA_SGL_KERNEL_INCLUDE OR NOT STRATA_TVM_FFI_INCLUDE OR NOT STRATA_TVM_FFI_LIB)
      message(FATAL_ERROR "STRATA_ENABLE_NVFP4 needs -DSTRATA_SGL_KERNEL_INCLUDE=... -DSTRATA_TVM_FFI_INCLUDE=... -DSTRATA_TVM_FFI_LIB=...")
    endif()
    target_sources(strata_kernels PRIVATE
      src/kernels/cuda/nvfp4_marlin_expert.cu
      src/kernels/cuda/nvfp4_stream_expert.cu)
    set_source_files_properties(src/kernels/cuda/nvfp4_marlin_expert.cu PROPERTIES
      INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/marlin/gemm/marlin_moe;${CMAKE_CURRENT_SOURCE_DIR}/../third_party/marlin/gemm/marlin;${STRATA_SGL_KERNEL_INCLUDE};${STRATA_TVM_FFI_INCLUDE}"
      COMPILE_OPTIONS "-DSGL_CUDA_ARCH=${STRATA_SGL_CUDA_ARCH};--expt-relaxed-constexpr;--expt-extended-lambda")
    target_compile_definitions(strata_core PUBLIC STRATA_ENABLE_NVFP4)
    target_link_libraries(strata_kernels PUBLIC ${STRATA_TVM_FFI_LIB})
    if(STRATA_BUILD_TESTS)
      add_executable(test_nvfp4_stream_smallm tests/test_nvfp4_stream_smallm.cu)
      target_include_directories(test_nvfp4_stream_smallm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/marlin/gemm/marlin_moe
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/marlin/gemm/marlin
        ${STRATA_SGL_KERNEL_INCLUDE}
        ${STRATA_TVM_FFI_INCLUDE}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
      target_compile_options(test_nvfp4_stream_smallm PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-DSGL_CUDA_ARCH=${STRATA_SGL_CUDA_ARCH};--expt-relaxed-constexpr;--expt-extended-lambda>)
      target_link_libraries(test_nvfp4_stream_smallm PRIVATE strata_kernels ${_strata_gpu_runtime_target})
      add_test(NAME test_nvfp4_stream_smallm COMMAND test_nvfp4_stream_smallm)
    endif()
  endif()
  # ---- P3: the NVFP4 cold-expert tier (Spark GB10).  Self-contained CUDA (no marlin/tvm_ffi),
  # ON for CUDA builds: nothing calls the new entry points unless the worker is handed an NVFP4
  # pack, so Q2_0 behaviour is unchanged.  HIP builds skip it (the worker is CUDA-only anyway).
  option(STRATA_ENABLE_NVFP4_COLD "NVFP4 cold-expert grouped path (self-contained CUDA)" ON)
  if(STRATA_ENABLE_NVFP4_COLD AND STRATA_ENABLE_CUDA)
    target_sources(strata_kernels PRIVATE src/kernels/cuda/nvfp4_cold_grouped.cu)
    target_compile_definitions(strata_kernels PUBLIC STRATA_ENABLE_NVFP4_COLD)
  endif()
  # Match the pinned llama.cpp CUDA arithmetic contract for this adapter only.
  if(STRATA_ENABLE_CUDA)
    set_source_files_properties(src/kernels/cuda/native_mmvq.cu src/kernels/cuda/native_bf16.cu
      src/kernels/cuda/native_gr_norm.cu
      src/kernels/cuda/native_moe.cu
      src/kernels/cuda/native_gdn.cu
      src/kernels/cuda/native_gr_postops.cu
      src/kernels/cuda/native_router.cu
      src/kernels/cuda/native_gdn_preprocess.cu
      src/kernels/cuda/native_qsa.cu
      src/kernels/cuda/native_rope.cu
      src/kernels/cuda/native_ple_postops.cu
      src/kernels/cuda/native_flash_attn.cu
      src/kernels/cuda/native_qsa_indexer.cu
      src/kernels/cuda/native_qsa_score.cu
      PROPERTIES COMPILE_OPTIONS "--use_fast_math")
  elseif(STRATA_ENABLE_HIP)
    # Keep Q8_K's multiply and magic-bias rounding as separate FP32 operations on HIP; contraction to an
    # FMA changes tie results. The embedding gather also relies on separately rounded operations.
    set_source_files_properties(src/kernels/cuda/elementwise.cu src/kernels/cuda/quantize_act.cu
      PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  endif()

  set(STRATA_ORACLE_SOURCE_DIR "" CACHE PATH "Pinned llama.cpp source for independent CUDA parity tests")
  set(STRATA_ORACLE_BUILD_DIR "" CACHE PATH "Matching existing shared llama.cpp build")
  # This oracle is a CUDA ggml build; do not pull CUDA libraries into a HIP build.
  if(STRATA_ENABLE_CUDA AND STRATA_ORACLE_SOURCE_DIR AND STRATA_ORACLE_BUILD_DIR)
    foreach(parity IN ITEMS native_mmvq native_bf16 native_shared native_gr_norm native_moe native_gdn native_gr_postops native_router native_gdn_preprocess native_gdn_pipeline native_qsa native_rope native_ple_postops native_flash_attn native_qsa_indexer native_qsa_score)
    add_executable(${parity}_parity bench/micro/${parity}_parity.cpp)
    target_include_directories(${parity}_parity PRIVATE
      "${STRATA_ORACLE_SOURCE_DIR}/ggml/include" ${_strata_gpu_include_directories})
    target_compile_definitions(${parity}_parity PRIVATE GGML_SHARED GGML_BACKEND_SHARED)
    target_link_libraries(${parity}_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
    foreach(component IN ITEMS ggml ggml-base ggml-cpu ggml-cuda)
      string(REPLACE "-" "_" variable "STRATA_ORACLE_${component}")
      find_library(${variable} NAMES ${component} PATHS
        "${STRATA_ORACLE_BUILD_DIR}/ggml/src" "${STRATA_ORACLE_BUILD_DIR}/ggml/src/${component}"
        "${STRATA_ORACLE_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
      target_link_libraries(${parity}_parity PRIVATE "${${variable}}")
    endforeach()
    add_test(NAME ${parity}_parity COMMAND ${parity}_parity --selftest)
    if(WIN32)
      set_tests_properties(${parity}_parity PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${STRATA_ORACLE_BUILD_DIR}/bin")
    endif()
    endforeach()
  endif()

  # ---- remote_stage transport ----------------------------------------------------
  # The data plane is RC verbs only.  TCP is used solely to exchange QP attributes;
  # when libibverbs is absent the target remains buildable but its runtime selftest
  # fails closed instead of silently falling back to a socket data path.
  find_path(STRATA_VERBS_INCLUDE_DIR infiniband/verbs.h)
  find_library(STRATA_VERBS_LIBRARY NAMES ibverbs)
  add_library(strata_remote_stage STATIC src/core/remote_stage.cpp src/core/rdma_expert_tier.cpp)
  target_include_directories(strata_remote_stage PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(strata_remote_stage PUBLIC strata_warnings)
  if(STRATA_VERBS_INCLUDE_DIR AND STRATA_VERBS_LIBRARY)
    target_compile_definitions(strata_remote_stage PRIVATE STRATA_REMOTE_STAGE_VERBS=1)
    target_include_directories(strata_remote_stage PRIVATE ${STRATA_VERBS_INCLUDE_DIR})
    target_link_libraries(strata_remote_stage PUBLIC ${STRATA_VERBS_LIBRARY})
    message(STATUS "Strata: libibverbs enabled for remote stage")
  else()
    message(STATUS "Strata: libibverbs not found; remote stage runtime is fail-closed")
  endif()
  add_executable(strata_remote_stage_tool tools/remote_stage.cpp)
  target_link_libraries(strata_remote_stage_tool PRIVATE strata_remote_stage)
  set_target_properties(strata_remote_stage_tool PROPERTIES OUTPUT_NAME "strata-remote-stage")
  if(STRATA_BUILD_TESTS)
    enable_testing()
    add_test(NAME remote_stage_selftest COMMAND strata_remote_stage_tool --selftest)
    set_tests_properties(remote_stage_selftest PROPERTIES TIMEOUT 30)
  endif()

  # ---- the RDMA cold-expert worker (docs/RDMA_EXPERTS.md) -------------------------------
  # The Spark half of the cross-machine tier: polls the flag-ring doorbell, computes the
  # requested Q2_0 experts on the GPU out of the pack's experts.bin arena, publishes the
  # rows into the result ring.  Also builds the --probe sender used as the end-to-end gate.
  add_executable(strata_cold_expert_worker tools/cold_expert_worker.cpp)
  target_include_directories(strata_cold_expert_worker PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata_cold_expert_worker PRIVATE strata_remote_stage strata_kernels
                        ${_strata_gpu_runtime_target})
  set_target_properties(strata_cold_expert_worker PROPERTIES OUTPUT_NAME "strata-cold-expert-worker")

  # ---- strata_engine (P2.S5) -----------------------------------------------------
  # The COMPOSED layer lives here and not in `strata_core`, because it calls kernels and `strata_kernels`
  # already links `strata_core` - putting `gdn_layer` in core would make the two mutually dependent.  A third
  # library is the honest shape of the dependency graph: core knows about devices and weights, kernels know
  # about arithmetic, and this one knows the ORDER the arithmetic happens in.
  add_library(strata_engine STATIC src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp src/core/remote_experts.cpp
    src/core/expert_cache.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp src/core/mtp.cpp src/core/dflash.cu src/core/cold_expert_host.cpp)
  target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu strata_remote_stage ${_strata_gpu_runtime_target})
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/file_expert_source_test.cpp")
    enable_testing()
    add_executable(file_expert_source_test tests/core/file_expert_source_test.cpp)
    target_link_libraries(file_expert_source_test PRIVATE strata_engine)
    add_test(NAME file_expert_source_test COMMAND file_expert_source_test)
    set_tests_properties(file_expert_source_test PROPERTIES TIMEOUT 60)
  endif()

  # `strata generate` (P2.S6).  The FIRST user-facing program: everything under it is a component, and this is
  # what composes them into a token.  No tokenizer - the prompt arrives as ids - because Gate C1 compares logits
  # against llama.cpp on the SAME ids and a tokenizer on one side of that comparison is a second variable.
  # plan v0.3 P5: batched prompt processing (tensor-core GEMMs through cuBLAS).
  add_library(strata_prefill STATIC src/prefill/gemm.cu src/prefill/kernels.cu src/prefill/prefill.cpp)
  target_include_directories(strata_prefill PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  target_link_libraries(strata_prefill PUBLIC strata_engine ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})
  if(STRATA_ENABLE_HIP AND STRATA_HIPBLASLT_AVAILABLE)
    target_compile_definitions(strata_prefill PRIVATE STRATA_HIPBLASLT_AVAILABLE=1)
    target_link_libraries(strata_prefill PRIVATE roc::hipblaslt)

    add_executable(tune_hipblaslt tools/hip/tune_hipblaslt.cpp)
    target_link_libraries(tune_hipblaslt PRIVATE hip::host roc::hipblas roc::hipblaslt)
  endif()

  add_executable(strata src/program/generate.cpp)
  target_include_directories(strata PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec ${_strata_gpu_runtime_target})

  if(STRATA_ENABLE_HIP)
    enable_testing()
    set(_strata_hip_tests)
    add_executable(hip_expert_cache_staging tests/hip/expert_cache_staging.cpp)
    target_link_libraries(hip_expert_cache_staging PRIVATE strata_engine)
    add_test(NAME hip_expert_cache_staging COMMAND hip_expert_cache_staging)
    list(APPEND _strata_hip_tests hip_expert_cache_staging)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/intrinsics.cpp")
      add_executable(hip_intrinsics tests/hip/intrinsics.cpp)
      target_link_libraries(hip_intrinsics PRIVATE strata_hip_runtime)
      add_test(NAME hip_intrinsics COMMAND hip_intrinsics)
      list(APPEND _strata_hip_tests hip_intrinsics)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/handoff.cpp")
      add_executable(hip_handoff tests/hip/handoff.cpp)
      target_link_libraries(hip_handoff PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_handoff COMMAND hip_handoff)
      list(APPEND _strata_hip_tests hip_handoff)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/native_qsa_score.cpp")
      add_executable(hip_native_qsa_score tests/hip/native_qsa_score.cpp)
      target_link_libraries(hip_native_qsa_score PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_native_qsa_score COMMAND hip_native_qsa_score)
      list(APPEND _strata_hip_tests hip_native_qsa_score)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/prefill_native_batch.cpp")
      add_executable(hip_prefill_native_batch tests/hip/prefill_native_batch.cpp)
      target_link_libraries(hip_prefill_native_batch PRIVATE strata_kernels strata_hip_runtime)
      add_test(NAME hip_prefill_native_batch COMMAND hip_prefill_native_batch)
      list(APPEND _strata_hip_tests hip_prefill_native_batch)
    endif()
    if(STRATA_HIPBLASLT_AVAILABLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/hip/prefill_hipblaslt_gemm.cpp")
      add_executable(hip_prefill_hipblaslt_gemm tests/hip/prefill_hipblaslt_gemm.cpp)
      target_include_directories(hip_prefill_hipblaslt_gemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill")
      target_link_libraries(hip_prefill_hipblaslt_gemm PRIVATE strata_prefill strata_hip_runtime roc::hipblaslt roc::hipblas)
      add_test(NAME hip_prefill_hipblaslt_gemm COMMAND hip_prefill_hipblaslt_gemm)
      set_tests_properties(hip_prefill_hipblaslt_gemm PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
      list(APPEND _strata_hip_tests hip_prefill_hipblaslt_gemm)
    endif()
    if(_strata_hip_tests)
      set_tests_properties(${_strata_hip_tests} PROPERTIES TIMEOUT 60)
    endif()
    if(TARGET hip_prefill_native_batch)
      set_tests_properties(hip_prefill_native_batch PROPERTIES TIMEOUT 120)
    endif()
    if(TARGET hip_prefill_hipblaslt_gemm)
      set_tests_properties(hip_prefill_hipblaslt_gemm PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()
  endif()

  add_executable(dequant_s2_parity src/kernels/dequant_s2_parity.cpp)
  # The tests are HOST C++ files that call cudaMalloc/cudaMemcpy themselves, so they need the toolkit headers
  # and cudart explicitly: enabling the CUDA language gives those to .cu sources automatically, but a .cpp
  # target is just a C++ target and gets neither.
  target_include_directories(dequant_s2_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(dequant_s2_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME dequant_s2_parity COMMAND dequant_s2_parity --selftest)

  add_executable(s2_gemv_parity src/kernels/s2_gemv_parity.cpp)
  target_include_directories(s2_gemv_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s2_gemv_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s2_gemv_parity COMMAND s2_gemv_parity --selftest)

  add_executable(iq_parity src/kernels/iq_parity.cpp)
  target_include_directories(iq_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(iq_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_executable(shared_expert_parity src/kernels/shared_expert_parity.cpp)
  target_include_directories(shared_expert_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(shared_expert_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME shared_expert_parity COMMAND shared_expert_parity --selftest)

  add_executable(gr_parity src/kernels/gr_parity.cpp)
  target_include_directories(gr_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(gr_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME gr_parity COMMAND gr_parity --selftest)

  add_executable(gdn_parity src/kernels/gdn_parity.cpp)
  target_include_directories(gdn_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(gdn_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME gdn_parity COMMAND gdn_parity --selftest)

  add_executable(s2_gemv_q8_parity src/kernels/s2_gemv_q8_parity.cpp)
  target_include_directories(s2_gemv_q8_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s2_gemv_q8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s2_gemv_q8_parity COMMAND s2_gemv_q8_parity --selftest)

  add_executable(sampler_parity src/kernels/sampler_parity.cpp)
  target_include_directories(sampler_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(sampler_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME sampler_parity COMMAND sampler_parity --selftest)

  add_executable(rope_parity src/kernels/rope_parity.cpp)
  target_include_directories(rope_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(rope_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME rope_parity COMMAND rope_parity --selftest)

  add_executable(quantize_act_parity src/kernels/quantize_act_parity.cpp)
  target_include_directories(quantize_act_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(quantize_act_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME quantize_act_parity COMMAND quantize_act_parity --selftest)

  add_executable(router_top10_parity src/kernels/router_top10_parity.cpp)
  target_include_directories(router_top10_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(router_top10_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME router_top10_parity COMMAND router_top10_parity --selftest)

  add_executable(s_gemv_parity src/kernels/s_gemv_parity.cpp)
  target_include_directories(s_gemv_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s_gemv_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s_gemv_parity COMMAND s_gemv_parity --selftest)

  add_executable(elementwise_parity src/kernels/elementwise_parity.cpp)
  target_include_directories(elementwise_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(elementwise_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME elementwise_parity COMMAND elementwise_parity --selftest)

  add_executable(bf16_gemv_parity src/kernels/bf16_gemv_parity.cpp)
  target_include_directories(bf16_gemv_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(bf16_gemv_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME bf16_gemv_parity COMMAND bf16_gemv_parity --selftest)

  add_executable(s_gemv_q8k_parity src/kernels/s_gemv_q8k_parity.cpp)
  target_include_directories(s_gemv_q8k_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(s_gemv_q8k_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME s_gemv_q8k_parity COMMAND s_gemv_q8k_parity --selftest)

  add_executable(qsa_parity src/kernels/qsa_parity.cpp)
  target_include_directories(qsa_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(qsa_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME qsa_parity COMMAND qsa_parity --selftest)

  # ---- P2.S4: the PLE n-gram hash, the IQ4_NL table read, and the PLE block ----
  # `ngram.cpp` is host-only but lives in `strata_kernels` because `ple.cu` shares its geometry constants, and
  # splitting them across two libraries would only add a layer.
  #
  # The pack, the original GGUF and the ggml capture are all REQUIRED inputs: this test's oracles are real
  # artifacts, so the working directory is pinned to the source tree and a missing input is a hard error
  # INSIDE the test.  A test that passes by finding nothing to compare is the failure mode this project keeps
  # re-learning.
  add_executable(ple_parity src/kernels/ple_parity.cpp)
  target_include_directories(ple_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ple_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  set(STRATA_PLE_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro" CACHE PATH
      "Directory containing required ple_in.bin and ple_out.bin parity fixtures")
  add_test(NAME ple_parity COMMAND ple_parity --selftest
    --in "${STRATA_PLE_FIXTURE_DIR}/ple_in.bin" --out "${STRATA_PLE_FIXTURE_DIR}/ple_out.bin"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  # P4b: the PLE table's FP8 E4M3 row format (NVFP4 stack) - synthetic loopback, no model, no RDMA.
  add_executable(test_ple_fp8 tests/test_ple_fp8.cpp)
  target_include_directories(test_ple_fp8 PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(test_ple_fp8 PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME test_ple_fp8 COMMAND test_ple_fp8)

  # Plan v0.3 P7: INT8 KV storage vs a host reference and vs the FP16 path (GPU, synthetic, no model).
  add_executable(kv_q8_parity src/kernels/kv_q8_parity.cpp)
  target_include_directories(kv_q8_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_q8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_q8_parity COMMAND kv_q8_parity)

  # FP8 (e4m3) KV storage vs a host reference and vs the FP16 path (GPU, synthetic, no model).
  add_executable(kv_fp8_parity src/kernels/kv_fp8_parity.cpp)
  target_include_directories(kv_fp8_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_fp8_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_fp8_parity COMMAND kv_fp8_parity)

  # KV streaming (docs/kv-streaming-design.md): streamed vs resident attention, bitwise (GPU, synthetic, no model).
  add_executable(kv_stream_parity src/kernels/kv_stream_parity.cpp)
  target_include_directories(kv_stream_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_stream_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_stream_parity COMMAND kv_stream_parity)

  # PR #120: the hybrid K8V4 KV (INT8 K + rotated Q4_0 V) vs host references (GPU, synthetic, no model).
  add_executable(kv_hybrid_parity src/kernels/kv_hybrid_parity.cpp)
  target_include_directories(kv_hybrid_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_hybrid_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_hybrid_parity COMMAND kv_hybrid_parity)
  # PR #21: Q4_0 KV with Walsh-Hadamard rotation vs a host reference (GPU, synthetic, no model).
  add_executable(kv_q4_parity src/kernels/kv_q4_parity.cpp)
  target_include_directories(kv_q4_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(kv_q4_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME kv_q4_parity COMMAND kv_q4_parity)
  add_executable(cvec_parity src/kernels/cvec_parity.cpp)
  target_include_directories(cvec_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(cvec_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME cvec_parity COMMAND cvec_parity)

  # Plan v0.3 P0.1: working-set lock of a host region (CPU only).
  add_executable(platform_memory_test src/platform/memory_test.cpp)
  target_link_libraries(platform_memory_test PRIVATE strata_core ${_strata_gpu_runtime_target})
  add_test(NAME platform_memory_test COMMAND platform_memory_test)

  # Plan v0.3 P2: the n-gram table read straight from the SSD. The selftest needs only the CPU and a temp file.
  add_executable(ple_reader_test src/ngram/ple_reader_test.cpp)
  target_include_directories(ple_reader_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ple_reader_test PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  add_test(NAME ple_reader_selftest COMMAND ple_reader_test --selftest)

  # The GPU test runs ON THE TARGET PC. `--selftest` allocates
  # and writes real VRAM, so a machine without a device fails this test rather than skipping it silently -
  # which is correct: an engine that targets one GPU should not report a pass for having no GPU.
  if(STRATA_ENABLE_CUDA)
    add_test(NAME cuda_device_selftest COMMAND strata-device --selftest)
  else()
    add_test(NAME hip_device_selftest COMMAND strata-device --selftest)
  endif()

  # ---- P2.S5: the GraphRegistry --------------------------------------------------
  # A .cu because the test needs real kernels to capture; it links `strata_core` and not `strata_kernels`,
  # because what it exercises is the capture machinery and not any kernel in it.
  if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/graph_registry_test.cu")
    add_executable(graph_registry_test bench/micro/graph_registry_test.cu)
    target_include_directories(graph_registry_test PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(graph_registry_test PRIVATE strata_core ${_strata_gpu_runtime_target})
    add_test(NAME graph_registry_test COMMAND graph_registry_test)
  endif()

  # ---- P2.S5's GATE: is a device<->pinned-memory handoff graph-capturable? --------------------
  # The doorbell protocol puts the D2H of `x_f` and the miss list INSIDE G[l] - that is the whole overlap.  The
  # phase doc says to verify this "before building on it", and this project has already lost two rounds to
  # building first (a memory-only spin that DEADLOCKS on Windows, and a `cudaEventRecord` inside a capture that
  # is SILENTLY DROPPED).  It links only cudart: it is about the capture machinery, not about any kernel.
  if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/pinned_capture.cu")
    add_executable(pinned_capture bench/micro/pinned_capture.cu)
    target_include_directories(pinned_capture PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(pinned_capture PRIVATE ${_strata_gpu_runtime_target})
    add_test(NAME pinned_capture COMMAND pinned_capture)
  endif()

  # Tells tests/CMakeLists.txt to add the CUDA-dependent Catch2 target.  It cannot be declared there
  # unconditionally: tests/ is added at the end of this file, but the flag has to be known before it runs.
  if(STRATA_ENABLE_CUDA)
    set(STRATA_CUDA_TESTS ON)
  endif()
endif()

# ---- strata_kernels_cpu (P2.S3) ----
#
# A SEPARATE LIBRARY from `strata_kernels`, and `/arch:AVX512` is set on the SOURCE rather than on the target
# so that the flag applies only to the TU that needs it.  Both parts matter:
#
#   * the CPU expert kernel uses `_mm256_dpbusd_epi32` (AVX512-VNNI) and `_mm256_multishift_epi64_epi8`
#     (AVX512-VBMI); without the flag MSVC reports undeclared intrinsics and never mentions AVX-512;
#   * a TU compiled WITH `/arch:AVX512` may use AVX-512 in ANY of its code, so the scalar fallback and the
#     test drivers must NOT be in it - otherwise the fallback is not a fallback, it is the same bet with
#     worse opt-out.  `cpu_require_expert_support()` therefore refuses at startup on a CPU without the
#     features, and the fallback exists for tests, exactly as P2.S3 specifies.
# Plan v0.3 P6: speculation components that need no GPU.
add_library(strata_spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp)
target_include_directories(strata_spec PUBLIC include)
if(STRATA_BUILD_TESTS)
  add_executable(suffix_drafter_test src/spec/suffix_drafter_test.cpp)
  target_link_libraries(suffix_drafter_test PRIVATE strata_spec)
  add_test(NAME suffix_drafter_test COMMAND suffix_drafter_test)
  add_executable(controller_test src/spec/controller_test.cpp)
  target_link_libraries(controller_test PRIVATE strata_spec)
  add_test(NAME controller_test COMMAND controller_test)
  add_executable(draft_policy_test src/spec/draft_policy_test.cpp)
  target_link_libraries(draft_policy_test PRIVATE strata_spec)
  add_test(NAME draft_policy_test COMMAND draft_policy_test)
  # the serve loop's conversation-cache retention policy (header-only, no library to link)
  add_executable(conv_cache_test src/program/conv_cache_test.cpp)
  target_include_directories(conv_cache_test PRIVATE include)
  add_test(NAME conv_cache_test COMMAND conv_cache_test)
endif()

add_library(strata_kernels_cpu STATIC src/kernels/cpu/pool.cpp src/kernels/cpu/expert_layout.cpp)
# The AVX-512/AVX2 expert kernels are x86-only.  On aarch64 (DGX Spark GB10) the engine runs GPU-only
# (--no-pool, every expert in the VRAM cache), so the kernels are replaced by stubs that refuse work loudly
# instead of failing to compile.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86|i386)$" OR WIN32)
  set(STRATA_X86 ON)
else()
  set(STRATA_X86 OFF)
endif()
if(STRATA_X86)
  target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/expert.cpp src/kernels/cpu/q2_avx2.cpp)
else()
  target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/arm_stubs.cpp)
endif()
target_include_directories(strata_kernels_cpu PUBLIC include)
# Plan v0.3 P6: only the kernel files get the AVX-512 / AVX2 flags; the pool and the probe run on any x86-64,
# and the engine picks the kernels at run time (cpu_avx512_ok).
if(STRATA_X86)
if(MSVC)
  set_source_files_properties(src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  # GCC/Clang spell it differently, and `-mavx512vnni` alone is not enough: VBMI is a separate feature bit.
  set_source_files_properties(src/kernels/cpu/expert.cpp
                              PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()
endif()
find_package(Threads REQUIRED)
target_link_libraries(strata_kernels_cpu PUBLIC Threads::Threads)

# ---- plan v0.3 P6: the i-quant models' CPU experts (IQ2_XS / IQ3_XXS files) ----
#
# The CPU computes a native expert with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so an IQ
# weight means exactly what it means in llama.cpp.  ggml is built static from a llama.cpp checkout: a local one
# with -DSTRATA_GGML_DIR=..., otherwise the pinned upstream commit (third_party/ggml/VERSION.txt).
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
if(STRATA_PREFILL_MMQ AND NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_NATIVE_EXPERTS=ON for GGML sources")
endif()
if(STRATA_NATIVE_EXPERTS)
  enable_language(C)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  if(STRATA_PORTABLE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
  else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  endif()
  if(STRATA_GGML_DIR)
    add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(strata_llamacpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
      GIT_SHALLOW FALSE)
    FetchContent_GetProperties(strata_llamacpp)
    if(NOT strata_llamacpp_POPULATED)
      FetchContent_Populate(strata_llamacpp)
    endif()
    add_subdirectory(${strata_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  endif()
  # part of strata_kernels_cpu (the pool calls these and they call its CPU probe: one library, so GNU ld's
  # one-pass static linking has no cycle to resolve)
  target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp)
  if(STRATA_X86)
    target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/iq_avx2.cpp)
  endif()
  target_include_directories(strata_kernels_cpu PRIVATE third_party/ggml)
  if(STRATA_X86)
  if(MSVC)
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  else()
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  endif()
  endif()
  target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
  if(STRATA_ENABLE_CUDA)
    # prompt-speed plan step 2b: llama.cpp's MMQ kernels (ggml-cuda, MIT) for the prompt path's experts, compiled
    # from the same checkout as ggml above - only mmq.cuh's instances for the packs' types and the quantizer, with
    # the few host symbols of ggml-cuda.cu they use in src/prefill/ggml_cuda_host.cu
    if(STRATA_GGML_DIR)
      set(_strata_gg ${STRATA_GGML_DIR}/ggml)
    else()
      set(_strata_gg ${strata_llamacpp_SOURCE_DIR}/ggml)
    endif()
    set(_strata_mmq ${_strata_gg}/src/ggml-cuda)
    add_library(strata_mmq STATIC src/prefill/moe_mmq.cu src/prefill/ggml_cuda_host.cu ${_strata_mmq}/quantize.cu)
    foreach(_t q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs)
      target_sources(strata_mmq PRIVATE ${_strata_mmq}/template-instances/mmq-instance-${_t}.cu)
    endforeach()
    target_include_directories(strata_mmq PRIVATE ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
                               PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    target_link_libraries(strata_mmq PUBLIC ggml-base ${_strata_gpu_runtime_target})
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1)
  elseif(STRATA_PREFILL_MMQ)
    # Compile only the pinned GGML MMQ and activation-quantizer sources used by prefill. This uses
    # common.cuh's native HIP path without enabling a second GGML HIP backend.
    if(STRATA_GGML_DIR)
      set(_strata_gg "${STRATA_GGML_DIR}/ggml")
    else()
      set(_strata_gg "${strata_llamacpp_SOURCE_DIR}/ggml")
    endif()
    set(_strata_mmq "${_strata_gg}/src/ggml-cuda")
    if(NOT EXISTS "${_strata_mmq}/common.cuh" OR NOT EXISTS "${_strata_mmq}/quantize.cu")
      message(FATAL_ERROR "The selected GGML source is missing the ggml-cuda MMQ sources: ${_strata_mmq}")
    endif()
    set(_strata_mmq_sources
      src/prefill/moe_mmq.cu
      src/prefill/ggml_cuda_host.cu
      "${_strata_mmq}/quantize.cu")
    foreach(_t q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs)
      list(APPEND _strata_mmq_sources "${_strata_mmq}/template-instances/mmq-instance-${_t}.cu")
    endforeach()
    set_source_files_properties(${_strata_mmq_sources} PROPERTIES LANGUAGE HIP)
    add_library(strata_mmq STATIC ${_strata_mmq_sources})
    target_include_directories(strata_mmq
      PRIVATE "${_strata_gg}/include" "${_strata_gg}/src" "${_strata_mmq}"
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(strata_mmq PRIVATE GGML_USE_HIP=1)
    target_link_libraries(strata_mmq PUBLIC ggml-base hip::host roc::hipblas)
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1)

    # Numerical diagnostic: synthetic layouts by default, optional first-expert IQ3_XXS/IQ4_NL check with a pack.
    add_executable(hip_prefill_mmq_parity tests/hip/prefill_mmq_parity.cpp)
    target_include_directories(hip_prefill_mmq_parity PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_mmq_parity PRIVATE strata_prefill)
    add_test(NAME hip_prefill_mmq_parity COMMAND hip_prefill_mmq_parity)
    set_tests_properties(hip_prefill_mmq_parity PROPERTIES TIMEOUT 120)
    list(APPEND _strata_hip_tests hip_prefill_mmq_parity)
  endif()
  if(STRATA_ENABLE_HIP)
    # Optional real-model diagnostic: pass the GGUF shard containing the native PLE key.
    add_executable(hip_ple_iq4 tests/hip/ple_iq4.cpp)
    target_link_libraries(hip_ple_iq4 PRIVATE strata_kernels strata_hip_runtime ggml-base)
  endif()
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)
    add_executable(native_expert_parity src/kernels/native_expert_parity.cpp)
    target_include_directories(native_expert_parity PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_expert_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
  endif()
endif()

if(STRATA_BUILD_TESTS)
  # Plan v0.3 P6: one expert applied to several tokens, bitwise vs the single-token kernel (synthetic, CPU only).
  # plan v0.3 P5: the device dequantizers vs the artifact's CPU dequantizers on the real shards (not a ctest: needs
  # the 38 GB model; run it by hand or from tools/gpu_session.py)
  if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)   # #93: it needs the device dequantizers and a GPU runtime
    add_executable(dequant_bf16_test src/kernels/dequant_bf16_test.cpp)
    target_link_libraries(dequant_bf16_test PRIVATE strata_kernels strata_artifact ${_strata_gpu_runtime_target})
  endif()

  add_executable(expert_multi_test src/kernels/cpu/expert_multi_test.cpp)
  target_link_libraries(expert_multi_test PRIVATE strata_kernels_cpu)
  add_test(NAME expert_multi_test COMMAND expert_multi_test)

  add_executable(expert_parity src/kernels/cpu/expert_parity.cpp)
  target_link_libraries(expert_parity PRIVATE strata_kernels_cpu)
  # The pack is a REQUIRED input, not an optional one: this test compares against a real expert blob, and a
  # missing file would otherwise be reported as a pass by a test that skipped everything.
  add_test(NAME expert_parity COMMAND expert_parity --selftest
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(pool_test src/kernels/cpu/pool_test.cpp)
  target_link_libraries(pool_test PRIVATE strata_kernels_cpu)
  add_test(NAME pool_test COMMAND pool_test --selftest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  # issue #29: tiny batches with sleeping workers, as on a big-VRAM card; a hang fails it
  add_executable(pool_stress src/kernels/cpu/pool_stress.cpp)
  target_link_libraries(pool_stress PRIVATE strata_kernels_cpu)
  add_test(NAME pool_stress COMMAND pool_stress 20)

  if(UNIX AND NOT APPLE)
    add_executable(direct_file_async_test tests/platform/direct_file_async.cpp src/platform/direct_file.cpp)
    target_include_directories(direct_file_async_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(direct_file_async_test PRIVATE Threads::Threads)
    add_test(NAME direct_file_async_test COMMAND direct_file_async_test)
  endif()
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  # Model-level checks may be absent from source distributions.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
  endif()
endif()

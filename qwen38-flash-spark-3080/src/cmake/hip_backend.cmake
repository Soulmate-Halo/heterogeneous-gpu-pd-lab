# Opt-in HIP configuration. Strata's CUDA-shaped kernels currently target RDNA3 gfx1100 wave32.
# CMake/compiler discovery stays machine-independent; pass CMAKE_HIP_COMPILER when it is not on PATH.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  set(CMAKE_HIP_ARCHITECTURES gfx1100 CACHE STRING "Strata HIP target architecture")
endif()
if(NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1100")
  message(FATAL_ERROR
    "Strata HIP currently supports only gfx1100 wave32; CMAKE_HIP_ARCHITECTURES is '${CMAKE_HIP_ARCHITECTURES}'")
endif()

enable_language(HIP)
find_package(hip CONFIG REQUIRED)
find_package(hipblas CONFIG REQUIRED)
find_package(hipblaslt CONFIG QUIET)

if(NOT TARGET hip::host)
  message(FATAL_ERROR "The ROCm hip CMake package did not provide hip::host")
endif()
if(NOT TARGET roc::hipblas)
  message(FATAL_ERROR "The ROCm hipblas CMake package did not provide roc::hipblas")
endif()
if(TARGET roc::hipblaslt)
  set(STRATA_HIPBLASLT_AVAILABLE ON)
else()
  set(STRATA_HIPBLASLT_AVAILABLE OFF)
  message(STATUS "Strata: hipBLASLt not found; solution-table dispatch is unavailable")
endif()

# HIP's link step produces a PIE; make Strata and ggml objects PIC for the ROCm linker.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(STRATA_HIP_COMPAT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/hip_compat")
add_library(strata_hip_runtime INTERFACE)
target_include_directories(strata_hip_runtime BEFORE INTERFACE
  "${STRATA_HIP_COMPAT_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(strata_hip_runtime INTERFACE STRATA_USE_HIP=1)
target_link_libraries(strata_hip_runtime INTERFACE hip::host)
foreach(_language IN ITEMS CXX HIP)
  target_compile_options(strata_hip_runtime INTERFACE
    "$<$<COMPILE_LANGUAGE:${_language}>:-include>"
    "$<$<COMPILE_LANGUAGE:${_language}>:${STRATA_HIP_COMPAT_INCLUDE_DIR}/cuda_runtime.h>")
endforeach()

# CMake does not infer HIP from Strata's existing CUDA-shaped .cu suffixes.
file(GLOB_RECURSE _strata_hip_sources CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu")
if(_strata_hip_sources)
  set_source_files_properties(${_strata_hip_sources} PROPERTIES LANGUAGE HIP)
endif()
foreach(_source IN ITEMS tests/hip/intrinsics.cpp tests/hip/native_qsa_score.cpp)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
    set_source_files_properties("${_source}" PROPERTIES LANGUAGE HIP)
  endif()
endforeach()

message(STATUS "Strata: HIP enabled, arch ${CMAKE_HIP_ARCHITECTURES}")

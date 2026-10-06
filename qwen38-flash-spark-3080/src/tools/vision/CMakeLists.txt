# tools/vision - strata-vision, the image encoder for Strata's multimodal path (optional; setup.py builds it when
# the vision encoder is chosen).  Uses llama.cpp's mtmd library at the same pinned commit as the engine's ggml.
#
#   cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp
#         [-DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native]
#   cmake --build build-vision --target strata-vision
cmake_minimum_required(VERSION 3.21)
project(strata_vision C CXX)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp" CACHE PATH "llama.cpp checkout")
option(STRATA_VISION_CUDA "run the vision encoder on the GPU (builds ggml-cuda)" OFF)
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
if(STRATA_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)        # no vcomp140.dll / libgomp needed on the user's PC
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} ON CACHE BOOL "" FORCE)
  endforeach()
else()
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
endif()
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${STRATA_VISION_CUDA} CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)

add_executable(strata-vision strata_vision.cpp)
target_link_libraries(strata-vision PRIVATE mtmd llama)
target_compile_features(strata-vision PRIVATE cxx_std_17)
set_target_properties(strata-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

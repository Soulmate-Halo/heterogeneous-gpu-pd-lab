#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

// The FP32-activation MMVF implementation below is adapted from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, ggml/src/ggml-cuda/{mmvf.cu,common.cuh}.
// Scope: ordinary contiguous BF16 matrix, one FP32 activation vector, no fusion/ids/channels.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
__device__ __forceinline__ float mmvf_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}

template <int BLOCK_SIZE>
__global__ void bf16_f32_mmvf_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                    float* __restrict__ y, int n_in) {
    const int t = threadIdx.x;
    const uint16_t* row = w + (size_t) blockIdx.x * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    const float2* inputs2 = reinterpret_cast<const float2*>(x);
    __shared__ float partials[32];
    if constexpr (BLOCK_SIZE > 32) {
        if (t < 32) partials[t] = 0.0f;
        __syncthreads();
    }
    float acc = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const float2 input = inputs2[pair];
        // Match the two ordered multiply-adds in ggml_cuda_mad, not a pair sum followed by one add.
        acc = __fmaf_rn(f32_from_bf16((uint16_t) weight), input.x, acc);
        acc = __fmaf_rn(f32_from_bf16((uint16_t) (weight >> 16)), input.y, acc);
    }
    acc = mmvf_warp_sum(acc);
    if constexpr (BLOCK_SIZE > 32) {
        // All lanes have the same reduced value; one store avoids a same-value shared-memory race.
        if ((t & 31) == 0) partials[t / 32] = acc;
        __syncthreads();
        if (t < 32) acc = mmvf_warp_sum(partials[t]);
    }
    if (t == 0) y[blockIdx.x] = acc;
}

// the same kernel for up to 8 activation rows - the weight row is read ONCE and every
// token keeps its own accumulator with exactly the single-row kernel's order (pairs, two ordered FMAs, the same warp
// and block reductions), so each output is bit-identical to a bf16_f32_mmvf_kernel launch of its own.
template <int BLOCK_SIZE, int NT>
__global__ void bf16_f32_mmvf_multi_kernel(const float* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ w,
                                          float* __restrict__ y, int64_t ldy, int n_in, int n_tok) {
    const int t = threadIdx.x;
    const uint16_t* row = w + (size_t) blockIdx.x * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    __shared__ float partials[NT][32];
    if constexpr (BLOCK_SIZE > 32) {
        if (t < 32)
#pragma unroll
            for (int k = 0; k < NT; ++k) partials[k][t] = 0.0f;
        __syncthreads();
    }
    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k < n_tok) {
                const float2 input = reinterpret_cast<const float2*>(x + (size_t) k * ldx)[pair];
                acc[k] = __fmaf_rn(w0, input.x, acc[k]);
                acc[k] = __fmaf_rn(w1, input.y, acc[k]);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = mmvf_warp_sum(acc[k]);
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0)
#pragma unroll
            for (int k = 0; k < NT; ++k) partials[k][t / 32] = acc[k];
        __syncthreads();
        if (t < 32)
#pragma unroll
            for (int k = 0; k < NT; ++k) acc[k] = mmvf_warp_sum(partials[k][t]);
    }
    if (t == 0)
#pragma unroll
        for (int k = 0; k < NT; ++k)
            if (k < n_tok) y[(size_t) k * ldy + blockIdx.x] = acc[k];
}

int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}

}  // namespace

void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                               int64_t n_in, int64_t n_out, int n_tok, void* stream) {
    if (n_tok == 1 && ldy >= n_out) { bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, stream); return; }
    if (n_tok < 1 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || n_out <= 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf_multi: 1..8 rows, even n_in/ldx, aligned pointers");
    const cudaStream_t st = (cudaStream_t) stream;
#define STRATA_MMVF_M(N) case N: \
    if (n_tok <= 4) bf16_f32_mmvf_multi_kernel<N, 4><<<(unsigned) n_out, N, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, n_tok); \
    else bf16_f32_mmvf_multi_kernel<N, 8><<<(unsigned) n_out, N, 0, st>>>(x, ldx, w, y, ldy, (int) n_in, n_tok); break
    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_M(32); STRATA_MMVF_M(64); STRATA_MMVF_M(96); STRATA_MMVF_M(128);
        STRATA_MMVF_M(160); STRATA_MMVF_M(192); STRATA_MMVF_M(224); STRATA_MMVF_M(256);
    }
#undef STRATA_MMVF_M
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf_multi launch: ") + cudaGetErrorString(result));
}

void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y,
                         int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_in > std::numeric_limits<int>::max() ||
        n_out <= 0 || n_out > std::numeric_limits<int>::max())
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out <= INT_MAX");
    if (x == nullptr || w == nullptr || y == nullptr ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(y) & 3u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null or misaligned pointer");
    const cudaStream_t st = (cudaStream_t) stream;
#define STRATA_MMVF_CASE(N) case N: \
    bf16_f32_mmvf_kernel<N><<<(unsigned) n_out, N, 0, st>>>(x, w, y, (int) n_in); break
    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_CASE(32);
        STRATA_MMVF_CASE(64);
        STRATA_MMVF_CASE(96);
        STRATA_MMVF_CASE(128);
        STRATA_MMVF_CASE(160);
        STRATA_MMVF_CASE(192);
        STRATA_MMVF_CASE(224);
        STRATA_MMVF_CASE(256);
    }
#undef STRATA_MMVF_CASE
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("bf16_gemv_fp32_mmvf launch: ") + cudaGetErrorString(result));
}


}  // namespace strata::kernels

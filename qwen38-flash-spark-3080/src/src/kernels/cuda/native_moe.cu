// Arithmetic adapted from the MIT-licensed pinned ggml CUDA
// moe-weighted-reduction.cu at 3cf03257f219afbe7334045ff7c6a06ac68c627d.
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
#include "strata/kernels/native_moe.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
__global__ void combine(const float* __restrict__ parts, const float* __restrict__ weights,
                        const float* __restrict__ shared, float* __restrict__ output,
                        int64_t n_embd, int k) {
    // blockIdx.y = the token of a multi-token launch (0 for the single one)
    const int64_t tk = blockIdx.y;
    parts += tk * k * n_embd; weights += tk * k; if (shared) shared += tk * n_embd; output += tk * n_embd;
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;
    float sum = parts[col] * weights[0];
    for (int expert = 1; expert < k; ++expert) {
        sum += parts[int64_t(expert) * n_embd + col] * weights[expert];
    }
    if (shared) sum += shared[col];
    output[col] = sum;
}
bool valid_span(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % alignof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}
void native_moe_combine_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_moe_combine_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_moe_combine(const float* parts, const float* weights, const float* shared,
                        float* output, int64_t n_embd, int64_t k, void* stream) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > 15)
        throw std::invalid_argument("native MoE combine requires a stream, positive width and 1..15 experts");
    const size_t row_bytes = size_t(n_embd) * sizeof(float);
    const size_t part_bytes = row_bytes * size_t(k), weight_bytes = size_t(k) * sizeof(float);
    if (!valid_span(parts, part_bytes) || !valid_span(weights, weight_bytes) || !valid_span(output, row_bytes)
            || (shared && !valid_span(shared, row_bytes))
            || overlap(output, row_bytes, parts, part_bytes)
            || overlap(output, row_bytes, weights, weight_bytes)
            || (shared && overlap(output, row_bytes, shared, row_bytes)))
        throw std::invalid_argument("native MoE combine requires aligned spans and disjoint output");
    combine<<<unsigned((n_embd + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        parts, weights, shared, output, n_embd, int(k));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("native MoE combine (multi) requires a stream, width, 1..15 experts, tokens");
    combine<<<dim3(unsigned((n_embd + 255) / 256), (unsigned) n_tok), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        parts, weights, shared, output, n_embd, int(k));
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}

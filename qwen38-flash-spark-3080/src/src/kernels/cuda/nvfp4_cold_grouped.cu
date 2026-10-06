// src/kernels/cuda/nvfp4_cold_grouped.cu - P3 baseline: grouped NVFP4 cold experts on the Spark GB10.
//
// **SHAPE.** Mirrors moe_grouped_s2's launch contract (s2_expert_grouped.cu) so the worker's
// host-side group assembly serves both formats unchanged: one gu pass, one swiglu pass, one down
// pass, three launches per window.  The per-row math is a warp-per-row dot product with in-kernel
// dequant (nibble -> e2m1, fp8 scale -> e4m3, fp32 ws2), fp32 FMA accumulation:
//
//   1. `gu_nvfp4_kernel`      rows 0..2*FF of one group (row < FF: gate plane, else up plane)
//   2. `swiglu_nvfp4_kernel`  h[e][i] = silu(gu[e][i]) * gu[e][FF + i]   (expf, the accurate one)
//   3. `down_nvfp4_kernel`    out[dst][r] = down_w[r] . h[e]
//
// **WHY WARP-PER-ROW WITH SERIAL ENTRY CHUNKS.**  The worker emits groups of at most 8 entries
// (WGMAX in cold_expert_worker.cpp - many small groups keep the GB10's latency-bound scheduler
// fed, measured 3-10x over whole-window big groups on the Q2 path).  Each lane owns two weight
// bytes (four k elements) per iteration, so the entry loop is a fixed 8-wide register chunk:
// weights are read once per 8 entries, and a group larger than 8 falls through the same loop in
// chunks - correct at any size, fast at the size that ships.
//
// Baseline = correctness first (P3 contract); the reads are already the minimum the format allows
// (2.4 MB of weight bytes per expert pass vs 1.35 MB for Q2_0 - the price of 4.5-bit weights at
// 16-element groups, paid once per expert per window, not per token).
#include "strata/kernels/nvfp4_cold.hpp"

#include <cstdio>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;          // 8 warps: one output row each per pass
constexpr int WARPS = THREADS / 32;
constexpr int ECHUNK = 8;             // entries served per weight read (the worker's WGMAX)

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// One block = WARPS rows of ONE group.  lane handles weight bytes {2*lane, 2*lane+1} + 64*iter,
// i.e. four consecutive k per iteration, so both the weight stream (coalesced bytes) and the
// activation stream (4 contiguous floats per lane) are cache-friendly.  Both bytes sit in the
// same 16-element scale group (2*lane % 8 is even), one scale fetch covers all four k.
__global__ void __launch_bounds__(THREADS) gu_nvfp4_kernel(
        const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
        const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_tok,
        const float* __restrict__ x, float* __restrict__ gu) {
    const int g = blockIdx.y;
    if (g >= n_groups[0]) return;
    const int start = grp_start[g], ne = grp_start[g + 1] - start;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const float gate_ws2 = ((const float*) (blob + NVFP4_OFF_GLOBAL))[0];
    const float up_ws2 = ((const float*) (blob + NVFP4_OFF_GLOBAL))[2];
    const int row = blockIdx.x * WARPS + (threadIdx.x >> 5);   // 0 .. 2*FF
    const int lane = threadIdx.x & 31;
    const bool is_gate = row < NVFP4_FF;
    const int lrow = is_gate ? row : row - NVFP4_FF;
    const uint8_t* __restrict__ wrow =
        (is_gate ? blob + NVFP4_OFF_GATE_W : blob + NVFP4_OFF_UP_W) + (int64_t) lrow * (NVFP4_H / 2);
    const uint8_t* __restrict__ srow =
        (is_gate ? blob + NVFP4_OFF_GATE_S : blob + NVFP4_OFF_UP_S) + (int64_t) lrow * (NVFP4_H / NVFP4_GROUP);
    const float ws2 = is_gate ? gate_ws2 : up_ws2;

    for (int e0 = 0; e0 < ne; e0 += ECHUNK) {
        const int ec = min(ne - e0, ECHUNK);
        float acc[ECHUNK] = {};
        for (int b = 2 * lane; b < NVFP4_H / 2; b += 64) {
            const uint32_t lo = wrow[b], hi = wrow[b + 1];
            const float scale = nvfp4_e4m3_f32(srow[b >> 3]) * ws2;
            const float w0 = nvfp4_e2m1_f32(lo & 15u) * scale;
            const float w1 = nvfp4_e2m1_f32(lo >> 4) * scale;
            const float w2 = nvfp4_e2m1_f32(hi & 15u) * scale;
            const float w3 = nvfp4_e2m1_f32(hi >> 4) * scale;
            const int k = 2 * b;
#pragma unroll
            for (int e = 0; e < ECHUNK; ++e) {
                if (e >= ec) break;
                const float* xr = x + (int64_t) ent_tok[start + e0 + e] * NVFP4_H + k;
                acc[e] = fmaf(w0, __ldg(xr), fmaf(w1, __ldg(xr + 1), fmaf(w2, __ldg(xr + 2),
                                                                    fmaf(w3, __ldg(xr + 3), acc[e]))));
            }
        }
#pragma unroll
        for (int e = 0; e < ECHUNK; ++e) {
            if (e >= ec) break;
            const float total = warp_sum(acc[e]);
            if (lane == 0) gu[(int64_t) (start + e0 + e) * (2 * NVFP4_FF) + row] = total;
        }
    }
}

// h[e][i] = silu(gu[e][i]) * gu[e][FF + i];  expf matches the Q2 swiglu's choice of the accurate
// exponential (its __expf sibling lives in the cpu-order path) - and the oracle's torch silu is
// the accurate one too.
__global__ void swiglu_nvfp4_kernel(const float* __restrict__ gu, float* __restrict__ h,
                                    long long n_pairs) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pairs) return;
    const long long e = i / NVFP4_FF, c = i % NVFP4_FF;
    const float g = gu[e * (2 * NVFP4_FF) + c];
    const float u = gu[e * (2 * NVFP4_FF) + NVFP4_FF + c];
    h[i] = g / (1.0f + expf(-g)) * u;
}

__global__ void __launch_bounds__(THREADS) down_nvfp4_kernel(
        const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
        const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_dst,
        const float* __restrict__ h, float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= n_groups[0]) return;
    const int start = grp_start[g], ne = grp_start[g + 1] - start;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const float ws2 = ((const float*) (blob + NVFP4_OFF_GLOBAL))[4];
    const int row = blockIdx.x * WARPS + (threadIdx.x >> 5);   // 0 .. H
    const int lane = threadIdx.x & 31;
    const uint8_t* __restrict__ wrow = blob + NVFP4_OFF_DOWN_W + (int64_t) row * (NVFP4_FF / 2);
    const uint8_t* __restrict__ srow = blob + NVFP4_OFF_DOWN_S + (int64_t) row * (NVFP4_FF / NVFP4_GROUP);

    for (int e0 = 0; e0 < ne; e0 += ECHUNK) {
        const int ec = min(ne - e0, ECHUNK);
        float acc[ECHUNK] = {};
        for (int b = 2 * lane; b < NVFP4_FF / 2; b += 64) {
            const uint32_t lo = wrow[b], hi = wrow[b + 1];
            const float scale = nvfp4_e4m3_f32(srow[b >> 3]) * ws2;
            const float w0 = nvfp4_e2m1_f32(lo & 15u) * scale;
            const float w1 = nvfp4_e2m1_f32(lo >> 4) * scale;
            const float w2 = nvfp4_e2m1_f32(hi & 15u) * scale;
            const float w3 = nvfp4_e2m1_f32(hi >> 4) * scale;
            const int k = 2 * b;
#pragma unroll
            for (int e = 0; e < ECHUNK; ++e) {
                if (e >= ec) break;
                const float* hr = h + (int64_t) (start + e0 + e) * NVFP4_FF + k;
                acc[e] = fmaf(w0, hr[0], fmaf(w1, hr[1], fmaf(w2, hr[2], fmaf(w3, hr[3], acc[e]))));
            }
        }
#pragma unroll
        for (int e = 0; e < ECHUNK; ++e) {
            if (e >= ec) break;
            const float total = warp_sum(acc[e]);
            if (lane == 0) out[(int64_t) ent_dst[start + e0 + e] * NVFP4_H + row] = total;
        }
    }
}

__global__ void q8_act_dequant_kernel(const uint8_t* __restrict__ blocks,
                                      const float* __restrict__ scales, float* __restrict__ x,
                                      long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long b = i >> 5;
    x[i] = (float) (int8_t) blocks[b * 34 + 2 + (i & 31)] * scales[b];
}

void check(const char* what, void* stream) {
    const cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(st));
        std::exit(1);
    }
    (void) stream;
}

}  // namespace

uint64_t moe_grouped_nvfp4_scratch_bytes(int64_t cap_entries) {
    const uint64_t gu = ((uint64_t) cap_entries * (2 * NVFP4_FF) * sizeof(float) + 255) & ~255ull;
    const uint64_t h = ((uint64_t) cap_entries * NVFP4_FF * sizeof(float) + 255) & ~255ull;
    return gu + h;
}

void moe_grouped_nvfp4(const unsigned long long* grp_ptr, const int32_t* grp_start,
                       const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok,
                       int64_t cap_groups, int64_t cap_entries, const float* x, void* scratch,
                       float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    cudaStream_t cs = (cudaStream_t) stream;
    const uint64_t gu_bytes = ((uint64_t) cap_entries * (2 * NVFP4_FF) * sizeof(float) + 255) & ~255ull;
    float* gu = (float*) scratch;
    float* h = (float*) ((uint8_t*) scratch + gu_bytes);
    gu_nvfp4_kernel<<<dim3((unsigned) (2 * NVFP4_FF / WARPS), (unsigned) cap_groups), THREADS, 0,
                      cs>>>(grp_ptr, grp_start, n_groups, ent_tok, x, gu);
    check("moe_grouped_nvfp4/gu", stream);
    const long long pairs = (long long) cap_entries * NVFP4_FF;
    swiglu_nvfp4_kernel<<<(unsigned) ((pairs + THREADS - 1) / THREADS), THREADS, 0, cs>>>(gu, h,
                                                                                         pairs);
    check("moe_grouped_nvfp4/swiglu", stream);
    down_nvfp4_kernel<<<dim3((unsigned) (NVFP4_H / WARPS), (unsigned) cap_groups), THREADS, 0,
                        cs>>>(grp_ptr, grp_start, n_groups, ent_dst, h, out);
    check("moe_grouped_nvfp4/down", stream);
}

void nvfp4_q8_act_dequant(const uint8_t* blocks, const float* scales, float* x, int64_t n,
                          void* stream) {
    if (n <= 0) return;
    if (n % 32 != 0) {
        std::fprintf(stderr, "nvfp4_q8_act_dequant: n %lld is not a multiple of 32\n",
                     (long long) n);
        std::exit(1);
    }
    q8_act_dequant_kernel<<<(unsigned) ((n + THREADS - 1) / THREADS), THREADS, 0,
                            (cudaStream_t) stream>>>(blocks, scales, x, n);
    check("nvfp4_q8_act_dequant", stream);
}

}  // namespace strata::kernels

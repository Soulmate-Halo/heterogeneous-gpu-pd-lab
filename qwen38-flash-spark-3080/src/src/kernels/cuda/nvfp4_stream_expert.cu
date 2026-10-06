// src/kernels/cuda/nvfp4_stream_expert.cu — small-M (batch, cap<=8) NVFP4 weight-streaming GEMM.
//
// Marlin's tile MMA barely fills the memory pipe at M<=8 (~73 GB/s on GB10). This kernel assigns one
// block to a 64-wide output tile (marlin tile_n) and streams that tile's K groups with coalesced
// 16-byte ld.global.cs loads — the physical marlin order is contiguous 512 B per (k_tile, n_tile).
// Four warps split the K groups of a part; several parts run as extra blocks and a deterministic
// fp32 reduce writes bf16. Numerics follow marlin's NVFP4 skip-flop dequant: fp4 nibble -> bf16
// without exponent bias, processed fp8 scale -> bf16, hmul, fp32 accumulate, then one bf16
// rounding and a bf16 multiply by the per-expert global scale (or global*topk when mul_topk).
#include "strata/kernels/nvfp4_marlin_expert.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace strata::kernels {
namespace {

__device__ __forceinline__ uint4 ldcs_u4(const uint32_t* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// Marlin dequant<bf16, FE2M1, skip_flop>: sign at bf16 bit 15, EEM at bits 8..6.
__device__ __forceinline__ __nv_bfloat16 fp4_raw_bf16(unsigned nib) {
    const unsigned bits = ((nib & 8u) << 12) | ((nib & 7u) << 6);
    return __ushort_as_bfloat16(static_cast<unsigned short>(bits));
}

// dequant_fp8_scales<bf16, FE4M3>: byte bit7 -> bf16 bit14, bits 6..0 -> bits 10..4.
__device__ __forceinline__ __nv_bfloat16 fp8_scale_bf16(unsigned byte) {
    const unsigned bits = ((byte & 0x80u) << 7) | ((byte & 0x7fu) << 4);
    return __ushort_as_bfloat16(static_cast<unsigned short>(bits));
}

// Inverse of marlin_permute_scales (group of 64) then nvfp4_marlin_process_scales' [0,2,1,3] swap.
// The bit transform itself is elementwise and applied by fp8_scale_bf16.
__device__ __forceinline__ int scale_index(int group, int n, int N) {
    const int chunk = n >> 6;
    const int local = n & 63;
    const int p = (local & 7) * 8 + (local >> 3);
    const int idx = group * N + (chunk << 6) + p;
    constexpr int dest[4] = {0, 2, 1, 3};
    return (idx & ~3) + dest[idx & 3];
}

template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32, 2)
nvfp4_stream_partial(const __nv_bfloat16* __restrict__ A, float* __restrict__ partial,
                     const uint32_t* __restrict__ W, const uint8_t* __restrict__ S,
                     const int32_t* __restrict__ sorted, const int32_t* __restrict__ experts,
                     int rows, int N, int K, int k_tiles, int parts) {
    constexpr int kTileN = 64;
    constexpr int kTileK = 16;
    const int n_tile = static_cast<int>(blockIdx.x);
    const int part = static_cast<int>(blockIdx.y);
    const int blk = static_cast<int>(blockIdx.z);
    const int expert = experts[blk];
    if (expert < 0) return;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n_tiles = N >> 6;

    __shared__ int s_rows[8];
    __shared__ int s_nrows;
    if (threadIdx.x == 0) {
        int n = 0;
        const int32_t* ids = sorted + blk * 16;
        for (int j = 0; j < 16 && n < 8; ++j) {
            const int id = ids[j];
            if (static_cast<unsigned>(id) < static_cast<unsigned>(rows)) s_rows[n++] = id;
        }
        s_nrows = n;
    }
    __syncthreads();
    const int nrows = s_nrows;
    if (nrows == 0) return;

    extern __shared__ __align__(16) char dyn[];
    float* wtile = reinterpret_cast<float*>(dyn);                  // [WARPS][16][64]
    float* wacc = wtile + WARPS * kTileK * kTileN;                 // [WARPS][8][64]
    float* xbuf = wacc + WARPS * 8 * kTileN;                       // [WARPS][8][16]

    const uint32_t* We = W + (static_cast<long long>(expert) * N * K) / 8;
    const uint8_t* Se = S + static_cast<long long>(expert) * (K / 16) * N;

    float acc[8][2];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        acc[r][0] = 0.f;
        acc[r][1] = 0.f;
    }

    const int kt_lo = part * (k_tiles / parts);
    const int kt_hi = kt_lo + k_tiles / parts;
    const int c0 = lane * 2;
    const int c1 = c0 + 1;
    float* my_w = wtile + warp * kTileK * kTileN;
    float* my_x = xbuf + warp * 8 * kTileK;

    for (int kt = kt_lo + warp; kt < kt_hi; kt += WARPS) {
        const uint32_t* tile = We + (static_cast<long long>(kt) * n_tiles + n_tile) * 128;
        const uint4 v = ldcs_u4(tile + lane * 4);
        const uint32_t words[4] = {v.x, v.y, v.z, v.w};
        // vals[v] sits at nibble inv_pack[v] (marlin pack_idx = {0,2,4,6,1,3,5,7}).
        constexpr int inv_pack[8] = {0, 4, 1, 5, 2, 6, 3, 7};
        constexpr int tc_off[4] = {0, 1, 8, 9};

#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const int warp_id = sub;
            const int tc_col = lane >> 2;
            const int tc_row = (lane & 3) << 1;
            const int cur_n = warp_id * 16 + tc_col;
            const uint32_t word = words[sub];
            unsigned nibs[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) nibs[inv_pack[i]] = (word >> (i * 4)) & 0xfu;

            const int n_lo = n_tile * kTileN + cur_n;
            const int n_hi = n_lo + 8;
            const __nv_bfloat16 sb_lo = fp8_scale_bf16(Se[scale_index(kt, n_lo, N)]);
            const __nv_bfloat16 sb_hi = fp8_scale_bf16(Se[scale_index(kt, n_hi, N)]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int k_in = tc_row + tc_off[i];
                my_w[k_in * kTileN + cur_n] =
                    __bfloat162float(__hmul(fp4_raw_bf16(nibs[i]), sb_lo));
                my_w[k_in * kTileN + cur_n + 8] =
                    __bfloat162float(__hmul(fp4_raw_bf16(nibs[4 + i]), sb_hi));
            }
        }

        for (int i = lane; i < nrows * kTileK; i += 32) {
            const int r = i / kTileK;
            const int k = i - r * kTileK;
            const int row = s_rows[r];
            my_x[r * kTileK + k] =
                __bfloat162float(__ldg(A + static_cast<long long>(row) * K + kt * kTileK + k));
        }
        __syncwarp();

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= nrows) continue;
            float a0 = acc[r][0];
            float a1 = acc[r][1];
#pragma unroll
            for (int k = 0; k < kTileK; ++k) {
                const float x = my_x[r * kTileK + k];
                a0 = fmaf(x, my_w[k * kTileN + c0], a0);
                a1 = fmaf(x, my_w[k * kTileN + c1], a1);
            }
            acc[r][0] = a0;
            acc[r][1] = a1;
        }
        __syncwarp();
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        if (r < nrows) {
            wacc[((warp * 8 + r) * kTileN) + c0] = acc[r][0];
            wacc[((warp * 8 + r) * kTileN) + c1] = acc[r][1];
        }
    }
    __syncthreads();

    if (warp == 0) {
        const long long part_base = (static_cast<long long>(part) * rows) * N + n_tile * kTileN;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= nrows) continue;
            float s0 = 0.f, s1 = 0.f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) {
                s0 += wacc[((w * 8 + r) * kTileN) + c0];
                s1 += wacc[((w * 8 + r) * kTileN) + c1];
            }
            const int row = s_rows[r];
            partial[part_base + static_cast<long long>(row) * N + c0] = s0;
            partial[part_base + static_cast<long long>(row) * N + c1] = s1;
        }
    }
}

__global__ void nvfp4_stream_epilogue(const float* __restrict__ partial, __nv_bfloat16* __restrict__ C,
                                      const int32_t* __restrict__ sorted, const int32_t* __restrict__ experts,
                                      const uint16_t* __restrict__ gm, const float* __restrict__ topk,
                                      int parts, int rows, int N, int nblk, int mul_topk) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= rows * N) return;
    const int row = idx / N;
    const int col = idx - row * N;
    int expert = -1;
    for (int b = 0; b < nblk && expert < 0; ++b) {
        const int32_t* ids = sorted + b * 16;
        for (int j = 0; j < 16; ++j) {
            if (ids[j] == row) {
                expert = experts[b];
                break;
            }
        }
    }
    if (expert < 0) return;
    float sum = 0.f;
    for (int p = 0; p < parts; ++p)
        sum += partial[(static_cast<long long>(p) * rows + row) * N + col];
    __nv_bfloat16 r = __float2bfloat16(sum);
    __nv_bfloat16 g = __ushort_as_bfloat16(gm[expert]);
    if (mul_topk) g = __hmul(g, __float2bfloat16(topk[row]));
    C[idx] = __hmul(r, g);
}

int pick_parts(int k_tiles, int rows, int N, long long work_floats) {
    const long long need_one = static_cast<long long>(rows) * N;
    if (work_floats < need_one) return 0;
    for (int p = 4; p >= 1; p >>= 1) {
        if (k_tiles % p == 0 && static_cast<long long>(p) * need_one <= work_floats) return p;
    }
    return 1;
}

}  // namespace

void nvfp4_stream_expert_gemm(const void* a_bf16, void* c_bf16, const uint8_t* w, const uint8_t* scales,
                              const uint16_t* global_scale, const int32_t* sorted, const int32_t* experts,
                              const float* topk, float* work, int64_t work_floats, int rows, int nblk,
                              int size_n, int size_k, int mul_topk, void* stream) {
    if (rows <= 0 || nblk <= 0 || size_n <= 0 || size_k <= 0) return;
    if ((size_n & 63) || (size_k & 15) || rows > 8) return;
    constexpr int kWarps = 4;
    const int n_tiles = size_n >> 6;
    const int k_tiles = size_k >> 4;
    const int parts = pick_parts(k_tiles, rows, size_n, work_floats);
    if (parts <= 0 || work == nullptr) return;

    const size_t smem = (static_cast<size_t>(kWarps) * 16 * 64
                         + static_cast<size_t>(kWarps) * 8 * 64
                         + static_cast<size_t>(kWarps) * 8 * 16) * sizeof(float);
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    nvfp4_stream_partial<kWarps><<<dim3(n_tiles, parts, nblk), kWarps * 32, smem, cs>>>(
        static_cast<const __nv_bfloat16*>(a_bf16), work, reinterpret_cast<const uint32_t*>(w), scales, sorted,
        experts, rows, size_n, size_k, k_tiles, parts);
    const int nout = rows * size_n;
    nvfp4_stream_epilogue<<<static_cast<unsigned>((nout + 255) / 256), 256, 0, cs>>>(
        work, static_cast<__nv_bfloat16*>(c_bf16), sorted, experts, global_scale, topk, parts, rows, size_n, nblk,
        mul_topk);
}

}  // namespace strata::kernels

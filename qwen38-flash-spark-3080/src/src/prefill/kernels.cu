// src/prefill/kernels.cu - see include/strata/prefill/kernels.hpp.
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/router_top10.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int S = 128, HK = 16, HV = 48, C = 10240;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ uint16_t bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ uint16_t hf(float f) { return __half_as_ushort(__float2half_rn(f)); }
// block-wide sum for blockDim.x <= 1024, result broadcast
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (w == 0) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    return sh[0];
}
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "prefill %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

// ---------------------------------------------------------------- hyper-connection
__global__ void gr_norm_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                               float* __restrict__ xn, uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    for (int d = threadIdx.x; d < N; d += blockDim.x) {
        const float v = r[d] * rs * w[c * N + d];
        xn[row * N + d] = v;
        xn16[row * N + d] = bf(v);
    }
}
// F-1: the row scale only (and the BF16 image); gr_mix_r_kernel recomputes r * rs * w itself, in the same order,
// so the FP32 copy of the normalized rows (T x 10240 floats) is neither written nor read
__global__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                                  float* __restrict__ rs_out, uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    for (int d = threadIdx.x; d < N; d += blockDim.x) xn16[row * N + d] = bf(r[d] * rs * w[c * N + d]);
}
__global__ void gr_mix_r_kernel(const float* __restrict__ R, const float* __restrict__ rs, const float* __restrict__ w,
                                const float* __restrict__ g, float* __restrict__ mixed, uint16_t* __restrict__ mixed16,
                                int64_t T, uint16_t* __restrict__ mixed_h) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        const float x = R[j] * rs[t * HC + c] * w[c * N + d];   // gr_norm_kernel's value, bit for bit
        s = fmaf(x, sigm(g[j]), s);
    }
    s /= (float) HC;
    mixed[i] = s;
    if (mixed16) mixed16[i] = bf(s);
    if (mixed_h) mixed_h[i] = hf(s);
}
// F-2: gr_write_kernel for one row (t, c), then gr_norm_rs_kernel's reduction over it with the next half's norm
// weights - the same thread-to-element mapping (256 threads, stride 256) and block_sum, so rs and the BF16 image are
// the same bits, and R is not read back
constexpr int GRW_PER = (N + 255) / 256;
__global__ void __launch_bounds__(256) gr_write_norm_rs_kernel(float* __restrict__ R, const float* __restrict__ bo,
                                                               const float* __restrict__ inj, int64_t inj_ld,
                                                               const float* __restrict__ w, float eps,
                                                               float* __restrict__ rs_out, uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int64_t t = row / HC;
    const int c = (int) (row % HC);
    float* r = R + row * N;
    const float sc = 2.0f * sigm(inj[t * inj_ld + c] / (float) HC);
    float v[GRW_PER];
    float ss = 0.0f;
    int k = 0;
#pragma unroll
    for (int d = threadIdx.x; d < N; d += 256, ++k) {
        const float x = fmaf(bo[t * N + d], sc, r[d]);
        r[d] = x;
        v[k] = x;
        ss += x * x;
    }
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    k = 0;
#pragma unroll
    for (int d = threadIdx.x; d < N; d += 256, ++k) xn16[row * N + d] = bf(v[k] * rs * w[c * N + d]);
}
__global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = lo[i] / (float) HC;
    lo16[i] = bf(x / (1.0f + __expf(-x)));
}
__global__ void gr_mix_kernel(const float* __restrict__ xn, const float* __restrict__ g, float* __restrict__ mixed,
                              uint16_t* __restrict__ mixed16, int64_t T, uint16_t* __restrict__ mixed_h) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        s = fmaf(xn[j], sigm(g[j]), s);
    }
    s /= (float) HC;
    mixed[i] = s;
    if (mixed16) mixed16[i] = bf(s);
    if (mixed_h) mixed_h[i] = hf(s);
}
__global__ void gr_write_kernel(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                                int64_t inj_ld, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, c = (i % D) / N, d = i % N;
    R[i] = fmaf(bo[t * N + d], 2.0f * sigm(inj[t * inj_ld + c] / (float) HC), R[i]);
}
__global__ void gr_broadcast_kernel(const float* __restrict__ e, float* __restrict__ R, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, d = i % N;
    R[i] = e[t * N + d];
}

// ---------------------------------------------------------------- GDN
__global__ void gdn_gates_kernel(const float* __restrict__ ab, const float* __restrict__ dt,
                                 const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta,
                                 int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * HV) return;
    const int64_t t = i / HV, h = i % HV;
    const float v = ab[t * 2 * HV + h] + dt[h];
    gate[i] = (v > 20.0f ? v : log1pf(__expf(v))) * ssm_a[h];
    beta[i] = sigm(ab[t * 2 * HV + HV + h]);
}
// one thread per channel, walks the chunk; then a second kernel normalises
__global__ void gdn_conv_kernel(float* __restrict__ hist, const float* __restrict__ qkv, const float* __restrict__ w,
                                float* __restrict__ h, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = 0; t < T; ++t) {
        const float x = qkv[t * C + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[t * C + c] = s / (1.0f + __expf(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
    hist[c * 3] = v0; hist[c * 3 + 1] = v1; hist[c * 3 + 2] = v2;
}
// C-3: the same 4-tap causal conv, tiled over tokens: thread (c, tile) reads its tile's 3 predecessors from the
// chunk (or the history before it) instead of carrying them - the conv reads inputs, not its own outputs, so the
// tiles are independent. The same expression per element (so the same bits); the history is written afterwards.
constexpr int CONV_TILE = 64;
__global__ void gdn_conv_tiled_kernel(const float* __restrict__ hist, const float* __restrict__ qkv,
                                      const float* __restrict__ w, float* __restrict__ h, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int64_t t0 = (int64_t) blockIdx.y * CONV_TILE;
    if (t0 >= T) return;
    const int64_t t1 = t0 + CONV_TILE < T ? t0 + CONV_TILE : T;
    auto input = [&](int64_t t) -> float { return t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)]; };
    float v0 = input(t0 - 3), v1 = input(t0 - 2), v2 = input(t0 - 1);
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = t0; t < t1; ++t) {
        const float x = qkv[t * C + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[t * C + c] = s / (1.0f + __expf(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
}
// the history after the chunk: its last three inputs (the older history where the chunk is shorter than 3)
__global__ void gdn_conv_hist_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v[3];
    for (int k = 0; k < 3; ++k) {
        const int64_t t = T - 3 + k;
        v[k] = t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)];
    }
    hist[c * 3] = v[0]; hist[c * 3 + 1] = v[1]; hist[c * 3 + 2] = v[2];
}
__global__ void gdn_l2_kernel(float* __restrict__ h, float eps) {
    // block (t, head) over the 32 q/k heads, 128 threads
    const int64_t t = blockIdx.y;
    const int head = blockIdx.x;
    float* x = h + t * C + head * S;
    const float v = x[threadIdx.x];
    float sq = warp_sum(v * v);
    __shared__ float part[4];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    x[threadIdx.x] = v * rsqrtf(ss + eps);
}
constexpr int RG = 4, RPG = S / RG;
__global__ void __launch_bounds__(S * RG) gdn_rec_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                         const float* __restrict__ gate,
                                                         const float* __restrict__ beta, const float* __restrict__ z,
                                                         const float* __restrict__ gamma, float eps,
                                                         float* __restrict__ y, uint16_t* __restrict__ y16, int64_t T) {
    __shared__ float sk[S], sq[S], red[RG][S], wsum[16];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    const float g_col = gamma[col];
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h + t * C;
        __syncthreads();
        if (tid < S) { sq[tid] = ht[qh * S + tid]; sk[tid] = ht[HK * S + qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[t * HV + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col] = kv;
        __syncthreads();
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * HK * S + head * S + col] - g * kv_col) * beta[t * HV + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sp = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * rsqrtf((float) S);
            sp = oc * oc;
        }
        sp = warp_sum(sp);
        if ((tid & 31) == 0) wsum[tid >> 5] = sp;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float v = oc * rsqrtf(ss / (float) S + eps) * g_col * sigm(z[t * HV * S + head * S + col]);
            y[t * HV * S + head * S + col] = v;
            y16[t * HV * S + head * S + col] = hf(v);
        }
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}

// D-2: the recurrence with the value columns split over 4 blocks per head (4x the blocks of the kernel above, a
// quarter of its threads per __syncthreads), and the output norm - the only step that couples the head's columns -
// in its own kernel. Per column the same arithmetic in the same order (the 4 row-group partial sums added as
// red[0] + red[1] + red[2] + red[3]; the norm's warp sums over the same 32-column warps): the same bits.
constexpr int CB = 32, NCB = S / CB;
__global__ void __launch_bounds__(CB * RG) gdn_rec_cols_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                                const float* __restrict__ gate,
                                                                const float* __restrict__ beta,
                                                                float* __restrict__ oc_out, int64_t T) {
    __shared__ float sk[S], sq[S], red[RG][CB];
    const int head = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, tid = rg * CB + c, col = cb * CB + c;
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h + t * C;
        __syncthreads();
        if (tid < S) { sq[tid] = ht[qh * S + tid]; sk[tid] = ht[HK * S + qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[t * HV + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][c] = kv;
        __syncthreads();
        const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
        const float delta = (ht[2 * HK * S + head * S + col] - g * kv_col) * beta[t * HV + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][c] = o;
        __syncthreads();
        if (rg == 0) oc_out[t * HV * S + head * S + col] = (red[0][c] + red[1][c] + red[2][c] + red[3][c]) * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}
__global__ void __launch_bounds__(S) gdn_out_norm_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                         float eps, float* __restrict__ y, uint16_t* __restrict__ y16) {
    __shared__ float wsum[4];
    const int64_t t = blockIdx.x;
    const int head = blockIdx.y, col = threadIdx.x;
    const size_t at = (size_t) t * HV * S + (size_t) head * S + col;
    const float oc = y[at];
    float sp = warp_sum(oc * oc);
    if ((col & 31) == 0) wsum[col >> 5] = sp;
    __syncthreads();
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float v = oc * rsqrtf(ss / (float) S + eps) * gamma[col] * sigm(z[t * HV * S + head * S + col]);
    y[at] = v;
    y16[at] = hf(v);
}

// ---------------------------------------------------------------- MoE
template <int REG>
__global__ void route_kernel(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ wout,
                             int64_t T) {
    const int64_t t = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (t >= T) return;
    const int lane = threadIdx.x & 31;
    const float* lg = logits + t * (REG * 32);
    float v[REG];
#pragma unroll
    for (int i = 0; i < REG; ++i) v[i] = lg[lane + i * 32];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < REG; ++i) mx = fmaxf(mx, v[i]);
    mx = warp_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < REG; ++i) { v[i] = expf(v[i] - mx); sum += v[i]; }
    const float rcp = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < REG; ++i) { v[i] *= rcp; if (isnan(v[i])) v[i] = -FLT_MAX; }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v[0];
        int ex = lane;
#pragma unroll
        for (int i = 1; i < REG; ++i) if (v[i] > best) { best = v[i]; ex = lane + i * 32; }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, m);
            const int oi = __shfl_xor_sync(0xffffffffu, ex, m);
            if (ob > best || (ob == best && oi < ex)) { best = ob; ex = oi; }
        }
        if ((ex & 31) == lane) { v[ex / 32] = -INFINITY; selected_sum += best; }
        if (lane == 0) ids[t * 10 + rank] = ex;
        if (rank == lane) selected = best;
    }
    selected_sum = fmaxf(warp_sum(selected_sum), 6.103515625e-5f);
    if (lane < 10) wout[t * 10 + lane] = selected / selected_sum;
}
// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales [2560][10] f16
template <bool HALF>
__global__ void blob_dequant_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu16,
                                    uint16_t* __restrict__ d16) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one thread per 4 weights (one code byte)
    const int64_t n_gu = 1280LL * 640, n_d = 2560LL * 160;
    if (i < n_gu) {
        const int64_t row = i / 640, byte = i % 640;
        const uint8_t c = blob[row * 640 + byte];
        const uint8_t* sp = blob + O_GU_SC + (size_t) (row * 40 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = gu16 + row * 2560 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 160, byte = j % 160;
        const uint8_t c = blob[O_D_CODES + row * 160 + byte];
        const uint8_t* sp = blob + O_D_SC + (size_t) (row * 10 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = d16 + row * 640 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    }
}
__global__ void swiglu_il_kernel(const float* __restrict__ gu, uint16_t* __restrict__ h16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const int64_t r = i / 640, k = i % 640;
    const float g = gu[r * 1280 + 2 * k], u = gu[r * 1280 + 2 * k + 1];
    h16[i] = hf(g / (1.0f + __expf(-g)) * u);
}
__global__ void swiglu_pair_kernel(const float* __restrict__ g, const float* __restrict__ u, uint16_t* __restrict__ h16,
                                   int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const float a = g[i];
    h16[i] = hf(a / (1.0f + __expf(-a)) * u[i]);
}
__global__ void gather_rows16_kernel(const uint16_t* __restrict__ x, const int32_t* __restrict__ src,
                                     uint16_t* __restrict__ dst, int64_t n, int64_t width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one uint4 (8 bf16)
    const int64_t per = width / 8;
    if (i >= n * per) return;
    const int64_t r = i / per, j = i % per;
    reinterpret_cast<uint4*>(dst)[r * per + j] = reinterpret_cast<const uint4*>(x)[(int64_t) src[r] * per + j];
}
__global__ void scatter_rows_f32_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                        const int32_t* __restrict__ dst_idx, int64_t n, int64_t width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one float4
    const int64_t per = width / 4;
    if (i >= n * per) return;
    const int64_t r = i / per, j = i % per;
    reinterpret_cast<float4*>(dst)[(int64_t) dst_idx[r] * per + j] =
        reinterpret_cast<const float4*>(src)[r * per + j];
}
__global__ void scatter_rows_f16_kernel(float* __restrict__ dst, const uint16_t* __restrict__ src,
                                        const int32_t* __restrict__ dst_idx, int64_t n, int64_t width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one uint2 (4 f16) -> one float4
    const int64_t per = width / 4;
    if (i >= n * per) return;
    const int64_t r = i / per, j = i % per;
    const uint2 v = reinterpret_cast<const uint2*>(src)[r * per + j];
    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&v.x));
    const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&v.y));
    reinterpret_cast<float4*>(dst)[(int64_t) dst_idx[r] * per + j] = make_float4(lo.x, lo.y, hi.x, hi.y);
}
__global__ void moe_combine_kernel(const float* __restrict__ Dm, const int32_t* __restrict__ slot,
                                   const float* __restrict__ w, const float* __restrict__ shared,
                                   const float* __restrict__ sg, float* __restrict__ bo, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 10; ++k) s = fmaf(w[t * 10 + k], Dm[(int64_t) slot[t * 10 + k] * N + d], s);
    bo[i] = s + shared[i] * sigm(sg[t]);
}

// ---------------------------------------------------------------- QSA helpers
__global__ void rms_rows_kernel(float* __restrict__ x, const float* __restrict__ w, int64_t cols, int64_t ld, float eps) {
    __shared__ float sh[32];
    float* r = x + (int64_t) blockIdx.x * ld;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) ss += r[c] * r[c];
    const float s = rsqrtf(block_sum(ss, sh) / (float) cols + eps);
    __syncthreads();
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) r[c] = s * r[c] * w[c];
}
__global__ void rope_kernel(float* __restrict__ x, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
                            float theta_scale, const int32_t* __restrict__ mtab) {
    const int64_t row = blockIdx.x;             // t * heads + h
    const int pair = threadIdx.x;               // 0..31
    const int64_t t = row / heads, h = row % heads;
    float* p = x + t * ld + h * dim;
    const float theta = (float) strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair) * powf(theta_scale, (float) pair);
    const float c = cosf(theta), s = sinf(theta);
    const float a = p[pair], b = p[pair + 32];
    p[pair] = a * c - b * s;
    p[pair + 32] = a * s + b * c;
}
__global__ void split_q_kernel(const float* __restrict__ qf, float* __restrict__ q, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    q[i] = qf[t * 24 * 512 + h * 512 + d];
}
__global__ void gate_attn_kernel(const float* __restrict__ a, const float* __restrict__ qf, uint16_t* __restrict__ o16,
                                 int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    o16[i] = hf(a[i] * (1.0f / (1.0f + expf(-qf[t * 24 * 512 + h * 512 + 256 + d]))));
}

// one block per (token, kv head, 64-value group); 64 threads. KV streaming: the pool page only if the block is
// resident (table >= 0), and the host copy and the prompt path's staging pool (both identity layout) when given.
__global__ void kv_append_kernel(const float* __restrict__ K, const float* __restrict__ V, int64_t pos0,
                                 const int32_t* __restrict__ table, int64_t page_size, uint16_t* k_pool,
                                 uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                                 strata::kernels::KvHostPools host, strata::kernels::KvHostPools stage) {
    const int64_t t = blockIdx.x;
    const int kvh = blockIdx.y, g = blockIdx.z >> 1;
    const bool is_v = (blockIdx.z & 1) != 0;
    const int d = g * 64 + threadIdx.x;
    const float x = (is_v ? V : K)[t * 512 + kvh * 256 + d];
    const int64_t pos = pos0 + t;
    const int64_t page = table[pos / page_size];
    const int64_t row = (page * 2 + kvh) * page_size + pos % page_size;
    const int64_t row_id = ((pos / page_size) * 2 + kvh) * page_size + pos % page_size;
    if (k_pool != nullptr) {
        const uint16_t h = hf(x);
        if (page >= 0) (is_v ? v_pool : k_pool)[row * 256 + d] = h;
        if (host.k_pool != nullptr) (is_v ? host.v_pool : host.k_pool)[row_id * 256 + d] = h;
        if (stage.k_pool != nullptr) (is_v ? stage.v_pool : stage.k_pool)[row_id * 256 + d] = h;
        return;
    }
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float wm[2];
    if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(wm[0], wm[1]);
    const uint16_t sb = hf(amax / 127.0f);
    const float sf = __half2float(__ushort_as_half(sb));
    int q = 0;
    if (sf > 0.0f) { q = __float2int_rn(x / sf); q = q < -127 ? -127 : (q > 127 ? 127 : q); }
    if (page >= 0) {
        (is_v ? v_q : k_q)[row * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? v_scale : k_scale)[row * 4 + g] = sb;
    }
    if (host.k_q != nullptr) {
        (is_v ? host.v_q : host.k_q)[row_id * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? host.v_scale : host.k_scale)[row_id * 4 + g] = sb;
    }
    if (stage.k_q != nullptr) {
        (is_v ? stage.v_q : stage.k_q)[row_id * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? stage.v_scale : stage.k_scale)[row_id * 4 + g] = sb;
    }
}
__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = hf(x[i]);
}
__global__ void round_f16_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = __half2float(__float2half_rn(x[i]));
}
__global__ void to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = bf(x[i]);
}

}  // namespace

void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, const strata::kernels::KvHostPools* host, const strata::kernels::KvHostPools* stage) {
    if (T <= 0) return;
    kv_append_kernel<<<dim3((unsigned) T, 2, 8), 64, 0, (cudaStream_t) stream>>>(
        K, V, pos0, page_table, page_size, k_pool, v_pool, k_q, v_q, k_scale, v_scale,
        host ? *host : strata::kernels::KvHostPools{}, stage ? *stage : strata::kernels::KvHostPools{});
    check("kv_append");
}
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_f16");
}
void round_f16(const float* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    round_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("round_f16");
}
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_bf16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_bf16");
}

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream) {
    gr_norm_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, xn, xn16);
    check("gr_norm");
}
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream) {
    gr_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, rs, xn16);
    check("gr_norm_rs");
}
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h) {
    gr_mix_r_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(R, rs, w_norm, gated, mixed, mixed16, T,
                                                                          mixed_h);
    check("gr_mix_r");
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream) {
    gr_write_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, w_norm_next, eps,
                                                                                      rs, xn16);
    check("gr_write_norm_rs");
}
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream) {
    gr_silu_kernel<<<blocks_for(T * LR), 256, 0, (cudaStream_t) stream>>>(lo, lo16, T * LR);
    check("gr_silu");
}
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h) {
    gr_mix_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(xn, gated, mixed, mixed16, T, mixed_h);
    check("gr_mix");
}
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream) {
    gr_write_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, T);
    check("gr_write");
}
void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    gr_broadcast_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(e, R, T);
    check("gr_broadcast");
}
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream) {
    gdn_gates_kernel<<<blocks_for(T * HV), 256, 0, (cudaStream_t) stream>>>(ab, dt, ssm_a, gate, beta, T);
    check("gdn_gates");
}
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream) {
    static const bool serial = std::getenv("STRATA_GDN_CONV_SERIAL") != nullptr;   // the old walk (A/B)
    if (serial || T <= CONV_TILE) {
        gdn_conv_kernel<<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, conv_w, h, T);
    } else {
        gdn_conv_tiled_kernel<<<dim3(C / 128, (unsigned) ((T + CONV_TILE - 1) / CONV_TILE)), 128, 0,
                                (cudaStream_t) stream>>>(history, qkv, conv_w, h, T);
        gdn_conv_hist_kernel<<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, T);
    }
    gdn_l2_kernel<<<dim3(2 * HK, (unsigned) T), S, 0, (cudaStream_t) stream>>>(h, eps);
    check("gdn_conv");
}
void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream) {
    static const bool serial = std::getenv("STRATA_GDN_REC_HEADS") != nullptr;   // the one-block-per-head kernel (A/B)
    if (serial || T <= 0) {
        gdn_rec_kernel<<<HV, dim3(S, RG), 0, (cudaStream_t) stream>>>(state, h, gate, beta, z, gamma, eps, y, y16, T);
    } else {
        gdn_rec_cols_kernel<<<HV * NCB, dim3(CB, RG), 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
        gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S, 0, (cudaStream_t) stream>>>(z, gamma, eps, y, y16);
    }
    check("gdn_recurrence");
}
void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream) {
    if (n_expert == 512)
        route_kernel<16><<<(unsigned) ((T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(logits, ids, weights, T);
    else if (n_expert == 256)
        route_kernel<8><<<(unsigned) ((T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(logits, ids, weights, T);
    else
        strata::kernels::router_top10(logits, (int) T, (int) n_expert, 10, ids, weights, stream);
    check("route");
}
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_kernel<false><<<blocks_for(1280LL * 640 + 2560LL * 160), 256, 0, (cudaStream_t) stream>>>(blob, gu16, down16);
    check("blob_dequant");
}
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_kernel<true><<<blocks_for(1280LL * 640 + 2560LL * 160), 256, 0, (cudaStream_t) stream>>>(blob, gu16, down16);
    check("blob_dequant_f16");
}
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    swiglu_il_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(gu, h16, n);
    check("swiglu_interleaved");
}
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    swiglu_pair_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(g, u, h16, n);
    check("swiglu_pair");
}
namespace {
__global__ void copy_i32_kernel(int32_t* __restrict__ dst, const int32_t* __restrict__ src, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[i];
}
}  // namespace
void copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const int64_t b = (n + 255) / 256;
    copy_i32_kernel<<<(unsigned) (b < 256 ? b : 256), 256, 0, (cudaStream_t) stream>>>(dst, src, n);
    check("copy_i32");
}
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    gather_rows16_kernel<<<blocks_for(n * (width / 8)), 256, 0, (cudaStream_t) stream>>>(x16, src, dst16, n, width);
    check("gather_rows16");
}
void scatter_rows_f32(float* dst, const float* src, const int32_t* dst_idx, int64_t n, int64_t width,
                      void* stream) {
    if (n <= 0) return;
    scatter_rows_f32_kernel<<<blocks_for(n * (width / 4)), 256, 0, (cudaStream_t) stream>>>(dst, src, dst_idx, n,
                                                                                            width);
    check("scatter_rows_f32");
}
void scatter_rows_f16(float* dst, const uint16_t* src, const int32_t* dst_idx, int64_t n, int64_t width,
                      void* stream) {
    if (n <= 0) return;
    scatter_rows_f16_kernel<<<blocks_for(n * (width / 4)), 256, 0, (cudaStream_t) stream>>>(dst, src, dst_idx, n,
                                                                                            width);
    check("scatter_rows_f16");
}
void moe_combine(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream) {
    moe_combine_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(Dm, slot, w, shared, sg, bo, T);
    check("moe_combine");
}
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    if (rows <= 0) return;
    rms_rows_kernel<<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(x, w, cols, ld, eps);
    check("rms_rows");
}
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, float freq_base, void* stream) {
    const float theta_scale = powf(freq_base, -2.0f / 64.0f);
    rope_kernel<<<(unsigned) (T * heads), 32, 0, (cudaStream_t) stream>>>(x, heads, dim, ld, pos0, theta_scale, strata::kernels::mrope_table());
    check("rope");
}
void split_q(const float* q_full, float* q, int64_t T, void* stream) {
    split_q_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(q_full, q, T);
    check("split_q");
}
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream) {
    gate_attn_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(attn, q_full, out16, T);
    check("gate_attn");
}

}  // namespace strata::prefill

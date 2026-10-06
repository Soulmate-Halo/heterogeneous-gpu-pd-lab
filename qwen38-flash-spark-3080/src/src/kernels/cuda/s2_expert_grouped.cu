// src/kernels/cuda/s2_expert_grouped.cu - R4's grouped GPU expert.  Read the header first.
//
// THE ARITHMETIC IS `src/kernels/cpu/expert.cpp`'s, over the same bytes, one warp per output row.
//
//     sum_j (c_j - 1) * d_w * xhat_j * d_x   ==   d_w * d_x * ( sum_j c_j*xhat_j  -  sum_j xhat_j )
//
// Both sums are INT8 x INT8, so both are `__dp4a` - four multiply-accumulates per instruction, exact in
// integer and with no dequantization inside the loop.  The weight scale `d_w` is fp16 (one per 64 elements)
// and the activation scale `d_x` is fp16 (one per 32 elements), so the float work is one multiply-add per
// 32-element chunk rather than one per element.
//
// **THE SUMMATION ORDER IS NOT THE CPU's AND CANNOT BE.**  Lane `L` takes chunks `L, L+32, ...` and the
// partials are reduced through shuffle; the CPU walks every chunk in order with its own accumulator shape.
// The two agree to float rounding and not to the bit, which is the same contract `s_gemv_parity` carries for
// the same reason.  `bench/micro/moe_hit_parity.cu` is the check.
#include "strata/kernels/s2_expert_grouped.hpp"

#include "strata/kernels/quantize_act.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

// THE BLOB'S OWN GEOMETRY, from `include/strata/kernels/cpu/expert.hpp`.  Restated as literals because that
// header is the CPU path's and this file must not silently follow it if the two ever disagree: the sizes below
// are what the CPU kernel's indexing computes, and `moe_hit_parity` compares the two end to end.
constexpr int H = 2560;
constexpr int FF = 640;
constexpr int QK = 64;                       // Q2_0's group: one fp16 scale per 64 weights
constexpr int ROW_GU = H / 4;                // 640 B of codes per gate/up row (2 bits per element)
constexpr int ROW_D = FF / 4;                // 160 B per down row
constexpr int SC_GU = H / QK;                // 40 fp16 scales per gate/up row
constexpr int SC_D = FF / QK;                // 10 per down row
constexpr size_t O_D_CODES = (size_t) 2 * FF * ROW_GU;
constexpr size_t O_GU_SCALES = O_D_CODES + (size_t) H * ROW_D;
constexpr size_t O_D_SCALES = O_GU_SCALES + (size_t) 2 * FF * SC_GU * 2;

__device__ __forceinline__ float f16_at(const uint8_t* p) {
    return __half2float(__ushort_as_half((uint16_t) (p[0] | (p[1] << 8))));
}

/// **ONE S2 ROW AGAINST A Q8_0 ACTIVATION, WARP-WIDE.**  Every lane accumulates its own float partial over a
/// strided set of 32-element chunks and the caller reduces; `chunk` indices are absolute so the caller can
/// start the lane at any offset.
///
/// Returns the lane's partial.  `codes` is `n_in/4` bytes and `scales` `n_in/QK` fp16, both for THIS row.
///
/// **`x_scales` IS R4.2h AND IT IS OPTIONAL ON PURPOSE.**  When it is non-null the activation's multiplier
/// comes from an fp32 array instead of the block's fp16 `d`.  The CPU pool multiplies by the fp32
/// `ActQ::scale` (`cpu/expert.cpp:92`), and the two disagreed by **4.761e-04 relative on 80 of 80 chunks**
/// (`bench/micro/act_quant_parity.cu`), which is what made a cache hit compute a different expert from a
/// cache miss.  Null keeps the previous fp16 behaviour **exactly**, so `moe_hit_parity` - which passes no
/// scales - still measures the kernel it always measured.
__device__ __forceinline__ float row_dot_s2_q8(const uint8_t* __restrict__ codes,
                                               const uint8_t* __restrict__ scales,
                                               const uint8_t* __restrict__ x_q8_0, int n_chunks, int lane,
                                               const float* __restrict__ x_scales = nullptr) {
    float acc = 0.0f;
    for (int c = lane; c < n_chunks; c += 32) {
        const uint8_t* cb = codes + (size_t) c * 8;             // 8 code bytes = 32 elements
        const uint8_t* xb = x_q8_0 + (size_t) c * 34;           // one block_q8_0
        const float dx = x_scales ? x_scales[c] : f16_at(xb);
        const int8_t* xq = (const int8_t*) (xb + 2);

        // ---- THE CODES, EXPANDED TO ONE BYTE PER ELEMENT, THEN FOUR AT A TIME INTO `dp4a`.
        //
        // The packed form is LSB-first: element 4j+k is bits [2k, 2k+2) of code byte j.  `dp4a` needs both
        // operands as packed int8, so each code byte becomes a word whose four bytes are its four 2-bit
        // fields - which is exactly `(c & 3) | ((c>>2)&3)<<8 | ((c>>4)&3)<<16 | ((c>>6)&3)<<24`.
        int s = 0;      // sum of code * x
        int hx = 0;     // sum of x        - the weight-independent term, as ones * x
        const int ones = 0x01010101;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned cbyte = cb[j];
            const int cw = (int) ((cbyte & 3u) | (((cbyte >> 2) & 3u) << 8) | (((cbyte >> 4) & 3u) << 16) |
                                  (((cbyte >> 6) & 3u) << 24));
            // **`memcpy`, NOT A CAST.**  A `block_q8_0` is 34 BYTES - two of header then 32 of int8 - so the
            // activation data at offset 2 is never 4-byte aligned, and `*(const int*)(xq + 4*j)` faults with
            // "misaligned address".  It faulted exactly that way on this kernel's first run.  `memcpy` of a
            // known 4 bytes compiles to whatever load is legal for the alignment, which is the point of using
            // it rather than reasoning about which cast happens to work.
            int xw;
            memcpy(&xw, xq + 4 * j, 4);
            s = __dp4a(cw, xw, s);
            hx = __dp4a(ones, xw, hx);
        }
        // One weight scale per 64 elements, so per TWO 32-element chunks.
        const float dw = f16_at(scales + (size_t) (c >> 1) * 2);
        acc += dw * dx * (float) (s - hx);
    }
    return acc;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    return v;
}

/// GATE AND UP, ONE WARP PER ROW.
///
/// **THE ROWS ARE INTERLEAVED AND THE FIRST VERSION OF THIS GOT IT WRONG.**  In the blob, row-SLOT `i` of the
/// `2*FF` gate/up rows is gate row `i/2` when `i` is even and up row `(i-1)/2` when it is odd - that is what
/// `O_GU_CODES + (2r)*ROW_GU` and `+ (2r+1)*ROW_GU` mean in `expert.cpp`.  This kernel decoded the correct
/// code row for slot `i` and then wrote it to output slot `i` of a layout whose first `FF` entries are gate and
/// whose last `FF` are up, which pairs `silu(gate[r]) * up[r]` with the WRONG `up` for every `r`.  It produced
/// finite, plausible numbers.  `moe_hit_parity` caught it on the first run, at worst relative error 2.2e+03.
///
/// So slot `i` is DECODED from row-slot `i` and WRITTEN to the output slot its parity says it belongs to.
__global__ void gu_kernel(const uint8_t* __restrict__ blob_base, const int32_t* __restrict__ slot_index,
                          long long blob_bytes, const uint8_t* __restrict__ x_q8_0,
                          const float* __restrict__ x_scales, float* __restrict__ gate_up, int n_hits,
                          const int32_t* __restrict__ d_count = nullptr,
                          const int32_t* __restrict__ dst_index = nullptr, int tok_div = 0) {
    const int warps_per_block = (int) (blockDim.x >> 5);
    const long long slot = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const long long rows_per_hit = 2LL * FF;
    const long long total = (long long) n_hits * rows_per_hit;
    if (slot >= total) return;
    const int h = (int) (slot / rows_per_hit);
    if (d_count != nullptr && h >= *d_count) return;     // token graph: capacity layout, device count
    const int i = (int) (slot % rows_per_hit);
    const int lane = threadIdx.x & 31;

    const uint8_t* blob = blob_base + (size_t) slot_index[h] * (size_t) blob_bytes;
    if (tok_div > 0) {   // plan v0.3 P6 verify window: each hit reads its own token's activation
        const int tok = dst_index[h] / tok_div;
        x_q8_0 += (size_t) tok * (size_t) (H / 32) * 34;
        if (x_scales != nullptr) x_scales += (size_t) tok * (size_t) (H / 32);
    }
    const float acc = row_dot_s2_q8(blob + (size_t) i * ROW_GU,
                                    blob + O_GU_SCALES + (size_t) i * SC_GU * 2, x_q8_0, H / 32, lane, x_scales);
    const float s = warp_sum(acc);
    if (lane != 0) return;
    // ---- THE OUTPUT LAYOUT IS GATE-MAJOR, AND THAT IS NOT COSMETIC.
    //
    // The first version wrote `gate_up[h*2FF + {0..FF-1}] = gate` and `[h*2FF + FF..] = up`, i.e. a per-hit
    // [gate | up] pair, and then called the shared `swiglu_kernel` and `quantize_q8_0` over the whole buffer.
    // Both of those walk a CONTIGUOUS range, so with more than one hit they read hit 0's UP rows where they
    // wanted hit 1's GATE rows - finite numbers, wrong expert.  With one hit it would have passed.
    //
    // Gate-major removes the mismatch instead of adding a stride to two other kernels: every hit's gate rows
    // are contiguous from 0, every hit's up rows are contiguous from `n_hits * FF`, and the swiglu's output
    // lands in the first `n_hits * FF` floats exactly where `quantize_q8_0` reads it.
    const int r = i >> 1;
    const size_t base = (i & 1) ? ((size_t) n_hits * FF + (size_t) h * FF) : ((size_t) h * FF);
    gate_up[base + (size_t) r] = s;
}

/// `silu(gate) * up`, in place, over a GATE-MAJOR buffer: `[0, n_pairs)` is every hit's gate and
/// `[n_pairs, 2*n_pairs)` is every hit's up, so hit `h`'s row `r` meets itself at `h*FF + r`.
///
/// SiLU on the GATE and multiplied by up - the reading `docs/semantics.md` records, and the one that is wrong
/// the other way round in a way that still produces a finite number.
__global__ void swiglu_kernel(float* __restrict__ gate_up, long long n_pairs) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pairs) return;
    const float g = gate_up[i];
    const float u = gate_up[n_pairs + i];
    gate_up[i] = (g / (1.0f + __expf(-g))) * u;
}

/// DOWN, ONE WARP PER ROW, reading the quantized intermediate the caller produced.
///
/// `dst_index[h]` is which row of the shared output buffer hit `h` fills - see the header.  It is the router's
/// slot, not `h`, and the two differ on every layer where some experts are resident and some are not.
__global__ void down_kernel(const uint8_t* __restrict__ blob_base, const int32_t* __restrict__ slot_index,
                            const int32_t* __restrict__ dst_index, long long blob_bytes,
                            const uint8_t* __restrict__ h_q8_0, const float* __restrict__ h_scales,
                            float* __restrict__ out, int n_hits, const int32_t* __restrict__ d_count = nullptr) {
    const int warps_per_block = (int) (blockDim.x >> 5);
    const long long row = (long long) blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const long long total = (long long) n_hits * H;
    if (row >= total) return;
    const int h = (int) (row / H);
    if (d_count != nullptr && h >= *d_count) return;
    const int r = (int) (row % H);
    const int lane = threadIdx.x & 31;

    const uint8_t* blob = blob_base + (size_t) slot_index[h] * (size_t) blob_bytes;
    const uint8_t* xb = h_q8_0 + (size_t) h * (size_t) (FF / 32) * 34;
    const float acc = row_dot_s2_q8(blob + O_D_CODES + (size_t) r * ROW_D,
                                    blob + O_D_SCALES + (size_t) r * SC_D * 2, xb, FF / 32, lane,
                                    h_scales ? h_scales + (size_t) h * (size_t) (FF / 32) : nullptr);
    const float s = warp_sum(acc);
    if (lane == 0) out[(size_t) dst_index[h] * H + r] = s;
}

// The CPU subtracts the weight bias after its eight FMA accumulators have been reduced. Moving the
// subtraction into each integer dot, as the legacy kernel does, changes rounding even with equal scales.
__global__ void activation_correction_kernel(const uint8_t* q8, const float* scales, float* hx, int chunks) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= chunks) return;
    const int8_t* q = (const int8_t*) (q8 + (size_t) c * 34 + 2);
    int sum = 0;
    for (int j = 0; j < 32; ++j) sum += q[j];
    hx[c] = __fmul_rn(scales[c], (float) sum);
}

__device__ __forceinline__ int dot4(const uint8_t* codes, const int8_t* q) {
    const unsigned c = *codes;
    const int cw = (int) ((c & 3u) | (((c >> 2) & 3u) << 8) |
                          (((c >> 4) & 3u) << 16) | (((c >> 6) & 3u) << 24));
    int xw;
    memcpy(&xw, q, sizeof xw);
    return __dp4a(cw, xw, 0);
}

__device__ __forceinline__ float row_dot_cpu_order(const uint8_t* codes, const uint8_t* scales,
                                                   const uint8_t* xq, const float* xs,
                                                   const float* hx, int blocks, int lane) {
    float acc = 0.0f;
    float corr = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const float d = f16_at(scales + 2 * b);
        const int lo = dot4(codes + b * 16 + lane,
                           (const int8_t*) (xq + (size_t) (2 * b) * 34 + 2) + lane * 4);
        const int hi = dot4(codes + b * 16 + 8 + lane,
                           (const int8_t*) (xq + (size_t) (2 * b + 1) * 34 + 2) + lane * 4);
        acc = __fmaf_rn(__fmul_rn(d, xs[2 * b]), (float) lo, acc);
        acc = __fmaf_rn(__fmul_rn(d, xs[2 * b + 1]), (float) hi, acc);
        if (lane == 0)
            corr = __fadd_rn(corr, __fmul_rn(d, __fadd_rn(hx[2 * b], hx[2 * b + 1])));
    }
    // _mm_add_ps(low128, high128), then two _mm_hadd_ps. The pair order is 4, 1, 2;
    // a standard shuffle tree in the order 4, 2, 1 is a different floating-point expression.
    constexpr unsigned mask = 0xffffffffu;
    acc = __fadd_rn(acc, __shfl_down_sync(mask, acc, 4, 8));
    acc = __fadd_rn(acc, __shfl_down_sync(mask, acc, 1, 8));
    acc = __fadd_rn(acc, __shfl_down_sync(mask, acc, 2, 8));
    return __fsub_rn(acc, corr);  // Only lane zero is consumed.
}

template <bool DOWN>
__global__ void cpu_order_projection_kernel(const uint8_t* blob_base, const int32_t* slots,
                                              const int32_t* destinations, long long blob_bytes,
                                              const uint8_t* xq, const float* xs, const float* hx,
                                              float* out, int n_hits) {
    constexpr int rows_per_hit = DOWN ? H : 2 * FF;
    const int row = blockIdx.x * (blockDim.x / 8) + threadIdx.x / 8;
    if (row >= n_hits * rows_per_hit) return;
    const int h = row / rows_per_hit;
    const int r = row % rows_per_hit;
    const int lane = threadIdx.x & 7;
    const uint8_t* blob = blob_base + (size_t) slots[h] * (size_t) blob_bytes;
    const int chunks_offset = DOWN ? h * (FF / 32) : 0;
    const uint8_t* codes = DOWN ? blob + O_D_CODES + (size_t) r * ROW_D : blob + (size_t) r * ROW_GU;
    const uint8_t* scales = DOWN ? blob + O_D_SCALES + (size_t) r * SC_D * 2
                                 : blob + O_GU_SCALES + (size_t) r * SC_GU * 2;
    const float value = row_dot_cpu_order(codes, scales, xq + (size_t) chunks_offset * 34,
                                           xs + chunks_offset, hx + chunks_offset,
                                           DOWN ? SC_D : SC_GU, lane);
    if (lane != 0) return;
    if (DOWN) out[(size_t) destinations[h] * H + r] = value;
    else out[((r & 1) ? (size_t) n_hits * FF : 0) + (size_t) h * FF + (r >> 1)] = value;
}

__global__ void cpu_order_swiglu_kernel(float* gu, int pairs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    const float g = gu[i];
    // Accurate fp32 exponential; __expf's approximation would introduce an additional source of error.
    // CPU/GPU libc last-bit differences are diagnosed separately by the micro, not hidden with FP64 here.
    const float eg = expf(-g);
    gu[i] = __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, eg)), gu[pairs + i]);
}

__global__ void cpu_order_quantize_kernel(const float* x, uint8_t* blocks, float* scales,
                                           float* hx, int chunks) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= chunks) return;
    const float* xb = x + c * 32;
    uint8_t* out = blocks + (size_t) c * 34;
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xb[j]));
    const float s = amax > 0.0f ? __fdiv_rn(amax, 127.0f) : 0.0f;
    const float inv = s > 0.0f ? __fdiv_rn(1.0f, s) : 0.0f;
    scales[c] = s;
    const uint16_t bits = __half_as_ushort(__float2half_rn(s));
    out[0] = (uint8_t) bits;
    out[1] = (uint8_t) (bits >> 8);
    int sum = 0;
    for (int j = 0; j < 32; ++j) {
        const float t = __fmul_rn(xb[j], inv);
        int v = (int) __fadd_rn(t, t >= 0.0f ? 0.5f : -0.5f);
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        out[2 + j] = (uint8_t) (int8_t) v;
        sum += v;
    }
    hx[c] = __fmul_rn(s, (float) sum);
}

constexpr int THREADS = 256;

void check(const char* who, void* stream) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", who, cudaGetErrorString(e));
        std::exit(1);
    }
    // Deliberately NOT synchronising for a non-null stream: this is called once per layer from a captured
    // graph's worth of work, and `finish()`'s null-stream sync in `s_gemv.cu` is the pattern that made a whole
    // round of measurements the driver's cost instead of the kernel's (Memory/ERRORS.md RC-7).
    (void) stream;
}

}  // namespace

uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    // R4.2h: the fp32 scales for the INTERMEDIATE's own quantization, one per 32-element chunk per hit.
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) +
           ((xh + 15) & ~15ull);
}

void moe_hit_grouped_s2(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                        int64_t n_hits, int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch, float* out,
                        void* stream, const float* x_scales) {
    if (n_hits <= 0) return;
    cudaStream_t cs = (cudaStream_t) stream;
    const int warps = THREADS / 32;

    const uint64_t gu_bytes = ((uint64_t) n_hits * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n_hits * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);

    // 1. gate + up, one launch for every row of every hit.
    {
        const long long rows = n_hits * 2LL * FF;
        const unsigned blocks = (unsigned) ((rows + warps - 1) / warps);
        gu_kernel<<<blocks, THREADS, 0, cs>>>(blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up,
                                              (int) n_hits);
        check("moe_hit_grouped_s2/gu", stream);
    }
    // 2. silu(gate) * up.
    {
        const long long pairs = n_hits * (long long) FF;
        const unsigned blocks = (unsigned) ((pairs + THREADS - 1) / THREADS);
        swiglu_kernel<<<blocks, THREADS, 0, cs>>>(gate_up, pairs);
        check("moe_hit_grouped_s2/swiglu", stream);
    }
    // 3. the intermediate's own contract, which is `ggml_mul_mat`'s rule and NOT a choice: the down weight is
    //    Q2_0, whose `vec_dot_type` is Q8_0.  `gate_up` holds the pairs; the product went into the first half.
    //    **R4.2h: the CPU quantizes this intermediate into `a2` with fp32 scales too** (`expert.cpp:232`), so
    //    when the caller supplies `x_scales` the intermediate gets the CPU's contract as well - otherwise the
    //    down projection would keep the very disagreement the gate/up projection just had removed.
    if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, n_hits * (int64_t) FF, stream);
    else quantize_q8_0(gate_up, h_q8_0, n_hits * (int64_t) FF, stream);
    // 4. down.
    {
        const long long rows = n_hits * (long long) H;
        const unsigned blocks = (unsigned) ((rows + warps - 1) / warps);
        down_kernel<<<blocks, THREADS, 0, cs>>>(blob_base, slot_index, dst_index, blob_bytes, h_q8_0,
                                                x_scales != nullptr ? h_scales : nullptr, out, (int) n_hits);
        check("moe_hit_grouped_s2/down", stream);
    }
}

namespace {
// Plan v0.3 P4 token graph: which of this layer's routed experts are resident, decided ON THE DEVICE from the
// static residency row, so no host step sits between the ring and the hit kernels.  One warp; k <= 32.
__global__ void hit_select_kernel(const int32_t* __restrict__ ids, const int32_t* __restrict__ res_row, int k,
                                  int n_expert, int32_t* __restrict__ slot, int32_t* __restrict__ dst,
                                  int32_t* __restrict__ count) {
    const int lane = threadIdx.x;
    int s = -1;
    if (lane < k) {
        const int e = ids[lane];
        if (e >= 0 && e < n_expert) s = res_row[e];
    }
    const unsigned hit = __ballot_sync(0xffffffffu, s >= 0);
    if (s >= 0) {
        const int at = __popc(hit & ((1u << lane) - 1u));
        slot[at] = s;
        dst[at] = lane;
    }
    if (lane == 0) *count = __popc(hit);
}

// Plan v0.3 P6: the same for up to 128 routed entries (a verify window of T tokens x k): four warps, ballots
// compacted in entry order.
__global__ void hit_select_multi_kernel(const int32_t* __restrict__ ids, const int32_t* __restrict__ res_row, int n,
                                        int n_expert, int32_t* __restrict__ slot, int32_t* __restrict__ dst,
                                        int32_t* __restrict__ count) {
    __shared__ int warp_count[4];
    const int i = threadIdx.x, lane = i & 31, warp = i >> 5;
    int s = -1;
    if (i < n) {
        const int e = ids[i];
        if (e >= 0 && e < n_expert) s = res_row[e];
    }
    const unsigned hit = __ballot_sync(0xffffffffu, s >= 0);
    if (lane == 0) warp_count[warp] = __popc(hit);
    __syncthreads();
    int before = 0;
    for (int w = 0; w < warp; ++w) before += warp_count[w];
    if (s >= 0) {
        const int at = before + __popc(hit & ((1u << lane) - 1u));
        slot[at] = s;
        dst[at] = i;
    }
    if (i == 0) *count = warp_count[0] + warp_count[1] + warp_count[2] + warp_count[3];
}

__global__ void add_hits_kernel(float* __restrict__ parts, const float* __restrict__ hit_out,
                                const int32_t* __restrict__ dst, const int32_t* __restrict__ count, int n_embd) {
    const int h = blockIdx.y;
    if (h >= *count) return;
    const size_t row = (size_t) dst[h] * (size_t) n_embd;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_embd; i += gridDim.x * blockDim.x)
        parts[row + i] += hit_out[row + i];
}
}  // namespace

void moe_hit_select(const int32_t* ids, const int32_t* res_row, int k, int n_expert, int32_t* slot, int32_t* dst,
                    int32_t* count, void* stream) {
    if (k < 1 || k > 32) { std::fprintf(stderr, "moe_hit_select: k must be 1..32\n"); std::exit(1); }
    hit_select_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(ids, res_row, k, n_expert, slot, dst, count);
    check("moe_hit_select", stream);
}

void moe_hit_grouped_s2_dev(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                            const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                            void* scratch, float* out, void* stream, const float* x_scales) {
    if (cap <= 0) return;
    cudaStream_t cs = (cudaStream_t) stream;
    const int warps = THREADS / 32;
    const uint64_t gu_bytes = ((uint64_t) cap * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    {
        const long long rows = cap * 2LL * FF;
        gu_kernel<<<(unsigned) ((rows + warps - 1) / warps), THREADS, 0, cs>>>(
            blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, (int) cap, d_count);
        check("moe_hit_grouped_s2_dev/gu", stream);
    }
    {
        const long long pairs = cap * (long long) FF;
        swiglu_kernel<<<(unsigned) ((pairs + THREADS - 1) / THREADS), THREADS, 0, cs>>>(gate_up, pairs);
        check("moe_hit_grouped_s2_dev/swiglu", stream);
    }
    if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap * (int64_t) FF, stream);
    else quantize_q8_0(gate_up, h_q8_0, cap * (int64_t) FF, stream);
    {
        const long long rows = cap * (long long) H;
        down_kernel<<<(unsigned) ((rows + warps - 1) / warps), THREADS, 0, cs>>>(
            blob_base, slot_index, dst_index, blob_bytes, h_q8_0, x_scales != nullptr ? h_scales : nullptr, out,
            (int) cap, d_count);
        check("moe_hit_grouped_s2_dev/down", stream);
    }
}

void moe_hit_select_multi(const int32_t* ids, const int32_t* res_row, int n, int n_expert, int32_t* slot, int32_t* dst,
                          int32_t* count, void* stream) {
    if (n < 1 || n > 128) { std::fprintf(stderr, "moe_hit_select_multi: n must be 1..128\n"); std::exit(1); }
    hit_select_multi_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(ids, res_row, n, n_expert, slot, dst, count);
    check("moe_hit_select_multi", stream);
}

void moe_hit_grouped_s2_multi(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                              const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                              const float* x_scales, int k_per_token, void* scratch, float* out, void* stream) {
    if (cap <= 0) return;
    cudaStream_t cs = (cudaStream_t) stream;
    const int warps = THREADS / 32;
    const uint64_t gu_bytes = ((uint64_t) cap * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    {
        const long long rows = cap * 2LL * FF;
        gu_kernel<<<(unsigned) ((rows + warps - 1) / warps), THREADS, 0, cs>>>(
            blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, (int) cap, d_count, dst_index, k_per_token);
        check("moe_hit_grouped_s2_multi/gu", stream);
    }
    {
        const long long pairs = cap * (long long) FF;
        swiglu_kernel<<<(unsigned) ((pairs + THREADS - 1) / THREADS), THREADS, 0, cs>>>(gate_up, pairs);
        check("moe_hit_grouped_s2_multi/swiglu", stream);
    }
    if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap * (int64_t) FF, stream);
    else quantize_q8_0(gate_up, h_q8_0, cap * (int64_t) FF, stream);
    {
        const long long rows = cap * (long long) H;
        down_kernel<<<(unsigned) ((rows + warps - 1) / warps), THREADS, 0, cs>>>(
            blob_base, slot_index, dst_index, blob_bytes, h_q8_0, x_scales != nullptr ? h_scales : nullptr, out,
            (int) cap, d_count);
        check("moe_hit_grouped_s2_multi/down", stream);
    }
}

namespace {
// Plan v0.3 P6: experts GROUPED - one blob pointer per group (a VRAM slot or a mapped host blob read over PCIe),
// every entry of the group (a token routed to that expert) computed from ONE read of each row.  Per entry the
// arithmetic is `row_dot_s2_q8`'s, chunk by chunk in the same lane order, so every entry is bitwise the per-entry
// hit kernel's.
constexpr int GU_CHUNKS = (H / 32 + 31) / 32;   // 3: chunks of a gate/up row per lane (80 chunks / 32 lanes)
constexpr int GMAX = 8;                          // group-size split: <=8 entries -> tile kernels, more -> big
constexpr int GU_ROWS = 32;                      // gate/up rows per block: 4 per warp
constexpr int D_ROWS = 64;                       // down rows per block: 8 per warp

// One activation chunk's contribution, `row_dot_s2_q8`'s inner body with the 32 int8 of the chunk already in
// aligned words: same dp4a sequence, same float expression, so the result is bitwise the per-entry kernel's.
__device__ __forceinline__ float chunk_dot(uint2 cb, const int* xw, float dw, float dx) {
    const uint8_t* cbytes = (const uint8_t*) &cb;
    int s = 0, hx = 0;
    const int ones = 0x01010101;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned cbyte = cbytes[j];
        const int cw = (int) ((cbyte & 3u) | (((cbyte >> 2) & 3u) << 8) | (((cbyte >> 4) & 3u) << 16) |
                              (((cbyte >> 6) & 3u) << 24));
        s = __dp4a(cw, xw[j], s);
        hx = __dp4a(ones, xw[j], hx);
    }
    return dw * dx * (float) (s - hx);
}

// Gate/up: a block = GU_ROWS rows of ONE group, at most GMAX entries (bigger groups are the big kernel's -
// see below).  The group's activations (each entry's token row of x_q8_0 and its fp32 scales) are staged once
// into shared memory as aligned words; each warp then walks its rows, loading each lane's code chunks once and
// dotting them with every entry.
__global__ void __launch_bounds__(256) gu_grouped_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                         const int32_t* __restrict__ grp_start,
                                                         const int32_t* __restrict__ n_groups,
                                                         const int32_t* __restrict__ ent_tok,
                                                         const uint8_t* __restrict__ x_q8_0,
                                                         const float* __restrict__ x_scales,
                                                         float* __restrict__ gate_up, int cap_entries) {
    __shared__ int xs_q[GMAX][H / 4];          // the entries' int8 activations as words (2560 B each)
    __shared__ float xs_d[GMAX][H / 32];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], ne = grp_start[g + 1] - e0;
    if (ne > GMAX) return;                     // big group: gu_grouped_big_kernel owns it
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < ne * (H / 32); i += blockDim.x) {
        const int k = i / (H / 32), c = i - k * (H / 32);
        const uint8_t* xb = x_q8_0 + (size_t) ent_tok[e0 + k] * (size_t) (H / 32) * 34 + (size_t) c * 34;
        xs_d[k][c] = x_scales ? x_scales[(size_t) ent_tok[e0 + k] * (H / 32) + c] : f16_at(xb);
        const int8_t* q = (const int8_t*) (xb + 2);
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            int v;
            memcpy(&v, q + 4 * w, 4);
            xs_q[k][c * 8 + w] = v;
        }
    }
    __syncthreads();
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = blockIdx.x * GU_ROWS;
    for (int rr = warp; rr < GU_ROWS; rr += 8) {
        const int i = row0 + rr;
        const uint8_t* codes = blob + (size_t) i * ROW_GU;
        const uint8_t* scales = blob + O_GU_SCALES + (size_t) i * SC_GU * 2;
        uint2 cb[GU_CHUNKS];
        float dw[GU_CHUNKS];
#pragma unroll
        for (int q = 0; q < GU_CHUNKS; ++q) {
            const int c = lane + 32 * q;
            if (c < H / 32) {
                cb[q] = *(const uint2*) (codes + (size_t) c * 8);
                dw[q] = f16_at(scales + (size_t) (c >> 1) * 2);
            }
        }
        for (int k = 0; k < ne; ++k) {
            float acc = 0.0f;
#pragma unroll
            for (int q = 0; q < GU_CHUNKS; ++q) {
                const int c = lane + 32 * q;
                if (c >= H / 32) break;
                acc += chunk_dot(cb[q], &xs_q[k][c * 8], dw[q], xs_d[k][c]);
            }
            const float sum = warp_sum(acc);
            if (lane == 0) {
                const int e = e0 + k, r = i >> 1;
                const size_t base = (i & 1) ? ((size_t) cap_entries * FF + (size_t) e * FF) : ((size_t) e * FF);
                gate_up[base + (size_t) r] = sum;
            }
        }
    }
}

// BIG GROUPS (prefill chunks route hundreds of entries to one expert): one lane per ENTRY, no shared staging,
// no warp reduction.  A warp holds 32 entries; each lane walks its entry's activation chunks straight from
// global (the whole group's activations stay hot in L2: every row re-reads them) while the row's weight chunks
// are warp-broadcast loads.  Every lane accumulates a full row dot in registers (chunks in order), so the
// 5-shfl warp_sum per (row, entry) of the tile kernel disappears and occupancy stays high.  Per entry the
// arithmetic is `chunk_dot`'s over the same bytes; the chunk order is sequential rather than lane-strided, so
// results agree with the tile kernel to float rounding, not to the bit.
constexpr int GB_ROWS = 64;                    // rows per block: 8 warps x 8 rows
__global__ void __launch_bounds__(256) gu_grouped_big_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                             const int32_t* __restrict__ grp_start,
                                                             const int32_t* __restrict__ n_groups,
                                                             const int32_t* __restrict__ ent_tok,
                                                             const uint8_t* __restrict__ x_q8_0,
                                                             const float* __restrict__ x_scales,
                                                             float* __restrict__ gate_up, int cap_entries) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], ne = grp_start[g + 1] - e0;
    if (ne <= GMAX) return;                    // small group: gu_grouped_kernel owns it
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = blockIdx.x * GB_ROWS;
    for (int et = 0; et < ne; et += 32) {
        const int e = e0 + et + lane;
        const bool live = (et + lane) < ne;
        const uint8_t* xb =
            live ? x_q8_0 + (size_t) ent_tok[e] * (size_t) (H / 32) * 34 : (const uint8_t*) nullptr;
        const float* xs = live && x_scales ? x_scales + (size_t) ent_tok[e] * (H / 32) : nullptr;
        float acc[GB_ROWS / 8];
#pragma unroll
        for (int r = 0; r < GB_ROWS / 8; ++r) acc[r] = 0.0f;
#pragma unroll
        for (int r = 0; r < GB_ROWS / 8; ++r) {
            const int i = row0 + warp + r * 8;
            const uint8_t* codes = blob + (size_t) i * ROW_GU;
            const uint8_t* scales = blob + O_GU_SCALES + (size_t) i * SC_GU * 2;
            for (int c = 0; c < H / 32; ++c) {
                const uint2 cb = *(const uint2*) (codes + (size_t) c * 8);      // warp-broadcast
                const float dw = f16_at(scales + (size_t) (c >> 1) * 2);        // warp-broadcast
                int xw[8];
                float dx = 0.0f;
                if (live) {
                    const uint8_t* xc = xb + (size_t) c * 34;
                    dx = xs ? xs[c] : f16_at(xc);
                    const int8_t* q = (const int8_t*) (xc + 2);
#pragma unroll
                    for (int w = 0; w < 8; ++w) memcpy(&xw[w], q + 4 * w, 4);
                } else {
#pragma unroll
                    for (int w = 0; w < 8; ++w) xw[w] = 0;
                }
                acc[r] += chunk_dot(cb, xw, dw, dx);
            }
        }
        if (live) {
#pragma unroll
            for (int r = 0; r < GB_ROWS / 8; ++r) {
                const int i = row0 + warp + r * 8, rr = i >> 1;
                const size_t base = (i & 1) ? ((size_t) cap_entries * FF + (size_t) e * FF) : ((size_t) e * FF);
                gate_up[base + (size_t) rr] = acc[r];
            }
        }
    }
}

// Down: a block = D_ROWS rows of ONE group (at most GMAX entries, as above); the entries' quantized
// intermediates staged once.  A down row is 20 chunks, so lanes 0..19 each hold one chunk, as in the
// per-entry kernel.
__global__ void __launch_bounds__(256) down_grouped_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                           const int32_t* __restrict__ grp_start,
                                                           const int32_t* __restrict__ n_groups,
                                                           const int32_t* __restrict__ ent_dst,
                                                           const uint8_t* __restrict__ h_q8_0,
                                                           const float* __restrict__ h_scales,
                                                           float* __restrict__ out) {
    __shared__ int hs_q[GMAX][FF / 4];
    __shared__ float hs_d[GMAX][FF / 32];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], ne = grp_start[g + 1] - e0;
    if (ne > GMAX) return;                     // big group: down_grouped_big_kernel owns it
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < ne * (FF / 32); i += blockDim.x) {
        const int k = i / (FF / 32), c = i - k * (FF / 32);
        const uint8_t* xb = h_q8_0 + (size_t) (e0 + k) * (size_t) (FF / 32) * 34 + (size_t) c * 34;
        hs_d[k][c] = h_scales ? h_scales[(size_t) (e0 + k) * (FF / 32) + c] : f16_at(xb);
        const int8_t* q = (const int8_t*) (xb + 2);
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            int v;
            memcpy(&v, q + 4 * w, 4);
            hs_q[k][c * 8 + w] = v;
        }
    }
    __syncthreads();
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = blockIdx.x * D_ROWS;
    for (int rr = warp; rr < D_ROWS; rr += 8) {
        const int r = row0 + rr;
        const uint8_t* codes = blob + O_D_CODES + (size_t) r * ROW_D;
        const uint8_t* scales = blob + O_D_SCALES + (size_t) r * SC_D * 2;
        const int c = lane;
        uint2 cb = make_uint2(0, 0);
        float dw = 0.0f;
        if (c < FF / 32) {
            cb = *(const uint2*) (codes + (size_t) c * 8);
            dw = f16_at(scales + (size_t) (c >> 1) * 2);
        }
        for (int k = 0; k < ne; ++k) {
            float acc = 0.0f;
            if (c < FF / 32) acc += chunk_dot(cb, &hs_q[k][c * 8], dw, hs_d[k][c]);
            const float sum = warp_sum(acc);
            if (lane == 0) out[(size_t) ent_dst[e0 + k] * H + r] = sum;
        }
    }
}

// Big-group down: the lane-per-entry twin of gu_grouped_big_kernel.  h_q8_0 is indexed by ENTRY NUMBER (the
// grouped intermediate), and a down row is FF/32 = 20 chunks.
__global__ void __launch_bounds__(256) down_grouped_big_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                               const int32_t* __restrict__ grp_start,
                                                               const int32_t* __restrict__ n_groups,
                                                               const int32_t* __restrict__ ent_dst,
                                                               const uint8_t* __restrict__ h_q8_0,
                                                               const float* __restrict__ h_scales,
                                                               float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], ne = grp_start[g + 1] - e0;
    if (ne <= GMAX) return;                    // small group: down_grouped_kernel owns it
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = blockIdx.x * GB_ROWS;
    for (int et = 0; et < ne; et += 32) {
        const int e = e0 + et + lane;
        const bool live = (et + lane) < ne;
        const uint8_t* xb =
            live ? h_q8_0 + (size_t) e * (size_t) (FF / 32) * 34 : (const uint8_t*) nullptr;
        const float* xs = live && h_scales ? h_scales + (size_t) e * (FF / 32) : nullptr;
        float acc[GB_ROWS / 8];
#pragma unroll
        for (int r = 0; r < GB_ROWS / 8; ++r) acc[r] = 0.0f;
#pragma unroll
        for (int r = 0; r < GB_ROWS / 8; ++r) {
            const int rr = row0 + warp + r * 8;
            const uint8_t* codes = blob + O_D_CODES + (size_t) rr * ROW_D;
            const uint8_t* scales = blob + O_D_SCALES + (size_t) rr * SC_D * 2;
            for (int c = 0; c < FF / 32; ++c) {
                const uint2 cb = *(const uint2*) (codes + (size_t) c * 8);      // warp-broadcast
                const float dw = f16_at(scales + (size_t) (c >> 1) * 2);        // warp-broadcast
                int xw[8];
                float dx = 0.0f;
                if (live) {
                    const uint8_t* xc = xb + (size_t) c * 34;
                    dx = xs ? xs[c] : f16_at(xc);
                    const int8_t* q = (const int8_t*) (xc + 2);
#pragma unroll
                    for (int w = 0; w < 8; ++w) memcpy(&xw[w], q + 4 * w, 4);
                } else {
#pragma unroll
                    for (int w = 0; w < 8; ++w) xw[w] = 0;
                }
                acc[r] += chunk_dot(cb, xw, dw, dx);
            }
        }
        if (live) {
#pragma unroll
            for (int r = 0; r < GB_ROWS / 8; ++r) {
                const int rr = row0 + warp + r * 8;
                out[(size_t) ent_dst[e] * H + rr] = acc[r];
            }
        }
    }
}
}  // namespace

namespace {
// Plan v0.3 P6: groups built on the device when every expert is resident at `base + id * blob` (the MTP layer):
// one block of 128 threads, groups in first-appearance order, entries of a group in routing order.
__global__ void group_resident_kernel(const int32_t* __restrict__ ids, int n, int k_per_tok, const uint8_t* base,
                                      long long blob, unsigned long long* __restrict__ grp_ptr,
                                      int32_t* __restrict__ grp_start, int32_t* __restrict__ counts,
                                      int32_t* __restrict__ ent_dst, int32_t* __restrict__ ent_tok) {
    __shared__ int e_s[128], first_s[128], size_s[128], gidx_s[128], gstart_s[129];
    const int i = threadIdx.x;
    const int e = i < n ? ids[i] : -1;
    e_s[i] = e;
    __syncthreads();
    int first = i, rank = 0, size = 0;
    if (i < n) {
        for (int j = 0; j < i; ++j)
            if (e_s[j] == e) { if (first == i) first = j; ++rank; }
        if (first == i)
            for (int j = i; j < n; ++j) size += e_s[j] == e;
    }
    first_s[i] = first;
    size_s[i] = (i < n && first == i) ? size : 0;
    __syncthreads();
    if (i == 0) {
        int gi = 0, acc = 0;
        for (int j = 0; j < n; ++j)
            if (first_s[j] == j) {
                gidx_s[j] = gi;
                gstart_s[gi] = acc;
                grp_ptr[gi] = (unsigned long long) (base + (size_t) e_s[j] * (size_t) blob);
                grp_start[gi] = acc;
                acc += size_s[j];
                ++gi;
            }
        grp_start[gi] = acc;
        counts[0] = gi;
        counts[1] = acc;
    }
    __syncthreads();
    if (i < n) {
        const int at = gstart_s[gidx_s[first]] + rank;
        ent_dst[at] = i;
        ent_tok[at] = i / k_per_tok;
    }
}
}  // namespace

void moe_group_resident(const int32_t* ids, int n, int k_per_tok, const uint8_t* base, int64_t blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* counts, int32_t* ent_dst,
                        int32_t* ent_tok, void* stream) {
    if (n < 1 || n > 128) { std::fprintf(stderr, "moe_group_resident: n must be 1..128\n"); std::exit(1); }
    group_resident_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(ids, n, k_per_tok, base, (long long) blob, grp_ptr,
                                                               grp_start, counts, ent_dst, ent_tok);
    check("moe_group_resident", stream);
}

void moe_grouped_s2(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                    const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    cudaStream_t cs = (cudaStream_t) stream;
    const uint64_t gu_bytes = ((uint64_t) cap_entries * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap_entries * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    {
        // small groups (ne <= GMAX) and big ones split device-side: both kernels see every group and the
        // wrong one exits on the first lines, so decode windows pay one extra empty wave, nothing more.
        gu_grouped_kernel<<<dim3((unsigned) (2 * FF / GU_ROWS), (unsigned) cap_groups), 256, 0, cs>>>(
            grp_ptr, grp_start, n_groups, ent_tok, x_q8_0, x_scales, gate_up, (int) cap_entries);
        gu_grouped_big_kernel<<<dim3((unsigned) (2 * FF / GB_ROWS), (unsigned) cap_groups), 256, 0, cs>>>(
            grp_ptr, grp_start, n_groups, ent_tok, x_q8_0, x_scales, gate_up, (int) cap_entries);
        check("moe_grouped_s2/gu", stream);
    }
    {
        const long long pairs = cap_entries * (long long) FF;
        swiglu_kernel<<<(unsigned) ((pairs + THREADS - 1) / THREADS), THREADS, 0, cs>>>(gate_up, pairs);
        check("moe_grouped_s2/swiglu", stream);
    }
    if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap_entries * (int64_t) FF, stream);
    else quantize_q8_0(gate_up, h_q8_0, cap_entries * (int64_t) FF, stream);
    {
        down_grouped_kernel<<<dim3((unsigned) (H / D_ROWS), (unsigned) cap_groups), 256, 0, cs>>>(
            grp_ptr, grp_start, n_groups, ent_dst, h_q8_0, x_scales != nullptr ? h_scales : nullptr, out);
        down_grouped_big_kernel<<<dim3((unsigned) (H / GB_ROWS), (unsigned) cap_groups), 256, 0, cs>>>(
            grp_ptr, grp_start, n_groups, ent_dst, h_q8_0, x_scales != nullptr ? h_scales : nullptr, out);
        check("moe_grouped_s2/down", stream);
    }
}

void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream) {
    if (cap <= 0) return;
    const dim3 grid((unsigned) ((n_embd + 255) / 256 < 8 ? (n_embd + 255) / 256 : 8), (unsigned) cap);
    add_hits_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(parts, hit_out, dst, count, (int) n_embd);
    check("moe_hit_add", stream);
}

void moe_hit_grouped_s2_cpu_order(const uint8_t* blob_base, const int32_t* slot_index,
                                 const int32_t* dst_index, int64_t n_hits, int64_t blob_bytes,
                                 const uint8_t* x_q8_0, void* scratch, float* out, void* stream,
                                 const float* x_scales, float* gate_up_trace) {
    if (n_hits <= 0) return;
    if (x_scales == nullptr) {
        std::fprintf(stderr, "moe_hit_grouped_s2_cpu_order requires fp32 activation scales\n");
        std::exit(1);
    }
    cudaStream_t cs = (cudaStream_t) stream;
    const uint64_t gu_bytes = ((uint64_t) n_hits * 2 * FF * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n_hits * (FF / 32) * 34 + 15) & ~15ull;
    const uint64_t scale_bytes = ((uint64_t) n_hits * (FF / 32) * 4 + 15) & ~15ull;
    float* gu = (float*) scratch;
    uint8_t* hq = (uint8_t*) scratch + gu_bytes;
    float* hs = (float*) (hq + q8_bytes);
    float* hh = (float*) ((uint8_t*) hs + scale_bytes);
    float* xh = (float*) ((uint8_t*) hh + scale_bytes);
    activation_correction_kernel<<<(H / 32 + THREADS - 1) / THREADS, THREADS, 0, cs>>>(
        x_q8_0, x_scales, xh, H / 32);
    check("cpu_order/input_correction", stream);
    const int rows_per_block = THREADS / 8;
    cpu_order_projection_kernel<false><<<(unsigned) ((n_hits * 2 * FF + rows_per_block - 1) / rows_per_block),
                                            THREADS, 0, cs>>>(
        blob_base, slot_index, dst_index, blob_bytes, x_q8_0, x_scales, xh, gu, (int) n_hits);
    check("cpu_order/gate_up", stream);
    if (gate_up_trace != nullptr &&
        cudaMemcpyAsync(gate_up_trace, gu, (size_t) n_hits * 2 * FF * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
        std::fprintf(stderr, "cpu_order/gate_up_trace copy failed\n");
        std::exit(1);
    }
    cpu_order_swiglu_kernel<<<(unsigned) ((n_hits * FF + THREADS - 1) / THREADS), THREADS, 0, cs>>>(
        gu, (int) n_hits * FF);
    check("cpu_order/swiglu", stream);
    cpu_order_quantize_kernel<<<(unsigned) ((n_hits * (FF / 32) + THREADS - 1) / THREADS), THREADS, 0, cs>>>(
        gu, hq, hs, hh, (int) n_hits * (FF / 32));
    check("cpu_order/intermediate_quantize", stream);
    cpu_order_projection_kernel<true><<<(unsigned) ((n_hits * H + rows_per_block - 1) / rows_per_block),
                                           THREADS, 0, cs>>>(
        blob_base, slot_index, dst_index, blob_bytes, hq, hs, hh, out, (int) n_hits);
    check("cpu_order/down", stream);
}

}  // namespace strata::kernels

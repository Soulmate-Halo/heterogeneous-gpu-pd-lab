// src/kernels/cuda/quantize_act.cu - P2.S2: quantize an activation the way ggml converts src1.
//
// WHY THIS EXISTS AT ALL.  `ggml_mul_mat` converts the ACTIVATION (src1) to the weight's `vec_dot_type`
// before the dot product - the rule established over rounds 119-136 and written into `ref/quant.py` as
// `apply_src1`.  The CPU expert path already honours it (`bench/micro/cpu_s2.cpp` quantizes to int8 and uses
// `vpdpbusd`), and its P0.T2 parity passes at 1.461e-06.  The GPU kernels added in rounds 166-177 take FP16
// activations instead.  **Left that way, the two paths would compute different numbers for different experts
// of the SAME token** - not a tolerance question, an inconsistency inside one forward pass.
//
// Q2_0 IS THE CASE THAT MATTERS: the routed experts are all Q2_0, `Q2_0 -> Q8_0`, and that is 31.64 GiB of the
// 38 GiB pack.  Q8_K (for the K-quants and IQ4_XS) is the other half of the table and is not here yet.
//
// THE THREE SUBTLETIES, each of which was paid for once already and is transcribed rather than recalled:
//
//   1. `d32 = amax / 127.0f` is computed in FP32, and the QUANTIZED INTEGERS divide by **d32**, not by the
//      fp16-rounded value that gets stored.  Conflating the two shifts every quant near a rounding boundary -
//      measured against ggml's own bytes, 18 of 80 blocks differed, and because Q2_0 experts convert to Q8_0
//      the error landed in all 48 MoE blocks and made the end-to-end KL WORSE.
//   2. The reference divides in FLOAT64 (`blk.astype(np.float64) / float(np.float32(d32))`) and rounds with
//      `rint` - half to even.  This kernel therefore divides in double too: an FP32 division followed by
//      `rintf` differs from the reference wherever the f32 quotient rounds across a .5 boundary.
//   3. The DEQUANTIZED value is `q * d16`, the fp16 scale, not `q * d32`.  The block stores fp16 and that is
//      what a reader multiplies by.
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK8_0 = 32;

// THE fp16 CONVERSION LIVES IN `strata/kernels/f16_bits.hpp`, and this file used to carry its own copy.
//
// Round 198 found the private copy wrong in a way no fixture here could see: it tested `if (exp >= 31)` to
// detect an out-of-range exponent, which conflates an f32 INF/NAN (raw exponent 255) with a FINITE value too
// large for fp16.  Every finite overflow - 1e30, 65536, 1e45 - came back as a NaN instead of saturating to
// inf, and this file's fixture is O(1) throughout, so it passed.  A numpy-generated oracle caught it on its
// first run.  Three copies of a converter whose failure mode is silent wrong bits was two copies too many.


__global__ void quantize_q8_0_kernel(const float* __restrict__ x, uint8_t* __restrict__ blocks,
                                     long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float* xb = x + b * QK8_0;
    uint8_t* out = blocks + b * 34;                 // { fp16 d ; int8 qs[32] }

    float amax = 0.0f;
    for (int i = 0; i < QK8_0; ++i) amax = fmaxf(amax, fabsf(xb[i]));
    if (amax == 0.0f) {
        // ggml leaves the block zeroed: d = 0 and every q = 0.  Writing the fp16 zero explicitly rather than
        // skipping keeps the block layout deterministic for the byte comparison.
        const uint16_t zb = f16_from_f32(0.0f);
        out[0] = (uint8_t) (zb & 0xFF);
        out[1] = (uint8_t) (zb >> 8);
        for (int i = 0; i < QK8_0; ++i) out[2 + i] = 0;
        return;
    }
    const float d32 = amax / 127.0f;
    const uint16_t d16bits = f16_from_f32(d32);
    out[0] = (uint8_t) (d16bits & 0xFF);
    out[1] = (uint8_t) (d16bits >> 8);

    // double division, matching the reference exactly (subtlety 2)
    for (int i = 0; i < QK8_0; ++i) {
        double q = rint((double) xb[i] / (double) d32);
        if (q > 127.0) q = 127.0;
        if (q < -128.0) q = -128.0;
        out[2 + i] = (uint8_t) (int8_t) q;
    }
}

/// **THE HIT PATH'S QUANTIZER, AND IT EXISTS TO REPRODUCE `act_quant_q8_1` EXACTLY (R4.2h, round 331).**
///
/// The engine computes every routed expert twice.  MISSES go through `cpu/expert.cpp:138 act_quant_q8_1`
/// into the VNNI kernel; HITS went through `quantize_q8_0` above.  **Those are not the same quantization**,
/// and round 330 measured the difference with both real implementations linked
/// (`bench/micro/act_quant_parity.cu`):
///
///     int8 activations differing : 0 of 2560                       <- this rule's ties are measure-zero
///     chunks whose SCALE differs : 80 of 80, max rel 4.761e-04     <- 2^-11: fp32 vs the block's fp16
///
/// The scale is the difference that matters: `ActQ::scale` is `float` and the CPU kernel multiplies by it
/// (`expert.cpp:92`), while `row_dot_s2_q8` read `f16_at(xb)` out of the `block_q8_0`.  So this variant
/// writes the SAME 34-byte blocks (the kernel's indexing is unchanged) **and** a parallel fp32 scale array
/// carrying `d32` unrounded.
///
/// It also adopts the CPU's ROUNDING RULE - multiply by the reciprocal, round half AWAY FROM ZERO - rather
/// than this file's `rint`.  That is deliberate and it is the opposite of the direction the comment above
/// argues for.  `quantize_q8_0` was tuned to match **ggml's Q8_0 bytes**, which is right for the pack and for
/// `moe_hit_parity`.  This function's job is different: it must match **this engine's own CPU reference**,
/// because a hit and a miss for the same expert on the same layer have to produce the same number.  The CPU
/// path is the reference - C1 passes on it - so the hit path is brought to it, not the reverse.
__global__ void quantize_q8_0_scaled_kernel(const float* __restrict__ x, uint8_t* __restrict__ blocks,
                                            float* __restrict__ scales, long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float* xb = x + b * QK8_0;
    uint8_t* out = blocks + b * 34;

    float amax = 0.0f;
    for (int i = 0; i < QK8_0; ++i) amax = fmaxf(amax, fabsf(xb[i]));
    // VERBATIM from `cpu/expert.cpp:144-145`, including the `amax > 0` guard, so the fp32 value written here
    // is bit-identical to the `s` the CPU path used.
    const float s = amax > 0.f ? amax / 127.f : 0.f;
    const float inv = s > 0.f ? 1.f / s : 0.f;
    scales[b] = s;

    const uint16_t d16bits = f16_from_f32(s);
    out[0] = (uint8_t) (d16bits & 0xFF);
    out[1] = (uint8_t) (d16bits >> 8);
    for (int i = 0; i < QK8_0; ++i) {
        // VERBATIM from `cpu/expert.cpp:159-162`: reciprocal multiply, then `t + copysign(0.5, t)` truncated
        // toward zero, which is `lround`'s rule - round half away from zero.
        const float t = xb[i] * inv;
        const float r = t + (t >= 0.f ? 0.5f : -0.5f);
        int v = (int) r;
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        out[2 + i] = (uint8_t) (int8_t) v;
    }
}

__global__ void dequant_q8_0_kernel(const uint8_t* __restrict__ blocks, float* __restrict__ x,
                                    long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const uint8_t* blk = blocks + b * 34;
    const uint16_t dbits = (uint16_t) (blk[0] | (blk[1] << 8));
    const float d = f32_from_f16(dbits);
    float* out = x + b * QK8_0;
    for (int i = 0; i < QK8_0; ++i) out[i] = (float) (int8_t) blk[2 + i] * d;
}

// ===================== Q8_K =====================
//
// `block_q8_K` = { float d ; int8_t qs[256] ; int16_t bsums[16] } = 292 bytes, no padding
// (`static_assert` in ggml-common.h).  QK_K = 256.

constexpr int QK_K = 256;
constexpr int Q8K_BYTES = 292;

/// ggml's `nearest_int` (ggml-quants.c L621), transcribed rather than replaced.
///
/// The magic number is 1.5 * 2^23: adding it forces the mantissa's integer part into the low bits, and the
/// mask/subtract recover it.  The reason it is transcribed and not written as `rintf` is that the two differ
/// on exact ties - and that is the whole point of the constant, so a "cleaner" rewrite would silently change
/// which way ties go.
__device__ __forceinline__ int nearest_int_dev(float fval) {
    const float val = fval + 12582912.0f;
    int i;
    memcpy(&i, &val, 4);
    return (i & 0x007fffff) - 0x00400000;
}

__global__ void quantize_q8_K_kernel(const float* __restrict__ x, uint8_t* __restrict__ blocks,
                                     long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float* xb = x + b * QK_K;
    uint8_t* out = blocks + b * Q8K_BYTES;
    float* d = (float*) out;
    int8_t* qs = (int8_t*) (out + 4);
    int16_t* bsums = (int16_t*) (out + 4 + QK_K);

    // `max` is the SIGNED value at the largest magnitude position, and the comparison is STRICTLY greater,
    // so a tie keeps the FIRST maximum - which is what `np.argmax` does in the reference transcription too.
    float max = 0.0f, amax = 0.0f;
    for (int j = 0; j < QK_K; ++j) {
        const float ax = fabsf(xb[j]);
        if (ax > amax) {
            amax = ax;
            max = xb[j];
        }
    }
    if (amax == 0.0f) {
        // ggml writes d = 0 and zeroes qs, then `continue`s - which LEAVES bsums UNWRITTEN.  A zeroed block
        // and an untouched one are indistinguishable to the dot product, but not to a byte comparison, so
        // this zeroes bsums as well and the parity test's reference does the same.  Recorded because it is a
        // deliberate divergence from the letter of the source.
        *d = 0.0f;
        for (int j = 0; j < QK_K; ++j) qs[j] = 0;
        for (int j = 0; j < QK_K / 16; ++j) bsums[j] = 0;
        return;
    }
    const float iscale = -127.0f / max;          // -127, NOT -128; see the header
    for (int j = 0; j < QK_K; ++j) {
        // `__fmul_rn`, NOT `iscale * xb[j]`.  ggml computes the product, ROUNDS IT TO F32, and then adds
        // 12582912.0f inside `nearest_int`.  Written as a plain expression, nvcc CONTRACTS the multiply into
        // the add as an FMA - which is a more accurate product but not the same one, and it flips the result
        // wherever the true product sits just off a .5 boundary.  The first version of this kernel differed
        // from the reference in 1 element of 524,288 for exactly this reason, and `__fmul_rn` pins the
        // rounding step the source actually performs.
        const int v = nearest_int_dev(__fmul_rn(iscale, xb[j]));
        qs[j] = (int8_t) min(127, v);            // MIN only - the source has no lower clamp
    }
    for (int j = 0; j < QK_K / 16; ++j) {
        int sum = 0;
        for (int ii = 0; ii < 16; ++ii) sum += qs[j * 16 + ii];
        bsums[j] = (int16_t) sum;
    }
    *d = 1.0f / iscale;
}

__global__ void dequant_q8_K_kernel(const uint8_t* __restrict__ blocks, float* __restrict__ x,
                                    long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const uint8_t* blk = blocks + b * Q8K_BYTES;
    float d;
    memcpy(&d, blk, 4);
    const int8_t* qs = (const int8_t*) (blk + 4);
    float* out = x + b * QK_K;
    for (int i = 0; i < QK_K; ++i) out[i] = (float) qs[i] * d;
}

}  // namespace

void quantize_q8_0(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "quantize_q8_0: n %lld is not a multiple of %d\n", (long long) n, QK8_0);
        std::exit(1);
    }
    const long long nb = n / QK8_0;
    const int threads = 128;
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    quantize_q8_0_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(x, blocks, nb);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "quantize_q8_0 launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

/// See `quantize_q8_0_scaled_kernel`.  Writes the same 34-byte `block_q8_0` layout as `quantize_q8_0`, plus
/// `scales[n/32]` carrying the fp32 `s` the CPU path used - so a hit can be computed with the CPU's
/// multiplier instead of the block's fp16 `d`.  `scales` must not be null.
void quantize_q8_0_scaled(const float* x, uint8_t* blocks, float* scales, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "quantize_q8_0_scaled: n %lld is not a multiple of %d\n", (long long) n, QK8_0);
        std::exit(1);
    }
    if (scales == nullptr) {
        std::fprintf(stderr, "quantize_q8_0_scaled: scales is null\n");
        std::exit(1);
    }
    const long long nb = n / QK8_0;
    const int threads = 128;
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    quantize_q8_0_scaled_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(x, blocks, scales, nb);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "quantize_q8_0_scaled launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

void dequant_q8_0(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const long long nb = n / QK8_0;
    const int threads = 128;
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    dequant_q8_0_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(blocks, x, nb);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dequant_q8_0 launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

void quantize_q8_K(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK_K != 0) {
        std::fprintf(stderr, "quantize_q8_K: n %lld is not a multiple of %d\n", (long long) n, QK_K);
        std::exit(1);
    }
    const long long nb = n / QK_K;
    const int threads = 64;                       // one block per thread, and a block is 256 elements
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    quantize_q8_K_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(x, blocks, nb);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "quantize_q8_K launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

void dequant_q8_K(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const long long nb = n / QK_K;
    const int threads = 64;
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    dequant_q8_K_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(blocks, x, nb);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dequant_q8_K launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace strata::kernels

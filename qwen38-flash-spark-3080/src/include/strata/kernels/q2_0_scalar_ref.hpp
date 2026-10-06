#pragma once

// include/strata/kernels/q2_0_scalar_ref.hpp - a PORTABLE scalar transcription of `s2_expert_scalar`
// (src/kernels/cpu/expert.cpp) with `quant_acts = false`: exact-fp32 activations at both stages, no SIMD,
// no AVX-512, no CUDA.  It exists so the RDMA cold-expert path can be checked on machines where NEITHER the
// VNNI kernel (3080 host: no AVX-512) NOR a second GPU implementation (the Spark worker validating its own
// CUDA path) is available - a kernel checked against another copy of itself proves nothing.
//
// The blob layout constants come from `strata/kernels/cpu/expert.hpp` and are not restated.

#include "strata/kernels/cpu/expert.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace strata::kernels {

/// IEEE float -> half, round-to-nearest-even, pure integer arithmetic (the conversion the CUDA quantizers
/// get from `f16_bits.hpp`).  Finite overflow saturates to inf; NaN stays NaN.
inline uint16_t q2_ref_f16_from_f32(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t mant = x & 0x007fffffu;
    const int exp = (int) ((x >> 23) & 0xffu);
    if (exp == 255) return (uint16_t) (sign | (mant ? 0x7e00u : 0x7c00u));
    const int e = exp - 127 + 15;
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t) sign;
        const uint32_t m = mant | 0x00800000u;
        const uint32_t shift = (uint32_t) (14 - e);
        uint32_t half = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1u))) ++half;
        return (uint16_t) (sign + half);
    }
    uint32_t half = ((uint32_t) e << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;
    return (uint16_t) (sign + half);
}

/// IEEE half -> float, pure integer arithmetic (no F16C, no CUDA).
inline float q2_ref_h2f(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t man = h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        // subnormal half: scale the mantissa into a normal float
        f = sign;
        if (man) {
            uint32_t m = man, e = 127 - 15 + 1;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3ffu;
            f = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 0x1fu) {
        f = sign | 0x7f800000u | (man << 13);   // inf / nan
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof out);
    return out;
}

inline float q2_ref_h2f_at(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return q2_ref_h2f(h);
}

/// One Q2_0 expert against an exact-fp32 activation: out(H) = down(swiglu(gate(x), up(x))).
/// Bit-identical formula to `s2_expert_scalar(blob, x, out, false)`; differs from the engine's quantized
/// paths by the activation contract only, which a relative-tolerance check absorbs.
inline void q2_0_expert_scalar_ref(const uint8_t* blob, const float* x, float* out) {
    using namespace strata::kernels::cpu;
    float ff[FF];
    for (int r = 0; r < FF; ++r) {
        const uint8_t* gc = blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU;
        const uint8_t* gs = blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2;
        const uint8_t* uc = blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU;
        const uint8_t* us = blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2;
        float sg = 0.f, su = 0.f;
        for (int b = 0; b < SC_GU; ++b) {
            const float dg = q2_ref_h2f_at(gs + 2 * b), du = q2_ref_h2f_at(us + 2 * b);
            for (int j = 0; j < QK; ++j) {
                const int o = b * QK + j;
                sg += (float) (((gc[b * 16 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1) * dg * x[o];
                su += (float) (((uc[b * 16 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1) * du * x[o];
            }
        }
        ff[r] = (sg / (1.f + std::exp(-sg))) * su;
    }
    for (int r = 0; r < H; ++r) {
        const uint8_t* dc = blob + O_D_CODES + (size_t) r * ROW_D;
        const uint8_t* ds = blob + O_D_SCALES + (size_t) r * SC_D * 2;
        float acc = 0.f;
        for (int b = 0; b < SC_D; ++b) {
            const float d = q2_ref_h2f_at(ds + 2 * b);
            for (int j = 0; j < QK; ++j)
                acc += (float) (((dc[b * 16 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1) * d * ff[b * QK + j];
        }
        out[r] = acc;
    }
}

/// Relative L2 between two H-wide rows, for the tolerance checks.
inline double q2_ref_rel_l2(const float* a, const float* b, int64_t n) {
    double num = 0.0, den = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = (double) a[i] - (double) b[i];
        num += d * d;
        den += (double) b[i] * (double) b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

/// Stage-1 input quantization, VERBATIM semantics of `quantize_q8_0_scaled_kernel`
/// (src/kernels/cuda/quantize_act.cu): fp32 scale `amax/127`, reciprocal multiply, round half AWAY from
/// zero, clamp +/-127.  `xq` receives the dequantized values the hit kernels effectively compute with.
inline void q2_ref_quant_scaled(const float* x, int64_t n, float* xq) {
    for (int64_t b = 0; b < n / 32; ++b) {
        const float* xb = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::fabs(xb[i]) > amax ? std::fabs(xb[i]) : amax;
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        for (int i = 0; i < 32; ++i) {
            const float t = xb[i] * inv;
            const float r = t + (t >= 0.f ? 0.5f : -0.5f);
            int v = (int) r;
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            xq[b * 32 + i] = s * (float) v;
        }
    }
}

/// Intermediate quantization, VERBATIM semantics of `quantize_q8_0_kernel`: scale `amax/127` stored fp16,
/// double-division `rint` (nearest-even) codes, clamp -128..127, and the dot multiplies by the fp16 d.
inline void q2_ref_quant_ggml(const float* x, int64_t n, float* xq) {
    for (int64_t b = 0; b < n / 32; ++b) {
        const float* xb = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::fabs(xb[i]) > amax ? std::fabs(xb[i]) : amax;
        if (amax == 0.0f) {
            for (int i = 0; i < 32; ++i) xq[b * 32 + i] = 0.0f;
            continue;
        }
        const float d32 = amax / 127.0f;
        const float d = q2_ref_h2f(q2_ref_f16_from_f32(d32));
        for (int i = 0; i < 32; ++i) {
            double q = std::rint((double) xb[i] / (double) d32);
            if (q > 127.0) q = 127.0;
            if (q < -128.0) q = -128.0;
            xq[b * 32 + i] = d * (float) (int8_t) (int) q;
        }
    }
}

/// The GPU hit path's computation, transcription for reference checks: BOTH stages quantized the way
/// `moe_hit_grouped_s2` (with fp32 x_scales) + `quantize_q8_0_scaled` + `quantize_q8_0` do it, so a correct
/// kernel lands at fp32-evaluation-order distance (~1e-4), not at the ~1e-2 raw-vs-quantized distance.
inline void q2_0_expert_gpu_ref(const uint8_t* blob, const float* x, float* out) {
    using namespace strata::kernels::cpu;
    float xq[H], ff[FF], ffq[FF];
    q2_ref_quant_scaled(x, H, xq);
    for (int r = 0; r < FF; ++r) {
        const uint8_t* gc = blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU;
        const uint8_t* gs = blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2;
        const uint8_t* uc = blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU;
        const uint8_t* us = blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2;
        float sg = 0.f, su = 0.f;
        for (int b = 0; b < SC_GU; ++b) {
            const float dg = q2_ref_h2f_at(gs + 2 * b), du = q2_ref_h2f_at(us + 2 * b);
            for (int j = 0; j < QK; ++j) {
                const int o = b * QK + j;
                sg += (float) (((gc[b * 16 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1) * dg * xq[o];
                su += (float) (((uc[b * 16 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1) * du * xq[o];
            }
        }
        ff[r] = (sg / (1.f + std::exp(-sg))) * su;
    }
    q2_ref_quant_ggml(ff, FF, ffq);
    for (int r = 0; r < H; ++r) {
        const uint8_t* dc = blob + O_D_CODES + (size_t) r * ROW_D;
        const uint8_t* ds = blob + O_D_SCALES + (size_t) r * SC_D * 2;
        float acc = 0.f;
        for (int b = 0; b < SC_D; ++b) {
            const float d = q2_ref_h2f_at(ds + 2 * b);
            for (int j = 0; j < QK; ++j)
                acc += (float) (((dc[b * 16 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1) * d * ffq[b * QK + j];
        }
        out[r] = acc;
    }
}

} // namespace strata::kernels

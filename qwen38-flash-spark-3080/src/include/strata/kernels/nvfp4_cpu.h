// include/strata/kernels/nvfp4_cpu.h - P5: ARM NEON CPU kernel for NVFP4 cold experts.
//
// **WHAT THIS IS FOR.** Decode-time split offload: the Spark worker hands a fraction of cold
// entries to the GB10 CPU cluster (20 cores) so they compute in parallel with the GPU grouped
// kernel inside the same service window.  The numerics contract is IDENTICAL to the GPU baseline
// (nvfp4_cold.hpp):  w = e2m1(nibble) * e4m3(scale[r][k/16]) * ws2, fp32 activations, fp32
// accumulate; no process_scales, no permutation (that is the hot flavor's marlin path).
//
// **LAYOUT.** Canonical v2 cold blob (2,765,056 B): header 64 B; gate_w/up_w/down_w nibble planes
// (low nibble = even k); e4m3 scale planes (1 B per row per 16 elements); globals[6] fp32
// {gate_ws2, gate_is, up_ws2, up_is, down_ws2, down_is}.  Offsets come from nvfp4_cold.hpp.
//
// **NEON TRICK.** e2m1 values times 2 are exact int8 ({0,1,2,3,4,6,8,12} with sign), so one
// vqtbl1q_u8 decodes 16 nibbles; widened int8->fp32 FMA against vld2q-deinterleaved x (bytes hold
// elements 2b/2b+1, so even/odd lanes stay separate until the horizontal sum).  The per-group
// scale is folded with one vfmaq: acc += (0.5*e4m3*ws2) * group_partial.
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

#include "strata/kernels/nvfp4_cold.hpp"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace strata::kernels {

// e2m1(nib) * 2 as exact int8 (index = nibble 0..15).
inline constexpr int8_t NVFP4_E2M1_X2[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                                             0, -1, -2, -3, -4, -6, -8, -12};

// 256-entry e4m3fn decode table (built once; NaN encodings mapped to NaN like the header).
inline void nvfp4_cpu_e4m3_table(float* lut256) {
    for (int i = 0; i < 256; ++i) lut256[i] = nvfp4_e4m3_f32((uint32_t) i);
}

#if defined(__aarch64__)

// One weight row dot: wrow = in_dim/2 packed nibbles, srow = in_dim/16 e4m3 scales,
// scale_mul = 0.5f * ws2 (the 0.5 folds E2M1_X2 back to e2m1), x = in_dim fp32.
// in_dim must be a multiple of 16.
inline float nvfp4_cpu_row_dot_neon(const uint8_t* wrow, const uint8_t* srow, float scale_mul,
                                    const float* x, int64_t in_dim, const float* e4m3_lut) {
    const uint8x16_t lut = vreinterpretq_u8_s8(vld1q_s8(NVFP4_E2M1_X2));
    const uint8x8_t msk = vdup_n_u8(0x0f);
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int64_t g = 0; g < in_dim; g += 16) {
        const float s = e4m3_lut[srow[g / NVFP4_GROUP]] * scale_mul;
        const uint8x8_t b = vld1_u8(wrow + g / 2);
        const uint8x8_t lo = vand_u8(b, msk);              // elements 0,2,..,14
        const uint8x8_t hi = vshr_n_u8(b, 4);              // elements 1,3,..,15
        const int8x16_t w = vreinterpretq_s8_u8(vqtbl1q_u8(lut, vcombine_u8(lo, hi)));
        const int16x8_t we16 = vmovl_s8(vget_low_s8(w));   // even elements
        const int16x8_t wo16 = vmovl_s8(vget_high_s8(w));  // odd elements
        const float32x4_t we0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(we16)));
        const float32x4_t we1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(we16)));
        const float32x4_t wo0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wo16)));
        const float32x4_t wo1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wo16)));
        const float32x4x2_t x0 = vld2q_f32(x + g);         // val[0]=k0,2,4,6  val[1]=k1,3,5,7
        const float32x4x2_t x1 = vld2q_f32(x + g + 8);     // val[0]=k8,10,12,14 ...
        float32x4_t part = vmulq_f32(we0, x0.val[0]);
        part = vfmaq_f32(part, wo0, x0.val[1]);
        part = vfmaq_f32(part, we1, x1.val[0]);
        part = vfmaq_f32(part, wo1, x1.val[1]);
        acc = vfmaq_f32(acc, part, vdupq_n_f32(s));
    }
    return vaddvq_f32(acc);
}
#endif

// Scalar fallback (any host); also the NEON path's correctness cross-check.
inline float nvfp4_cpu_row_dot_scalar(const uint8_t* wrow, const uint8_t* srow, float scale_mul,
                                      const float* x, int64_t in_dim, const float* e4m3_lut) {
    float acc = 0.0f;
    for (int64_t g = 0; g < in_dim; g += NVFP4_GROUP) {
        const float s = e4m3_lut[srow[g / NVFP4_GROUP]] * scale_mul;
        float part = 0.0f;
        for (int64_t k = 0; k < NVFP4_GROUP; k += 2) {
            const uint8_t packed = wrow[(g + k) / 2];
            part += (float) NVFP4_E2M1_X2[packed & 15u] * x[g + k] +
                    (float) NVFP4_E2M1_X2[packed >> 4] * x[g + k + 1];
        }
        acc += s * part;
    }
    return acc;
}

using Nvfp4RowDot = float (*)(const uint8_t*, const uint8_t*, float, const float*, int64_t,
                              const float*);

#if defined(__aarch64__)
inline constexpr Nvfp4RowDot NVFP4_CPU_ROW_DOT = &nvfp4_cpu_row_dot_neon;
#else
inline constexpr Nvfp4RowDot NVFP4_CPU_ROW_DOT = &nvfp4_cpu_row_dot_scalar;
#endif

// One expert, one token: out = down . (silu(gate.x) * (up.x)).  `blob` is one canonical cold
// blob in HOST memory, `x` NVFP4_H fp32, `out` NVFP4_H fp32, `h` caller scratch of NVFP4_FF fp32,
// `e4m3_lut` from nvfp4_cpu_e4m3_table.
inline void nvfp4_cpu_expert(const uint8_t* blob, const float* x, float* out, float* h,
                             const float* e4m3_lut) {
    const uint8_t* gate_w = blob + NVFP4_OFF_GATE_W;
    const uint8_t* up_w = blob + NVFP4_OFF_UP_W;
    const uint8_t* down_w = blob + NVFP4_OFF_DOWN_W;
    const uint8_t* gate_s = blob + NVFP4_OFF_GATE_S;
    const uint8_t* up_s = blob + NVFP4_OFF_UP_S;
    const uint8_t* down_s = blob + NVFP4_OFF_DOWN_S;
    float ws2[6];
    std::memcpy(ws2, blob + NVFP4_OFF_GLOBAL, sizeof ws2);
    for (int64_t i = 0; i < NVFP4_FF; ++i) {
        const float g = NVFP4_CPU_ROW_DOT(gate_w + i * (NVFP4_H / 2),
                                          gate_s + i * (NVFP4_H / NVFP4_GROUP), 0.5f * ws2[0], x,
                                          NVFP4_H, e4m3_lut);
        const float u = NVFP4_CPU_ROW_DOT(up_w + i * (NVFP4_H / 2),
                                          up_s + i * (NVFP4_H / NVFP4_GROUP), 0.5f * ws2[2], x,
                                          NVFP4_H, e4m3_lut);
        h[i] = g / (1.0f + std::exp(-g)) * u;   // silu(g) * u
    }
    for (int64_t r = 0; r < NVFP4_H; ++r)
        out[r] = NVFP4_CPU_ROW_DOT(down_w + r * (NVFP4_FF / 2),
                                   down_s + r * (NVFP4_FF / NVFP4_GROUP), 0.5f * ws2[4], h,
                                   NVFP4_FF, e4m3_lut);
}

}  // namespace strata::kernels

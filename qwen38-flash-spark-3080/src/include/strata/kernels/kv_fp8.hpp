// include/strata/kernels/kv_fp8.hpp - FP8 e4m3 KV storage for the QSA layers (`--kv fp8`).
//
// Same cell layout as INT8 (`kv_q8.hpp`): per KV head, groups of 64 values are 64 e4m3 bytes plus one FP16
// scale. sm_86 has no FP8 tensor core, so the conversion is the software path in cuda_fp8.h
// (`__nv_cvt_float_to_fp8` with `__NV_SATFINITE`, `__nv_fp8_e4m3`, `__nv_cvt_fp8_to_halfraw`).
//
//     scale = fp16(max|x| / 448),  code = satfinite_e4m3(x / scale),  x' = float(code_e4m3) * scale
// A stored scale of 0 (the group was all zeros, or flushed) writes zeros and does not divide.
// Byte count matches INT8: 1,056 B per cell (2 heads x 256 x K and V, plus 32 B of scales). Attention reads
// the pools directly as KV_MODE 2 (qsa_decode_attn.cu, qsa_prompt_attn.cu); the gather below is the
// non-fast path's FP16 scratch, same contract as `kv_gather_q8_step`.
#pragma once

#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"

#include <cstdint>

#if defined(__CUDACC__) && !defined(__HIPCC__)
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#endif

namespace strata::kernels {

inline constexpr int KV_FP8_GROUP = 64;
inline constexpr float KV_FP8_EMAX = 448.0f;   // largest finite e4m3 value

/// Bytes per cell (one token, one layer): K and V codes plus their scales. Same arithmetic as
/// `kv_q8_bytes_per_cell` (one byte per value, one fp16 scale per 64).
inline uint64_t kv_fp8_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * s.head_dim * 2 +
           (uint64_t) s.n_head_kv * (s.head_dim / KV_FP8_GROUP) * 2 * 2;
}

#if defined(__CUDACC__) && !defined(__HIPCC__)
/// satfinite e4m3 byte. `x` is already divided by the stored scale.
__device__ __forceinline__ uint8_t kv_fp8_encode(float x) {
    __nv_fp8_e4m3 packed;
    packed.__x = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    return static_cast<uint8_t>(packed.__x);
}

/// float(code_e4m3), before the group's scale multiply.
__device__ __forceinline__ float kv_fp8_decode(uint8_t code) {
    __nv_fp8_e4m3 packed;
    packed.__x = static_cast<__nv_fp8_storage_t>(code);
    const __half_raw hr = __nv_cvt_fp8_to_halfraw(packed.__x, __NV_E4M3);
    return __half2float(__half(hr));
}
#elif defined(__CUDACC__)
// HIP has no cuda_fp8.h. The tensor prompt kernel is not launched there; these exist so the
// instantiation of KV_MODE 2 still compiles. Same bit layout as the CUDA software conversion.
__device__ __forceinline__ float kv_fp8_decode(uint8_t code) {
    const float sign = (code & 0x80) ? -1.f : 1.f;
    const int exp = (code >> 3) & 0xf;
    const int man = code & 7;
    float v;
    if (exp == 0) v = ldexpf((float) man, -9);
    else if (exp == 15 && man == 7) v = nanf("");
    else v = ldexpf(1.f + (float) man * 0.125f, exp - 7);
    return sign * v;
}
__device__ __forceinline__ uint8_t kv_fp8_encode(float x) {
    if (!(x == x)) return 0x7f;
    const int neg = signbit(x) ? 0x80 : 0;
    float a = fabsf(x);
    if (a >= KV_FP8_EMAX) return (uint8_t) (neg | 0x7e);
    uint8_t best = (uint8_t) neg;
    float best_e = a;
    for (int c = 0; c < 127; ++c) {
        const float e = fabsf(kv_fp8_decode((uint8_t) c) - a);
        if (e < best_e) { best_e = e; best = (uint8_t) (neg | c); }
    }
    return best;
}
#endif

/// Append the cell at step[kStepPos] (graph-capturable: position and page come from device memory). With a host
/// copy (KV streaming, `kv_stream.hpp`) the cell is written there too, and to VRAM only if its block is resident.
void kv_append_fp8_step(uint8_t* k_fp8, uint8_t* v_fp8, uint16_t* k_scale, uint16_t* v_scale,
                        const int32_t* page_table, const int32_t* step, const float* kcur, const float* vcur,
                        const QsaShapes& s, void* stream, const KvHostPools* host = nullptr);

/// The prompt path: T consecutive cells from pos0, K/V [T, n_head_kv, head_dim]. Also into `stage` (identity
/// layout) when given. Not capturable.
void kv_append_fp8(uint8_t* k_fp8, uint8_t* v_fp8, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                   int64_t pos0, int64_t T, const float* K, const float* V, const QsaShapes& s, void* stream,
                   const KvHostPools* host = nullptr, const KvHostPools* stage = nullptr);

/// Gather step[kStepWidth] cells named by `ids` into FP16 scratch `[id][kv_head][head_dim]`; the grid is sized by
/// `max_ids` (capacity), the kernel reads the real count from `step`.
void kv_gather_fp8_step(const uint8_t* k_fp8, const uint8_t* v_fp8, const uint16_t* k_scale, const uint16_t* v_scale,
                        const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                        const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream);

}  // namespace strata::kernels

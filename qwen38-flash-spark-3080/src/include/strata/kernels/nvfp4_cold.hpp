// include/strata/kernels/nvfp4_cold.hpp - P3: the Spark cold-expert tier for NVFP4 packs.
//
// **WHAT THIS IS FOR.** The Q2_0 cold pack (experts.bin, cpuk::BLOB=1,382,400 B/expert, q8_0
// activation contract) is replaced on the NVFP4 stack by the P1 cold pack: per-layer files
// `layer-%05d-cold-nvfp4.bin`, each 512 sequential canonical v2 blobs of NVFP4_COLD_BLOB_BYTES
// (2,765,056 B).  This header pins the cold blob layout (the C++ restatement of
// tools/nvfp4_pack.py's build_blob for flavor "cold"), the e2m1/e4m3 decode shared by the GPU
// kernels and the host reference, the host scalar reference itself (the selftest's yardstick - a
// kernel checked against another copy of itself proves nothing), and the grouped entry points the
// cold worker launches.
//
// **SCALE SEMANTICS (the one thing that must match sglang).**  A cold blob's scales are RAW:
//     weight[r,k] = e2m1(nibble) * e4m3(scale[r][k/GROUP]) * weight_scale_2 (fp32, per projection)
// No process_scales, no permutation - that is the hot flavor's marlin path and does not apply
// here.  input_scale rides along in the blob but is unused (W4A16: activations stay floating).
// This formula is what the P3 oracle (tests/nvfp4_coldpath_ref.py) cross-checks against sglang's
// own dequantize_nvfp4.
//
// **COMPUTE PATH (baseline, correctness first).**  The grouped launch shape mirrors
// moe_grouped_s2 (grp_ptr/grp_start/ent_dst/ent_tok, counts on device) so the worker's group
// assembly is untouched; the per-group compute reads fp32 activations and dequantizes weights
// in-kernel (registers, fp32 FMA) - no materialized BF16 stage, no q8 re-quantization of the
// intermediate.  cutlass FP4 is the P3-latter evaluation item, not this baseline.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define NVFP4_HD __host__ __device__
#else
#define NVFP4_HD
#endif

namespace strata::kernels {

// ---- geometry (NVFP4-PORT-PLAN.md section 0; validated against the pack tool) -----------------
inline constexpr int NVFP4_H = 2560;         // n_embd
inline constexpr int NVFP4_FF = 640;         // expert intermediate width
inline constexpr int NVFP4_GROUP = 16;       // elements per fp8 scale

// ---- canonical v2 cold blob layout (tools/nvfp4_pack.py: 64 B header + 6 planes + 24 B of
// ---- fp32 globals, zero-padded to a 256 B multiple) -------------------------------------------
inline constexpr int64_t NVFP4_HEADER = 64;
inline constexpr int64_t NVFP4_W_GU_BYTES = (int64_t) NVFP4_FF * NVFP4_H / 2;   // 819,200: gate or up codes
inline constexpr int64_t NVFP4_W_D_BYTES = (int64_t) NVFP4_H * NVFP4_FF / 2;    // 819,200: down codes
inline constexpr int64_t NVFP4_S_GU_BYTES = (int64_t) NVFP4_FF * (NVFP4_H / NVFP4_GROUP);   // 102,400
inline constexpr int64_t NVFP4_S_D_BYTES = (int64_t) NVFP4_H * (NVFP4_FF / NVFP4_GROUP);    // 102,400
inline constexpr int64_t NVFP4_OFF_GATE_W = NVFP4_HEADER;
inline constexpr int64_t NVFP4_OFF_UP_W = NVFP4_OFF_GATE_W + NVFP4_W_GU_BYTES;
inline constexpr int64_t NVFP4_OFF_DOWN_W = NVFP4_OFF_UP_W + NVFP4_W_GU_BYTES;
inline constexpr int64_t NVFP4_OFF_GATE_S = NVFP4_OFF_DOWN_W + NVFP4_W_D_BYTES;
inline constexpr int64_t NVFP4_OFF_UP_S = NVFP4_OFF_GATE_S + NVFP4_S_GU_BYTES;
inline constexpr int64_t NVFP4_OFF_DOWN_S = NVFP4_OFF_UP_S + NVFP4_S_GU_BYTES;
inline constexpr int64_t NVFP4_OFF_GLOBAL = NVFP4_OFF_DOWN_S + NVFP4_S_D_BYTES;             // 2,764,864
// global[6] fp32, PROJ_ORDER = gate, up, down:
//   {gate_ws2, gate_input_scale, up_ws2, up_input_scale, down_ws2, down_input_scale}
inline constexpr int64_t NVFP4_COLD_BLOB_BYTES = 2765056;   // == pack tool's canonical_size()

// ---- scalar decode (one definition, host and device) ------------------------------------------

// e2m1 (1 sign, 2 exp bias 1, 1 mantissa): magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}.
NVFP4_HD inline float nvfp4_e2m1_f32(uint32_t nib) {
    const uint32_t e = (nib >> 1) & 3u, m = nib & 1u;
    const float v = (e == 0u) ? 0.5f * (float) m
                              : (1.0f + 0.5f * (float) m) * (float) (1u << (e - 1u));
    return (nib & 8u) ? -v : v;
}

// e4m3fn (1 sign, 4 exp bias 7, 3 mantissa; 0x7F/0xFF are NaN - never present in scales,
// mapped anyway so a corrupt blob poisons the output instead of silently computing).
NVFP4_HD inline float nvfp4_e4m3_f32(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    float v;
    if (e == 0u) v = 0x1p-9f * (float) m;                        // subnormal: m/8 * 2^-6
    else if (e == 15u && m == 7u) v = 0.0f / 0.0f;               // NaN encoding
    else v = (float) (8u + m) * (float) (1u << (e >= 10u ? e - 10u : 0u)) /
             (float) (1u << (e >= 10u ? 0u : 10u - e));          // (1+m/8) * 2^(e-7)
    return (b & 128u) ? -v : v;
}

// ---- host scalar reference (selftest yardstick; fp64 accumulation, libm exp) ------------------
// One expert, one token: out = down . (silu(gate.x) * (up.x)).  `blob` is one canonical cold
// blob (host memory), `x` NVFP4_H floats, `out` NVFP4_H floats.  Row-major checkpoint order:
// weight plane row r is output row r, byte (r, k/2) holds elements k (low nibble) and k+1.
namespace nvfp4_cold_detail {
inline double ref_dot(const uint8_t* wplane, const uint8_t* splane, float ws2, int64_t out_row,
                      int64_t in_dim, const float* x) {
    const uint8_t* wrow = wplane + out_row * (in_dim / 2);
    const uint8_t* srow = splane + out_row * (in_dim / NVFP4_GROUP);
    double acc = 0.0;
    for (int64_t k = 0; k < in_dim; k += 2) {
        const uint8_t packed = wrow[k / 2];
        const double scale = (double) nvfp4_e4m3_f32(srow[k / NVFP4_GROUP]) * (double) ws2;
        acc += (double) nvfp4_e2m1_f32(packed & 15u) * scale * (double) x[k];
        acc += (double) nvfp4_e2m1_f32(packed >> 4) * scale * (double) x[k + 1];
    }
    return acc;
}
}  // namespace nvfp4_cold_detail

inline void nvfp4_cold_expert_host_ref(const uint8_t* blob, const float* x, float* out) {
    const uint8_t* gate_w = blob + NVFP4_OFF_GATE_W;
    const uint8_t* up_w = blob + NVFP4_OFF_UP_W;
    const uint8_t* down_w = blob + NVFP4_OFF_DOWN_W;
    const uint8_t* gate_s = blob + NVFP4_OFF_GATE_S;
    const uint8_t* up_s = blob + NVFP4_OFF_UP_S;
    const uint8_t* down_s = blob + NVFP4_OFF_DOWN_S;
    float ws2[6];
    std::memcpy(ws2, blob + NVFP4_OFF_GLOBAL, sizeof ws2);
    double h[NVFP4_FF];
    for (int64_t i = 0; i < NVFP4_FF; ++i) {
        const double g = nvfp4_cold_detail::ref_dot(gate_w, gate_s, ws2[0], i, NVFP4_H, x);
        const double u = nvfp4_cold_detail::ref_dot(up_w, up_s, ws2[2], i, NVFP4_H, x);
        h[i] = g / (1.0 + std::exp(-g)) * u;
    }
    float hf[NVFP4_FF];
    for (int64_t i = 0; i < NVFP4_FF; ++i) hf[i] = (float) h[i];
    for (int64_t r = 0; r < NVFP4_H; ++r)
        out[r] = (float) nvfp4_cold_detail::ref_dot(down_w, down_s, ws2[4], r, NVFP4_FF, hf);
}

// ---- device entry points (src/kernels/cuda/nvfp4_cold_grouped.cu) ------------------------------

/// Scratch layout: gate/up activations (cap_entries x 2*FF fp32) then the swiglu result
/// (cap_entries x FF fp32), 256 B aligned.  Caller-owned; no allocation on the request path.
uint64_t moe_grouped_nvfp4_scratch_bytes(int64_t cap_entries);

/// **GROUPED COLD EXPERTS, NVFP4 BASELINE.**  Same group/entry contract as moe_grouped_s2:
/// group g's blob is at device address grp_ptr[g] (a cold blob in the arena), its entries are
/// [grp_start[g], grp_start[g+1]); entry e reads token ent_tok[e]'s fp32 row of `x`
/// (n_tok x NVFP4_H) and writes NVFP4_H floats to out + ent_dst[e] * NVFP4_H.  Group and entry
/// counts are read from `n_groups` (device: [0]=groups, [1]=entries).  Any group size works;
/// entries are served in chunks of 8 per weight pass.
void moe_grouped_nvfp4(const unsigned long long* grp_ptr, const int32_t* grp_start,
                       const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok,
                       int64_t cap_groups, int64_t cap_entries, const float* x, void* scratch,
                       float* out, void* stream);

/// Slim-wire adapter: the engine's bind-v2 push carries quantize_q8_0_scaled output
/// (34 B block_q8_0 + a parallel fp32 scale per 32 elements).  The Q2 path consumes the codes
/// directly; the NVFP4 path needs fp32 activations, and x[i] = qs[i] * scales[i/32] is EXACTLY
/// the activation value the Q2 kernels multiply against (the codes' fp16 block d is bypassed the
/// same way x_scales bypasses it there).  `blocks` is (n/32) x 34 B, `scales` (n/32) fp32,
/// `x` n fp32; n must be a multiple of 32.
void nvfp4_q8_act_dequant(const uint8_t* blocks, const float* scales, float* x, int64_t n,
                          void* stream);

}  // namespace strata::kernels

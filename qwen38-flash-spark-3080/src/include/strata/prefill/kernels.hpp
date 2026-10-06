// include/strata/prefill/kernels.hpp - plan v0.3 P5: the element-wise and stateful kernels of a prompt chunk.
//
// T tokens at a time, all row-major with the token as the slow index.  The arithmetic mirrors the per-token native
// kernels (the decode path), batched; where a step is a recurrence in time (the GDN conv and state, the PLE conv)
// the kernel walks the chunk's tokens in order inside one launch.
#pragma once

#include "strata/kernels/kv_stream.hpp"

#include <cstdint>

namespace strata::prefill {

// ---- hyper-connection (n_embd 2560, hc 4, hc_lr 320)
/// xn[t, c*2560 + d] = R[t,c,d] * rsqrt(mean_d R[t,c,:]^2 + eps) * w_norm[c*2560 + d]; also its BF16 image.
void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream);
/// F-1: gr_norm without its FP32 output: the row scales rs[t*4 + c] and the BF16 image; gr_mix_r then reads R.
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream);
/// gr_mix with xn recomputed from R, rs and w_norm exactly as gr_norm computes it (the same bits).
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h = nullptr);
/// F-2: gr_write, then gr_norm_rs of the next half (its norm weights) over the rows just written - the same bits as
/// the two calls, without reading R back.
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream);
/// lo16[t, k] = bf16(silu(lo[t, k] / hc))
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream);
/// mixed[t, d] = mean_c xn[t, c, d] * sigmoid(gated[t, c, d]); FP32, BF16 and FP16 (either image may be null).
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h = nullptr);
/// R[t, c, d] += bo[t, d] * 2 sigmoid(inj[t, c] / hc)   (inj has row stride inj_ld)
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream);
/// R[t, c, :] = e[t, :] for all four streams (the embedding broadcast).
void gr_broadcast(const float* e, float* R, int64_t T, void* stream);

// ---- GDN (state 128, 16 k heads, 48 v heads, 10240 conv channels, 4 taps)
/// gate[t,h] = softplus(ab[t,h] + dt[h]) * ssm_a[h];  beta[t,h] = sigmoid(ab[t, 48 + h])  (ab: [T, 96])
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream);
/// The 4-tap causal conv + SiLU over the chunk (history [C][3] in, updated to the chunk's last three inputs), then
/// the L2 norm of the q and k heads of every token.  h: [T, C].
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream);
/// The recurrence over the chunk, block per value head, state in registers; y[t] = rmsnorm(o) * gamma * sigmoid(z)
/// (FP32 and FP16 bits: the out projection is quantized).
void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream);

// ---- MoE
/// softmax over 512, top-10 (ties to the lower id), weights renormalised over the ten (the native router).
void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream);
/// Expert blob (Strata pack layout, Q2_0) -> BF16 matrices: gate/up interleaved [1280, 2560], down [2560, 640].
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream);
/// h16[n, r] = fp16(silu(gu[n, 2r]) * gu[n, 2r + 1])   (the interleaved expert gate/up)
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream);
/// h16[n, r] = fp16(silu(g[n, r]) * u[n, r])   (the shared expert, gate and up separate, width 640)
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream);
/// dst[i] = src[i] for n int32s, as a kernel: either side may be mapped host memory, and the copy never waits
/// behind the copy engine's queue (the prompt path's grouping tables, while the expert stream fills it).
void copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream);
/// Gather rows: dst16[i, :] = x16[src[i], :] (n rows of `width` BF16).
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream);
/// Scatter rows: dst[dst_idx[i], :] = src[i, :] (n rows of `width` FP32, width a multiple of 4).  `src` may be
/// MAPPED host memory (the RDMA cold-expert tier's result block: the peer NIC wrote it, the flag was observed
/// on the host before this launch).
void scatter_rows_f32(float* dst, const float* src, const int32_t* dst_idx, int64_t n, int64_t width,
                      void* stream);
/// The f16 twin of scatter_rows_f32 (the RDMA tier's slim wire: the peer writes f16 rows, the conversion to
/// f32 happens on load here).  Same mapped-host allowance, width a multiple of 4.
void scatter_rows_f16(float* dst, const uint16_t* src, const int32_t* dst_idx, int64_t n, int64_t width,
                      void* stream);
/// bo[t, :] = shared[t, :] * sigmoid(sg[t]) + sum_k w[t, k] * D[slot[t, k], :]
void moe_combine(const float* D, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream);

// ---- QSA helpers
/// In place: x[r, :] = x[r, :] * rsqrt(mean x^2 + eps) * w  over rows of `cols` (row stride `ld`).
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream);
/// NEOX rotary on the first 64 dims of each head: x [T, heads, dim] at positions pos0 + t.
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, float freq_base, void* stream);
/// q_full [T, 24, 512] (q | gate per head) -> q [T, 24, 256]
void split_q(const float* q_full, float* q, int64_t T, void* stream);
/// attn[t, h, d] *= sigmoid(q_full[t, h, 256 + d]) -> out16 (fp16 bits: the o-projection is quantized)
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream);

/// K and V of T consecutive cells (positions pos0..pos0+T-1; K normed and rotated) into the paged pools: FP16
/// (`k_pool`/`v_pool`) or INT8 codes + FP16 scale per 64 (`k_q`...), the decode append's arithmetic.
/// K, V: [T, 2, 256].
void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, const strata::kernels::KvHostPools* host = nullptr,
               const strata::kernels::KvHostPools* stage = nullptr);

/// fp32 -> fp16 bits and fp32 -> bf16, n elements (the two activation images of the prompt GEMMs).
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream);
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream);
/// y = fp32(fp16(x)): what an FP16 store of x would read back (the FP16 indexer-key experiment)
void round_f16(const float* x, float* y, int64_t n, void* stream);
/// Expert blob -> FP16 (Q2_0 values are exact in FP16).
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream);

}  // namespace strata::prefill

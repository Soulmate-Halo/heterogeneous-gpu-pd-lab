// include/strata/kernels/nvfp4_marlin_expert.hpp - P2: NVFP4 hot experts on the marlin W4A16 kernel.
//
// **WHAT THIS IS.** The VRAM expert tier's second weight format.  `moe_hit_grouped_s2` computes a layer's
// resident experts from Q2_0 blobs; this file computes them from NVFP4 blobs (the `pack-hot` flavor of
// `tools/nvfp4_pack.py`) through the marlin MoE GEMM extracted to `third_party/marlin` (P2前半, bit-exact
// against the production JIT at evidence/marlin-extract-p2-20261002.json).  The Q2_0 path is untouched:
// callers select this path by handing the driver a `Nvfp4HotArena`, and every entry point below is new code.
//
// **THE BLOB IS THE PACK'S, NOT THE KERNEL'S.**  One hot blob is 2,765,056 bytes: a 64-byte `STRNVFP4`
// header, four marlin-ready planes (fused w13 weight, w2 weight, w13 scale, w2 scale - fused means the
// marlin tile permutation spans the whole 1280-wide w13, so the planes must NOT be split per projection),
// and a 24-byte tail of six raw f32 global scalars (gate/up/down weight_scale_2 + input_scale, in that
// interleave).  Marlin requires each weight/scale array contiguous across experts (`is_contiguous()` is a
// hard RuntimeCheck), so the VRAM tier stores the four planes in four PLANAR arenas - all slots' w13 back to
// back, then all w2, and so on (`ExpertCache::open_planar`).  The header and tail are never copied to the
// device: the header is validated on the host at fill time, and the tail's two scalars the kernel needs are
// folded into the per-slot `gm13`/`gm2` arrays by `nvfp4_process_globals`.
//
// **GLOBAL-SCALE RECIPE, VERIFIED AGAINST sglang.**  The kernel reads `scale2_ptr[expert_id]` as one bf16
// and multiplies it into the dequant.  sglang's `nvfp4_marlin_process_global_scale` computes
// `bf16(g) * 2**119` (target bf16 exponent bias 126, FP8 group-scale bias 7).  Multiplying by a power of two
// only adds to the exponent, so the host fold is: round g to bf16 (RNE), then add 119 to its exponent field.
// The fused w13 has ONE global per expert, which is only lossless when gate and up share weight_scale_2 -
// the oracle stack refuses the collapse when they differ (`worker/kernel.py:61`), and so does
// `nvfp4_process_globals`.  input_scale is NOT part of the marlin path (P2前半's golden case had none and
// matched the production JIT bitwise); it rides the blob for the cold flavor's cutlass path.
//
// **ROUTING, AND WHY IT IS TRIVIAL HERE.**  The kernel consumes vLLM-style MoE routing: `sorted_token_ids`
// (block-aligned, values are (token*top_k + k) pair indices, padding = size_m*top_k), one `expert_ids` entry
// per block, and `num_tokens_post_padded`.  Strata's hits are DISTINCT experts, one token each, so block b
// holds exactly hit b: `sorted[b*BM] = b`, the rest padding, `expert_ids[b] = slot[b]`, and the whole thing
// is filled by one tiny device kernel from the same (slot, dst, count) list `moe_hit_grouped_s2_dev` takes -
// no host readback, so the sequence is CUDA-graph capturable.  `topk_weights` is a constant 1.0 array
// (fp32 - the kernel reads it as `const float*`): the router weight is applied later by `moe_combine`, as on
// the Q2_0 path.  The activation is the layer's f32 `mixed` converted straight to bf16 - marlin is a W4A16
// kernel, there is no q8_0 activation contract in this path.
//
// **THE COUNT-OF-ZERO CASE IS FREE.**  `parallel = num_post / BM = 0` gives `iters = 0` and every block
// returns before touching memory (marlin_template.h:567-569), so a layer with no hits launches the same
// captured nodes and does nothing.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata::kernels {

// ---- geometry, fixed by the artifact and shared with the Q2_0 path ----
inline constexpr int NV4_H = 2560;          // n_embd
inline constexpr int NV4_FF = 640;          // expert intermediate width
inline constexpr int NV4_GROUP = 16;        // NVFP4 group size (modelopt W4A16)
inline constexpr int NV4_BLOCKM = 16;       // marlin MoE block_m (P2前半's proven config)

// ---- the hot blob's canonical v2 layout (tools/nvfp4_pack.py: HEADER_BYTES/ALIGN/GLOBAL_BYTES) ----
inline constexpr size_t NV4_HEADER = 64;
inline constexpr size_t NV4_W13_BYTES = 1638400;    // 2*FF x H, marlin repacked (u8 view of int32 plane)
inline constexpr size_t NV4_W2_BYTES = 819200;      // H x FF, marlin repacked
inline constexpr size_t NV4_W13S_BYTES = 204800;    // (H/16) x 2*FF processed F8 scales
inline constexpr size_t NV4_W2S_BYTES = 102400;     // (FF/16) x H processed F8 scales
inline constexpr size_t NV4_GLOBALS = 24;           // 6 x f32: gate/up/down ws2 + input_scale interleaved
inline constexpr size_t NV4_O_W13 = NV4_HEADER;
inline constexpr size_t NV4_O_W2 = NV4_O_W13 + NV4_W13_BYTES;
inline constexpr size_t NV4_O_W13S = NV4_O_W2 + NV4_W2_BYTES;
inline constexpr size_t NV4_O_W2S = NV4_O_W13S + NV4_W13S_BYTES;
inline constexpr size_t NV4_O_GLOBALS = NV4_O_W2S + NV4_W2S_BYTES;      // 2,764,864
inline constexpr size_t NV4_BLOB = 2765056;                             // 256 B aligned total
static_assert(NV4_O_GLOBALS + NV4_GLOBALS <= NV4_BLOB && NV4_BLOB % 256 == 0, "blob v2 geometry");

/// Slices one hot blob into the four marlin plane pointers+sizes, after validating the v2 header (magic,
/// version 2, flavor hot, plane geometry).  `layer`/`expert` are cross-checked against the header when
/// non-negative, so a fill aimed at the wrong expert refuses rather than caching a plausible wrong answer.
/// Host-side; `blob` must point at NV4_BLOB readable bytes.
bool nvfp4_hot_plane_views(const uint8_t* blob, const uint8_t* planes[4], size_t sizes[4], int64_t layer = -1,
                           int64_t expert = -1);

/// The 24-byte globals tail of a hot blob (valid only after `nvfp4_hot_plane_views` accepted the blob).
inline const uint8_t* nvfp4_blob_globals(const uint8_t* blob) { return blob + NV4_O_GLOBALS; }

/// Folds the blob tail into the kernel's per-slot globals.  `gm13` = bf16(gate_ws2) * 2^119 (requires
/// gate_ws2 == up_ws2 bitwise, the oracle stack's own refusal), `gm2` = bf16(down_ws2) * 2^119, both
/// returned as bf16 bit patterns.  Returns false on the scale mismatch or a non-finite/out-of-range value.
/// Host-side, exact: rounding to bf16 then adding 119 to the exponent is bit-identical to sglang's
/// `bf16(g) * 2.0**119` because the multiplier is a power of two.
bool nvfp4_process_globals(const uint8_t* tail24, uint16_t& gm13, uint16_t& gm2);

/// The device-side weight arrays the kernel indexes by slot (= marlin expert id).  All four plane arenas
/// are `n_slots x plane_bytes` contiguous; the gm arrays are `n_slots` bf16.  Produced by the caller from
/// the planar ExpertCache plus its gm staging.
struct Nvfp4HotArena {
    const uint8_t* w13 = nullptr;
    const uint8_t* w2 = nullptr;
    const uint8_t* w13s = nullptr;
    const uint8_t* w2s = nullptr;
    const uint16_t* gm13 = nullptr;
    const uint16_t* gm2 = nullptr;
    int64_t n_slots = 0;
    bool on() const { return w13 && w2 && w13s && w2s && gm13 && gm2 && n_slots > 0; }
};

/// Bytes of caller-owned scratch `moe_hit_grouped_nvfp4*` needs for `cap` hits.  Carved, never allocated
/// on the token path (P2.T10), same rule as `moe_hit_grouped_scratch_bytes`.
uint64_t moe_hit_nvfp4_scratch_bytes(int64_t cap, int sms);

/// One-time scratch initialisation, OUTSIDE any graph capture: zeroes the marlin locks, fills topk_weights
/// with 1.0f, and clears the count cells.  The fp32-reduce protocol returns its locks to zero after every
/// launch (P2前半 ran the dirty-workspace case bit-exact), and c_tmp is written before it is read within
/// every launch, so neither needs re-clearing per layer.
void moe_hit_nvfp4_scratch_init(void* scratch, int64_t cap, int sms, void* stream);

/// **THE GRAPH-PATH ENTRY, drop-in for `moe_hit_grouped_s2_dev` on an NVFP4 tier.**  For each hit h in
/// `[0, *d_count)` (device count, `cap` the captured maximum): expert `arena` slot `d_slot[h]` against the
/// f32 activation `x_f` (NV4_H), result scattered as f32 to `hit_out + d_dst[h] * NV4_H` - the exact
/// write pattern `moe_hit_add` consumes, and bitwise the same routing contract as the s2 dev kernel.
/// Sequence per call: x->bf16 convert, routing-prep kernel, marlin w13 GEMM, silu*up, marlin w2 GEMM
/// (x per-expert global scales), scatter epilogue.  Capture-safe: no allocation, no sync, no host read of
/// device memory; the marlin host entry launches on `stream` via TVMFFIEnvSetStream (set and restored).
void moe_hit_grouped_nvfp4_dev(const Nvfp4HotArena& arena, const int32_t* d_slot, const int32_t* d_dst,
                               const int32_t* d_count, int64_t cap, const float* x_f, void* scratch,
                               float* hit_out, void* stream);

/// The host-count variant, drop-in for `moe_hit_grouped_s2` (the `expert_hit_run` path).  `n_hits <= cap`;
/// the count is uploaded into the scratch's own cell, after which the device sequence is identical.
void moe_hit_grouped_nvfp4(const Nvfp4HotArena& arena, const int32_t* d_slot, const int32_t* d_dst,
                           int64_t n_hits, int64_t cap, const float* x_f, void* scratch, float* hit_out,
                           void* stream);

// Batched prefill: host arrays contain grouped expert slots, source token rows and Dm destinations.
// At most 1024 real rows per call; padding is generated per expert at the marlin tile boundary.
// Scratch is separate from the decode graph. Router weights remain applied by moe_combine.
inline constexpr int NV4_PREFILL_ROWS = 1024;
uint64_t moe_prefill_nvfp4_scratch_bytes(int sms);
void moe_prefill_grouped_nvfp4(const Nvfp4HotArena& arena, const int32_t* h_slot,
                              const int32_t* h_dst, const int32_t* h_tok, int n_rows,
                              const float* x_f, void* scratch, float* out, void* stream);

/// **THE DECODE-PUSH STAGED PAIR, for callers that capture the GPU sequence into a CUDA graph.**  The
/// batch entry above expands the (slot,dst,tok) rows into per-block routing on the fly from pageable host
/// memory, which stream capture forbids.  `nvfp4_push_expand` does that expansion into process-persistent
/// PINNED staging (same values, same order - numerics are untouched) and returns num_post, or -1 if the
/// pinned staging could not be allocated or `cap` is outside 1..128; the caller must then treat the staged
/// path as unavailable and fall back to `moe_prefill_grouped_nvfp4`.  `nvfp4_push_issue_staged` issues the
/// exact GPU sequence the batch entry would (H2D from the pinned staging, gather, the two marlin GEMMs,
/// silu, scatter) with no host-memory reads, so it is legal inside cudaStreamBeginCapture/EndCapture.
/// NOT thread-safe: the expansion staging is shared process state, so expand+issue (or expand+capture)
/// must run on one thread at a time - the cold-expert doorbell's single thread is the intended caller.
int32_t nvfp4_push_expand(const int32_t* h_slot, const int32_t* h_dst, const int32_t* h_tok, int32_t cap);
void nvfp4_push_issue_staged(const Nvfp4HotArena& arena, int64_t cap, const float* x_f, void* scratch,
                             float* out, void* stream);

/// Small-M weight-streaming GEMM used by `run()` when the call is the batch path and `rows`(=cap)<=8.
/// `a_bf16` is [rows, size_k], `c_bf16` is [rows, size_n], both row-major. `w` / `scales` are the marlin
/// planes (`w` int32 tiles, `scales` processed fp8). `sorted` is `nblk * 16` row ids (padding >= rows),
/// `experts` is one slot per block. `work` holds `work_floats` fp32 partials (the marlin c_tmp region).
/// `mul_topk == 0` multiplies the per-expert global bf16 scale; `mul_topk != 0` multiplies
/// global_scale * bf16(topk[row]), matching moe_wna16_marlin_gemm. No host branch on device data.
void nvfp4_stream_expert_gemm(const void* a_bf16, void* c_bf16, const uint8_t* w, const uint8_t* scales,
                              const uint16_t* global_scale, const int32_t* sorted, const int32_t* experts,
                              const float* topk, float* work, int64_t work_floats, int rows, int nblk,
                              int size_n, int size_k, int mul_topk, void* stream);

}  // namespace strata::kernels
